from __future__ import annotations

import datetime

import numpy as np
import polars as pl
from scipy.optimize import linprog

from dgup._internal.constants import (
    COL_DATE,
    COL_INVENTORY,
    FORECAST_HORIZON,
    MAX_INJECTION,
    MAX_WITHDRAWAL,
    MONTHLY_MAX,
    MONTHLY_MIN,
    STORAGE_CAPACITY,
)

# Quantile column names produced by the forecast pipeline
Q25 = "q25"
Q50 = "q50"
Q75 = "q75"

# Months where inventory-too-low is the binding risk (high EOM minimum)
_MONTHS_HIGH_MIN = {1, 2, 11, 12}
# Months where inventory-too-high is the binding risk (low EOM maximum)
_MONTHS_LOW_MAX = {3, 4}


def select_active_quantile_v1(month: int) -> str:
    """Return which forecast quantile the optimizer should use for a given month.

    The selection is driven by the asymmetry of the monthly penalty band:

    - Months with a high EOM inventory minimum (Jan, Feb, Nov, Dec): use Q75
      so the optimizer plans for higher-than-expected usage and builds a
      larger delivery buffer to protect the inventory floor.
    - Months with a very low EOM inventory maximum (Mar, Apr): use Q25 so
      the optimizer avoids over-delivering and over-filling storage.
    - All other months: use Q50 (median).

    Args:
        month: Integer month (1=Jan … 12=Dec).

    Returns:
        One of ``"q25"``, ``"q50"``, or ``"q75"``.
    """
    if month in _MONTHS_HIGH_MIN:
        return Q75
    if month in _MONTHS_LOW_MAX:
        return Q25
    return Q50


def optimize_week_lp_v1(
    forecast_7day: pl.DataFrame,
    inventory_t3: float,
    date_t: datetime.date,
) -> pl.DataFrame:
    """Plan 7 daily deliveries using a linear programme over the forecast window.

    The LP minimises total delivery subject to:

    * Daily injection limit per month
    * Daily withdrawal limit per month
    * Non-negative inventory every day
    * Non-negative delivery every day
    * Soft end-of-month inventory band (slack variables on both bounds; penalised
      heavily in the objective so the solver finds the least-bad solution if the
      hard constraints are infeasible)

    The month-aware quantile is selected via ``select_active_quantile_v1`` for
    each day individually (the 7-day window may cross a month boundary).

    Args:
        forecast_7day: DataFrame with columns ``Date``, ``q25``, ``q50``, ``q75``
            covering the 7 days starting at ``date_t``. Must have exactly 7 rows.
        inventory_t3: Last confirmed inventory level (therms), from day t-3.
        date_t: The first day of the planning window (nomination day = t-1, gas
            day = t).

    Returns:
        7-row DataFrame with columns: Date, optimized_delivery,
        projected_inventory, active_quantile.
    """
    if len(forecast_7day) != FORECAST_HORIZON:
        msg = f"forecast_7day must have {FORECAST_HORIZON} rows, got {len(forecast_7day)}"
        raise ValueError(msg)

    forecast_7day = forecast_7day.sort(COL_DATE)
    dates = forecast_7day[COL_DATE].to_list()
    months = [d.month for d in dates]

    # Select active quantile per day
    active_quantiles = [select_active_quantile_v1(m) for m in months]
    usage_forecast = np.array([
        forecast_7day[q][i] for i, q in enumerate(active_quantiles)
    ], dtype=float)

    # Build per-day activity limits
    max_inj = np.array([MAX_INJECTION[m] for m in months], dtype=float)
    max_wit = np.array([MAX_WITHDRAWAL[m] for m in months], dtype=float)

    # ------------------------------------------------------------------
    # LP formulation
    # Variables: x = [delivery_0..delivery_6, slack_lo_j, slack_hi_j]
    #   where j indexes end-of-month days within the window.
    #
    # inventory[t] = inventory_t3 + sum_{k=0}^{t} (delivery[k] - usage[k])
    #              = inventory_t3 - cumsum(usage)[t] + cumsum(delivery)[t]
    # ------------------------------------------------------------------

    n = FORECAST_HORIZON
    cum_usage = np.cumsum(usage_forecast)

    # Identify end-of-month indices (last occurrence of each month in window)
    eom_indices: list[int] = []
    for i in range(n - 1, -1, -1):
        m = months[i]
        if i == n - 1 or months[i + 1] != m:
            eom_indices.append(i)
    eom_indices = sorted(eom_indices)
    n_eom = len(eom_indices)

    # Total variable count: n deliveries + 2 * n_eom slack variables
    n_vars = n + 2 * n_eom
    PENALTY = 1e6  # large penalty for violating EOM band

    # Objective: minimise total delivery + penalise EOM slack
    c = np.zeros(n_vars)
    c[:n] = 1.0  # minimise sum of deliveries
    c[n : n + n_eom] = PENALTY  # slack_lo penalty (inventory below min)
    c[n + n_eom :] = PENALTY  # slack_hi penalty (inventory above max)

    # ------------------------------------------------------------------
    # Inequality constraints A_ub @ x <= b_ub
    # ------------------------------------------------------------------
    ineq_rows: list[np.ndarray] = []
    ineq_rhs: list[float] = []

    # Cumulative sum matrix: C[t, k] = 1 if k <= t else 0
    C = np.tril(np.ones((n, n)))  # C[t] @ delivery = cumulative delivery through t

    for t in range(n):
        inv_base = inventory_t3 - cum_usage[t]  # inventory if all deliveries were 0

        # injection[t] = delivery[t] - usage[t] <= MAX_INJECTION[months[t]]
        # equivalently: delivery[t] <= usage[t] + max_inj[t]
        row = np.zeros(n_vars)
        row[t] = 1.0
        ineq_rows.append(row)
        ineq_rhs.append(usage_forecast[t] + max_inj[t])

        # withdrawal[t] = usage[t] - delivery[t] <= MAX_WITHDRAWAL[months[t]]
        # equivalently: -delivery[t] <= -usage[t] + max_wit[t]
        row = np.zeros(n_vars)
        row[t] = -1.0
        ineq_rows.append(row)
        ineq_rhs.append(-usage_forecast[t] + max_wit[t])

        # inventory[t] >= 0  →  -cumsum(delivery)[:t+1] <= inv_base
        row = np.zeros(n_vars)
        row[:t + 1] = -1.0
        ineq_rows.append(row)
        ineq_rhs.append(inv_base)

    # EOM band constraints (soft — with slack variables)
    for j, eom_idx in enumerate(eom_indices):
        eom_month = months[eom_idx]
        inv_base_eom = inventory_t3 - cum_usage[eom_idx]

        # inventory at eom_idx = inv_base_eom + C[eom_idx] @ delivery
        # Must be >= MONTHLY_MIN: -C @ delivery + slack_lo >= -inv_base - MONTHLY_MIN
        # → C @ delivery - slack_lo <= inv_base + inv_base_eom - MONTHLY_MIN ... rearranged:
        # Soft lower: inv_base_eom + C[eom_idx]@delivery + slack_lo[j] >= MONTHLY_MIN
        # → -C[eom_idx]@delivery - slack_lo[j] <= -MONTHLY_MIN + inv_base_eom (but we flip)
        # For linprog, all ineq are <=:
        # Lower bound violation: MONTHLY_MIN - (inv_base_eom + C[eom_idx]@delivery) <= slack_lo[j]
        # → -C[eom_idx]@delivery - slack_lo[j] <= -MONTHLY_MIN + inv_base_eom  (×-1 on both sides first)
        # Actually more clearly:
        # inventory >= MONTHLY_MIN - slack_lo  → -inventory + (-slack_lo) <= -MONTHLY_MIN
        row_lo = np.zeros(n_vars)
        row_lo[: n] = -C[eom_idx]
        row_lo[n + j] = -1.0  # -slack_lo
        ineq_rows.append(row_lo)
        ineq_rhs.append(-MONTHLY_MIN[eom_month] + inv_base_eom)

        # Upper bound: inventory <= MONTHLY_MAX + slack_hi
        # → C[eom_idx]@delivery - slack_hi <= MONTHLY_MAX - inv_base_eom
        row_hi = np.zeros(n_vars)
        row_hi[:n] = C[eom_idx]
        row_hi[n + n_eom + j] = -1.0  # -slack_hi
        ineq_rows.append(row_hi)
        ineq_rhs.append(MONTHLY_MAX[eom_month] - inv_base_eom)

    A_ub = np.array(ineq_rows)
    b_ub = np.array(ineq_rhs)

    # Bounds: delivery >= 0; slacks >= 0
    bounds = [(0.0, None)] * n + [(0.0, None)] * (2 * n_eom)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        deliveries = result.x[:n]
    else:
        # Fallback: match usage exactly (net_flow = 0), clipped to activity limits
        deliveries = np.clip(usage_forecast, usage_forecast - max_wit, usage_forecast + max_inj)

    # Compute projected inventory
    projected_inv = inventory_t3 + np.cumsum(deliveries - usage_forecast)

    return pl.DataFrame(
        {
            COL_DATE: dates,
            "optimized_delivery": deliveries.tolist(),
            "projected_inventory": projected_inv.tolist(),
            "active_quantile": active_quantiles,
        }
    )


def batch_optimize_deliveries_v1(
    df: pl.DataFrame,
    forecast_df: pl.DataFrame,
) -> pl.DataFrame:
    """Simulate optimized daily deliveries over the full historical period.

    For each day in ``df``, the function loads the 7-day forecast window
    starting at that day from ``forecast_df``, runs the LP, and records the
    first day's optimized delivery. Storage simulation and penalty flags are
    then computed on the resulting delivery series.

    Args:
        df: Historical DataFrame with at least COL_DATE, COL_TOTAL, and
            COL_INVENTORY columns (from simulate_storage_v1). Must be sorted
            ascending by date.
        forecast_df: Pre-computed forecast DataFrame with columns Date, q25,
            q50, q75. Typically produced by predict_7day_v1.

    Returns:
        DataFrame with optimized_delivery, projected_inventory, and
        active_quantile columns appended, plus re-simulated inventory and
        penalty columns using the optimized deliveries.
    """
    from dgup._internal.storage import compute_daily_penalty_v1, compute_monthly_penalty_v1, simulate_storage_v1
    from dgup._internal.constants import COL_DELIVERY

    df = df.sort(COL_DATE)
    forecast_df = forecast_df.sort(COL_DATE)

    date_list = df[COL_DATE].to_list()
    inventory_series = df[COL_INVENTORY].to_list()

    opt_deliveries: list[float] = []
    opt_quantiles: list[str] = []

    for i, date in enumerate(date_list):
        # Use the last confirmed inventory: t-3 if available, else initial
        inv_t3 = inventory_series[max(0, i - 3)]

        # Extract 7-day forecast window starting at this date
        fcast_window = forecast_df.filter(
            (pl.col(COL_DATE) >= date)
            & (pl.col(COL_DATE) < date + datetime.timedelta(days=FORECAST_HORIZON))
        )
        if len(fcast_window) < FORECAST_HORIZON:
            # Not enough forecast data; fall through to naive
            opt_deliveries.append(df[COL_DELIVERY][i] if COL_DELIVERY in df.columns else 0.0)
            opt_quantiles.append(select_active_quantile_v1(date.month))
            continue

        opt_result = optimize_week_lp_v1(fcast_window, inv_t3, date)
        opt_deliveries.append(opt_result["optimized_delivery"][0])
        opt_quantiles.append(opt_result["active_quantile"][0])

    df = df.with_columns(
        pl.Series("optimized_delivery", opt_deliveries),
        pl.Series("opt_active_quantile", opt_quantiles),
    )

    # Re-simulate storage using optimized deliveries
    from dgup._internal.constants import COL_USAGE_1, COL_USAGE_2, COL_USAGE_2_1

    opt_sim = df.rename({"optimized_delivery": COL_DELIVERY})
    opt_sim = simulate_storage_v1(opt_sim)
    opt_sim = compute_daily_penalty_v1(opt_sim)
    opt_sim = compute_monthly_penalty_v1(opt_sim)

    return opt_sim.rename(
        {
            COL_DELIVERY: "optimized_delivery",
            COL_INVENTORY: "opt_inventory",
            "daily_penalty": "opt_daily_penalty",
            "monthly_penalty": "opt_monthly_penalty",
        }
    )
