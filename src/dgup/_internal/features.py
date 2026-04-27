from __future__ import annotations

import math

import polars as pl

from dgup._internal.constants import COL_DATE, COL_TOTAL, FORECAST_HORIZON

# Lag offsets used for feature engineering (all ≥ lag_min enforced at call time)
_LAG_OFFSETS: list[int] = [3, 4, 5, 6, 7, 14, 21, 28, 364]

# Feature column names exposed for downstream use
FEATURE_COLS: list[str] = [
    *[f"lag_{n}" for n in _LAG_OFFSETS],
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "month",
    "day_of_week",
    "is_weekend",
    "sin_month",
    "cos_month",
    "sin_dow",
    "cos_dow",
]

TARGET_COLS: list[str] = [f"horizon_{h}" for h in range(1, FORECAST_HORIZON + 1)]


def build_features_v1(df: pl.DataFrame, lag_min: int = 3) -> pl.DataFrame:
    """Build the supervised feature matrix for 7-day multi-step forecasting.

    Adds lag, rolling, and calendar features to the DataFrame, then appends
    7 target columns (horizon_1..horizon_7). Rows with any NaN introduced by
    lags or future targets are dropped.

    The ``lag_min`` parameter enforces the observability constraint: no lag
    shorter than ``lag_min`` days is included. The nomination for day ``t``
    is filed on day ``t-1`` by 1:00 PM; usage through ``t-3`` is the last
    confirmed data at that point, so ``lag_min=3`` is the production default.

    Args:
        df: DataFrame with at least COL_DATE and COL_TOTAL columns, sorted by
            date ascending.
        lag_min: Minimum lag offset to include. Must be >= 1. Defaults to 3.

    Returns:
        DataFrame with feature and target columns, NaN rows dropped.
    """
    if lag_min < 1:
        msg = f"lag_min must be >= 1, got {lag_min}"
        raise ValueError(msg)

    df = df.sort(COL_DATE)
    usage = pl.col(COL_TOTAL)

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------
    lag_exprs = [
        usage.shift(n).alias(f"lag_{n}")
        for n in _LAG_OFFSETS
        if n >= lag_min
    ]

    # ------------------------------------------------------------------
    # Rolling features (window ends at t-lag_min, i.e. shift then roll)
    # ------------------------------------------------------------------
    shifted = usage.shift(lag_min)
    rolling_exprs = [
        shifted.rolling_mean(window_size=7).alias("rolling_mean_7"),
        shifted.rolling_mean(window_size=28).alias("rolling_mean_28"),
        shifted.rolling_std(window_size=7).alias("rolling_std_7"),
    ]

    # ------------------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------------------
    month = pl.col(COL_DATE).dt.month().alias("month")
    dow = pl.col(COL_DATE).dt.weekday().alias("day_of_week")  # 0=Mon, 6=Sun
    is_weekend = (pl.col(COL_DATE).dt.weekday() >= 5).alias("is_weekend")

    sin_month = (pl.col(COL_DATE).dt.month().cast(pl.Float64) * (2 * math.pi / 12)).sin().alias("sin_month")
    cos_month = (pl.col(COL_DATE).dt.month().cast(pl.Float64) * (2 * math.pi / 12)).cos().alias("cos_month")
    sin_dow = (pl.col(COL_DATE).dt.weekday().cast(pl.Float64) * (2 * math.pi / 7)).sin().alias("sin_dow")
    cos_dow = (pl.col(COL_DATE).dt.weekday().cast(pl.Float64) * (2 * math.pi / 7)).cos().alias("cos_dow")

    calendar_exprs = [month, dow, is_weekend, sin_month, cos_month, sin_dow, cos_dow]

    # ------------------------------------------------------------------
    # Multi-step target labels
    # horizon_h at row T = usage[T + h - 1] for h=1..7
    #   horizon_1 = usage[T]       (shift 0)  — the gas day being planned
    #   horizon_7 = usage[T + 6]   (shift -6) — 6 days ahead
    # All features are available at time T-1 when the nomination deadline is.
    # ------------------------------------------------------------------
    target_exprs = [
        usage.shift(-(h - 1)).alias(f"horizon_{h}")
        for h in range(1, FORECAST_HORIZON + 1)
    ]

    df = df.with_columns(*lag_exprs, *rolling_exprs, *calendar_exprs, *target_exprs)

    # Drop any row with NaN in feature or target columns
    all_new_cols = (
        [f"lag_{n}" for n in _LAG_OFFSETS if n >= lag_min]
        + ["rolling_mean_7", "rolling_mean_28", "rolling_std_7"]
        + ["month", "day_of_week", "is_weekend", "sin_month", "cos_month", "sin_dow", "cos_dow"]
        + [f"horizon_{h}" for h in range(1, FORECAST_HORIZON + 1)]
    )
    return df.drop_nulls(subset=all_new_cols)


def build_inference_features_v1(df: pl.DataFrame, lag_min: int = 3) -> pl.DataFrame:
    """Build the feature matrix for inference (no target columns).

    Identical to build_features_v1 but omits the horizon target columns and
    does not drop rows for missing targets. Use this when generating live
    forecasts where future actuals are not yet available.

    Args:
        df: DataFrame with at least COL_DATE and COL_TOTAL columns, sorted by
            date ascending.
        lag_min: Minimum lag offset. Defaults to 3.

    Returns:
        DataFrame with feature columns only.
    """
    if lag_min < 1:
        msg = f"lag_min must be >= 1, got {lag_min}"
        raise ValueError(msg)

    df = df.sort(COL_DATE)
    usage = pl.col(COL_TOTAL)

    lag_exprs = [
        usage.shift(n).alias(f"lag_{n}")
        for n in _LAG_OFFSETS
        if n >= lag_min
    ]

    shifted = usage.shift(lag_min)
    rolling_exprs = [
        shifted.rolling_mean(window_size=7).alias("rolling_mean_7"),
        shifted.rolling_mean(window_size=28).alias("rolling_mean_28"),
        shifted.rolling_std(window_size=7).alias("rolling_std_7"),
    ]

    month = pl.col(COL_DATE).dt.month().alias("month")
    dow = pl.col(COL_DATE).dt.weekday().alias("day_of_week")
    is_weekend = (pl.col(COL_DATE).dt.weekday() >= 5).alias("is_weekend")
    sin_month = (pl.col(COL_DATE).dt.month().cast(pl.Float64) * (2 * math.pi / 12)).sin().alias("sin_month")
    cos_month = (pl.col(COL_DATE).dt.month().cast(pl.Float64) * (2 * math.pi / 12)).cos().alias("cos_month")
    sin_dow = (pl.col(COL_DATE).dt.weekday().cast(pl.Float64) * (2 * math.pi / 7)).sin().alias("sin_dow")
    cos_dow = (pl.col(COL_DATE).dt.weekday().cast(pl.Float64) * (2 * math.pi / 7)).cos().alias("cos_dow")

    df = df.with_columns(*lag_exprs, *rolling_exprs, month, dow, is_weekend, sin_month, cos_month, sin_dow, cos_dow)

    feature_cols = (
        [f"lag_{n}" for n in _LAG_OFFSETS if n >= lag_min]
        + ["rolling_mean_7", "rolling_mean_28", "rolling_std_7"]
        + ["month", "day_of_week", "is_weekend", "sin_month", "cos_month", "sin_dow", "cos_dow"]
    )
    return df.drop_nulls(subset=feature_cols)
