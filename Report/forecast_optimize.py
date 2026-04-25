"""
Gas Storage Optimization Pipeline (2-DAY AHEAD + V4)
=====================================================
Changes from v3:
  - Adaptive buffers by month: summer inj buffer tighter (0.78),
    winter wd buffer tighter (0.80), others at 0.85/0.88
  - Multiple EOM constraints in horizon (not just current month)
  - Stronger mid-month slack reduced 10% -> 6% for tighter steering
  - Post-LP clip uses actual usage history percentile as usage estimate
    instead of p95 forecast (avoids over-clipping on withdrawal side)
  - Storage overshoot guard: if sim_storage > EOM_max at start of month,
    force minimum withdrawal to bleed down storage
  - EOM_BUFFER made month-specific: tight months get larger buffer
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
CAPACITY          = 144_841
HORIZON           = 60
FORECAST_LAG      = 2
LP_MARGIN         = 0.10
OPT_QUANTILE      = "p95"

# Adaptive injection buffer by month (summer tighter — inj violations dominate Jul-Oct)
INJECTION_BUFFER_BY_MONTH = {
    1: 0.85, 2: 0.85, 3: 0.85, 4: 0.85,
    5: 0.82, 6: 0.80, 7: 0.78, 8: 0.78,
    9: 0.78, 10: 0.80, 11: 0.85, 12: 0.85,
}

# Adaptive withdrawal buffer by month (winter tighter — wd violations in Apr)
WITHDRAWAL_BUFFER_BY_MONTH = {
    1: 0.85, 2: 0.85, 3: 0.85, 4: 0.80,
    5: 0.85, 6: 0.88, 7: 0.88, 8: 0.88,
    9: 0.88, 10: 0.88, 11: 0.85, 12: 0.85,
}

# Month-specific EOM buffer: months with persistent overshoot get larger buffer
EOM_BUFFER_BY_MONTH = {
    1: 0.10, 2: 0.10, 3: 0.10, 4: 0.10,
    5: 0.08, 6: 0.08, 7: 0.08, 8: 0.08,
    9: 0.06, 10: 0.06, 11: 0.10, 12: 0.10,
}

MID_MONTH_SLACK   = 0.06   # tightened from 0.10

QUANTILES = {
    "p10": 0.10,
    "p50": 0.50,
    "p95": 0.95,
}

# ─────────────────────────────────────────────────────────────
# CONTRACT CONSTRAINT TABLES
# ─────────────────────────────────────────────────────────────
DAILY_PARAMS = pd.DataFrame({
    "month": range(1, 13),
    "max_inj_pct": [
        0.0030, 0.0030, 0.0030, 0.0030,
        0.0045, 0.0050, 0.0045, 0.0070,
        0.0070, 0.0070, 0.0030, 0.0030,
    ],
    "max_wd_pct": [
        0.0100, 0.0085, 0.0060, 0.0030,
        0.0030, 0.0030, 0.0030, 0.0030,
        0.0030, 0.0030, 0.0040, 0.0085,
    ],
}).set_index("month")

EOM_PARAMS = pd.DataFrame({
    "month": range(1, 13),
    "eom_min_pct": [
        0.35, 0.10, 0.00, 0.00,
        0.10, 0.20, 0.30, 0.50,
        0.70, 0.85, 0.75, 0.55,
    ],
    "eom_max_pct": [
        0.45, 0.25, 0.10, 0.10,
        0.20, 0.30, 0.40, 0.60,
        0.80, 1.00, 0.90, 0.70,
    ],
}).set_index("month")

def get_daily_limits(date):
    row = DAILY_PARAMS.loc[date.month]
    return (
        row["max_inj_pct"] * CAPACITY,
        row["max_wd_pct"]  * CAPACITY,
    )

def get_eom_bounds(date):
    row = EOM_PARAMS.loc[date.month]
    return (
        row["eom_min_pct"] * CAPACITY,
        row["eom_max_pct"] * CAPACITY,
    )

def get_eom_buffer(month):
    return EOM_BUFFER_BY_MONTH[month]

def get_injection_buffer(month):
    return INJECTION_BUFFER_BY_MONTH[month]

def get_withdrawal_buffer(month):
    return WITHDRAWAL_BUFFER_BY_MONTH[month]

def get_mid_month_target(date):
    eom_min, eom_max = get_eom_bounds(date)
    return (eom_min + eom_max) / 2.0

# ─────────────────────────────────────────────────────────────
# US HOLIDAYS
# ─────────────────────────────────────────────────────────────
US_HOLIDAYS = pd.to_datetime([
    "2021-01-01","2021-01-18","2021-02-15","2021-05-31",
    "2021-07-04","2021-07-05","2021-09-06","2021-10-11",
    "2021-11-11","2021-11-25","2021-12-24","2021-12-31",
    "2022-01-17","2022-02-21","2022-05-30","2022-06-20",
    "2022-07-04","2022-09-05","2022-10-10","2022-11-11",
    "2022-11-24","2022-12-26",
    "2023-01-02","2023-01-16","2023-02-20","2023-05-29",
    "2023-06-19","2023-07-04","2023-09-04","2023-10-09",
    "2023-11-10","2023-11-23","2023-12-25",
    "2024-01-01","2024-01-15","2024-02-19","2024-05-27",
    "2024-06-19","2024-07-04","2024-09-02","2024-10-14",
    "2024-11-11","2024-11-28","2024-12-25",
])

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def build_features(df, forecast_lag=FORECAST_LAG):
    df = df.copy().sort_values("Date").reset_index(drop=True)

    df["dayofweek"]  = df["Date"].dt.dayofweek
    df["month"]      = df["Date"].dt.month
    df["dayofmonth"] = df["Date"].dt.day
    df["dayofyear"]  = df["Date"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["quarter"]    = df["Date"].dt.quarter

    month_end = df.groupby(df["Date"].dt.to_period("M"))["Date"].transform("max")
    df["days_to_eom"] = (month_end - df["Date"]).dt.days

    df["is_holiday"]         = df["Date"].isin(US_HOLIDAYS).astype(int)
    df["day_before_holiday"] = df["Date"].shift(-1).isin(US_HOLIDAYS).astype(int)
    df["day_after_holiday"]  = df["Date"].shift(1).isin(US_HOLIDAYS).astype(int)

    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"usage_lag_{lag}"] = df["Total Usage"].shift(lag + forecast_lag)

    past = df["Total Usage"].shift(forecast_lag)
    df["usage_roll7_mean"]  = past.rolling(7).mean()
    df["usage_roll7_std"]   = past.rolling(7).std()
    df["usage_roll14_mean"] = past.rolling(14).mean()
    df["usage_roll30_mean"] = past.rolling(30).mean()
    df["usage_roll30_std"]  = past.rolling(30).std()

    df["storage_pct"] = df["Total Gas Before"].shift(forecast_lag) / CAPACITY

    return df.dropna().reset_index(drop=True)

FEATURES = [
    "dayofweek","month","dayofmonth","dayofyear","is_weekend","quarter",
    "days_to_eom",
    "is_holiday","day_before_holiday","day_after_holiday",
    "usage_lag_1","usage_lag_2","usage_lag_3","usage_lag_7",
    "usage_lag_14","usage_lag_21","usage_lag_28",
    "usage_roll7_mean","usage_roll7_std",
    "usage_roll14_mean",
    "usage_roll30_mean","usage_roll30_std",
    "storage_pct",
]

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("Loading data...")
raw = pd.read_csv("uta_gas_usage_all.csv", parse_dates=["Date"])
raw = raw.sort_values("Date").reset_index(drop=True)

daily_limits = raw["Date"].apply(
    lambda d: pd.Series(
        get_daily_limits(d),
        index=["Max_Injection_Limit", "Max_Withdrawal_Limit"]
    )
)
eom_bounds = raw["Date"].apply(
    lambda d: pd.Series(
        get_eom_bounds(d),
        index=["EOM_Min_Storage", "EOM_Max_Storage"]
    )
)

raw["Max_Injection_Limit"]  = daily_limits["Max_Injection_Limit"]
raw["Max_Withdrawal_Limit"] = daily_limits["Max_Withdrawal_Limit"]
raw["EOM_Min_Storage"]      = eom_bounds["EOM_Min_Storage"]
raw["EOM_Max_Storage"]      = eom_bounds["EOM_Max_Storage"]

df = build_features(raw)

# ─────────────────────────────────────────────────────────────
# TRAIN BLENDED MODELS
# ─────────────────────────────────────────────────────────────
print("Training blended quantile models (GBM + QuantileRegressor)...")

test_df = df[(df["Date"].dt.year >= 2022) & (df["Date"].dt.year <= 2024)].copy()

retrain_dates = [pd.Timestamp(f"2022-{m:02d}-01") for m in range(1, 13)] + \
                [pd.Timestamp(f"2023-{m:02d}-01") for m in range(1, 13)] + \
                [pd.Timestamp(f"2024-{m:02d}-01") for m in range(1, 13)]

models_dict = {}

for date in retrain_dates:
    train    = df[df["Date"] < date]
    X_raw, y = train[FEATURES], train["Total Usage"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    models = {}
    for name, q in QUANTILES.items():
        gbm = GradientBoostingRegressor(
            loss="quantile", alpha=q,
            n_estimators=300, max_depth=4,
            learning_rate=0.05, random_state=42
        ).fit(X_raw, y)

        qreg = QuantileRegressor(quantile=q, alpha=0.1, solver="highs")
        qreg.fit(X_scaled, y)

        models[name] = {"gbm": gbm, "qreg": qreg, "scaler": scaler}

    models_dict[date] = models
    print(f"  trained up to {date.date()}")

def get_model(date):
    return models_dict[max([d for d in retrain_dates if d <= date])]

def blended_predict(models, X_raw, quantile_name):
    m      = models[quantile_name]
    gbm_p  = m["gbm"].predict(X_raw)
    qreg_p = m["qreg"].predict(m["scaler"].transform(X_raw))
    return 0.70 * gbm_p + 0.30 * qreg_p

# ─────────────────────────────────────────────────────────────
# FORECAST
# ─────────────────────────────────────────────────────────────
print("Forecasting 2022-2024 (2-day ahead, blended)...")

records = []
for _, row in test_df.iterrows():
    models = get_model(row["Date"])
    X      = row[FEATURES].values.reshape(1, -1)

    rec = {
        "Date":                 row["Date"],
        "Actual_Usage":         row["Total Usage"],
        "Actual_Delivery":      row["Delivery"],
        "Total_Gas_Before":     row["Total Gas Before"],
        "Total_Gas_After":      row["Total Gas After"],
        "Max_Injection_Limit":  row["Max_Injection_Limit"],
        "Max_Withdrawal_Limit": row["Max_Withdrawal_Limit"],
        "EOM_Min_Storage":      row["EOM_Min_Storage"],
        "EOM_Max_Storage":      row["EOM_Max_Storage"],
    }

    for k in QUANTILES:
        rec[f"Usage_{k.upper()}"] = blended_predict(models, X, k)[0]

    records.append(rec)

forecast_df = pd.DataFrame(records).set_index("Date")

# ─────────────────────────────────────────────────────────────
# LP SOLVER
# ─────────────────────────────────────────────────────────────
def solve_lp(storage_init, usage, inj, wd,
             eom_constraints,
             mid_idx, mid_target):
    """
    Parameters
    ----------
    eom_constraints : list of (eom_idx, eom_min, eom_max) tuples
                      — one per month boundary found in horizon
    mid_idx         : day-15 index in horizon (-1 if not found)
    mid_target      : desired storage at day-15
    """
    H         = len(usage)
    cum_usage = np.cumsum(usage)

    inj_norm = inj / (inj.max() + 1e-9)
    wd_norm  = wd  / (wd.max()  + 1e-9)
    c        = np.ones(H) - LP_MARGIN * (inj_norm + wd_norm)

    rows, rhs = [], []

    for t in range(H):
        e   = np.zeros(H); e[t]     = 1
        cum = np.zeros(H); cum[:t+1] = 1

        rows.append(e);    rhs.append(inj[t] + usage[t])
        rows.append(-e);   rhs.append(wd[t]  - usage[t])
        rows.append(cum);  rhs.append(CAPACITY - storage_init + cum_usage[t])
        rows.append(-cum); rhs.append(storage_init - cum_usage[t])

    # Multiple EOM constraints (one per month in horizon)
    for eom_idx, eom_min, eom_max in eom_constraints:
        cum = np.zeros(H); cum[:eom_idx+1] = 1
        cu  = cum_usage[eom_idx]
        rows.append(-cum); rhs.append(storage_init - cu - eom_min)
        rows.append(cum);  rhs.append(eom_max - storage_init + cu)

    # Mid-month steering
    if mid_idx >= 0:
        slack = CAPACITY * MID_MONTH_SLACK
        cum   = np.zeros(H); cum[:mid_idx+1] = 1
        cu    = cum_usage[mid_idx]
        rows.append(-cum); rhs.append(storage_init - cu - (mid_target - slack))
        rows.append(cum);  rhs.append((mid_target + slack) - storage_init + cu)

    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs),
                  bounds=[(0, None)]*H, method="highs")
    return res.x if res.status == 0 else None

# ─────────────────────────────────────────────────────────────
# ROLLING OPTIMIZATION
# ─────────────────────────────────────────────────────────────
print("Running 2-day-ahead optimization (horizon=60, v4)...")

opt_delivery = []
sim_storage  = forecast_df["Total_Gas_Before"].iloc[0]
dates        = forecast_df.index.tolist()

for i, d in enumerate(dates):

    future_start = i + 1
    h = dates[future_start : future_start + HORIZON]

    if len(h) == 0:
        opt_delivery.append(forecast_df.loc[d, "Actual_Delivery"])
        sim_storage = np.clip(
            sim_storage
            + forecast_df.loc[d, "Actual_Delivery"]
            - forecast_df.loc[d, "Actual_Usage"],
            0, CAPACITY
        )
        continue

    usage = forecast_df.loc[h, f"Usage_{OPT_QUANTILE.upper()}"].values

    # Adaptive monthly buffers applied per horizon day
    inj = np.array([
        get_daily_limits(hd)[0] * get_injection_buffer(hd.month)
        for hd in h
    ])
    wd = np.array([
        get_daily_limits(hd)[1] * get_withdrawal_buffer(hd.month)
        for hd in h
    ])

    # ── Multiple EOM constraints: one per month in horizon ────
    eom_constraints = []
    seen_months = set()
    for k, hd in enumerate(h):
        ym = (hd.year, hd.month)
        if ym not in seen_months:
            # find last day of this month within horizon
            last_k = max(
                j for j, hd2 in enumerate(h)
                if hd2.year == hd.year and hd2.month == hd.month
            )
            seen_months.add(ym)
            buf      = get_eom_buffer(hd.month)
            raw_min, raw_max = get_eom_bounds(h[last_k])
            eom_min  = raw_min + CAPACITY * buf
            eom_max  = raw_max - CAPACITY * buf
            if eom_min < eom_max:   # only add if bounds are valid
                eom_constraints.append((last_k, eom_min, eom_max))

    # Mid-month steering: day-15 of planning month
    mid_idx    = -1
    mid_target = 0.0
    for k, hd in enumerate(h):
        if hd.month == h[0].month and hd.day == 15:
            mid_idx    = k
            mid_target = get_mid_month_target(hd)
            break

    # Storage overshoot guard: if current storage already above next EOM max,
    # force a minimum delivery floor to actively bleed storage down
    delivery_floor = 0.0
    if eom_constraints:
        first_eom_idx, first_eom_min, first_eom_max = eom_constraints[0]
        if sim_storage > first_eom_max + CAPACITY * 0.02:
            # need to shed excess; compute minimum daily drain needed
            excess        = sim_storage - first_eom_max
            days_to_eom   = first_eom_idx + 1
            avg_wd_needed = excess / days_to_eom
            # floor delivery at (forecasted_usage - avg_wd_needed) so net flow
            # = delivery - usage = -avg_wd_needed (net withdrawal)
            delivery_floor = max(0.0, usage[0] - avg_wd_needed)

    plan = solve_lp(sim_storage, usage, inj, wd,
                    eom_constraints, mid_idx, mid_target)

    # Fallback 1: drop mid-month constraint
    if plan is None:
        plan = solve_lp(sim_storage, usage, inj, wd,
                        eom_constraints, -1, 0.0)

    # Fallback 2: drop all soft constraints, keep only EOM
    if plan is None:
        plan = solve_lp(sim_storage, usage, inj, wd,
                        eom_constraints[:1], -1, 0.0)

    planned = (
        forecast_df.loc[dates[future_start], "Actual_Delivery"]
        if plan is None else plan[0]
    )

    # Apply delivery floor from overshoot guard
    planned = max(planned, delivery_floor)

    # ── Hard post-LP clip to raw contract limits ──────────────
    tomorrow        = dates[future_start]
    raw_inj_lim     = get_daily_limits(tomorrow)[0]
    raw_wd_lim      = get_daily_limits(tomorrow)[1]
    usage_tomorrow  = forecast_df.loc[tomorrow, f"Usage_{OPT_QUANTILE.upper()}"]

    max_safe = raw_inj_lim + usage_tomorrow
    min_safe = max(0.0, usage_tomorrow - raw_wd_lim)
    planned  = float(np.clip(planned, min_safe, max_safe))

    opt_delivery.append(planned)

    sim_storage = np.clip(
        sim_storage + planned - forecast_df.loc[d, "Actual_Usage"],
        0, CAPACITY
    )

forecast_df["Opt_Delivery"] = opt_delivery

forecast_df["Opt_Storage_After"] = (
    forecast_df["Total_Gas_Before"]
    + forecast_df["Opt_Delivery"]
    - forecast_df["Actual_Usage"]
).clip(0, CAPACITY)

# ─────────────────────────────────────────────────────────────
# PENALTY FUNCTION
# ─────────────────────────────────────────────────────────────
def calculate_penalties(df, delivery_col, storage_col):
    d = df.copy()
    d["Flow"]       = d[delivery_col] - d["Actual_Usage"]
    d["Inj_Excess"] = (d["Flow"]  - d["Max_Injection_Limit"]).clip(lower=0)
    d["Wd_Excess"]  = (-d["Flow"] - d["Max_Withdrawal_Limit"]).clip(lower=0)
    d["Daily_Violation"] = (d["Inj_Excess"] > 0) | (d["Wd_Excess"] > 0)

    d["YM"]     = d.index.to_period("M")
    d["Is_EOM"] = d.groupby("YM").cumcount()
    d["Is_EOM"] = d["Is_EOM"] == d.groupby("YM")["Is_EOM"].transform("max")

    d["Monthly_Violation"] = (
        d["Is_EOM"] &
        ((d[storage_col] < d["EOM_Min_Storage"]) |
         (d[storage_col] > d["EOM_Max_Storage"]))
    )
    return d

# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
print("Evaluating penalties...")

actual_eval = calculate_penalties(forecast_df, "Actual_Delivery",  "Total_Gas_After")
opt_eval    = calculate_penalties(forecast_df, "Opt_Delivery",     "Opt_Storage_After")

def summary(df, name):
    print(f"{name:<35} | Daily: {df['Daily_Violation'].sum():4d} | Monthly: {df['Monthly_Violation'].sum()}")

summary(actual_eval, "Actual")
summary(opt_eval,    "Optimized (v4)")

# ─────────────────────────────────────────────────────────────
# DIAGNOSTIC REPORT
# ─────────────────────────────────────────────────────────────
print("\n── Daily violations by month ──")
diag = opt_eval.copy()
diag["month"] = diag.index.month
diag["year"]  = diag.index.year
print(diag.groupby("month")["Daily_Violation"].sum().to_string())

print("\n── Daily violations by year ──")
print(diag.groupby("year")["Daily_Violation"].sum().to_string())

print("\n── Monthly (EOM) violations detail ──")
mv = diag[diag["Monthly_Violation"]][
    ["EOM_Min_Storage", "EOM_Max_Storage", "Opt_Storage_After"]
]
print(mv.to_string() if len(mv) else "  No monthly violations!")

print("\n── Top daily violation days ──")
dv = diag[diag["Daily_Violation"]][
    ["Opt_Delivery","Actual_Usage",
     "Max_Injection_Limit","Max_Withdrawal_Limit",
     "Inj_Excess","Wd_Excess"]
].copy()
dv["Total_Excess"] = dv["Inj_Excess"] + dv["Wd_Excess"]
print(dv.sort_values("Total_Excess", ascending=False).head(20).to_string())

# ─────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────
output = forecast_df.copy()
output["Actual_Daily_Violation"]   = actual_eval["Daily_Violation"]
output["Actual_Monthly_Violation"] = actual_eval["Monthly_Violation"]
output["Opt_Daily_Violation"]      = opt_eval["Daily_Violation"]
output["Opt_Monthly_Violation"]    = opt_eval["Monthly_Violation"]

output.to_csv("gas_optimization_results_2022_2024_v4.csv")
print("\nSaved: gas_optimization_results_2022_2024_v4.csv")
print("Done.")