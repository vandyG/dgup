---
description: "Use when working on any dgup analysis, modeling, or optimization task. Covers the Scot Forge gas usage domain, data schema, Nicor Gas storage bank mechanics, penalty rules, and the forecast-then-optimize problem-solving approach."
applyTo: "**/*.py"
---

# dgup Problem Domain

## Background

Scot Forge (Spring Grove, IL) uses large natural gas furnaces for forging and heat treatment. Gas is sourced via Nicor Gas transportation service. Scot Forge holds a **Storage Banking Service (SBS)** account: excess delivered gas is injected into storage; shortfalls are covered by withdrawal. Nicor Gas imposes **daily and monthly inventory limits** — violating them triggers penalties.

The goal of this project is to **minimize penalties by optimizing the daily Delivery value**, using a forecast of next-day usage as input.

---

## Data: `data/silver/uta_gas_usage.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `Date` | `date` | Gas day (2015-08-01 → 2024-11-30, 3,410 rows) |
| `Nom` | `f64` | Nominated volume (therms). Linear function of Delivery — compute *after* optimization, not before. |
| `Delivery` | `f64` | Gas delivered to Scot Forge that day (therms) |
| `Usage - 1` | `f64` | Gas used at Facility 1 (therms) |
| `Usage - 2` | `f64` | Gas used at Facility 2, Meter A (therms) |
| `Usage - 2_1` | `f64` | Gas used at Facility 2, Meter B (therms) — 2 null values, fill with 0 |

**Total daily usage:**
```python
total_usage = pl.col("Usage - 1") + pl.col("Usage - 2") + pl.col("Usage - 2_1").fill_null(0)
```

---

## Storage Bank Mechanics

### Parameters

```python
INITIAL_INVENTORY = 81569.123976   # therms, starting balance before data begins
STORAGE_CAPACITY  = 144841.0       # therms, total SBS capacity (all % limits scale by this)
```

### Daily inventory update

```
net_flow[t]     = Delivery[t] - total_usage[t]
injection[t]    = max(0,  net_flow[t])   # gas going INTO storage
withdrawal[t]   = max(0, -net_flow[t])   # gas coming OUT of storage
inventory[t]    = inventory[t-1] + net_flow[t]
```

### Daily storage activity limits (as % of `STORAGE_CAPACITY`)

| Month | Max Injection | Max Withdrawal |
|-------|:------------:|:--------------:|
| Jan   | 0.30%        | 1.00%          |
| Feb   | 0.30%        | 0.85%          |
| Mar   | 0.30%        | 0.60%          |
| Apr   | 0.30%        | 0.30%          |
| May   | 0.45%        | 0.30%          |
| Jun   | 0.50%        | 0.30%          |
| Jul   | 0.45%        | 0.30%          |
| Aug   | 0.70%        | 0.30%          |
| Sep   | 0.70%        | 0.30%          |
| Oct   | 0.70%        | 0.30%          |
| Nov   | 0.30%        | 0.40%          |
| Dec   | 0.30%        | 0.85%          |

Minimum injection and minimum withdrawal are both 0% every month — no penalty for low activity.

### Month-end inventory limits (as % of `STORAGE_CAPACITY`)

| Month | Min   | Max    |
|-------|:-----:|:------:|
| Jan   | 35%   | 45%    |
| Feb   | 10%   | 25%    |
| Mar   |  0%   | 10%    |
| Apr   |  0%   | 10%    |
| May   | 10%   | 20%    |
| Jun   | 20%   | 30%    |
| Jul   | 30%   | 40%    |
| Aug   | 50%   | 60%    |
| Sep   | 70%   | 80%    |
| Oct   | 85%   | 100%   |
| Nov   | 75%   | 90%    |
| Dec   | 55%   | 70%    |

---

## Penalty Rules

**Do not compute cash-out prices or amounts.** Only track whether a penalty *event* occurred (boolean / count).

### Daily penalty (soft constraint)
```
daily_penalty[t] = True  if  injection[t]  > MAX_INJECTION[month]  * STORAGE_CAPACITY
                         or  withdrawal[t] > MAX_WITHDRAWAL[month] * STORAGE_CAPACITY
```

### Monthly penalty (hard constraint — prioritize eliminating these first)
```
monthly_penalty[m] = True  if  inventory[last_day_of_m] < MONTHLY_MIN[m] * STORAGE_CAPACITY
                            or  inventory[last_day_of_m] > MONTHLY_MAX[m] * STORAGE_CAPACITY
```

Monthly penalties are more costly and must be treated as hard constraints in any optimization.

---

## Observability Constraint

Current-day usage is **not available** until partway through the following day.

The nomination for day `t` is submitted on day `t-1` (deadline 1:00 PM). If the forecast runs at the **start** of day `t-1`, usage for day `t-2` has not yet been reported either. The safe assumption is:

> When forecasting `total_usage[t]`, only `total_usage[0..t-3]` (up through **three days prior**) should be treated as known.

Design all feature engineering and model pipelines to respect this lag — never use `t-2` or `t-1` usage as a feature. This means:

1. **Forecast** `total_usage[t]` using only data through `t-3` (lag-3 and greater).
2. **Optimize** `Delivery[t]` using the forecast and the last confirmed `inventory[t-3]` projected forward.

---

## Problem-Solving Approach

> **Status:** EDA (Step 1), hypothesis testing (Step 2), and the baseline model (Step 3) are complete. Active work starts at Step 4.

### Step 4 — Forecasting model
- Target: `total_usage[t]` (next-day total gas usage, in therms).
- Features: lagged usage (1, 2, 7 days), day-of-week, month, season.
- Try `statsmodels` SARIMAX or similar time-series model first.
- `jax` / `numpy` are available for more advanced models if needed.
- Evaluate with time-series cross-validation (no data leakage — walk-forward splits only).
- Metrics: MAE and MAPE are most interpretable for stakeholders.

### Step 5 — Delivery optimization
- Given `forecast_usage[t]` and `inventory[t-1]`, choose `Delivery[t]` to:
  - **Hard:** keep projected month-end inventory within `[MONTHLY_MIN, MONTHLY_MAX] * STORAGE_CAPACITY`.
  - **Soft:** keep daily injection/withdrawal within daily activity limits.
  - In practice: spread required monthly injection/withdrawal evenly across remaining days in the month.
- Nom can be derived after solving: it is a linear function of Delivery.

### Step 6 — Evaluation
- Compare penalty counts: baseline vs. optimized (historical simulation with actual usage).
- Visualize inventory trajectory against monthly bands.

---

## Code Conventions

- All reusable logic (storage bank simulation, penalty calculation, optimization) lives in `src/dgup/_internal/`.
- **Every incremental change is a new function** — never overwrite old logic. Old functions must remain for fallback.
- Results (simulated inventory, penalty flags, model outputs) saved to `data/silver/*.parquet`.
- Notebooks in `notebooks/` call package functions — no reusable logic defined inline.

### Useful column name constants (define in a module)
```python
COL_DATE       = "Date"
COL_NOM        = "Nom"
COL_DELIVERY   = "Delivery"
COL_USAGE_1    = "Usage - 1"
COL_USAGE_2    = "Usage - 2"
COL_USAGE_2_1  = "Usage - 2_1"
COL_TOTAL      = "total_usage"
COL_NET_FLOW   = "net_flow"
COL_INJECTION  = "injection"
COL_WITHDRAWAL = "withdrawal"
COL_INVENTORY  = "inventory"
```
