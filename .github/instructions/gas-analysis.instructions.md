---
description: "Use when working on Scot Forge gas-usage analysis, forecasting notebooks, storage-bank calculations, penalty logic, or tariff-driven feature engineering. Covers the project objective, authoritative input files, and the Nicor storage constraints that must stay consistent across analysis work."
name: "Gas Analysis Guidelines"
applyTo: "notebooks/**,data/docs/**"
---

# Gas Analysis Guidelines

- Treat [data/docs/problem_statement.md](../../data/docs/problem_statement.md) as the business objective, with the operational goal made explicit: create analysis, forecasts, and decision support that help Scot Forge minimize Nicor nomination, storage, and month-end penalty exposure rather than optimizing forecast accuracy in isolation.
- Use [data/silver/uta_gas_usage.parquet](../../data/silver/uta_gas_usage.parquet) as the authoritative input dataset unless a task explicitly introduces another source.
- The parquet contains 3410 daily records from 2015-08-01 through 2024-11-30 with these source columns: `Date`, `Nom`, `Delivery`, `Usage - 1`, `Usage - 2`, and `Usage - 2_1`.
- Interpret `Nom` as the daily nomination series and `Delivery` as the daily delivered-gas series.
- Treat `Usage - 1`, `Usage - 2`, and `Usage - 2_1` as the three forge-level daily usage series. If a task needs total demand, derive `Total Usage` explicitly as the row-wise sum of those three columns because the parquet does not store a separate total-usage field.
- When framing operational forecasting or nomination decisions, assume a given day's actual gas usage is not available until midway through the following day. Do not rely on same-day actual usage for decisions that must be made before that reporting lag clears.
- Prefer workspace-relative paths in notebooks and scripts. Replace hard-coded machine-specific paths with repo paths rooted at `data/`.
- Keep exploratory work in `notebooks/`. If logic becomes reusable or operational, move it into package code under `src/` only when the task explicitly calls for productionization.

## Storage Assumptions

- Use storage capacity `144841` therms.
- Use initial gas in bank `81569.123976` therms before the first observed record.
- When reconstructing storage balances, make the accounting explicit: prior balance, deliveries/injections, usage/withdrawals, and resulting ending balance.
- Distinguish daily activity limits from month-end inventory limits; do not mix them into one penalty rule.

## Tariff Constraints

- Treat [data/docs/penalty_calc.md](../../data/docs/penalty_calc.md) as the source of truth for storage and penalty rules.
- Daily storage activity percentages are applied against total storage capacity and vary by calendar month.
- Month-end storage inventory bands are also percentages of total storage capacity and must be checked on end-of-month balances.
- Use these daily limits by month:

| Month | Max Injection | Max Withdrawal |
|---|---:|---:|
| January | 0.30% | 1.00% |
| February | 0.30% | 0.85% |
| March | 0.30% | 0.60% |
| April | 0.30% | 0.30% |
| May | 0.45% | 0.30% |
| June | 0.50% | 0.30% |
| July | 0.45% | 0.30% |
| August | 0.70% | 0.30% |
| September | 0.70% | 0.30% |
| October | 0.70% | 0.30% |
| November | 0.30% | 0.40% |
| December | 0.30% | 0.85% |

- Use these month-end inventory bands:

| Month | Minimum | Maximum |
|---|---:|---:|
| January | 35% | 45% |
| February | 10% | 25% |
| March | 0% | 10% |
| April | 0% | 10% |
| May | 10% | 20% |
| June | 20% | 30% |
| July | 30% | 40% |
| August | 50% | 60% |
| September | 70% | 80% |
| October | 85% | 100% |
| November | 75% | 90% |
| December | 55% | 70% |

## Analysis Expectations

- Optimize for penalty-aware decisions. When evaluating a model or heuristic, connect forecast outputs to nomination sizing, storage-bank trajectory, end-of-month compliance, and expected penalty reduction.
- When discussing model quality, compare against a simple baseline that reflects Scot Forge's current heuristic process, not only against more complex models.
- Keep explanations business-readable. Translate statistical findings into operational implications such as likely storage violations, nomination pressure, or penalty exposure.
- If an analysis depends on a tariff interpretation that is not explicit in the provided documents, call out the assumption instead of implying the rule is settled.