from __future__ import annotations

import pandas as pd
import polars as pl

from dgup._internal.forecasting import (
    _build_penalty_flag_report,
    _optimize_delivery_for_month,
    _run_rolling_lightgbm_planning_prototype,
)
from dgup._internal.storage import _reconstruct_storage
from dgup._internal.tariffs import _daily_storage_limits, _month_end_inventory_limits


def test_optimize_delivery_for_month_respects_limits() -> None:
    dates = [pd.Timestamp("2024-01-30"), pd.Timestamp("2024-01-31")]
    forecast_usage = [100.0, 100.0]
    optimized_delivery = _optimize_delivery_for_month(
        dates=dates,
        forecast_usage=forecast_usage,
        prior_balance=400.0,
        capacity=1000.0,
    )

    max_injection, max_withdrawal = _daily_storage_limits(1, 1000.0)
    final_balance = 400.0
    for forecast, delivery in zip(forecast_usage, optimized_delivery, strict=True):
        assert delivery - forecast <= max_injection + 1e-9
        assert forecast - delivery <= max_withdrawal + 1e-9
        final_balance += delivery - forecast

    min_inventory, max_inventory = _month_end_inventory_limits(1, 1000.0)
    assert min_inventory - 1e-9 <= final_balance <= max_inventory + 1e-9


def test_optimize_delivery_for_month_respects_positive_reserve_floor() -> None:
    dates = [pd.Timestamp("2024-01-29"), pd.Timestamp("2024-01-30"), pd.Timestamp("2024-01-31")]
    forecast_usage = [100.0, 100.0, 100.0]
    optimized_delivery = _optimize_delivery_for_month(
        dates=dates,
        forecast_usage=forecast_usage,
        prior_balance=400.0,
        capacity=1000.0,
        reserve_floor=390.0,
    )

    reconstructed = _reconstruct_storage(
        pl.DataFrame({"Date": dates, "Delivery": optimized_delivery, "Total Usage": forecast_usage}),
        initial_balance=400.0,
        capacity=1000.0,
    ).to_pandas()
    assert (reconstructed["ending_balance"] >= 390.0 - 1e-9).all()


def test_run_rolling_lightgbm_planning_prototype_returns_window() -> None:
    artifacts = _run_rolling_lightgbm_planning_prototype(
        start_date="2024-01-01",
        end_date="2024-01-05",
        retrain_frequency_days=7,
        short_horizon_days=7,
        reserve_floor=1000.0,
    )

    assert len(artifacts.delivery_plan) == 5
    assert {"Date", "optimized_delivery", "conservative_forecast_total_usage", "reserve_floor"} <= set(
        artifacts.delivery_plan.columns,
    )
    assert artifacts.delivery_plan["model_retrained"].any()
    assert not artifacts.forecast_frame.empty
    assert not artifacts.violation_totals.empty


def test_build_penalty_flag_report_uses_boolean_violations() -> None:
    actual_strategy = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-30", "2024-01-31"]),
            "Delivery": [200.0, 200.0],
            "Total Usage": [100.0, 100.0],
            "ending_balance": [500.0, 120.0],
            "max_injection": [50.0, 50.0],
            "max_withdrawal": [50.0, 50.0],
            "min_inventory": [100.0, 150.0],
            "max_inventory": [900.0, 300.0],
            "is_month_end": [False, True],
            "injection_limit_exceeded": [True, False],
            "withdrawal_limit_exceeded": [False, False],
            "below_min_inventory": [False, True],
            "above_max_inventory": [False, False],
        },
    )
    planned_strategy = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-30", "2024-01-31"]),
            "Delivery": [100.0, 100.0],
            "Total Usage": [100.0, 100.0],
            "ending_balance": [200.0, 200.0],
            "max_injection": [50.0, 50.0],
            "max_withdrawal": [50.0, 50.0],
            "min_inventory": [100.0, 150.0],
            "max_inventory": [900.0, 300.0],
            "is_month_end": [False, True],
            "injection_limit_exceeded": [False, False],
            "withdrawal_limit_exceeded": [False, False],
            "below_min_inventory": [False, False],
            "above_max_inventory": [False, False],
        },
    )

    _, _, violation_totals, violation_by_date, _ = _build_penalty_flag_report(
        actual_strategy=actual_strategy,
        planned_strategy=planned_strategy,
        actual_label="Actual delivery",
        planned_label="Planned delivery",
    )

    daily_count = int(
        violation_totals.loc[
            (violation_totals["strategy"] == "Actual delivery")
            & (violation_totals["violation_type"] == "Daily activity violation"),
            "violation_count",
        ].iloc[0],
    )
    monthly_count = int(
        violation_totals.loc[
            (violation_totals["strategy"] == "Actual delivery")
            & (violation_totals["violation_type"] == "Month-end inventory violation"),
            "violation_count",
        ].iloc[0],
    )
    any_day_count = int(
        violation_totals.loc[
            (violation_totals["strategy"] == "Actual delivery")
            & (violation_totals["violation_type"] == "Any penalty day"),
            "violation_count",
        ].iloc[0],
    )

    assert daily_count == 1
    assert monthly_count == 1
    assert any_day_count == 2
    assert violation_by_date["penalty_occurred"].sum() == 2