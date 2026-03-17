from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from dgup._internal.gas_data import _default_data_path, _read_gas_usage
from dgup._internal.storage import _reconstruct_storage
from dgup._internal.tariffs import _daily_storage_limits, _month_end_inventory_limits


def test_default_data_path_exists() -> None:
    assert _default_data_path().exists()


def test_read_gas_usage_derives_total_usage(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
            "Nom": [100.0, 120.0],
            "Delivery": [110.0, 115.0],
            "Usage - 1": [10.0, None],
            "Usage - 2": [20.0, 30.0],
            "Usage - 2_1": [5.0, 7.0],
        },
    )
    data_path = tmp_path / "sample.parquet"
    frame.write_parquet(data_path)

    result = _read_gas_usage(data_path)

    assert result["Total Usage"].to_list() == [35.0, 37.0]


def test_reconstruct_storage_balances() -> None:
    frame = pl.DataFrame(
        {
            "Date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "Nom": [100.0, 80.0, 50.0],
            "Delivery": [100.0, 80.0, 50.0],
            "Total Usage": [90.0, 100.0, 40.0],
        },
    )

    result = _reconstruct_storage(frame, initial_balance=1000.0, capacity=10000.0)

    assert result["storage_delta"].to_list() == [10.0, -20.0, 10.0]
    assert result["prior_balance"].to_list() == [1000.0, 1010.0, 990.0]
    assert result["ending_balance"].to_list() == [1010.0, 990.0, 1000.0]
    assert result["injection"].to_list() == [10.0, 0.0, 10.0]
    assert result["withdrawal"].to_list() == [0.0, 20.0, 0.0]


def test_reconstruct_storage_flags_daily_limits() -> None:
    frame = pl.DataFrame(
        {
            "Date": [date(2024, 1, 1), date(2024, 1, 2)],
            "Nom": [50.0, 50.0],
            "Delivery": [25.0, 0.0],
            "Total Usage": [0.0, 15.0],
        },
    )

    result = _reconstruct_storage(frame, initial_balance=400.0, capacity=1000.0)

    assert result["injection_limit_exceeded"].to_list() == [True, False]
    assert result["withdrawal_limit_exceeded"].to_list() == [False, True]


def test_reconstruct_storage_flags_month_end_inventory() -> None:
    frame = pl.DataFrame(
        {
            "Date": [date(2024, 1, 30), date(2024, 1, 31)],
            "Nom": [0.0, 0.0],
            "Delivery": [0.0, 0.0],
            "Total Usage": [0.0, 60.0],
        },
    )

    result = _reconstruct_storage(frame, initial_balance=400.0, capacity=1000.0)

    assert result["is_month_end"].to_list() == [False, True]
    assert result["below_min_inventory"].to_list() == [False, True]
    assert result["above_max_inventory"].to_list() == [False, False]


@pytest.mark.parametrize(
    ("month", "expected_injection", "expected_withdrawal", "expected_min", "expected_max"),
    [
        (1, 30.0, 100.0, 350.0, 450.0),
        (6, 50.0, 30.0, 200.0, 300.0),
        (10, 70.0, 30.0, 850.0, 1000.0),
    ],
)
def test_tariff_limit_helpers(
    month: int,
    expected_injection: float,
    expected_withdrawal: float,
    expected_min: float,
    expected_max: float,
) -> None:
    assert _daily_storage_limits(month, 1000.0) == (expected_injection / 10, expected_withdrawal / 10)
    assert _month_end_inventory_limits(month, 1000.0) == (expected_min, expected_max)