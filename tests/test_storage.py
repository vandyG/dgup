"""Tests for storage simulation and penalty calculation."""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from dgup import (
    add_total_usage_v1,
    compute_daily_penalty_v1,
    compute_monthly_penalty_v1,
    simulate_storage_v1,
)
from dgup._internal.constants import (
    COL_DATE,
    COL_DAILY_PENALTY,
    COL_DELIVERY,
    COL_INJECTION,
    COL_INVENTORY,
    COL_MONTHLY_PENALTY,
    COL_NET_FLOW,
    COL_TOTAL,
    COL_USAGE_1,
    COL_USAGE_2,
    COL_USAGE_2_1,
    COL_WITHDRAWAL,
    INITIAL_INVENTORY,
    MAX_INJECTION,
    MAX_WITHDRAWAL,
    MONTHLY_MAX,
    MONTHLY_MIN,
    STORAGE_CAPACITY,
)


@pytest.fixture()
def simple_df() -> pl.DataFrame:
    """Three-row daily DataFrame for deterministic inventory tests."""
    return pl.DataFrame(
        {
            COL_DATE: [
                datetime.date(2020, 1, 10),
                datetime.date(2020, 1, 11),
                datetime.date(2020, 1, 12),
            ],
            COL_DELIVERY: [1000.0, 1500.0, 800.0],
            COL_TOTAL: [1200.0, 1200.0, 1200.0],
        }
    )


# ---------------------------------------------------------------------------
# simulate_storage_v1
# ---------------------------------------------------------------------------


def test_net_flow_computed_correctly(simple_df: pl.DataFrame) -> None:
    result = simulate_storage_v1(simple_df)
    expected = [-200.0, 300.0, -400.0]
    assert result[COL_NET_FLOW].to_list() == expected


def test_injection_withdrawal_non_negative(simple_df: pl.DataFrame) -> None:
    result = simulate_storage_v1(simple_df)
    assert all(v >= 0 for v in result[COL_INJECTION].to_list())
    assert all(v >= 0 for v in result[COL_WITHDRAWAL].to_list())


def test_injection_withdrawal_exclusive(simple_df: pl.DataFrame) -> None:
    """At most one of injection/withdrawal should be non-zero per row."""
    result = simulate_storage_v1(simple_df)
    for inj, wit in zip(result[COL_INJECTION].to_list(), result[COL_WITHDRAWAL].to_list()):
        assert inj == 0 or wit == 0


def test_inventory_starts_from_initial(simple_df: pl.DataFrame) -> None:
    result = simulate_storage_v1(simple_df)
    inventories = result[COL_INVENTORY].to_list()
    # First row: INITIAL_INVENTORY + (-200) = INITIAL_INVENTORY - 200
    assert inventories[0] == pytest.approx(INITIAL_INVENTORY - 200.0)
    # Second row: first inventory + 300
    assert inventories[1] == pytest.approx(INITIAL_INVENTORY - 200.0 + 300.0)
    # Third row: second inventory + (-400)
    assert inventories[2] == pytest.approx(INITIAL_INVENTORY - 200.0 + 300.0 - 400.0)


def test_simulate_preserves_row_count(simple_df: pl.DataFrame) -> None:
    result = simulate_storage_v1(simple_df)
    assert len(result) == len(simple_df)


# ---------------------------------------------------------------------------
# compute_daily_penalty_v1
# ---------------------------------------------------------------------------


def test_no_daily_penalty_within_limits() -> None:
    # January limits: injection ≤ MAX_INJECTION[1], withdrawal ≤ MAX_WITHDRAWAL[1]
    # Use values well below limits
    df = pl.DataFrame(
        {
            COL_DATE: [datetime.date(2020, 1, 5)],
            COL_DELIVERY: [1100.0],
            COL_TOTAL: [1000.0],
            COL_NET_FLOW: [100.0],
            COL_INJECTION: [100.0],
            COL_WITHDRAWAL: [0.0],
            COL_INVENTORY: [INITIAL_INVENTORY + 100.0],
        }
    )
    result = compute_daily_penalty_v1(df)
    assert result[COL_DAILY_PENALTY][0] is False


def test_daily_penalty_excess_injection() -> None:
    # January MAX_INJECTION = 0.30% × 144841 ≈ 434.5 therms
    # Inject 500 → should trigger penalty
    df = pl.DataFrame(
        {
            COL_DATE: [datetime.date(2020, 1, 5)],
            COL_DELIVERY: [1500.0],
            COL_TOTAL: [1000.0],
            COL_NET_FLOW: [500.0],
            COL_INJECTION: [500.0],
            COL_WITHDRAWAL: [0.0],
            COL_INVENTORY: [INITIAL_INVENTORY + 500.0],
        }
    )
    result = compute_daily_penalty_v1(df)
    assert result[COL_DAILY_PENALTY][0] is True


def test_daily_penalty_excess_withdrawal() -> None:
    # January MAX_WITHDRAWAL = 1.00% × 144841 ≈ 1448.4 therms
    # Withdraw 1500 → should trigger penalty
    df = pl.DataFrame(
        {
            COL_DATE: [datetime.date(2020, 1, 5)],
            COL_DELIVERY: [0.0],
            COL_TOTAL: [1500.0],
            COL_NET_FLOW: [-1500.0],
            COL_INJECTION: [0.0],
            COL_WITHDRAWAL: [1500.0],
            COL_INVENTORY: [INITIAL_INVENTORY - 1500.0],
        }
    )
    result = compute_daily_penalty_v1(df)
    assert result[COL_DAILY_PENALTY][0] is True


# ---------------------------------------------------------------------------
# compute_monthly_penalty_v1
# ---------------------------------------------------------------------------


def test_no_monthly_penalty_within_band() -> None:
    # January band: [35%, 45%] × 144841 = [50694, 65178]
    mid_jan_inventory = (MONTHLY_MIN[1] + MONTHLY_MAX[1]) / 2
    df = pl.DataFrame(
        {
            COL_DATE: [datetime.date(2020, 1, 30), datetime.date(2020, 1, 31)],
            COL_INVENTORY: [mid_jan_inventory, mid_jan_inventory],
            COL_NET_FLOW: [0.0, 0.0],
            COL_INJECTION: [0.0, 0.0],
            COL_WITHDRAWAL: [0.0, 0.0],
        }
    )
    result = compute_monthly_penalty_v1(df)
    # Jan 30 is not last day — must be False
    assert result[COL_MONTHLY_PENALTY][0] is False
    # Jan 31 is last day, inventory in band — must be False
    assert result[COL_MONTHLY_PENALTY][1] is False


def test_monthly_penalty_below_minimum() -> None:
    # January min = 35% × 144841 ≈ 50694 therms; inventory well below
    df = pl.DataFrame(
        {
            COL_DATE: [datetime.date(2020, 1, 31)],
            COL_INVENTORY: [10000.0],
            COL_NET_FLOW: [0.0],
            COL_INJECTION: [0.0],
            COL_WITHDRAWAL: [0.0],
        }
    )
    result = compute_monthly_penalty_v1(df)
    assert result[COL_MONTHLY_PENALTY][0] is True


def test_monthly_penalty_above_maximum() -> None:
    # January max = 45% × 144841 ≈ 65178 therms; inventory well above
    df = pl.DataFrame(
        {
            COL_DATE: [datetime.date(2020, 1, 31)],
            COL_INVENTORY: [STORAGE_CAPACITY],
            COL_NET_FLOW: [0.0],
            COL_INJECTION: [0.0],
            COL_WITHDRAWAL: [0.0],
        }
    )
    result = compute_monthly_penalty_v1(df)
    assert result[COL_MONTHLY_PENALTY][0] is True


def test_non_last_day_always_false() -> None:
    # Feb 10 is not month-end — should always be False regardless of inventory
    df = pl.DataFrame(
        {
            COL_DATE: [datetime.date(2020, 2, 10)],
            COL_INVENTORY: [0.0],  # purposely outside band
            COL_NET_FLOW: [0.0],
            COL_INJECTION: [0.0],
            COL_WITHDRAWAL: [0.0],
        }
    )
    result = compute_monthly_penalty_v1(df)
    assert result[COL_MONTHLY_PENALTY][0] is False


# ---------------------------------------------------------------------------
# add_total_usage_v1
# ---------------------------------------------------------------------------


def test_add_total_usage_sums_correctly() -> None:
    df = pl.DataFrame(
        {
            COL_DATE: [datetime.date(2020, 1, 1)],
            COL_USAGE_1: [100.0],
            COL_USAGE_2: [800.0],
            COL_USAGE_2_1: [50.0],
        }
    )
    result = add_total_usage_v1(df)
    assert result[COL_TOTAL][0] == pytest.approx(950.0)


def test_add_total_usage_fills_null() -> None:
    df = pl.DataFrame(
        {
            COL_DATE: [datetime.date(2020, 1, 1)],
            COL_USAGE_1: [100.0],
            COL_USAGE_2: [800.0],
            COL_USAGE_2_1: [None],
        }
    )
    result = add_total_usage_v1(df)
    assert result[COL_TOTAL][0] == pytest.approx(900.0)
