from __future__ import annotations

import polars as pl

from dgup._internal.constants import (
    COL_DATE,
    COL_DELIVERY,
    COL_DAILY_PENALTY,
    COL_INJECTION,
    COL_INVENTORY,
    COL_MONTHLY_PENALTY,
    COL_NET_FLOW,
    COL_TOTAL,
    COL_WITHDRAWAL,
    INITIAL_INVENTORY,
    MAX_INJECTION,
    MAX_WITHDRAWAL,
    MONTHLY_MAX,
    MONTHLY_MIN,
    STORAGE_CAPACITY,
)


def simulate_storage_v1(df: pl.DataFrame) -> pl.DataFrame:
    """Simulate daily storage bank inventory from delivery and usage.

    Adds net_flow, injection, withdrawal, and inventory columns. Assumes
    the DataFrame contains COL_DELIVERY and COL_TOTAL columns and is sorted
    by date ascending.

    Args:
        df: DataFrame with at least COL_DATE, COL_DELIVERY, and COL_TOTAL columns.

    Returns:
        DataFrame with additional columns: net_flow, injection, withdrawal, inventory.
    """
    df = df.sort(COL_DATE)
    df = df.with_columns(
        (pl.col(COL_DELIVERY) - pl.col(COL_TOTAL)).alias(COL_NET_FLOW),
    )
    df = df.with_columns(
        pl.col(COL_NET_FLOW).clip(lower_bound=0).alias(COL_INJECTION),
        (-pl.col(COL_NET_FLOW)).clip(lower_bound=0).alias(COL_WITHDRAWAL),
    )
    # Cumulative inventory starting from INITIAL_INVENTORY
    df = df.with_columns(
        (INITIAL_INVENTORY + pl.col(COL_NET_FLOW).cum_sum()).alias(COL_INVENTORY),
    )
    return df


def compute_daily_penalty_v1(sim_df: pl.DataFrame) -> pl.DataFrame:
    """Add a boolean daily_penalty column to a simulated storage DataFrame.

    A daily penalty occurs when injection or withdrawal exceeds the monthly
    activity limits defined in the storage contract.

    Args:
        sim_df: DataFrame produced by simulate_storage_v1.

    Returns:
        DataFrame with an additional boolean COL_DAILY_PENALTY column.
    """
    month = pl.col(COL_DATE).dt.month()

    max_inj = month.replace_strict(
        old=list(MAX_INJECTION.keys()), new=list(MAX_INJECTION.values()), return_dtype=pl.Float64
    )
    max_wit = month.replace_strict(
        old=list(MAX_WITHDRAWAL.keys()), new=list(MAX_WITHDRAWAL.values()), return_dtype=pl.Float64
    )

    return sim_df.with_columns(
        (
            (pl.col(COL_INJECTION) > max_inj) | (pl.col(COL_WITHDRAWAL) > max_wit)
        ).alias(COL_DAILY_PENALTY),
    )


def compute_monthly_penalty_v1(sim_df: pl.DataFrame) -> pl.DataFrame:
    """Add a boolean monthly_penalty column to a simulated storage DataFrame.

    The monthly penalty is evaluated only on the last calendar day of each
    month. It is True when the closing inventory falls outside the
    [MONTHLY_MIN, MONTHLY_MAX] band for that month.

    Args:
        sim_df: DataFrame produced by simulate_storage_v1.

    Returns:
        DataFrame with an additional boolean COL_MONTHLY_PENALTY column. Rows
        that are not the last day of their month carry False.
    """
    month = pl.col(COL_DATE).dt.month()
    next_day_month = (pl.col(COL_DATE) + pl.duration(days=1)).dt.month()
    is_last_day = month != next_day_month

    min_inv = month.replace_strict(
        old=list(MONTHLY_MIN.keys()), new=list(MONTHLY_MIN.values()), return_dtype=pl.Float64
    )
    max_inv = month.replace_strict(
        old=list(MONTHLY_MAX.keys()), new=list(MONTHLY_MAX.values()), return_dtype=pl.Float64
    )

    return sim_df.with_columns(
        (
            is_last_day
            & ((pl.col(COL_INVENTORY) < min_inv) | (pl.col(COL_INVENTORY) > max_inv))
        ).alias(COL_MONTHLY_PENALTY),
    )


def add_total_usage_v1(df: pl.DataFrame) -> pl.DataFrame:
    """Compute total_usage from the three facility columns and add it to the DataFrame.

    Usage - 2_1 may contain nulls which are filled with 0 per the data schema.

    Args:
        df: Raw gas usage DataFrame.

    Returns:
        DataFrame with COL_TOTAL column added.
    """
    from dgup._internal.constants import COL_USAGE_1, COL_USAGE_2, COL_USAGE_2_1

    return df.with_columns(
        (
            pl.col(COL_USAGE_1)
            + pl.col(COL_USAGE_2)
            + pl.col(COL_USAGE_2_1).fill_null(0)
        ).alias(COL_TOTAL),
    )
