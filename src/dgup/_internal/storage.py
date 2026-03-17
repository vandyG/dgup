from __future__ import annotations

from pathlib import Path

import polars as pl

from dgup._internal.gas_data import _USAGE_COLUMNS, _read_gas_usage
from dgup._internal.tariffs import (
    _INITIAL_GAS_IN_BANK,
    _MAX_INJECTION_RATES,
    _MAX_INVENTORY_RATES,
    _MAX_WITHDRAWAL_RATES,
    _MIN_INVENTORY_RATES,
    _STORAGE_CAPACITY,
)


def _date_expression(frame: pl.DataFrame) -> pl.Expr:
    dtype = frame.schema.get("Date")
    if dtype == pl.Date:
        return pl.col("Date")
    if dtype is not None and dtype.is_temporal():
        return pl.col("Date").dt.date()
    return pl.col("Date").str.to_date(strict=False)


def _ensure_total_usage(frame: pl.DataFrame) -> pl.DataFrame:
    if "Total Usage" in frame.columns:
        return frame.fill_null(0)

    missing_columns = [column for column in _USAGE_COLUMNS if column not in frame.columns]
    if missing_columns:
        msg = "frame must contain 'Total Usage' or all usage columns"
        raise ValueError(msg)

    return frame.fill_null(0).with_columns(
        sum((pl.col(column) for column in _USAGE_COLUMNS), start=pl.lit(0.0)).alias("Total Usage"),
    )


def _tariff_rules_frame() -> pl.DataFrame:
    months = list(range(1, 13))
    return pl.DataFrame(
        {
            "month": months,
            "max_injection_rate": [_MAX_INJECTION_RATES[month] for month in months],
            "max_withdrawal_rate": [_MAX_WITHDRAWAL_RATES[month] for month in months],
            "min_inventory_rate": [_MIN_INVENTORY_RATES[month] for month in months],
            "max_inventory_rate": [_MAX_INVENTORY_RATES[month] for month in months],
        },
    )


def _reconstruct_storage(
    frame: pl.DataFrame,
    *,
    initial_balance: float = _INITIAL_GAS_IN_BANK,
    capacity: float = _STORAGE_CAPACITY,
) -> pl.DataFrame:
    required_columns = {"Date", "Delivery"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        msg = f"frame is missing required columns: {', '.join(missing_columns)}"
        raise ValueError(msg)

    prepared = (
        _ensure_total_usage(frame)
        .lazy()
        .with_columns(_date_expression(frame).alias("Date"))
        .sort("Date")
        .with_columns(pl.col("Date").dt.month().alias("month"))
        .join(_tariff_rules_frame().lazy(), on="month", how="left")
        .with_columns((pl.col("Delivery") - pl.col("Total Usage")).alias("storage_delta"))
        .with_columns(
            pl.when(pl.col("storage_delta") > 0).then(pl.col("storage_delta")).otherwise(0.0).alias("injection"),
            pl.when(pl.col("storage_delta") < 0).then(-pl.col("storage_delta")).otherwise(0.0).alias("withdrawal"),
        )
        .with_columns((pl.lit(initial_balance) + pl.col("storage_delta").cum_sum()).alias("ending_balance"))
        .with_columns((pl.col("ending_balance") - pl.col("storage_delta")).alias("prior_balance"))
        .with_columns(
            (pl.col("max_injection_rate") * capacity).alias("max_injection"),
            (pl.col("max_withdrawal_rate") * capacity).alias("max_withdrawal"),
            (pl.col("min_inventory_rate") * capacity).alias("min_inventory"),
            (pl.col("max_inventory_rate") * capacity).alias("max_inventory"),
        )
        .with_columns(
            (pl.col("Date").dt.month() != pl.col("Date").shift(-1).dt.month())
            .fill_null(True)
            .alias("is_month_end"),
        )
        .with_columns(
            (pl.col("injection") > pl.col("max_injection")).alias("injection_limit_exceeded"),
            (pl.col("withdrawal") > pl.col("max_withdrawal")).alias("withdrawal_limit_exceeded"),
            (pl.col("ending_balance") < 0).alias("balance_below_zero"),
            (pl.col("ending_balance") > capacity).alias("balance_above_capacity"),
            (
                pl.col("is_month_end") & (pl.col("ending_balance") < pl.col("min_inventory"))
            ).alias("below_min_inventory"),
            (
                pl.col("is_month_end") & (pl.col("ending_balance") > pl.col("max_inventory"))
            ).alias("above_max_inventory"),
        )
    )
    return prepared.collect()


def _load_storage_series(
    data_path: str | Path | None = None,
    *,
    initial_balance: float = _INITIAL_GAS_IN_BANK,
    capacity: float = _STORAGE_CAPACITY,
) -> pl.DataFrame:
    return _reconstruct_storage(
        _read_gas_usage(data_path),
        initial_balance=initial_balance,
        capacity=capacity,
    )