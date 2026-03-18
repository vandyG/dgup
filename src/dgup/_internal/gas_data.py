from __future__ import annotations

from pathlib import Path

import polars as pl

_DATA_FILE = Path("data/silver/uta_gas_usage.parquet")
_USAGE_COLUMNS = ("Usage - 1", "Usage - 2", "Usage - 2_1")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_data_path() -> Path:
    return _repo_root() / _DATA_FILE


def _with_total_usage(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.fill_null(0).with_columns(
        sum((pl.col(column) for column in _USAGE_COLUMNS), start=pl.lit(0.0)).alias("Total Usage"),
    )


def _scan_gas_usage(data_path: str | Path | None = None) -> pl.LazyFrame:
    path = Path(data_path) if data_path is not None else _default_data_path()
    return _with_total_usage(pl.scan_parquet(path))


def _read_gas_usage(data_path: str | Path | None = None) -> pl.DataFrame:
    return _scan_gas_usage(data_path).collect()  # ty:ignore[invalid-return-type]
