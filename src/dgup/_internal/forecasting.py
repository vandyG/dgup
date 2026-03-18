from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMRegressor
from scipy.optimize import linprog
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from dgup._internal.gas_data import _read_gas_usage, _repo_root
from dgup._internal.storage import _reconstruct_storage
from dgup._internal.tariffs import (
    _INITIAL_GAS_IN_BANK,
    _STORAGE_CAPACITY,
    _daily_storage_limits,
    _month_end_inventory_limits,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


_LOOKBACK = 42
_ROLLING_WINDOWS = (7, 14, 28)
_TARGET_COLUMNS = ("Total Usage", "Usage - 1", "Usage - 2", "Usage - 2_1")
_COMPONENT_COLUMNS = ("Usage - 1", "Usage - 2", "Usage - 2_1")
_FOLDS = (("2024-01-01", "2024-11-30", "CY2024"),)
_ARTIFACT_VERSION = "v2"
_SEED = 42
_MONTH_END_PENALTY = 1_000.0
_WEEKEND_START_DAY = 5
_DEFAULT_RESERVE_FLOOR = 2_500.0
_DEFAULT_RESERVE_FLOOR_PENALTY = 25.0
_DEFAULT_RETRAIN_FREQUENCY_DAYS = 7
_DEFAULT_SHORT_HORIZON_DAYS = 14
_DEFAULT_CONSERVATIVE_QUANTILE = 0.8
_DEFAULT_RECENT_RESIDUAL_WINDOW = 84
_DEFAULT_FALLBACK_LOOKBACK_DAYS = 28


@dataclass(frozen=True)
class _ForecastArtifacts:
    metrics_frame: pd.DataFrame
    predictions_frame: pd.DataFrame
    summary_frame: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    delivery_plan: pd.DataFrame
    delivery_summary: pd.DataFrame


@dataclass(frozen=True)
class _RollingPlanningArtifacts:
    forecast_frame: pd.DataFrame
    delivery_plan: pd.DataFrame
    delivery_summary: pd.DataFrame
    daily_violations: pd.DataFrame
    monthly_violations: pd.DataFrame
    violation_totals: pd.DataFrame
    violation_by_date: pd.DataFrame
    month_end_balance: pd.DataFrame


def _artifact_directory() -> Path:
    return _repo_root() / "data" / "silver" / "forecast_artifacts"


def _artifact_paths() -> dict[str, Path]:
    artifact_directory = _artifact_directory()
    prefix = f"forecast-{_ARTIFACT_VERSION}"
    return {
        "metrics": artifact_directory / f"{prefix}-metrics.parquet",
        "predictions": artifact_directory / f"{prefix}-predictions.parquet",
        "summary": artifact_directory / f"{prefix}-summary.parquet",
        "aggregate": artifact_directory / f"{prefix}-aggregate.parquet",
        "delivery_plan": artifact_directory / f"{prefix}-delivery-plan.parquet",
        "delivery_summary": artifact_directory / f"{prefix}-delivery-summary.parquet",
        "metadata": artifact_directory / f"{prefix}-metadata.json",
    }


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    safe_denominator = np.where(denominator == 0, 1.0, denominator)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / safe_denominator) * 100.0)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    safe_target = np.where(np.abs(y_true) < 1.0, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / safe_target)) * 100.0)


def _metric_frame(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_true - y_pred
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "smape": _smape(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
    }


@lru_cache(maxsize=1)
def _load_usage_frame() -> pd.DataFrame:
    pandas_frame = _read_gas_usage().to_pandas()
    pandas_frame["Date"] = pd.to_datetime(pandas_frame["Date"])
    return pandas_frame[["Date", "Delivery", *_TARGET_COLUMNS]].copy()


def _build_feature_frame(series_name: str) -> pd.DataFrame:
    usage_frame = _load_usage_frame()
    feature_frame = pd.DataFrame(
        {
            "Date": usage_frame["Date"],
            "target": usage_frame[series_name].astype(np.float32),
        },
    )

    for lag in range(1, _LOOKBACK + 1):
        feature_frame[f"lag_{lag}"] = feature_frame["target"].shift(lag)

    shifted_target = feature_frame["target"].shift(1)
    for window in _ROLLING_WINDOWS:
        feature_frame[f"rolling_mean_{window}"] = shifted_target.rolling(window).mean()
        feature_frame[f"rolling_std_{window}"] = shifted_target.rolling(window).std()
        feature_frame[f"rolling_min_{window}"] = shifted_target.rolling(window).min()
        feature_frame[f"rolling_max_{window}"] = shifted_target.rolling(window).max()

    day_of_week = feature_frame["Date"].dt.dayofweek.astype(np.float32)
    day_of_year = feature_frame["Date"].dt.dayofyear.astype(np.float32)
    month = feature_frame["Date"].dt.month.astype(np.float32)
    elapsed_days = (feature_frame["Date"] - feature_frame["Date"].min()).dt.days.astype(np.float32)

    feature_frame["is_weekend"] = (day_of_week >= _WEEKEND_START_DAY).astype(np.float32)
    feature_frame["is_month_end"] = feature_frame["Date"].dt.is_month_end.astype(np.float32)
    feature_frame["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    feature_frame["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    feature_frame["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    feature_frame["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    feature_frame["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 366.0)
    feature_frame["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 366.0)
    feature_frame["trend"] = elapsed_days / elapsed_days.max()

    return feature_frame.dropna().reset_index(drop=True)


def _lag_columns() -> list[str]:
    return [f"lag_{lag}" for lag in range(1, _LOOKBACK + 1)]


def _context_columns() -> list[str]:
    context_columns = []
    for window in _ROLLING_WINDOWS:
        context_columns.extend(
            [
                f"rolling_mean_{window}",
                f"rolling_std_{window}",
                f"rolling_min_{window}",
                f"rolling_max_{window}",
            ],
        )
    context_columns.extend(
        [
            "is_weekend",
            "is_month_end",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "doy_sin",
            "doy_cos",
            "trend",
        ],
    )
    return context_columns


def _tabular_columns() -> list[str]:
    return [*_lag_columns(), *_context_columns()]


def _sequence_columns() -> list[str]:
    return [f"lag_{lag}" for lag in range(_LOOKBACK, 0, -1)]


def _make_lightgbm_model() -> LGBMRegressor:
    return LGBMRegressor(
        learning_rate=0.05,
        n_estimators=400,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=_SEED,
        verbosity=-1,
    )


def _build_tabular_models() -> dict[str, Any]:
    return {
        "seasonal_naive_7": None,
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-3, 4, 20))),
            ],
        ),
        "hist_gbm": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=400,
            l2_regularization=0.1,
            min_samples_leaf=20,
            random_state=_SEED,
        ),
        "lightgbm": _make_lightgbm_model(),
        "xgboost": XGBRegressor(
            learning_rate=0.05,
            max_depth=6,
            n_estimators=400,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=_SEED,
            tree_method="hist",
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=1e-4,
                        batch_size=64,
                        early_stopping=True,
                        learning_rate_init=1e-3,
                        max_iter=500,
                        n_iter_no_change=20,
                        random_state=_SEED,
                    ),
                ),
            ],
        ),
    }


def _split_train_validation(
    sequence_values: np.ndarray,
    context_values: np.ndarray,
    targets: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    validation_rows = max(64, int(len(targets) * 0.15))
    validation_rows = min(validation_rows, max(64, len(targets) // 4))
    training_cutoff = len(targets) - validation_rows
    if training_cutoff <= 0:
        training_cutoff = len(targets) - 1
    return (
        sequence_values[:training_cutoff],
        context_values[:training_cutoff],
        targets[:training_cutoff],
    ), (
        sequence_values[training_cutoff:],
        context_values[training_cutoff:],
        targets[training_cutoff:],
    )


def _make_lstm_model(sequence_length: int, context_width: int) -> keras.Model:
    sequence_input = keras.Input(shape=(sequence_length, 1), name="sequence")
    context_input = keras.Input(shape=(context_width,), name="context")

    sequence_branch = keras.layers.LayerNormalization()(sequence_input)
    sequence_branch = keras.layers.LSTM(32, dropout=0.1)(sequence_branch)

    context_branch = keras.layers.Dense(16, activation="relu")(context_input)
    joined = keras.layers.Concatenate()([sequence_branch, context_branch])
    joined = keras.layers.Dense(32, activation="relu")(joined)
    output = keras.layers.Dense(1)(joined)

    model = keras.Model(inputs=[sequence_input, context_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mae")
    return model


def _make_transformer_model(sequence_length: int, context_width: int) -> keras.Model:
    sequence_input = keras.Input(shape=(sequence_length, 1), name="sequence")
    context_input = keras.Input(shape=(context_width,), name="context")

    sequence_branch = keras.layers.Dense(32)(sequence_input)
    attention_output = keras.layers.MultiHeadAttention(num_heads=4, key_dim=8, dropout=0.1)(
        sequence_branch,
        sequence_branch,
    )
    sequence_branch = keras.layers.Add()([sequence_branch, attention_output])
    sequence_branch = keras.layers.LayerNormalization()(sequence_branch)

    feed_forward = keras.layers.Dense(64, activation="relu")(sequence_branch)
    feed_forward = keras.layers.Dense(32)(feed_forward)
    sequence_branch = keras.layers.Add()([sequence_branch, feed_forward])
    sequence_branch = keras.layers.LayerNormalization()(sequence_branch)
    sequence_branch = keras.layers.GlobalAveragePooling1D()(sequence_branch)

    context_branch = keras.layers.Dense(16, activation="relu")(context_input)
    joined = keras.layers.Concatenate()([sequence_branch, context_branch])
    joined = keras.layers.Dense(32, activation="relu")(joined)
    output = keras.layers.Dense(1)(joined)

    model = keras.Model(inputs=[sequence_input, context_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mae")
    return model


def _make_timesnet_model(sequence_length: int, context_width: int) -> keras.Model:
    sequence_input = keras.Input(shape=(sequence_length, 1), name="sequence")
    context_input = keras.Input(shape=(context_width,), name="context")

    normalized_sequence = keras.layers.LayerNormalization()(sequence_input)
    branch_short = keras.layers.Conv1D(24, 3, padding="same", activation="relu")(normalized_sequence)
    branch_medium = keras.layers.Conv1D(24, 7, padding="same", activation="relu")(normalized_sequence)
    branch_long = keras.layers.Conv1D(24, 14, padding="same", activation="relu")(normalized_sequence)
    sequence_branch = keras.layers.Concatenate()([branch_short, branch_medium, branch_long])
    sequence_branch = keras.layers.Conv1D(48, 1, padding="same", activation="relu")(sequence_branch)
    sequence_branch = keras.layers.GlobalAveragePooling1D()(sequence_branch)

    context_branch = keras.layers.Dense(24, activation="relu")(context_input)
    joined = keras.layers.Concatenate()([sequence_branch, context_branch])
    joined = keras.layers.Dense(32, activation="relu")(joined)
    output = keras.layers.Dense(1)(joined)

    model = keras.Model(inputs=[sequence_input, context_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mae")
    return model


def _fit_predict_sequence_model(
    model_name: str,
    train_sequence: np.ndarray,
    train_context: np.ndarray,
    train_target: np.ndarray,
    test_sequence: np.ndarray,
    test_context: np.ndarray,
) -> np.ndarray:
    keras.utils.set_random_seed(_SEED)

    train_mean = float(train_target.mean())
    train_std = float(train_target.std())
    if train_std == 0:
        train_std = 1.0

    standardized_train_sequence = (train_sequence - train_mean) / train_std
    standardized_test_sequence = (test_sequence - train_mean) / train_std

    context_scaler = StandardScaler()
    standardized_train_context = context_scaler.fit_transform(train_context)
    standardized_test_context = context_scaler.transform(test_context)

    standardized_target = (train_target - train_mean) / train_std
    training_split, validation_split = _split_train_validation(
        standardized_train_sequence,
        standardized_train_context,
        standardized_target,
    )
    train_seq_split, train_ctx_split, train_target_split = training_split
    validation_seq_split, validation_ctx_split, validation_target_split = validation_split

    builders = {
        "lstm": _make_lstm_model,
        "transformer": _make_transformer_model,
        "timesnet": _make_timesnet_model,
    }
    model = builders[model_name](train_sequence.shape[1], train_context.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]
    model.fit(
        {"sequence": train_seq_split, "context": train_ctx_split},
        train_target_split,
        validation_data=(
            {"sequence": validation_seq_split, "context": validation_ctx_split},
            validation_target_split,
        ),
        batch_size=128,
        epochs=20,
        verbose=0,
        callbacks=callbacks,
    )
    predictions = model.predict(
        {"sequence": standardized_test_sequence, "context": standardized_test_context},
        verbose=0,
    ).reshape(-1)
    return predictions * train_std + train_mean


def _backtest_series(series_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_frame = _build_feature_frame(series_name)
    tabular_columns = _tabular_columns()
    sequence_columns = _sequence_columns()
    context_columns = _context_columns()

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_start, fold_end, fold_name in _FOLDS:
        fold_start_ts = pd.Timestamp(fold_start)
        fold_end_ts = pd.Timestamp(fold_end)
        train_mask = feature_frame["Date"] < fold_start_ts
        test_mask = (feature_frame["Date"] >= fold_start_ts) & (feature_frame["Date"] <= fold_end_ts)

        train_frame = feature_frame.loc[train_mask].copy()
        test_frame = feature_frame.loc[test_mask].copy()
        if train_frame.empty or test_frame.empty:
            continue

        x_train = train_frame[tabular_columns].astype(np.float32)
        y_train = train_frame["target"].to_numpy(dtype=np.float32)
        x_test = test_frame[tabular_columns].astype(np.float32)
        y_test = test_frame["target"].to_numpy(dtype=np.float32)

        for model_name, model in _build_tabular_models().items():
            if model_name == "seasonal_naive_7":
                predictions = test_frame["lag_7"].to_numpy(dtype=np.float32)
            else:
                if model is None:
                    msg = f"missing model implementation for {model_name}"
                    raise RuntimeError(msg)
                fitted_model = model.fit(x_train, y_train)
                predictions = fitted_model.predict(x_test).astype(np.float32)

            fold_metrics = _metric_frame(y_test, predictions)
            metrics_rows.append(
                {
                    "series": series_name,
                    "model": model_name,
                    "fold": fold_name,
                    **fold_metrics,
                },
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "Date": test_frame["Date"].to_numpy(),
                        "series": series_name,
                        "fold": fold_name,
                        "model": model_name,
                        "actual": y_test,
                        "predicted": predictions,
                    },
                ),
            )

        train_sequence = train_frame[sequence_columns].to_numpy(dtype=np.float32).reshape(-1, _LOOKBACK, 1)
        test_sequence = test_frame[sequence_columns].to_numpy(dtype=np.float32).reshape(-1, _LOOKBACK, 1)
        train_context = train_frame[context_columns].to_numpy(dtype=np.float32)
        test_context = test_frame[context_columns].to_numpy(dtype=np.float32)

        for model_name in ("lstm", "transformer", "timesnet"):
            predictions = _fit_predict_sequence_model(
                model_name,
                train_sequence,
                train_context,
                y_train,
                test_sequence,
                test_context,
            ).astype(np.float32)
            fold_metrics = _metric_frame(y_test, predictions)
            metrics_rows.append(
                {
                    "series": series_name,
                    "model": model_name,
                    "fold": fold_name,
                    **fold_metrics,
                },
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "Date": test_frame["Date"].to_numpy(),
                        "series": series_name,
                        "fold": fold_name,
                        "model": model_name,
                        "actual": y_test,
                        "predicted": predictions,
                    },
                ),
            )

    return pd.DataFrame(metrics_rows), pd.concat(prediction_frames, ignore_index=True)


def _optimize_delivery_for_month(
    *,
    dates: Sequence[pd.Timestamp],
    forecast_usage: np.ndarray,
    prior_balance: float,
    capacity: float = _STORAGE_CAPACITY,
    reserve_floor: float = 0.0,
    reserve_floor_penalty: float = _DEFAULT_RESERVE_FLOOR_PENALTY,
) -> np.ndarray:
    steps = len(dates)
    delivery_offset = 0
    balance_offset = steps
    deviation_offset = steps * 2
    reserve_slack_offset = steps * 3
    slack_offset = steps * 4
    variable_count = steps * 4 + 2

    objective = np.zeros(variable_count, dtype=float)
    objective[deviation_offset:slack_offset] = 1.0
    objective[reserve_slack_offset:slack_offset] = reserve_floor_penalty
    objective[slack_offset:slack_offset + 2] = _MONTH_END_PENALTY

    lower_bounds = [0.0] * steps + [0.0] * steps + [0.0] * steps + [0.0] * steps + [0.0, 0.0]
    upper_bounds = [None] * steps + [capacity] * steps + [None] * steps + [None] * steps + [None, None]
    bounds = list(zip(lower_bounds, upper_bounds, strict=False))

    eq_matrix: list[list[float]] = []
    eq_rhs: list[float] = []
    ub_matrix: list[list[float]] = []
    ub_rhs: list[float] = []

    for index, date in enumerate(dates):
        month = int(pd.Timestamp(date).month)
        max_injection, max_withdrawal = _daily_storage_limits(month, capacity)

        balance_row = [0.0] * variable_count
        balance_row[balance_offset + index] = 1.0
        balance_row[delivery_offset + index] = -1.0
        if index > 0:
            balance_row[balance_offset + index - 1] = -1.0
            eq_rhs.append(-float(forecast_usage[index]))
        else:
            eq_rhs.append(prior_balance - float(forecast_usage[index]))
        eq_matrix.append(balance_row)

        injection_row = [0.0] * variable_count
        injection_row[delivery_offset + index] = 1.0
        ub_matrix.append(injection_row)
        ub_rhs.append(float(forecast_usage[index]) + max_injection)

        withdrawal_row = [0.0] * variable_count
        withdrawal_row[delivery_offset + index] = -1.0
        ub_matrix.append(withdrawal_row)
        ub_rhs.append(max_withdrawal - float(forecast_usage[index]))

        positive_deviation_row = [0.0] * variable_count
        positive_deviation_row[delivery_offset + index] = 1.0
        positive_deviation_row[deviation_offset + index] = -1.0
        ub_matrix.append(positive_deviation_row)
        ub_rhs.append(float(forecast_usage[index]))

        negative_deviation_row = [0.0] * variable_count
        negative_deviation_row[delivery_offset + index] = -1.0
        negative_deviation_row[deviation_offset + index] = -1.0
        ub_matrix.append(negative_deviation_row)
        ub_rhs.append(-float(forecast_usage[index]))

        reserve_row = [0.0] * variable_count
        reserve_row[balance_offset + index] = -1.0
        reserve_row[reserve_slack_offset + index] = -1.0
        ub_matrix.append(reserve_row)
        ub_rhs.append(-reserve_floor)

    last_month = int(pd.Timestamp(dates[-1]).month)
    min_inventory, max_inventory = _month_end_inventory_limits(last_month, capacity)

    month_low_row = [0.0] * variable_count
    month_low_row[balance_offset + steps - 1] = -1.0
    month_low_row[slack_offset] = -1.0
    ub_matrix.append(month_low_row)
    ub_rhs.append(-min_inventory)

    month_high_row = [0.0] * variable_count
    month_high_row[balance_offset + steps - 1] = 1.0
    month_high_row[slack_offset + 1] = -1.0
    ub_matrix.append(month_high_row)
    ub_rhs.append(max_inventory)

    optimization = linprog(
        c=objective,
        A_ub=np.asarray(ub_matrix, dtype=float),
        b_ub=np.asarray(ub_rhs, dtype=float),
        A_eq=np.asarray(eq_matrix, dtype=float),
        b_eq=np.asarray(eq_rhs, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not optimization.success:
        msg = f"delivery optimization failed: {optimization.message}"
        raise RuntimeError(msg)

    return optimization.x[delivery_offset:balance_offset]


def _trend_scale_days(usage_frame: pd.DataFrame) -> int:
    return max(int((usage_frame["Date"].max() - usage_frame["Date"].min()).days), 1)


def _build_tabular_feature_row(
    *,
    history_values: Sequence[float],
    forecast_date: pd.Timestamp,
    series_start_date: pd.Timestamp,
    trend_scale_days: int,
) -> dict[str, float]:
    if len(history_values) < _LOOKBACK:
        msg = f"at least {_LOOKBACK} history values are required to build a forecast row"
        raise ValueError(msg)

    history_series = pd.Series(history_values, dtype=np.float32)
    feature_row: dict[str, float] = {}
    for lag in range(1, _LOOKBACK + 1):
        feature_row[f"lag_{lag}"] = float(history_series.iloc[-lag])

    for window in _ROLLING_WINDOWS:
        window_slice = history_series.iloc[-window:]
        feature_row[f"rolling_mean_{window}"] = float(window_slice.mean())
        feature_row[f"rolling_std_{window}"] = float(window_slice.std())
        feature_row[f"rolling_min_{window}"] = float(window_slice.min())
        feature_row[f"rolling_max_{window}"] = float(window_slice.max())

    day_of_week = float(forecast_date.dayofweek)
    day_of_year = float(forecast_date.dayofyear)
    month = float(forecast_date.month)
    elapsed_days = float((forecast_date - series_start_date).days)

    feature_row["is_weekend"] = float(day_of_week >= _WEEKEND_START_DAY)
    feature_row["is_month_end"] = float(forecast_date.is_month_end)
    feature_row["dow_sin"] = float(np.sin(2.0 * np.pi * day_of_week / 7.0))
    feature_row["dow_cos"] = float(np.cos(2.0 * np.pi * day_of_week / 7.0))
    feature_row["month_sin"] = float(np.sin(2.0 * np.pi * month / 12.0))
    feature_row["month_cos"] = float(np.cos(2.0 * np.pi * month / 12.0))
    feature_row["doy_sin"] = float(np.sin(2.0 * np.pi * day_of_year / 366.0))
    feature_row["doy_cos"] = float(np.cos(2.0 * np.pi * day_of_year / 366.0))
    feature_row["trend"] = elapsed_days / float(trend_scale_days)
    return feature_row


def _recursive_lightgbm_forecast(
    *,
    fitted_model: LGBMRegressor,
    history_values: Sequence[float],
    forecast_dates: Sequence[pd.Timestamp],
    series_start_date: pd.Timestamp,
    trend_scale_days: int,
) -> np.ndarray:
    simulated_history = [float(value) for value in history_values]
    predictions: list[float] = []
    for forecast_date in forecast_dates:
        feature_row = _build_tabular_feature_row(
            history_values=simulated_history,
            forecast_date=pd.Timestamp(forecast_date),
            series_start_date=series_start_date,
            trend_scale_days=trend_scale_days,
        )
        prediction = float(
            fitted_model.predict(pd.DataFrame([feature_row], columns=_tabular_columns()).astype(np.float32))[0],
        )
        prediction = max(prediction, 0.0)
        predictions.append(prediction)
        simulated_history.append(prediction)
    return np.asarray(predictions, dtype=np.float32)


def _estimate_conservative_uplift(
    train_frame: pd.DataFrame,
    fitted_model: LGBMRegressor,
    *,
    quantile: float = _DEFAULT_CONSERVATIVE_QUANTILE,
    recent_window: int = _DEFAULT_RECENT_RESIDUAL_WINDOW,
) -> float:
    recent_frame = train_frame.tail(max(_LOOKBACK, recent_window)).copy()
    if recent_frame.empty:
        return 0.0
    predicted = fitted_model.predict(recent_frame[_tabular_columns()].astype(np.float32))
    residuals = recent_frame["target"].to_numpy(dtype=np.float32) - predicted
    positive_residuals = residuals[residuals > 0]
    if positive_residuals.size == 0:
        return 0.0
    return float(np.quantile(positive_residuals, quantile))


def _fallback_usage_assumption(
    history_values: Sequence[float],
    *,
    uplift: float,
    lookback_days: int = _DEFAULT_FALLBACK_LOOKBACK_DAYS,
) -> float:
    recent_history = pd.Series(list(history_values)[-max(lookback_days, 7) :], dtype=np.float32)
    baseline = max(float(recent_history.tail(min(7, len(recent_history))).mean()), float(recent_history.quantile(0.75)))
    return max(baseline + uplift, 0.0)


def _build_penalty_flag_report(
    *,
    actual_strategy: pd.DataFrame,
    planned_strategy: pd.DataFrame,
    actual_label: str,
    planned_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _build_daily_violations(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
        working = frame.copy()
        working["strategy"] = strategy
        if "storage_delta" not in working.columns:
            working["storage_delta"] = working["Delivery"] - working["Total Usage"]
        working["injection"] = working["storage_delta"].clip(lower=0.0)
        working["withdrawal"] = (-working["storage_delta"]).clip(lower=0.0)
        working["excess_injection"] = (working["injection"] - working["max_injection"]).clip(lower=0.0)
        working["excess_withdrawal"] = (working["withdrawal"] - working["max_withdrawal"]).clip(lower=0.0)

        if "injection_limit_exceeded" not in working.columns:
            working["injection_limit_exceeded"] = working["injection"] > working["max_injection"]
        if "withdrawal_limit_exceeded" not in working.columns:
            working["withdrawal_limit_exceeded"] = working["withdrawal"] > working["max_withdrawal"]

        working["daily_violation_occurred"] = (
            working["injection_limit_exceeded"] | working["withdrawal_limit_exceeded"]
        )
        working["daily_violation_class"] = np.select(
            [
                working["injection_limit_exceeded"] & working["withdrawal_limit_exceeded"],
                working["injection_limit_exceeded"],
                working["withdrawal_limit_exceeded"],
            ],
            ["Over both limits", "Over-injection", "Over-withdrawal"],
            default="Within limit",
        )
        return working

    def _build_monthly_violations(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
        month_end = frame.loc[frame["is_month_end"]].copy()
        month_end["strategy"] = strategy
        month_end["below_minimum"] = (month_end["min_inventory"] - month_end["ending_balance"]).clip(lower=0.0)
        month_end["above_maximum"] = (month_end["ending_balance"] - month_end["max_inventory"]).clip(lower=0.0)

        if "below_min_inventory" not in month_end.columns:
            month_end["below_min_inventory"] = month_end["ending_balance"] < month_end["min_inventory"]
        if "above_max_inventory" not in month_end.columns:
            month_end["above_max_inventory"] = month_end["ending_balance"] > month_end["max_inventory"]

        month_end["monthly_violation_occurred"] = (
            month_end["below_min_inventory"] | month_end["above_max_inventory"]
        )
        month_end["monthly_violation_class"] = np.select(
            [
                month_end["below_min_inventory"] & month_end["above_max_inventory"],
                month_end["below_min_inventory"],
                month_end["above_max_inventory"],
            ],
            ["Outside band", "Below minimum", "Above maximum"],
            default="Within band",
        )
        return month_end

    actual_daily = _build_daily_violations(actual_strategy, actual_label)
    planned_daily = _build_daily_violations(planned_strategy, planned_label)
    actual_monthly = _build_monthly_violations(actual_strategy, actual_label)
    planned_monthly = _build_monthly_violations(planned_strategy, planned_label)

    daily_totals = pd.concat(
        [
            pd.DataFrame(
                {
                    "strategy": [actual_label, planned_label],
                    "violation_type": "Daily activity violation",
                    "violation_count": [
                        int(actual_daily["daily_violation_occurred"].sum()),
                        int(planned_daily["daily_violation_occurred"].sum()),
                    ],
                },
            ),
            pd.DataFrame(
                {
                    "strategy": [actual_label, planned_label],
                    "violation_type": "Month-end inventory violation",
                    "violation_count": [
                        int(actual_monthly["monthly_violation_occurred"].sum()),
                        int(planned_monthly["monthly_violation_occurred"].sum()),
                    ],
                },
            ),
        ],
        ignore_index=True,
    )

    violation_by_date = pd.concat(
        [
            actual_daily.loc[:, ["Date", "strategy", "daily_violation_occurred"]].rename(
                columns={"daily_violation_occurred": "daily_violation_occurred"},
            ),
            planned_daily.loc[:, ["Date", "strategy", "daily_violation_occurred"]].rename(
                columns={"daily_violation_occurred": "daily_violation_occurred"},
            ),
            actual_monthly.loc[:, ["Date", "strategy", "monthly_violation_occurred"]].rename(
                columns={"monthly_violation_occurred": "monthly_violation_occurred"},
            ),
            planned_monthly.loc[:, ["Date", "strategy", "monthly_violation_occurred"]].rename(
                columns={"monthly_violation_occurred": "monthly_violation_occurred"},
            ),
        ],
        ignore_index=True,
    )
    violation_by_date = (
        violation_by_date.groupby(["strategy", "Date"], as_index=False)
        .agg(
            daily_violation_occurred=("daily_violation_occurred", "max"),
            monthly_violation_occurred=("monthly_violation_occurred", "max"),
        )
        .fillna(value=False)
        .sort_values(["strategy", "Date"])
        .reset_index(drop=True)
    )
    violation_by_date["penalty_occurred"] = (
        violation_by_date["daily_violation_occurred"] | violation_by_date["monthly_violation_occurred"]
    )
    violation_by_date["cumulative_penalty_days"] = violation_by_date.groupby("strategy")["penalty_occurred"].cumsum()

    any_day_totals = (
        violation_by_date.groupby("strategy", as_index=False)["penalty_occurred"]
        .sum()
        .rename(columns={"penalty_occurred": "violation_count"})
        .assign(violation_type="Any penalty day")
    )
    violation_totals = pd.concat([daily_totals, any_day_totals], ignore_index=True)

    month_end_balance = pd.concat(
        [
            actual_monthly.loc[:, ["Date", "strategy", "ending_balance", "min_inventory", "max_inventory"]],
            planned_monthly.loc[:, ["Date", "strategy", "ending_balance", "min_inventory", "max_inventory"]],
        ],
        ignore_index=True,
    )
    daily_violations = pd.concat([actual_daily, planned_daily], ignore_index=True)
    monthly_violations = pd.concat([actual_monthly, planned_monthly], ignore_index=True)
    return daily_violations, monthly_violations, violation_totals, violation_by_date, month_end_balance


def _run_rolling_lightgbm_planning_prototype(
    *,
    start_date: str = "2024-01-01",
    end_date: str = "2024-11-30",
    retrain_frequency_days: int = _DEFAULT_RETRAIN_FREQUENCY_DAYS,
    short_horizon_days: int = _DEFAULT_SHORT_HORIZON_DAYS,
    conservative_quantile: float = _DEFAULT_CONSERVATIVE_QUANTILE,
    reserve_floor: float = _DEFAULT_RESERVE_FLOOR,
    reserve_floor_penalty: float = _DEFAULT_RESERVE_FLOOR_PENALTY,
    fallback_lookback_days: int = _DEFAULT_FALLBACK_LOOKBACK_DAYS,
    recent_residual_window: int = _DEFAULT_RECENT_RESIDUAL_WINDOW,
) -> _RollingPlanningArtifacts:
    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date)
    usage_frame = _load_usage_frame().copy()
    feature_frame = _build_feature_frame("Total Usage")
    tabular_columns = _tabular_columns()
    trend_scale_days = _trend_scale_days(usage_frame)
    series_start_date = pd.Timestamp(usage_frame["Date"].min())

    actual_usage = usage_frame[["Date", "Delivery", "Total Usage"]].copy()
    actual_storage = _reconstruct_storage(pl.from_pandas(actual_usage)).to_pandas()
    actual_storage["Date"] = pd.to_datetime(actual_storage["Date"])
    prior_balance_lookup = actual_storage.set_index("Date")["prior_balance"].to_dict()

    evaluation_frame = actual_usage.loc[
        actual_usage["Date"].between(start_timestamp, end_timestamp),
        ["Date", "Delivery", "Total Usage"],
    ].copy()
    if evaluation_frame.empty:
        msg = "no rows available for the requested prototype window"
        raise ValueError(msg)

    current_balance = float(prior_balance_lookup[evaluation_frame.iloc[0]["Date"]])
    fitted_model: LGBMRegressor | None = None
    last_refit_date: pd.Timestamp | None = None
    conservative_uplift = 0.0
    forecast_rows: list[dict[str, object]] = []
    plan_rows: list[dict[str, object]] = []

    for _, actual_row in evaluation_frame.iterrows():
        current_date = pd.Timestamp(actual_row["Date"])
        should_refit = fitted_model is None or last_refit_date is None
        if last_refit_date is not None:
            should_refit = should_refit or (current_date - last_refit_date).days >= retrain_frequency_days

        if should_refit:
            train_frame = feature_frame.loc[feature_frame["Date"] < current_date].copy()
            if train_frame.empty:
                msg = f"no training data available before {current_date.date()}"
                raise ValueError(msg)
            fitted_model = _make_lightgbm_model().fit(
                train_frame[tabular_columns].astype(np.float32),
                train_frame["target"].to_numpy(dtype=np.float32),
            )
            conservative_uplift = _estimate_conservative_uplift(
                train_frame,
                fitted_model,
                quantile=conservative_quantile,
                recent_window=recent_residual_window,
            )
            last_refit_date = current_date

        if fitted_model is None:
            msg = "lightgbm model was not fitted"
            raise RuntimeError(msg)

        month_dates = list(pd.date_range(current_date, current_date + pd.offsets.MonthEnd(0), freq="D"))
        modeled_dates = month_dates[: min(short_horizon_days, len(month_dates))]
        history_values = actual_usage.loc[actual_usage["Date"] < current_date, "Total Usage"].astype(np.float32).tolist()
        point_forecast = _recursive_lightgbm_forecast(
            fitted_model=fitted_model,
            history_values=history_values,
            forecast_dates=modeled_dates,
            series_start_date=series_start_date,
            trend_scale_days=trend_scale_days,
        )
        conservative_forecast = point_forecast + conservative_uplift
        fallback_usage = _fallback_usage_assumption(
            history_values,
            uplift=conservative_uplift,
            lookback_days=fallback_lookback_days,
        )
        if len(month_dates) > len(modeled_dates):
            optimizer_forecast = np.concatenate(
                [
                    conservative_forecast,
                    np.full(len(month_dates) - len(modeled_dates), fallback_usage, dtype=np.float32),
                ],
            )
        else:
            optimizer_forecast = conservative_forecast

        effective_reserve_floor = max(reserve_floor, conservative_uplift * float(short_horizon_days))
        optimized_deliveries = _optimize_delivery_for_month(
            dates=month_dates,
            forecast_usage=optimizer_forecast,
            prior_balance=current_balance,
            reserve_floor=effective_reserve_floor,
            reserve_floor_penalty=reserve_floor_penalty,
        )

        optimized_delivery = float(optimized_deliveries[0])
        point_forecast_first = float(point_forecast[0])
        conservative_forecast_first = float(optimizer_forecast[0])
        actual_total_usage = float(actual_row["Total Usage"])
        actual_delivery = float(actual_row["Delivery"])
        predicted_ending_balance = current_balance + optimized_delivery - conservative_forecast_first
        realized_ending_balance = current_balance + optimized_delivery - actual_total_usage
        plan_rows.append(
            {
                "Date": current_date,
                "prior_balance": current_balance,
                "actual_delivery": actual_delivery,
                "optimized_delivery": optimized_delivery,
                "actual_total_usage": actual_total_usage,
                "point_forecast_total_usage": point_forecast_first,
                "conservative_forecast_total_usage": conservative_forecast_first,
                "fallback_usage_assumption": float(fallback_usage),
                "conservative_uplift": float(conservative_uplift),
                "reserve_floor": float(effective_reserve_floor),
                "predicted_ending_balance": predicted_ending_balance,
                "realized_ending_balance": realized_ending_balance,
                "delivery_delta": optimized_delivery - actual_delivery,
                "forecast_error": point_forecast_first - actual_total_usage,
                "model_retrained": bool(should_refit),
            },
        )
        for horizon_index, horizon_date in enumerate(month_dates, start=1):
            point_value = float(point_forecast[horizon_index - 1]) if horizon_index <= len(point_forecast) else np.nan
            forecast_rows.append(
                {
                    "decision_date": current_date,
                    "forecast_date": pd.Timestamp(horizon_date),
                    "horizon_day": horizon_index,
                    "point_forecast_total_usage": point_value,
                    "conservative_forecast_total_usage": float(optimizer_forecast[horizon_index - 1]),
                    "forecast_source": "rolling_lightgbm" if horizon_index <= len(modeled_dates) else "fallback_assumption",
                    "conservative_uplift": float(conservative_uplift),
                    "reserve_floor": float(effective_reserve_floor),
                    "model_retrained": bool(should_refit),
                },
            )
        current_balance = realized_ending_balance

    delivery_plan = pd.DataFrame(plan_rows).sort_values("Date").reset_index(drop=True)
    optimized_storage = _reconstruct_storage(
        pl.from_pandas(
            delivery_plan[["Date", "optimized_delivery", "actual_total_usage"]].rename(
                columns={
                    "optimized_delivery": "Delivery",
                    "actual_total_usage": "Total Usage",
                },
            ),
        ),
        initial_balance=_INITIAL_GAS_IN_BANK,
        capacity=_STORAGE_CAPACITY,
    ).to_pandas()
    optimized_storage["Date"] = pd.to_datetime(optimized_storage["Date"])
    delivery_plan = delivery_plan.merge(
        optimized_storage[
            [
                "Date",
                "ending_balance",
                "injection_limit_exceeded",
                "withdrawal_limit_exceeded",
                "below_min_inventory",
                "above_max_inventory",
                "balance_below_zero",
                "balance_above_capacity",
                "is_month_end",
                "max_injection",
                "max_withdrawal",
                "min_inventory",
                "max_inventory",
            ]
        ],
        on="Date",
        how="left",
    )

    actual_window = actual_storage.loc[
        actual_storage["Date"].between(start_timestamp, end_timestamp),
    ].copy()
    actual_window["Date"] = pd.to_datetime(actual_window["Date"])
    actual_strategy = actual_window.loc[:, ["Date", "Delivery", "Total Usage", "ending_balance", "max_injection", "max_withdrawal", "min_inventory", "max_inventory", "is_month_end", "injection_limit_exceeded", "withdrawal_limit_exceeded", "below_min_inventory", "above_max_inventory"]].copy()
    planned_strategy = delivery_plan.loc[:, ["Date", "optimized_delivery", "actual_total_usage", "ending_balance", "max_injection", "max_withdrawal", "min_inventory", "max_inventory", "is_month_end", "injection_limit_exceeded", "withdrawal_limit_exceeded", "below_min_inventory", "above_max_inventory"]].rename(
        columns={
            "optimized_delivery": "Delivery",
            "actual_total_usage": "Total Usage",
        },
    )
    daily_violations, monthly_violations, violation_totals, violation_by_date, month_end_balance = _build_penalty_flag_report(
        actual_strategy=actual_strategy,
        planned_strategy=planned_strategy,
        actual_label="Actual delivery",
        planned_label="Rolling LightGBM prototype",
    )

    actual_summary = {
        "strategy": "actual_delivery",
        "mean_delivery": float(actual_window["Delivery"].mean()),
        "mean_ending_balance": float(actual_window["ending_balance"].mean()),
        "injection_limit_exceeded": int(actual_window["injection_limit_exceeded"].sum()),
        "withdrawal_limit_exceeded": int(actual_window["withdrawal_limit_exceeded"].sum()),
        "below_min_inventory": int(actual_window["below_min_inventory"].sum()),
        "above_max_inventory": int(actual_window["above_max_inventory"].sum()),
        "balance_below_zero": int(actual_window["balance_below_zero"].sum()),
        "balance_above_capacity": int(actual_window["balance_above_capacity"].sum()),
    }
    prototype_summary = {
        "strategy": "rolling_lightgbm_prototype",
        "mean_delivery": float(delivery_plan["optimized_delivery"].mean()),
        "mean_ending_balance": float(delivery_plan["ending_balance"].mean()),
        "injection_limit_exceeded": int(delivery_plan["injection_limit_exceeded"].sum()),
        "withdrawal_limit_exceeded": int(delivery_plan["withdrawal_limit_exceeded"].sum()),
        "below_min_inventory": int(delivery_plan["below_min_inventory"].sum()),
        "above_max_inventory": int(delivery_plan["above_max_inventory"].sum()),
        "balance_below_zero": int(delivery_plan["balance_below_zero"].sum()),
        "balance_above_capacity": int(delivery_plan["balance_above_capacity"].sum()),
    }
    return _RollingPlanningArtifacts(
        forecast_frame=pd.DataFrame(forecast_rows).sort_values(["decision_date", "forecast_date"]).reset_index(drop=True),
        delivery_plan=delivery_plan,
        delivery_summary=pd.DataFrame([actual_summary, prototype_summary]),
        daily_violations=daily_violations,
        monthly_violations=monthly_violations,
        violation_totals=violation_totals,
        violation_by_date=violation_by_date,
        month_end_balance=month_end_balance,
    )


def _build_delivery_plan(predictions_frame: pd.DataFrame, summary_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_models = (
        summary_frame.sort_values(["series", "rank"]).groupby("series", as_index=False).first()[["series", "model"]]
    )
    direct_total_model = best_models.loc[best_models["series"] == "Total Usage", "model"].iloc[0]
    direct_predictions = predictions_frame.loc[
        (predictions_frame["series"] == "Total Usage") & (predictions_frame["model"] == direct_total_model)
    ].copy()
    direct_predictions["Date"] = pd.to_datetime(direct_predictions["Date"])

    usage_frame = _load_usage_frame()
    actual_usage = usage_frame[["Date", "Delivery", "Total Usage"]].copy()
    actual_usage["Date"] = pd.to_datetime(actual_usage["Date"])
    actual_storage = _reconstruct_storage(pl.from_pandas(actual_usage[["Date", "Delivery", "Total Usage"]])).to_pandas()
    actual_storage["Date"] = pd.to_datetime(actual_storage["Date"])
    prior_balance_lookup = actual_storage.set_index("Date")["prior_balance"].to_dict()

    plan_rows: list[dict[str, object]] = []
    for fold_name, fold_predictions in direct_predictions.groupby("fold"):
        sorted_fold = fold_predictions.sort_values("Date").reset_index(drop=True)
        current_balance = float(prior_balance_lookup[sorted_fold.loc[0, "Date"]])
        index = 0
        while index < len(sorted_fold):
            current_date = pd.Timestamp(sorted_fold.loc[index, "Date"])
            month_end_date = current_date + pd.offsets.MonthEnd(0)
            horizon_frame = sorted_fold.loc[
                (sorted_fold["Date"] >= current_date) & (sorted_fold["Date"] <= month_end_date)
            ].reset_index(drop=True)
            optimized_deliveries = _optimize_delivery_for_month(
                dates=horizon_frame["Date"].tolist(),
                forecast_usage=horizon_frame["predicted"].to_numpy(dtype=np.float32),
                prior_balance=current_balance,
            )
            forecast_usage = float(horizon_frame.loc[0, "predicted"])
            actual_row = actual_usage.loc[actual_usage["Date"] == current_date].iloc[0]
            actual_total_usage = float(actual_row["Total Usage"])
            actual_delivery = float(actual_row["Delivery"])
            optimized_delivery = float(optimized_deliveries[0])
            predicted_ending_balance = current_balance + optimized_delivery - forecast_usage
            realized_ending_balance = current_balance + optimized_delivery - actual_total_usage
            plan_rows.append(
                {
                    "Date": current_date,
                    "fold": fold_name,
                    "model": direct_total_model,
                    "prior_balance": current_balance,
                    "forecast_total_usage": forecast_usage,
                    "actual_total_usage": actual_total_usage,
                    "actual_delivery": actual_delivery,
                    "optimized_delivery": optimized_delivery,
                    "predicted_ending_balance": predicted_ending_balance,
                    "realized_ending_balance": realized_ending_balance,
                    "delivery_delta": optimized_delivery - actual_delivery,
                    "forecast_error": forecast_usage - actual_total_usage,
                },
            )
            current_balance = realized_ending_balance
            index += 1

    delivery_plan = pd.DataFrame(plan_rows).sort_values("Date").reset_index(drop=True)
    optimized_storage = _reconstruct_storage(
        pl.from_pandas(
            delivery_plan[["Date", "optimized_delivery", "actual_total_usage"]].rename(
                columns={
                    "optimized_delivery": "Delivery",
                    "actual_total_usage": "Total Usage",
                },
            ),
        ),
        initial_balance=_INITIAL_GAS_IN_BANK,
        capacity=_STORAGE_CAPACITY,
    ).to_pandas()
    optimized_storage["Date"] = pd.to_datetime(optimized_storage["Date"])
    delivery_plan = delivery_plan.merge(
        optimized_storage[
            [
                "Date",
                "ending_balance",
                "injection_limit_exceeded",
                "withdrawal_limit_exceeded",
                "below_min_inventory",
                "above_max_inventory",
                "balance_below_zero",
                "balance_above_capacity",
                "is_month_end",
            ]
        ],
        on="Date",
        how="left",
    )

    actual_summary = {
        "strategy": "actual_delivery",
        "mean_delivery": float(actual_storage["Delivery"].mean()),
        "mean_ending_balance": float(actual_storage["ending_balance"].mean()),
        "injection_limit_exceeded": int(actual_storage["injection_limit_exceeded"].sum()),
        "withdrawal_limit_exceeded": int(actual_storage["withdrawal_limit_exceeded"].sum()),
        "below_min_inventory": int(actual_storage["below_min_inventory"].sum()),
        "above_max_inventory": int(actual_storage["above_max_inventory"].sum()),
        "balance_below_zero": int(actual_storage["balance_below_zero"].sum()),
        "balance_above_capacity": int(actual_storage["balance_above_capacity"].sum()),
    }
    optimized_summary = {
        "strategy": "optimized_delivery",
        "mean_delivery": float(delivery_plan["optimized_delivery"].mean()),
        "mean_ending_balance": float(delivery_plan["ending_balance"].mean()),
        "injection_limit_exceeded": int(delivery_plan["injection_limit_exceeded"].sum()),
        "withdrawal_limit_exceeded": int(delivery_plan["withdrawal_limit_exceeded"].sum()),
        "below_min_inventory": int(delivery_plan["below_min_inventory"].sum()),
        "above_max_inventory": int(delivery_plan["above_max_inventory"].sum()),
        "balance_below_zero": int(delivery_plan["balance_below_zero"].sum()),
        "balance_above_capacity": int(delivery_plan["balance_above_capacity"].sum()),
    }
    delivery_summary = pd.DataFrame([actual_summary, optimized_summary])
    return delivery_plan, delivery_summary


def _save_artifacts(artifacts: _ForecastArtifacts) -> None:
    paths = _artifact_paths()
    paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
    artifacts.metrics_frame.to_parquet(paths["metrics"], index=False)
    artifacts.predictions_frame.to_parquet(paths["predictions"], index=False)
    artifacts.summary_frame.to_parquet(paths["summary"], index=False)
    artifacts.aggregate_metrics.to_parquet(paths["aggregate"], index=False)
    artifacts.delivery_plan.to_parquet(paths["delivery_plan"], index=False)
    artifacts.delivery_summary.to_parquet(paths["delivery_summary"], index=False)
    metadata = {
        "artifact_version": _ARTIFACT_VERSION,
        "model_names": sorted(artifacts.summary_frame["model"].unique().tolist()),
        "series_names": sorted(artifacts.summary_frame["series"].unique().tolist()),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_artifacts() -> _ForecastArtifacts | None:
    paths = _artifact_paths()
    if not all(path.exists() for path in paths.values()):
        return None
    return _ForecastArtifacts(
        metrics_frame=pd.read_parquet(paths["metrics"]),
        predictions_frame=pd.read_parquet(paths["predictions"]),
        summary_frame=pd.read_parquet(paths["summary"]),
        aggregate_metrics=pd.read_parquet(paths["aggregate"]),
        delivery_plan=pd.read_parquet(paths["delivery_plan"]),
        delivery_summary=pd.read_parquet(paths["delivery_summary"]),
    )


def _compute_artifacts() -> _ForecastArtifacts:
    metrics_frames = []
    prediction_frames = []
    for series_name in _TARGET_COLUMNS:
        series_metrics, series_predictions = _backtest_series(series_name)
        metrics_frames.append(series_metrics)
        prediction_frames.append(series_predictions)

    metrics_frame = pd.concat(metrics_frames, ignore_index=True)
    predictions_frame = pd.concat(prediction_frames, ignore_index=True)
    summary_frame = (
        metrics_frame.groupby(["series", "model"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            smape=("smape", "mean"),
            mape=("mape", "mean"),
            bias=("bias", "mean"),
        )
        .sort_values(["series", "mae", "smape", "rmse"])
        .reset_index(drop=True)
    )
    summary_frame["rank"] = summary_frame.groupby("series")["mae"].rank(method="first")

    best_models = (
        summary_frame.sort_values(["series", "rank"]).groupby("series", as_index=False).first()[["series", "model"]]
    )
    direct_total_model = best_models.loc[best_models["series"] == "Total Usage", "model"].iloc[0]
    direct_total_predictions = predictions_frame.loc[
        (predictions_frame["series"] == "Total Usage") & (predictions_frame["model"] == direct_total_model)
    ].copy()
    direct_total_predictions["strategy"] = "direct_total_best"

    component_best_predictions = predictions_frame.merge(
        best_models[best_models["series"].isin(_COMPONENT_COLUMNS)],
        on=["series", "model"],
        how="inner",
    )
    aggregate_component_predictions = (
        component_best_predictions.groupby(["Date", "fold"], as_index=False)
        .agg(actual=("actual", "sum"), predicted=("predicted", "sum"))
    )
    aggregate_component_predictions["strategy"] = "sum_of_component_bests"

    aggregate_comparison = pd.concat(
        [
            direct_total_predictions[["Date", "fold", "actual", "predicted", "strategy"]],
            aggregate_component_predictions,
        ],
        ignore_index=True,
    )
    aggregate_metrics_rows = []
    for strategy_name, strategy_frame in aggregate_comparison.groupby("strategy"):
        aggregate_metrics_rows.append(
            {
                "strategy": strategy_name,
                **_metric_frame(
                    strategy_frame["actual"].to_numpy(dtype=np.float32),
                    strategy_frame["predicted"].to_numpy(dtype=np.float32),
                ),
            },
        )
    aggregate_metrics = pd.DataFrame(aggregate_metrics_rows).sort_values("mae").reset_index(drop=True)
    delivery_plan, delivery_summary = _build_delivery_plan(predictions_frame, summary_frame)
    return _ForecastArtifacts(
        metrics_frame=metrics_frame,
        predictions_frame=predictions_frame,
        summary_frame=summary_frame,
        aggregate_metrics=aggregate_metrics,
        delivery_plan=delivery_plan,
        delivery_summary=delivery_summary,
    )


@lru_cache(maxsize=1)
def _run_forecast_suite_cached(*, force_refresh: bool) -> _ForecastArtifacts:
    if not force_refresh:
        cached_artifacts = _load_artifacts()
        if cached_artifacts is not None:
            return cached_artifacts
    artifacts = _compute_artifacts()
    _save_artifacts(artifacts)
    return artifacts


def run_forecast_suite(*, force_refresh: bool = False) -> _ForecastArtifacts:
    return _run_forecast_suite_cached(force_refresh)


def print_cli_summary(*, force_refresh: bool = False) -> None:
    artifacts = run_forecast_suite(force_refresh=force_refresh)
    summary_frame = artifacts.summary_frame.copy()
    summary_frame[["mae", "rmse", "smape", "mape", "bias"]] = summary_frame[
        ["mae", "rmse", "smape", "mape", "bias"]
    ].round(3)
    sys.stdout.write("Model summary by series\n")
    sys.stdout.write(f"{summary_frame.to_string(index=False)}\n\n")
    sys.stdout.write("Direct vs component-aggregate comparison\n")
    sys.stdout.write(f"{artifacts.aggregate_metrics.round(3).to_string(index=False)}\n\n")
    sys.stdout.write("Delivery optimization summary\n")
    sys.stdout.write(f"{artifacts.delivery_summary.round(3).to_string(index=False)}\n")
