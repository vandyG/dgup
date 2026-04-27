from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from dgup._internal.constants import COL_DATE, COL_TOTAL, FORECAST_HORIZON
from dgup._internal.features import (
    FEATURE_COLS,
    TARGET_COLS,
    build_features_v1,
    build_inference_features_v1,
)

# ---------------------------------------------------------------------------
# Model name registries
# ---------------------------------------------------------------------------
BASELINE_MODELS: list[str] = ["naive_lag7", "seasonal_mean"]
TREE_MODELS: list[str] = ["xgboost", "lightgbm", "random_forest"]
NF_MODELS: list[str] = ["mlp", "nhits", "timesnet", "fedformer", "patchtst", "itransformer"]
ALL_MODEL_NAMES: list[str] = BASELINE_MODELS + TREE_MODELS + NF_MODELS

# Naive lag mapping: horizon_h (1..7) → lag offset from row T to use as naive forecast
# h=1..5: lag_{8-h} from row T;  h=6: lag_9;  h=7: lag_8
# Rationale: horizon_h predicts usage[T+h-1].  "Same day last week" = usage[T+h-1-7].
# Expressed as lag from T: (T - (T+h-1-7)) = 8-h.  For h=6,7 this drops below lag_min=3
# so we fall back to two-weeks-ago (lag 16-h).
_NAIVE_LAG_MAP: dict[int, str] = {
    1: "lag_7",
    2: "lag_6",
    3: "lag_5",
    4: "lag_4",
    5: "lag_3",
    6: "lag_9",  # two-weeks-ago same day for h=6
    7: "lag_8",  # two-weeks-ago same day for h=7
}

_QUANTILES: list[float] = [0.25, 0.50, 0.75]
_Q_COL_NAMES: list[str] = ["q25", "q50", "q75"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _available_feature_cols(df: pl.DataFrame) -> list[str]:
    """Return feature column names present in df (lag availability depends on lag_min)."""
    return [c for c in FEATURE_COLS if c in df.columns]


def _to_xy(feat_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target matrix Y from a built feature DataFrame."""
    X = feat_df.select(_available_feature_cols(feat_df)).to_numpy(allow_copy=True).astype(float)
    Y = feat_df.select(TARGET_COLS).to_numpy(allow_copy=True).astype(float)
    return X, Y


def _to_nf_df(df: pl.DataFrame) -> pl.DataFrame:
    """Convert a standard DataFrame to NeuralForecast long format (unique_id, ds, y)."""
    return df.select(
        pl.lit("sf").alias("unique_id"),
        pl.col(COL_DATE).cast(pl.Datetime).alias("ds"),
        pl.col(COL_TOTAL).alias("y"),
    )


def _parse_nf_quantile_cols(pred_df: pl.DataFrame, model_alias: str) -> tuple[str, str, str]:
    """Return (q25_col, q50_col, q75_col) column names from a NF predict output.

    NeuralForecast with MQLoss(quantiles=[0.25, 0.5, 0.75]) outputs columns:
    ``{alias}-lo-50.0``, ``{alias}-median``, ``{alias}-hi-50.0``
    """
    q25_col = f"{model_alias}-lo-50.0"
    q50_col = f"{model_alias}-median"
    q75_col = f"{model_alias}-hi-50.0"
    # Fallback: if naming differs, use the first three non-id columns
    cols = [c for c in pred_df.columns if c not in ("unique_id", "ds")]
    if q25_col not in pred_df.columns and len(cols) >= 3:
        q25_col, q50_col, q75_col = cols[0], cols[1], cols[2]
    return q25_col, q50_col, q75_col


# ---------------------------------------------------------------------------
# Tree model builders
# ---------------------------------------------------------------------------


def _make_tree_model(model_name: str, quantile: float) -> Any:
    """Return an untrained multi-output quantile estimator for tree-based models."""
    if model_name == "xgboost":
        import xgboost as xgb  # noqa: PLC0415

        base = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantile,
            max_depth=6,
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            verbosity=0,
        )
        return MultiOutputRegressor(base, n_jobs=-1)

    if model_name == "lightgbm":
        import lightgbm as lgb  # noqa: PLC0415

        base = lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            max_depth=6,
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            verbose=-1,
        )
        return MultiOutputRegressor(base, n_jobs=-1)

    if model_name == "random_forest":
        # RF uses a single model; quantiles are derived from tree spread at predict time
        return RandomForestRegressor(max_depth=12, n_estimators=300, n_jobs=-1, random_state=42)

    msg = f"Unknown tree model: {model_name}"
    raise ValueError(msg)


def _rf_quantile_predict(rf: RandomForestRegressor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict Q25/Q50/Q75 from a fitted RandomForestRegressor using tree spread.

    Returns three arrays each of shape (n_samples, n_targets).
    """
    # tree_preds shape: (n_estimators, n_samples, n_targets) for multi-output RF
    tree_preds = np.array([t.predict(X) for t in rf.estimators_])
    # Handle single-output RF (shape might be (n_estimators, n_samples))
    if tree_preds.ndim == 2:
        tree_preds = tree_preds[:, :, np.newaxis]
    q25 = np.percentile(tree_preds, 25, axis=0)
    q50 = np.percentile(tree_preds, 50, axis=0)
    q75 = np.percentile(tree_preds, 75, axis=0)
    return q25, q50, q75


def _post_process_quantiles(
    q25: np.ndarray, q50: np.ndarray, q75: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip quantiles to prevent crossing: q25 <= q50 <= q75."""
    q25 = np.minimum(q25, q50)
    q75 = np.maximum(q75, q50)
    return q25, q50, q75


# ---------------------------------------------------------------------------
# NF model builders
# ---------------------------------------------------------------------------


def _make_nf_model(model_name: str, h: int = FORECAST_HORIZON, input_size: int = 28) -> Any:
    """Return an untrained NeuralForecast model instance."""
    from neuralforecast.losses.pytorch import MQLoss  # noqa: PLC0415

    loss = MQLoss(quantiles=_QUANTILES)
    max_steps = 200  # conservative default; tune champion separately

    if model_name == "mlp":
        from neuralforecast.models import MLP  # noqa: PLC0415

        return MLP(h=h, input_size=input_size, loss=loss, max_steps=max_steps)

    if model_name == "nhits":
        from neuralforecast.models import NHITS  # noqa: PLC0415

        return NHITS(h=h, input_size=input_size, loss=loss, max_steps=max_steps)

    if model_name == "timesnet":
        from neuralforecast.models import TimesNet  # noqa: PLC0415

        return TimesNet(h=h, input_size=input_size, loss=loss, max_steps=max_steps)

    if model_name == "fedformer":
        from neuralforecast.models import FEDformer  # noqa: PLC0415

        return FEDformer(h=h, input_size=input_size, loss=loss, max_steps=max_steps)

    if model_name == "patchtst":
        from neuralforecast.models import PatchTST  # noqa: PLC0415

        return PatchTST(h=h, input_size=input_size, loss=loss, max_steps=max_steps)

    if model_name == "itransformer":
        from neuralforecast.models import iTransformer  # noqa: PLC0415

        return iTransformer(h=h, input_size=input_size, loss=loss, max_steps=max_steps, n_series=1)

    msg = f"Unknown NF model: {model_name}"
    raise ValueError(msg)


def _nf_rolling_predict(nf_model: Any, train_df: pl.DataFrame, test_df: pl.DataFrame) -> pl.DataFrame:
    """Run rolling 7-day predictions across a test set using a fitted NF model.

    After fitting on train_df, the function steps through the test set in
    FORECAST_HORIZON-day chunks, extending the context with actual values
    between each prediction. Returns a DataFrame with columns:
    Date, q25_pred, q50_pred, q75_pred.
    """
    from neuralforecast import NeuralForecast  # noqa: PLC0415

    nf_train = _to_nf_df(train_df)
    nf = NeuralForecast(models=[nf_model], freq="1d")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nf.fit(nf_train)

    model_alias = type(nf_model).__name__
    test_dates = test_df[COL_DATE].to_list()
    test_usage = test_df[COL_TOTAL].to_list()

    all_dates: list[datetime.date] = []
    all_q25: list[float] = []
    all_q50: list[float] = []
    all_q75: list[float] = []

    context = nf_train.clone()
    i = 0
    while i < len(test_dates):
        chunk_size = min(FORECAST_HORIZON, len(test_dates) - i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = nf.predict(df=context)

        q25_col, q50_col, q75_col = _parse_nf_quantile_cols(pred, model_alias)
        # pred has `h` rows for each unique_id; extract first `chunk_size` rows
        pred_sorted = pred.sort("ds")
        for j in range(chunk_size):
            all_dates.append(test_dates[i + j])
            all_q25.append(float(pred_sorted[q25_col][j]))
            all_q50.append(float(pred_sorted[q50_col][j]))
            all_q75.append(float(pred_sorted[q75_col][j]))

        # Extend context with actuals from this chunk
        new_rows = pl.DataFrame(
            {
                "unique_id": ["sf"] * chunk_size,
                "ds": [datetime.datetime(d.year, d.month, d.day) for d in test_dates[i : i + chunk_size]],
                "y": test_usage[i : i + chunk_size],
            },
        )
        context = pl.concat([context, new_rows])
        i += chunk_size

    return pl.DataFrame(
        {
            COL_DATE: all_dates,
            "q25_pred": all_q25,
            "q50_pred": all_q50,
            "q75_pred": all_q75,
        },
    )


# ---------------------------------------------------------------------------
# Baseline predictors
# ---------------------------------------------------------------------------


def _baseline_predict(model_name: str, train_df: pl.DataFrame, test_df: pl.DataFrame) -> pl.DataFrame:
    """Generate predictions for baseline models over the test set.

    Args:
        model_name: One of "naive_lag7" or "seasonal_mean".
        train_df: Training data (for seasonal mean lookup).
        test_df: Test data to predict over.

    Returns:
        DataFrame with columns: Date, q25_pred, q50_pred, q75_pred.
    """
    if model_name == "naive_lag7":
        return _naive_lag7_predict(test_df)
    if model_name == "seasonal_mean":
        return _seasonal_mean_predict(train_df, test_df)
    msg = f"Unknown baseline model: {model_name}"
    raise ValueError(msg)


def _naive_lag7_predict(test_df: pl.DataFrame) -> pl.DataFrame:
    """Naive lag-7 baseline: use same-weekday values from the prior week (or prior 2 weeks
    for horizons 6 and 7 where lag_min=3 prevents using the 1-week-ago value).
    """
    feat_df = build_inference_features_v1(test_df)
    dates: list[datetime.date] = []
    q25_list: list[float] = []
    q50_list: list[float] = []
    q75_list: list[float] = []

    for row in feat_df.iter_rows(named=True):
        dates.append(row[COL_DATE])
        preds = []
        for h in range(1, FORECAST_HORIZON + 1):
            lag_col = _NAIVE_LAG_MAP[h]
            if lag_col in row and row[lag_col] is not None:
                preds.append(float(row[lag_col]))
            else:
                # fallback: use whatever lag_7 is
                preds.append(float(row.get("lag_7", 0.0) or 0.0))
        # Naive doesn't produce a distributional forecast; use point prediction for all quantiles
        # Add ±10% as a rough uncertainty interval (recalibrate with actual quantile models)
        base = np.array(preds)
        q50_list.append(float(np.mean(base)))
        q25_list.append(float(np.mean(base) * 0.90))
        q75_list.append(float(np.mean(base) * 1.10))

    return pl.DataFrame({COL_DATE: dates, "q25_pred": q25_list, "q50_pred": q50_list, "q75_pred": q75_list})


def _seasonal_mean_predict(train_df: pl.DataFrame, test_df: pl.DataFrame) -> pl.DataFrame:
    """Seasonal mean baseline: predict using mean of same weekday × month from training set."""
    # Build (month, weekday) → mean table from train
    lookup = (
        train_df.with_columns(
            pl.col(COL_DATE).dt.month().alias("month"),
            pl.col(COL_DATE).dt.weekday().alias("weekday"),
        )
        .group_by(["month", "weekday"])
        .agg(pl.col(COL_TOTAL).mean().alias("mean_usage"))
    )
    lookup_dict: dict[tuple[int, int], float] = {
        (row["month"], row["weekday"]): row["mean_usage"] for row in lookup.iter_rows(named=True)
    }

    dates: list[datetime.date] = []
    q25_list: list[float] = []
    q50_list: list[float] = []
    q75_list: list[float] = []

    for row in test_df.iter_rows(named=True):
        d = row[COL_DATE]
        preds = []
        for h in range(FORECAST_HORIZON):
            target_date = d + datetime.timedelta(days=h)
            m, wd = target_date.month, target_date.weekday()
            val = lookup_dict.get((m, wd), float(train_df[COL_TOTAL].mean()))
            preds.append(val)
        base = np.array(preds)
        dates.append(d)
        q50_list.append(float(np.mean(base)))
        q25_list.append(float(np.mean(base) * 0.90))
        q75_list.append(float(np.mean(base) * 1.10))

    return pl.DataFrame({COL_DATE: dates, "q25_pred": q25_list, "q50_pred": q50_list, "q75_pred": q75_list})


# ---------------------------------------------------------------------------
# Walk-forward CV metrics
# ---------------------------------------------------------------------------


def _compute_fold_metrics(
    feat_test_df: pl.DataFrame,
    q50_pred_rows: np.ndarray,
    q25_pred_rows: np.ndarray,
    q75_pred_rows: np.ndarray,
    fold: int,
    model_name: str,
) -> list[dict[str, Any]]:
    """Compute per-horizon MAE and MAPE and return a list of metric dicts."""
    records = []
    Y_true = feat_test_df.select(TARGET_COLS).to_numpy().astype(float)
    for h_idx in range(FORECAST_HORIZON):
        y_true = Y_true[:, h_idx]
        y_q50 = q50_pred_rows[:, h_idx] if q50_pred_rows.ndim == 2 else q50_pred_rows
        y_q25 = q25_pred_rows[:, h_idx] if q25_pred_rows.ndim == 2 else q25_pred_rows
        y_q75 = q75_pred_rows[:, h_idx] if q75_pred_rows.ndim == 2 else q75_pred_rows
        mae = float(np.abs(y_true - y_q50).mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = float(np.where(y_true != 0, np.abs((y_true - y_q50) / y_true), 0).mean())
        records.append(
            {
                "fold": fold,
                "model": model_name,
                "horizon": h_idx + 1,
                "mae": mae,
                "mape": mape,
                "q25_mae": float(np.maximum(0.25 * (y_q25 - y_true), 0.75 * (y_true - y_q25)).mean()),
                "q75_mae": float(np.maximum(0.75 * (y_q75 - y_true), 0.25 * (y_true - y_q75)).mean()),
            },
        )
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_walk_forward_cv_v1(
    df: pl.DataFrame,
    model_names: list[str] | None = None,
    n_splits: int = 5,
) -> pl.DataFrame:
    """Run walk-forward (expanding window) cross-validation for all requested models.

    Each fold expands the training window by one year, with a test set of 364
    days. The minimum training set is 3 full years.

    Args:
        df: DataFrame with at least COL_DATE and COL_TOTAL, sorted ascending.
        model_names: Which models to evaluate. Defaults to ALL_MODEL_NAMES.
        n_splits: Number of folds. Defaults to 5.

    Returns:
        DataFrame with columns: fold, model, horizon, mae, mape, q25_mae, q75_mae.
    """
    if model_names is None:
        model_names = ALL_MODEL_NAMES

    df = df.sort(COL_DATE)
    start_year = df[COL_DATE][0].year
    # First test year = start_year + 3 (3 full years of training)
    first_test_year = start_year + 3

    all_records: list[dict[str, Any]] = []

    for fold_idx in range(n_splits):
        test_year = first_test_year + fold_idx
        test_start = datetime.date(test_year, 1, 1)
        test_end = datetime.date(test_year, 12, 31)
        if test_start > df[COL_DATE][-1]:
            break  # no more data

        train_df = df.filter(pl.col(COL_DATE) < test_start)
        test_df = df.filter((pl.col(COL_DATE) >= test_start) & (pl.col(COL_DATE) <= test_end))

        if len(train_df) < 365 * 3 or len(test_df) < 30:
            continue

        feat_train = build_features_v1(train_df)
        feat_test = build_features_v1(pl.concat([train_df, test_df])).filter(
            pl.col(COL_DATE) >= test_start,
        )
        fcols = _available_feature_cols(feat_train)
        X_train, Y_train = (
            feat_train.select(fcols).to_numpy().astype(float),
            feat_train.select(TARGET_COLS).to_numpy().astype(float),
        )
        X_test = feat_test.select(fcols).to_numpy().astype(float)

        for model_name in model_names:
            try:
                if model_name in TREE_MODELS:
                    _records = _cv_tree_model(model_name, X_train, Y_train, X_test, feat_test, fold_idx)
                elif model_name in NF_MODELS:
                    _records = _cv_nf_model(model_name, train_df, test_df, feat_test, fold_idx)
                else:
                    _records = _cv_baseline(model_name, train_df, test_df, feat_test, fold_idx)
                all_records.extend(_records)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Fold {fold_idx} model {model_name} failed: {exc}", stacklevel=2)

    return pl.DataFrame(all_records)


def _cv_tree_model(
    model_name: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    feat_test: pl.DataFrame,
    fold: int,
) -> list[dict[str, Any]]:
    if model_name == "random_forest":
        rf = RandomForestRegressor(max_depth=12, n_estimators=300, n_jobs=-1, random_state=42)
        rf.fit(X_train, Y_train)
        q25, q50, q75 = _rf_quantile_predict(rf, X_test)
    else:
        models = {}
        for q, q_col in zip(_QUANTILES, _Q_COL_NAMES):
            m = _make_tree_model(model_name, q)
            m.fit(X_train, Y_train)
            models[q_col] = m
        q50 = models["q50"].predict(X_test)
        q25 = models["q25"].predict(X_test)
        q75 = models["q75"].predict(X_test)
        q25, q50, q75 = _post_process_quantiles(q25, q50, q75)

    return _compute_fold_metrics(feat_test, q50, q25, q75, fold, model_name)


def _cv_nf_model(
    model_name: str,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feat_test: pl.DataFrame,
    fold: int,
) -> list[dict[str, Any]]:
    nf_model = _make_nf_model(model_name)
    pred_df = _nf_rolling_predict(nf_model, train_df, test_df)
    # Align predictions to feat_test rows by date
    pred_aligned = pred_df.join(feat_test.select([COL_DATE]), on=COL_DATE, how="inner")
    q25 = pred_aligned["q25_pred"].to_numpy().reshape(-1, 1).repeat(FORECAST_HORIZON, axis=1)
    q50 = pred_aligned["q50_pred"].to_numpy().reshape(-1, 1).repeat(FORECAST_HORIZON, axis=1)
    q75 = pred_aligned["q75_pred"].to_numpy().reshape(-1, 1).repeat(FORECAST_HORIZON, axis=1)
    # NF predicts the full 7-day window per step; use directly
    return _compute_fold_metrics(feat_test, q50, q25, q75, fold, model_name)


def _cv_baseline(
    model_name: str,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feat_test: pl.DataFrame,
    fold: int,
) -> list[dict[str, Any]]:
    pred_df = _baseline_predict(model_name, train_df, test_df)
    q25 = pred_df["q25_pred"].to_numpy().reshape(-1, 1).repeat(FORECAST_HORIZON, axis=1)
    q50 = pred_df["q50_pred"].to_numpy().reshape(-1, 1).repeat(FORECAST_HORIZON, axis=1)
    q75 = pred_df["q75_pred"].to_numpy().reshape(-1, 1).repeat(FORECAST_HORIZON, axis=1)
    return _compute_fold_metrics(feat_test, q50, q25, q75, fold, model_name)


def select_champion_v1(cv_results: pl.DataFrame) -> str:
    """Select the champion model from walk-forward CV results.

    Primary criterion: lowest mean Q50 MAE across all folds and horizons.
    Tiebreaker: lowest standard deviation of MAE across folds (most consistent).
    Any model worse than naive_lag7 across all folds is eliminated.

    Args:
        cv_results: DataFrame produced by run_walk_forward_cv_v1.

    Returns:
        Name of the champion model.
    """
    summary = (
        cv_results.group_by("model")
        .agg(
            pl.col("mae").mean().alias("mean_mae"),
            pl.col("mae").std().alias("std_mae"),
        )
        .sort("mean_mae")
    )
    # Eliminate models worse than naive_lag7
    naive_mae_row = summary.filter(pl.col("model") == "naive_lag7")
    if len(naive_mae_row) > 0:
        naive_mae = naive_mae_row["mean_mae"][0]
        summary = summary.filter(pl.col("mean_mae") <= naive_mae * 1.05)  # 5% tolerance

    if len(summary) == 0:
        return "naive_lag7"

    return str(summary["model"][0])


def train_champion_v1(df: pl.DataFrame, model_name: str) -> Any:
    """Train the champion model on the full dataset.

    Args:
        df: Full historical DataFrame with COL_DATE and COL_TOTAL.
        model_name: Name of the champion model (from ALL_MODEL_NAMES).

    Returns:
        A trained model artifact. For tree models: a dict with keys
        'model_name', 'q25', 'q50', 'q75', 'feature_cols'.
        For NF models: a dict with 'model_name', 'nf', 'train_df'.
        For baselines: a dict with 'model_name', 'train_df'.
    """
    df = df.sort(COL_DATE)

    if model_name in TREE_MODELS:
        feat_df = build_features_v1(df)
        fcols = _available_feature_cols(feat_df)
        X, Y = feat_df.select(fcols).to_numpy().astype(float), feat_df.select(TARGET_COLS).to_numpy().astype(float)

        if model_name == "random_forest":
            model = RandomForestRegressor(max_depth=12, n_estimators=300, n_jobs=-1, random_state=42)
            model.fit(X, Y)
            return {"model_name": model_name, "model_type": "rf", "model": model, "feature_cols": fcols}

        models = {}
        for q, q_col in zip(_QUANTILES, _Q_COL_NAMES):
            m = _make_tree_model(model_name, q)
            m.fit(X, Y)
            models[q_col] = m
        return {"model_name": model_name, "model_type": "tree", "models": models, "feature_cols": fcols}

    if model_name in NF_MODELS:
        from neuralforecast import NeuralForecast  # noqa: PLC0415

        nf_model = _make_nf_model(model_name)
        nf = NeuralForecast(models=[nf_model], freq="1d")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nf.fit(_to_nf_df(df))
        return {"model_name": model_name, "model_type": "nf", "nf": nf, "train_df": df}

    # Baselines
    return {"model_name": model_name, "model_type": "baseline", "train_df": df}


def predict_7day_v1(trained_model: dict[str, Any], df: pl.DataFrame) -> pl.DataFrame:
    """Generate a 7-day quantile forecast using a trained model artifact.

    The forecast covers dates T through T+6 where T is determined by
    appending a virtual row for today + 1 (the next gas day) and extracting
    its lag features.

    Args:
        trained_model: Artifact returned by train_champion_v1.
        df: Full historical context DataFrame (COL_DATE, COL_TOTAL).

    Returns:
        7-row DataFrame with columns: Date, q25, q50, q75.
    """
    df = df.sort(COL_DATE)
    last_date = df[COL_DATE][-1]
    model_name = trained_model["model_name"]
    model_type = trained_model["model_type"]

    # Next gas day T = last known + 4 (skipping T-3, T-2, T-1 observability gap)
    # Actually: last confirmed = last_date = T-3, so T = last_date + 3
    gas_day_t = last_date + datetime.timedelta(days=3)
    forecast_dates = [gas_day_t + datetime.timedelta(days=h) for h in range(FORECAST_HORIZON)]

    if model_type in ("tree", "rf"):
        # Append virtual rows for T..T+6 with NaN usage to compute lag features
        virtual_rows = pl.DataFrame(
            {
                COL_DATE: forecast_dates,
                COL_TOTAL: [None] * FORECAST_HORIZON,
            },
            schema={COL_DATE: pl.Date, COL_TOTAL: pl.Float64},
        )
        extended = pl.concat([df, virtual_rows])
        feat_df = build_inference_features_v1(extended)
        # Take the FORECAST_HORIZON rows for the forecast dates
        feat_rows = feat_df.filter(pl.col(COL_DATE).is_in(forecast_dates))
        if len(feat_rows) == 0:
            # fallback: use the last available row
            feat_rows = feat_df.tail(FORECAST_HORIZON)

        fcols = trained_model["feature_cols"]
        X = feat_rows.select(fcols).to_numpy().astype(float)

        if model_type == "rf":
            rf = trained_model["model"]
            q25_arr, q50_arr, q75_arr = _rf_quantile_predict(rf, X)
        else:
            models = trained_model["models"]
            q50_arr = models["q50"].predict(X)
            q25_arr = models["q25"].predict(X)
            q75_arr = models["q75"].predict(X)
            q25_arr, q50_arr, q75_arr = _post_process_quantiles(q25_arr, q50_arr, q75_arr)

        # Each row predicts horizon_1..horizon_7 for its own gas day.
        # For the multi-day forecast, use horizon_1 from each row's predictions
        # (row i → gas day T+i → horizon_1 = usage[T+i]).
        q25_vals = [
            float(q25_arr[i, 0]) if q25_arr.ndim == 2 else float(q25_arr[i])
            for i in range(min(len(feat_rows), FORECAST_HORIZON))
        ]
        q50_vals = [
            float(q50_arr[i, 0]) if q50_arr.ndim == 2 else float(q50_arr[i])
            for i in range(min(len(feat_rows), FORECAST_HORIZON))
        ]
        q75_vals = [
            float(q75_arr[i, 0]) if q75_arr.ndim == 2 else float(q75_arr[i])
            for i in range(min(len(feat_rows), FORECAST_HORIZON))
        ]

        return pl.DataFrame(
            {
                COL_DATE: forecast_dates[: len(q50_vals)],
                "q25": q25_vals,
                "q50": q50_vals,
                "q75": q75_vals,
            },
        )

    if model_type == "nf":
        from neuralforecast import NeuralForecast  # noqa: PLC0415

        nf: NeuralForecast = trained_model["nf"]
        model_alias = type(nf.models[0]).__name__

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = nf.predict()

        q25_col, q50_col, q75_col = _parse_nf_quantile_cols(pred, model_alias)
        pred_sorted = pred.sort("ds")
        dates_out = [d.date() if hasattr(d, "date") else d for d in pred_sorted["ds"].to_list()]

        return pl.DataFrame(
            {
                COL_DATE: dates_out[:FORECAST_HORIZON],
                "q25": pred_sorted[q25_col][:FORECAST_HORIZON].to_list(),
                "q50": pred_sorted[q50_col][:FORECAST_HORIZON].to_list(),
                "q75": pred_sorted[q75_col][:FORECAST_HORIZON].to_list(),
            },
        )

    # Baselines
    train_df = trained_model["train_df"]
    virtual_test = pl.DataFrame(
        {COL_DATE: forecast_dates, COL_TOTAL: [float(train_df[COL_TOTAL].mean())] * FORECAST_HORIZON},
    )
    pred_df = _baseline_predict(model_name, train_df, virtual_test)
    return pred_df.rename({"q25_pred": "q25", "q50_pred": "q50", "q75_pred": "q75"})


def save_forecast_v1(forecast_df: pl.DataFrame, path: str) -> None:
    """Save a forecast DataFrame to a Parquet file.

    Args:
        forecast_df: 7-row forecast DataFrame produced by predict_7day_v1.
        path: Absolute or relative path to write the Parquet file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    forecast_df.write_parquet(path)
