from __future__ import annotations

import datetime
import pickle
import re
from pathlib import Path
from typing import Any

import polars as pl

from dgup._internal.constants import COL_DATE
from dgup._internal.forecast import predict_7day_v1, save_forecast_v1, train_champion_v1

_MODELS_DIR = Path("data/silver/models")


def retrain_weekly_v1(model_name: str, df: pl.DataFrame) -> dict[str, Any]:
    """Retrain the champion model on the full expanded training history.

    Intended to run every Monday morning. The supplied DataFrame should
    include all actuals available at the time of retraining (training history
    grows week by week — no sliding window).

    The trained model artifact is persisted to
    ``data/silver/models/{model_name}_v{yyyymmdd}.pkl`` where the date is
    today. Neuralforecast models (which are not pickle-serialisable in all
    versions) are saved using their own ``save`` method if available;
    otherwise they are pickled.

    Args:
        model_name: Name of the champion model to retrain.
        df: Full historical DataFrame with COL_DATE and COL_TOTAL, sorted
            ascending by date.

    Returns:
        The trained model artifact dict (same format as train_champion_v1).
    """
    trained = train_champion_v1(df, model_name)
    _save_model(trained, model_name)
    return trained


def _save_model(trained: dict[str, Any], model_name: str) -> None:
    """Persist a trained model artifact to disk."""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().strftime("%Y%m%d")
    artifact_path = _MODELS_DIR / f"{model_name}_v{today}.pkl"

    # NeuralForecast objects have a native save method
    if trained.get("model_type") == "nf":
        nf = trained["nf"]
        if hasattr(nf, "save"):
            save_dir = _MODELS_DIR / f"{model_name}_v{today}_nf"
            save_dir.mkdir(parents=True, exist_ok=True)
            nf.save(str(save_dir))
            # Pickle the metadata dict (without the heavy nf object)
            meta = {k: v for k, v in trained.items() if k != "nf"}
            meta["nf_save_path"] = str(save_dir)
            with artifact_path.open("wb") as f:
                pickle.dump(meta, f)
            return

    with artifact_path.open("wb") as f:
        pickle.dump(trained, f)


def load_latest_model_v1(model_name: str) -> dict[str, Any]:
    """Load the most recently saved model artifact for the given model name.

    Args:
        model_name: The model name whose latest artifact should be loaded.

    Returns:
        The trained model artifact dict.

    Raises:
        FileNotFoundError: If no saved artifacts exist for the given model.
    """
    pattern = re.compile(rf"^{re.escape(model_name)}_v(\d{{8}})\.pkl$")
    candidates = sorted(
        (p for p in _MODELS_DIR.iterdir() if pattern.match(p.name)),
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        msg = f"No saved model artifacts found for '{model_name}' in {_MODELS_DIR}"
        raise FileNotFoundError(msg)

    artifact_path = candidates[0]
    with artifact_path.open("rb") as f:
        artifact = pickle.load(f)  # noqa: S301

    # Restore NeuralForecast object if it was saved via nf.save()
    if "nf_save_path" in artifact:
        from neuralforecast import NeuralForecast  # noqa: PLC0415

        artifact["nf"] = NeuralForecast.load(artifact["nf_save_path"])

    return artifact


def run_daily_forecast_v1(
    model_name: str,
    df: pl.DataFrame,
    output_path: str | None = None,
) -> pl.DataFrame:
    """Load the latest trained model and produce a 7-day quantile forecast.

    This is the daily entry-point: it loads the most recently saved model
    artifact (trained on the preceding Monday), generates a 7-day forecast,
    and optionally saves it to a date-stamped Parquet file.

    Args:
        model_name: Champion model name.
        df: Full historical context DataFrame (COL_DATE, COL_TOTAL).
        output_path: Optional path override for saving the forecast Parquet.
            If not given, saves to
            ``data/silver/forecast_{model_name}_{yyyymmdd}.parquet``.

    Returns:
        7-row forecast DataFrame with columns: Date, q25, q50, q75.
    """
    trained = load_latest_model_v1(model_name)
    forecast_df = predict_7day_v1(trained, df)

    today = datetime.date.today().strftime("%Y%m%d")
    path = output_path or f"data/silver/forecast_{model_name}_{today}.parquet"
    save_forecast_v1(forecast_df, path)
    return forecast_df
