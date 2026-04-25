"""Endpoint handler for ``POST /predict``.

Runs inference on a single Iris feature vector using the loaded champion model
and appends the prediction — including all input features and a UTC timestamp —
to the rolling prediction log at ``/public/prediction_storage/predicted.csv``.

Routes:
    POST /predict: Classify a single Iris sample.
"""

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from iris_classifier_api.metadata.constants import CLASS_LABELS, FEATURE_COLUMNS
from iris_classifier_api.metadata.models import PredictRequest
from iris_classifier_api.metadata.state import state
from python.tasks.metadata.constants import IrisConstants

logger = logging.getLogger(__name__)

router = APIRouter()

# File-level lock — sufficient because uvicorn runs with a single worker
# (no forked processes) and async endpoints execute in a thread pool.
_predictions_lock = threading.Lock()


def _log_prediction(
    features: dict[str, float],
    pred_index: int,
    predicted_species: str,
) -> None:
    """Append a single prediction record to the prediction log CSV.

    Creates ``/public/prediction_storage/predicted.csv`` on the first call,
    then appends subsequent rows.  Writes are serialised with a threading lock
    to prevent data corruption when concurrent requests arrive.

    The CSV schema is:
        SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm,
        predicted_class, predicted_species, prediction_at

    Args:
        features: Dict mapping each feature column name to its float value.
        pred_index: Integer class index returned by the model (0–2).
        predicted_species: Human-readable species label for ``pred_index``.
    """
    predictions_path = (
        IrisConstants.PREDICTION_STORAGE_DIR / IrisConstants.PREDICTIONS_FILENAME
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        **features,
        "predicted_class": pred_index,
        "predicted_species": predicted_species,
        "prediction_at": datetime.now(timezone.utc).isoformat(),
    }

    df = pd.DataFrame([row])

    with _predictions_lock:
        write_header = not predictions_path.exists()
        df.to_csv(predictions_path, mode="a", header=write_header, index=False)


@router.post("/predict", summary="Classify an Iris sample")
def predict(request: PredictRequest) -> dict[str, Any]:
    """Run inference on a single Iris feature vector.

    The feature values are forwarded to the loaded
    :class:`mlflow.pyfunc.PyFuncModel` in the column order expected by the
    trained pipeline (``SepalLengthCm``, ``SepalWidthCm``,
    ``PetalLengthCm``, ``PetalWidthCm``).  Each prediction is persisted to
    the rolling log at ``/public/prediction_storage/predicted.csv`` for
    production drift monitoring.

    Args:
        request: Four numeric Iris measurements.

    Returns:
        A dictionary containing:

        - ``prediction`` (int): Integer class index (0–2).
        - ``species`` (str): Human-readable species label.

    Raises:
        HTTPException: 503 if the model is not loaded.
        HTTPException: 500 if the underlying model raises during inference.
    """
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check application startup logs.",
        )

    features = {
        "SepalLengthCm": request.sepal_length_cm,
        "SepalWidthCm": request.sepal_width_cm,
        "PetalLengthCm": request.petal_length_cm,
        "PetalWidthCm": request.petal_width_cm,
    }

    features_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    try:
        raw_prediction = state.model.predict(features_df)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    pred_index = int(raw_prediction[0])
    species = (
        CLASS_LABELS[pred_index]
        if 0 <= pred_index < len(CLASS_LABELS)
        else str(pred_index)
    )

    try:
        _log_prediction(features, pred_index, species)
    except Exception as exc:  # noqa: BLE001
        # Prediction logging must never block the inference response.
        logger.warning("Failed to log prediction: %s", exc)

    return {"prediction": pred_index, "species": species}
