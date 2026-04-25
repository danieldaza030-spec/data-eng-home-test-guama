"""Endpoint handler for ``GET /info``.

Returns the name and MLflow metadata of the currently loaded champion model.

Routes:
    GET /info: Current model metadata.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from iris_classifier_api.metadata.state import state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/info", summary="Current model metadata")
def info() -> dict[str, Any]:
    """Return the name and MLflow metadata of the currently loaded model.

    Returns:
        A dictionary containing:

        - ``model_name`` (str): Registered model name.
        - ``model_version`` (str): Registry version number.
        - ``run_id`` (str): MLflow run ID that produced this version.
        - ``metrics`` (dict): Logged metrics from the training run.
        - ``params`` (dict): Logged hyperparameters from the training run.
        - ``tags`` (dict): User-defined run tags (system tags excluded).

    Raises:
        HTTPException: 503 if no model has been loaded yet.
    """
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check application startup logs.",
        )

    return {
        "model_name": state.model_name,
        "model_version": state.model_version,
        "run_id": state.run_id,
        "metrics": state.metrics,
        "params": state.params,
        "tags": state.tags,
    }
