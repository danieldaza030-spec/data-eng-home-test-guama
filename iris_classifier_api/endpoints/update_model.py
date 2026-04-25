"""Endpoint handler for ``POST /update_model``.

Hot-reloads the in-memory champion model without restarting the API process.

Routes:
    POST /update_model: Reload the champion model from the MLflow registry.
"""

import logging

from fastapi import APIRouter, HTTPException
from mlflow.exceptions import MlflowException

from iris_classifier_api.metadata.models import UpdateModelRequest
from iris_classifier_api.metadata.state import load_model, state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/update_model", summary="Hot-reload the in-memory model")
def update_model(request: UpdateModelRequest) -> dict[str, str]:
    """Reload the champion model (or a named model) without restarting the API.

    This endpoint implements the standard MLflow alias-based update workflow:
    promote a new model version to ``@champion`` in the Model Registry, then
    call this endpoint — no downtime, no restart required.

    Note:
        MLflow's own ``mlflow models serve`` CLI does **not** expose a
        hot-reload endpoint; redeployment is its intended update path.
        For programmatic updates, reassigning the ``@champion`` alias and
        calling this endpoint is the recommended pattern when self-hosting.

    Args:
        request: Optional body with a ``model_name`` field.  Omit or set to
            ``null`` to reload the default champion model.

    Returns:
        A confirmation dictionary with keys ``status``, ``model_name``,
        and ``model_version``.

    Raises:
        HTTPException: 404 if the model or alias does not exist in the
            registry.
        HTTPException: 500 for any other MLflow tracking error.
    """
    try:
        load_model(request.model_name)
    except MlflowException as exc:
        status_code = 404 if "RESOURCE_DOES_NOT_EXIST" in str(exc) else 500
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    return {
        "status": "updated",
        "model_name": state.model_name,
        "model_version": str(state.model_version),
    }
