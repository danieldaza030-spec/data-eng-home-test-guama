"""Application state and MLflow model loading helpers.

This module owns the single shared :data:`state` instance and the two
functions that resolve the tracking URI and load a registered model into
memory.  Both the lifespan hook in ``main`` and the update-model endpoint
import from here so that there is a single source of truth for the active
model.

Attributes:
    state: Module-level singleton of :class:`~metadata.models.ModelState`
        shared across all request handlers.

Functions:
    resolve_tracking_uri: Return the MLflow tracking URI from the environment.
    load_model: Load a registered champion model into :data:`state`.
"""

import logging
import os

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from python.tasks.metadata.constants import MLFlowConstants

from iris_classifier_api.metadata.models import ModelState

logger = logging.getLogger(__name__)

state: ModelState = ModelState()


def resolve_tracking_uri() -> str:
    """Return the MLflow tracking URI from the environment or fall back to default.

    Returns:
        The tracking URI string to pass to :func:`mlflow.set_tracking_uri`.
    """
    return os.environ.get(
        MLFlowConstants.TRACKING_URI_ENV,
        MLFlowConstants.DEFAULT_TRACKING_URI,
    )


def load_model(model_name: str | None = None) -> None:
    """Load a registered MLflow model into :data:`state`.

    Resolves the ``@champion`` alias when no specific model name is given.
    After loading, updates all metadata fields on :data:`state` atomically.

    Args:
        model_name: Optional registered model name override.  When ``None``,
            the default :attr:`~python.data_models.constants.MLFlowConstants.REGISTERED_MODEL_NAME`
            is used together with the ``@champion`` alias.

    Raises:
        MlflowException: If the model or alias cannot be found in the
            registry, or if the tracking server is unreachable.
    """
    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    resolved_name = model_name or MLFlowConstants.REGISTERED_MODEL_NAME
    alias = MLFlowConstants.CHAMPION_ALIAS
    model_uri = f"models:/{resolved_name}@{alias}"

    logger.info("Loading model from URI '%s'.", model_uri)
    loaded = mlflow.pyfunc.load_model(model_uri)

    version_info = client.get_model_version_by_alias(resolved_name, alias)
    run = client.get_run(version_info.run_id)

    # Commit state atomically after all remote calls succeed
    state.model = loaded
    state.model_name = resolved_name
    state.model_version = version_info.version
    state.run_id = version_info.run_id
    state.metrics = dict(run.data.metrics)
    state.params = dict(run.data.params)
    state.tags = {
        k: v
        for k, v in run.data.tags.items()
        if not k.startswith("mlflow.")
    }

    logger.info(
        "Model '%s' v%s loaded (run_id=%s).",
        resolved_name,
        version_info.version,
        version_info.run_id,
    )
