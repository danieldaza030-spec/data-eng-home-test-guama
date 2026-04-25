"""FastAPI application for serving the Iris classification champion model.

The model is resolved from the MLflow Model Registry using the
``@champion`` alias and loaded exactly once during application startup via
FastAPI's lifespan context manager.  All three endpoints share the same
in-memory model instance stored in :data:`~iris_classifier_api.metadata.state.state`.

Endpoints:
    GET  /info          – Current model name and MLflow run metadata.
    POST /update_model  – Hot-reload the in-memory model without restarting.
    POST /predict       – Classify a single Iris feature vector.

Environment variables:
    MLFLOW_TRACKING_URI: MLflow tracking server URI.
        Defaults to ``http://mlflow:5000``.

Note on native MLflow model updates:
    MLflow itself provides ``mlflow models serve`` (CLI) which starts a
    REST server exposing ``/invocations``.  That server resolves the model
    URI on startup, so re-deploying with a new alias is the intended update
    mechanism — there is no built-in hot-reload endpoint.  The idiomatic
    approach in the Model Registry is to **reassign the** ``@champion``
    **alias** to a newer version; any service that loads by alias picks up
    the new version on the next ``mlflow.pyfunc.load_model`` call, which is
    exactly what ``POST /update_model`` triggers here.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from mlflow.exceptions import MlflowException

from iris_classifier_api.endpoints import info, predict, update_model
from iris_classifier_api.metadata.state import load_model, state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context: load the champion model on startup.

    The model is loaded synchronously during the startup phase so that the
    first request is never blocked by a cold-load.  If the ``@champion``
    alias does not yet exist in the registry (e.g. no training run has been
    promoted yet), startup continues without a model — ``GET /info`` and
    ``POST /predict`` will return **503** until a model is loaded via
    ``POST /update_model``.  On shutdown the in-memory reference is released.

    Args:
        app: The FastAPI application instance (injected by the framework).

    Yields:
        Control to the application router after the startup phase completes.
    """
    try:
        load_model()
    except MlflowException as exc:
        logger.warning(
            "Model could not be loaded at startup — the API will run without "
            "an active model until POST /update_model is called. Reason: %s",
            exc,
        )
    yield
    state.model = None
    logger.info("Model released from memory.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Iris Classifier API",
    description=(
        "REST API that serves the Iris classification champion model "
        "from the MLflow Model Registry."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(info.router)
app.include_router(update_model.router)
app.include_router(predict.router)
