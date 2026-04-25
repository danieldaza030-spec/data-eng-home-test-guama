"""Pydantic request/response models and in-memory model state for the API.

This module defines the data contracts used by the API endpoints as well as
the :class:`ModelState` container that holds the active MLflow model and its
associated metadata between requests.

Classes:
    ModelState: Singleton container for the active in-memory model and its
        MLflow run metadata.
    UpdateModelRequest: Request body schema for ``POST /update_model``.
    PredictRequest: Request body schema for ``POST /predict``.
"""

from typing import Any

from pydantic import BaseModel, Field

from python.tasks.metadata.constants import MLFlowConstants


class ModelState:
    """Singleton container for the active in-memory model and its metadata.

    Attributes:
        model: Loaded :class:`mlflow.pyfunc.PyFuncModel` instance, or
            ``None`` before the first successful load.
        model_name: Registered model name currently loaded.
        model_version: Registry version string of the loaded model.
        run_id: MLflow run ID that produced the loaded model version.
        metrics: Logged metrics dict from the originating run.
        params: Logged parameters dict from the originating run.
        tags: User-defined tags from the originating run (``mlflow.*``
            system tags are excluded).
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.model_name: str = MLFlowConstants.REGISTERED_MODEL_NAME
        self.model_version: str | None = None
        self.run_id: str | None = None
        self.metrics: dict[str, Any] = {}
        self.params: dict[str, Any] = {}
        self.tags: dict[str, Any] = {}


class UpdateModelRequest(BaseModel):
    """Request body for ``POST /update_model``.

    Attributes:
        model_name: Optional registered model name.  When ``None`` the API
            reloads the ``@champion`` version of the default model.
    """

    model_name: str | None = Field(
        default=None,
        description=(
            "Registered MLflow model name to load.  "
            "Omit to reload the @champion of the default model."
        ),
    )


class PredictRequest(BaseModel):
    """Feature vector for a single Iris classification request.

    Attributes:
        sepal_length_cm: Sepal length in centimetres.
        sepal_width_cm: Sepal width in centimetres.
        petal_length_cm: Petal length in centimetres.
        petal_width_cm: Petal width in centimetres.
    """

    sepal_length_cm: float = Field(..., gt=0, description="Sepal length in cm.")
    sepal_width_cm: float = Field(..., gt=0, description="Sepal width in cm.")
    petal_length_cm: float = Field(..., gt=0, description="Petal length in cm.")
    petal_width_cm: float = Field(..., gt=0, description="Petal width in cm.")
