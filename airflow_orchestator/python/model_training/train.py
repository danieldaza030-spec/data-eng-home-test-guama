"""Core training services for the Iris classification pipeline.

This module loads the persisted feature store, performs dataset splitting, and
trains scikit-learn classifiers while logging parameters, metrics, plots, and
artifacts to MLflow. Structured responses are represented with Pydantic models
to avoid undocumented dictionary payloads across the training flow.
"""

import itertools
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend required in containerised environments.
import matplotlib.pyplot as plt  # noqa: E402
import mlflow
import mlflow.sklearn
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from python.tasks.metadata.constants import IrisConstants, MLFlowConstants
from python.tasks.metadata.iris_schema import IRIS_FEATURE_COLS, IRIS_TARGET_COL
from python.tasks.metadata.pipeline_responses import TrainingMetrics

logger = logging.getLogger(__name__)

_TEST_SIZE: float = 0.2
_RANDOM_STATE: int = 42
_CLASS_LABELS: list[str] = [
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica",
]

# ---------------------------------------------------------------------------
# Hyperparameter grids (one grid + fixed-params pair per model family)
# ---------------------------------------------------------------------------

SVM_PARAM_GRID: dict[str, list] = {
    "C": [0.1, 1.0, 10.0, 100.0],
    "kernel": ["linear", "rbf"],
}
SVM_FIXED_PARAMS: dict = {"random_state": 42}

LR_PARAM_GRID: dict[str, list] = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "solver": ["lbfgs", "saga"],
}
LR_FIXED_PARAMS: dict = {"max_iter": 200, "random_state": 42}

KNN_PARAM_GRID: dict[str, list] = {
    "n_neighbors": [3, 5, 7, 11],
    "weights": ["uniform", "distance"],
}
KNN_FIXED_PARAMS: dict = {}


def load_feature_store(
    parquet_path: str | None = None,
    training_data: Literal["all", "latest"] = "all",
) -> pd.DataFrame:
    """Load the Iris feature store from a Parquet file.

    Args:
        parquet_path: Absolute path to the Parquet file.  When *None*, defaults
            to ``/public/iris_features.parquet`` as defined by
            :attr:`~python.data_models.constants.IrisConstants.PUBLIC_DIR` and
            :attr:`~python.data_models.constants.IrisConstants.PARQUET_FILENAME`.
        training_data: ``"all"`` (default) loads every row in the feature
            store; ``"latest"`` filters the DataFrame to only the rows from
            the most recently ingested batch, identified by comparing
            ``processed_at`` dates.  Use ``"latest"`` to train exclusively on
            the newest data when the feature store accumulates multiple batches
            via ``write_mode="append"``.

    Returns:
        A :class:`~pandas.DataFrame` containing the raw numeric feature
        columns, an integer-encoded ``Species`` target column, and a
        ``processed_at`` UTC timestamp column.

    Raises:
        FileNotFoundError: If the resolved Parquet path does not exist on the
            filesystem.
        ValueError: If the file is empty, the required feature and target
            columns are missing, or no rows match the latest batch date.
    """
    path = (
        Path(parquet_path)
        if parquet_path
        else IrisConstants.FEATURE_STORAGE_DIR / IrisConstants.FEATURE_STORE_FILENAME
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Feature store not found at '{path}'. "
            "Run an ingest DAG first to generate it."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Feature store at '{path}' is empty.")

    required_cols = set(IRIS_FEATURE_COLS + [IRIS_TARGET_COL])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Feature store is missing required columns: {sorted(missing)}. "
            f"Columns present: {sorted(df.columns.tolist())}"
        )

    if training_data == "latest" and "processed_at" in df.columns:
        latest_date = pd.to_datetime(df["processed_at"]).dt.date.max()
        df = df[pd.to_datetime(df["processed_at"]).dt.date == latest_date].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"No rows found for the latest batch date '{latest_date}' "
                "in the feature store."
            )
        logger.info(
            "training_data='latest' — filtered to batch date %s (%d rows).",
            latest_date,
            len(df),
        )

    logger.info(
        "Feature store loaded from '%s' — %d rows, columns: %s.",
        path,
        len(df),
        df.columns.tolist(),
    )
    return df


def split_features_target(
    df: pd.DataFrame,
    test_size: float = _TEST_SIZE,
    random_state: int = _RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the feature-store DataFrame into stratified train and test sets.

    Selects only the columns listed in ``IRIS_FEATURE_COLS`` as input features
    and ``IRIS_TARGET_COL`` as the target. The ``processed_at`` column and any
    other auxiliary columns are ignored.

    Features in the feature store are stored as raw (unscaled) numeric values.
    Scaling is delegated to the sklearn ``Pipeline`` that wraps each classifier,
    so the fitted scaler is persisted alongside the model artefact and applied
    automatically at inference time — no leakage is possible.

    Args:
        df: Feature-store DataFrame returned by :func:`load_feature_store`.
            Numeric feature columns must contain raw, unscaled values.
        test_size: Proportion of the dataset to include in the test split.
            Must be in the range ``(0.0, 1.0)``. Defaults to ``0.2``.
        random_state: Random seed for reproducibility. Defaults to ``42``.
            Varying this seed is a useful sanity check: if metrics are
            consistently perfect across many seeds the dataset is genuinely
            separable; if they vary, evaluate further for data quality issues.

    Returns:
        A four-element tuple ``(X_train, X_test, y_train, y_test)`` where:

        - ``X_train`` and ``X_test`` are :class:`~pandas.DataFrame` objects
          containing the raw (unscaled) numeric features.
        - ``y_train`` and ``y_test`` are :class:`~pandas.Series` objects
          containing the integer-encoded target labels.
    """
    X = df[IRIS_FEATURE_COLS]
    y = df[IRIS_TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        "Train/test split — train: %d rows, test: %d rows (test_size=%.2f).",
        len(X_train),
        len(X_test),
        test_size,
    )
    return X_train, X_test, y_train, y_test


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: list[str],
    title: str,
) -> str:
    """Render a confusion matrix as a PNG and return the temporary file path.

    The caller is responsible for removing the returned temporary file after
    it has been logged to MLFlow.

    Args:
        cm: Confusion matrix array of shape ``(n_classes, n_classes)`` as
            returned by :func:`sklearn.metrics.confusion_matrix`.
        class_labels: Human-readable class label strings in the same order as
            the confusion matrix axes.
        title: Figure title string, typically including the model name.

    Returns:
        Absolute path to the written temporary PNG file.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_labels))
    ax.set(
        xticks=tick_marks,
        yticks=tick_marks,
        xticklabels=class_labels,
        yticklabels=class_labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()

    safe_title = title.replace(" ", "_").replace("—", "").replace("/", "_")
    tmp = tempfile.NamedTemporaryFile(
        suffix=".png",
        prefix=f"confusion_matrix_{safe_title}_",
        delete=False,
    )
    fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def _ensure_experiment_with_proxy_uri(
    experiment_name: str,
    tracking_uri: str,
) -> None:
    """Create or migrate an MLFlow experiment to use a proxy-compatible artifact URI.

    Queries ALL experiments (active + soft-deleted) to handle every state that
    can accumulate across retried Airflow task attempts:

    * **Not found**: created fresh with ``mlflow-artifacts:/``.
    * **Active, correct URI**: no-op.
    * **Active, local URI**: renamed to free the name, then recreated.
    * **Soft-deleted holding the name**: restored, renamed, re-deleted to unblock
      the ``UNIQUE (name)`` constraint, then a fresh experiment is created.

    Args:
        experiment_name: MLFlow experiment name.
        tracking_uri: Fully-qualified MLFlow tracking server URI.
    """
    client = MlflowClient(tracking_uri=tracking_uri)

    all_matching = [
        exp
        for exp in client.search_experiments(view_type=ViewType.ALL)
        if exp.name == experiment_name
    ]
    active = [e for e in all_matching if e.lifecycle_stage == "active"]
    deleted = [e for e in all_matching if e.lifecycle_stage == "deleted"]

    if active:
        exp = active[0]
        if exp.artifact_location.startswith("mlflow-artifacts:"):
            return  # Already correct — nothing to do.

        stale_name = f"{experiment_name}_stale_{exp.experiment_id}"
        logger.warning(
            "Experiment '%s' has a local artifact location ('%s'). "
            "Renaming to '%s' and recreating with 'mlflow-artifacts:/' URI.",
            experiment_name,
            exp.artifact_location,
            stale_name,
        )
        client.rename_experiment(exp.experiment_id, stale_name)

    # Free any soft-deleted rows holding the name (they block the UNIQUE constraint).
    for exp in deleted:
        stale_name = f"{experiment_name}_stale_{exp.experiment_id}"
        try:
            client.restore_experiment(exp.experiment_id)
            client.rename_experiment(exp.experiment_id, stale_name)
            client.delete_experiment(exp.experiment_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not clear soft-deleted experiment %s: %s",
                exp.experiment_id,
                exc,
            )

    try:
        client.create_experiment(
            name=experiment_name,
            artifact_location="mlflow-artifacts:/",
        )
        logger.info(
            "Created MLFlow experiment '%s' with artifact_location='mlflow-artifacts:/'.",
            experiment_name,
        )
    except MlflowException as exc:
        if "UniqueViolation" in str(exc) or "already exists" in str(exc).lower():
            logger.info(
                "Experiment '%s' was created by a concurrent task; continuing.",
                experiment_name,
            )
        else:
            raise


def register_best_model(
    model_name: str,
    best_run_id: str,
    candidate_f1: float,
    dataset_version: str,
    tracking_uri: str,
    training_dates: list[str] | None = None,
) -> tuple[str, int]:
    """Register the overall best model in the MLflow Model Registry.

    All classifiers share a single registered model entry
    (:attr:`~python.data_models.constants.MLFlowConstants.REGISTERED_MODEL_NAME`,
    e.g. ``iris_classifier``). A new version is **always** created from the
    candidate run's ``model/`` artefact. The ``champion`` alias is moved to
    the new version only when *candidate_f1* strictly exceeds the metric of
    the current champion, ensuring the alias always points to the best model
    seen so far.

    The ``training_dates`` list (sorted ISO date strings of every
    ``processed_at`` date present in the training set) is persisted as the
    ``"training_dates"`` tag on the model version. The drift-monitor DAG reads
    this tag to filter the feature store to exactly the rows the model was
    trained on, avoiding false-positive or false-negative drift signals caused
    by comparing predictions against a different data slice.

    Registration steps:

    1. Ensure the registered model entry exists (created on first run).
    2. Create a new auto-incremented model version from the best child run's
       ``model/`` artefact.
    3. Tag the new version with ``dataset_version``, ``source_run_id``,
       ``model_type``, and ``training_dates`` for traceability.
    4. Move the ``champion`` alias to the new version only if it beats the
       current champion.

    Args:
        model_name: Short classifier identifier (e.g. ``"svm"``). Stored as a
            ``model_type`` tag on the new version for traceability.
        best_run_id: MLflow run ID of the best child run whose ``model/``
            artefact will be registered.
        candidate_f1: ``f1_macro`` score of the candidate model. Used to
            decide whether to promote the new version to champion.
        dataset_version: ISO-8601 string of the ``processed_at`` timestamp
            of the feature-store snapshot used for training.
        tracking_uri: Fully-qualified MLflow tracking server URI.
        training_dates: Sorted list of ISO date strings (``YYYY-MM-DD``) for
            every ``processed_at`` date present in the training set. Used by
            the drift-monitor to reconstruct the exact training distribution.
            When ``None`` or empty, the tag is omitted.

    Returns:
        A two-element tuple ``(registered_model_name, version_number)`` where
        *version_number* is the newly created integer version.

    Raises:
        mlflow.exceptions.MlflowException: If the tracking server is
            unreachable or the model artefact cannot be located.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    registered_name = MLFlowConstants.REGISTERED_MODEL_NAME

    # ------------------------------------------------------------------
    # Ensure the registered model entry exists.
    # ------------------------------------------------------------------
    try:
        client.get_registered_model(registered_name)
    except Exception:  # noqa: BLE001 — model entry does not exist yet
        client.create_registered_model(
            name=registered_name,
            description="Best Iris classifier across all model families.",
        )
        logger.info(
            "Created registered model '%s' in the MLflow Model Registry.",
            registered_name,
        )

    # ------------------------------------------------------------------
    # Always register a new version.
    # ------------------------------------------------------------------
    # Build version tags — include training_dates when available.
    version_tags: dict[str, str] = {
        "dataset_version": dataset_version,
        "source_run_id": best_run_id,
        "model_type": model_name,
    }
    if training_dates:
        version_tags["training_dates"] = ",".join(training_dates)

    model_version = client.create_model_version(
        name=registered_name,
        source=f"runs:/{best_run_id}/model",
        run_id=best_run_id,
        tags=version_tags,
    )
    logger.info(
        "Registered '%s' version %s (model_type=%s, f1_macro=%.4f, run_id=%s).",
        registered_name,
        model_version.version,
        model_name,
        candidate_f1,
        best_run_id,
    )

    # ------------------------------------------------------------------
    # Promote to champion only when the candidate beats the current one.
    # ------------------------------------------------------------------
    champion_f1: float = -1.0
    try:
        champion_version = client.get_model_version_by_alias(
            name=registered_name,
            alias=MLFlowConstants.CHAMPION_ALIAS,
        )
        champion_run = client.get_run(champion_version.run_id)
        champion_f1 = float(champion_run.data.metrics.get("f1_macro", -1.0))
        logger.info(
            "Current champion: version %s (run_id=%s, f1_macro=%.4f). "
            "Candidate: model_type=%s, f1_macro=%.4f.",
            champion_version.version,
            champion_version.run_id,
            champion_f1,
            model_name,
            candidate_f1,
        )
    except Exception:  # noqa: BLE001 — no champion alias set yet
        logger.info(
            "No '%s' champion found — promoting version %s as first champion.",
            MLFlowConstants.CHAMPION_ALIAS,
            model_version.version,
        )

    if candidate_f1 > champion_f1:
        client.set_registered_model_alias(
            name=registered_name,
            alias=MLFlowConstants.CHAMPION_ALIAS,
            version=model_version.version,
        )
        logger.info(
            "New champion: '%s' version %s (model_type=%s, f1_macro=%.4f). "
            "Alias '%s' updated. source_run_id=%s.",
            registered_name,
            model_version.version,
            model_name,
            candidate_f1,
            MLFlowConstants.CHAMPION_ALIAS,
            best_run_id,
        )
    else:
        logger.info(
            "Candidate f1_macro=%.4f does not beat champion f1_macro=%.4f "
            "— version %s registered but alias '%s' not moved.",
            candidate_f1,
            champion_f1,
            model_version.version,
            MLFlowConstants.CHAMPION_ALIAS,
        )

    return registered_name, int(model_version.version)


def train_and_log_grid(
    model_name: str,
    model_class: type,
    param_grid: dict[str, list],
    fixed_params: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    dataset_version: str,
    experiment_name: str = MLFlowConstants.EXPERIMENT_NAME,
) -> dict[str, Any]:
    """Train a classifier over a hyperparameter grid and log each combination.

    Configures the MLFlow tracking URI and experiment (same as
    :func:`train_and_log_model`), then creates a **parent run** named
    *model_name*.  For every combination in the Cartesian product of
    *param_grid* values a **nested child run** is started, logging:

    * all hyperparameters (fixed + variable),
    * accuracy, macro precision, macro recall, and macro F1,
    * a confusion-matrix PNG under ``plots/``,
    * the fitted model artefact under ``model/``.

    After all child runs finish, the parent run is annotated with
    ``best_*`` metrics and ``best_*`` params (winner by ``f1_macro``) and
    a ``best_child_run_id`` tag, enabling direct comparison in the MLFlow UI
    parallel-coordinates chart.

    Args:
        model_name: Human-readable identifier used as the parent run name
            and ``model_type`` tag (e.g. ``"svm"``).
        model_class: Scikit-learn estimator **class** (not an instance).
            Instantiated once per grid combination with ``{**fixed_params,
            **combo}``.
        param_grid: Hyperparameter grid to sweep.  Each key maps to a list
            of candidate values; every combination is trained as a child run.
        fixed_params: Hyperparameters kept constant across all combinations
            (e.g. ``{"random_state": 42}``).
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target vector (integer-encoded class labels).
        y_test: Test target vector (integer-encoded class labels).
        dataset_version: ISO-8601 string of the ``processed_at`` timestamp.
            Logged as a tag on the parent run for data lineage.
        experiment_name: MLFlow experiment name.  Defaults to
            :attr:`~python.data_models.constants.MLFlowConstants.EXPERIMENT_NAME`.

    Returns:
        Dictionary with the parent run metadata and the best child metrics.

    Raises:
        mlflow.exceptions.MlflowException: If the MLFlow tracking server is
            unreachable or any run operation fails.
    """
    tracking_uri = os.environ.get(
        MLFlowConstants.TRACKING_URI_ENV,
        MLFlowConstants.DEFAULT_TRACKING_URI,
    )
    mlflow.set_tracking_uri(tracking_uri)
    _ensure_experiment_with_proxy_uri(experiment_name, tracking_uri)
    mlflow.set_experiment(experiment_name)

    keys = list(param_grid.keys())
    combinations: list[dict] = [
        dict(zip(keys, combo))
        for combo in itertools.product(*param_grid.values())
    ]

    best_metrics: TrainingMetrics | None = None
    best_params: dict[str, Any] | None = None
    best_run_id: str | None = None

    with mlflow.start_run(run_name=model_name) as parent_run:
        mlflow.set_tags(
            {
                "model_type": model_name,
                "dataset_version": dataset_version,
                "grid_search": "true",
                "n_combinations": str(len(combinations)),
            }
        )
        mlflow.log_params(
            {f"fixed_{k}": v for k, v in fixed_params.items()}
        )

        for combo in combinations:
            run_name = f"{model_name}_" + "_".join(
                f"{k}={v}" for k, v in combo.items()
            )
            with mlflow.start_run(run_name=run_name, nested=True) as child_run:
                all_params = {**fixed_params, **combo}
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", model_class(**all_params)),
                ])
                mlflow.log_params(all_params)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = TrainingMetrics(
                    accuracy=float(accuracy_score(y_test, y_pred)),
                    precision_macro=float(
                        precision_score(
                            y_test, y_pred, average="macro", zero_division=0
                        )
                    ),
                    recall_macro=float(
                        recall_score(
                            y_test, y_pred, average="macro", zero_division=0
                        )
                    ),
                    f1_macro=float(
                        f1_score(
                            y_test, y_pred, average="macro", zero_division=0
                        )
                    ),
                )
                mlflow.log_metrics(metrics.model_dump())

                cm = confusion_matrix(y_test, y_pred)
                cm_png_path = _plot_confusion_matrix(
                    cm,
                    _CLASS_LABELS,
                    title=run_name,
                )
                try:
                    mlflow.log_artifact(cm_png_path, artifact_path="plots")
                finally:
                    os.unlink(cm_png_path)

                mlflow.sklearn.log_model(model, artifact_path="model")

                if (
                    best_metrics is None
                    or metrics.f1_macro > best_metrics.f1_macro
                ):
                    best_metrics = metrics
                    best_params = combo
                    best_run_id = child_run.info.run_id

                logger.info(
                    "  [%s] %s — f1_macro=%.4f",
                    model_name,
                    run_name,
                    metrics.f1_macro,
                )

        mlflow.log_metrics(
            {
                f"best_{metric_name}": metric_value
                for metric_name, metric_value in best_metrics.model_dump().items()
            }
        )
        mlflow.log_params({f"best_{k}": str(v) for k, v in best_params.items()})
        mlflow.set_tag("best_child_run_id", best_run_id)

        logger.info(
            "Grid search for '%s' complete — %d runs, best f1_macro=%.4f, params=%s.",
            model_name,
            len(combinations),
            best_metrics.f1_macro,
            best_params,
        )

    return {
        "run_id": parent_run.info.run_id,
        "model_name": model_name,
        "experiment_name": experiment_name,
        "best_metrics": best_metrics.model_dump(mode="json"),
        "best_params": best_params,
        "best_run_id": best_run_id,
        "n_combinations": len(combinations),
    }
