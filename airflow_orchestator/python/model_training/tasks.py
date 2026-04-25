"""Airflow task callables for the Iris model training pipeline.

This module contains Airflow 3.x orchestration logic only. Domain services for
feature loading, training, and MLflow logging live in
``python.model_training.train``. Shared nested payloads are typed with
Pydantic, while task-level responses remain dictionaries for XCom.

DAG params consumed by training tasks (resolved from ``context["params"]``):
    test_size (float): Train/test split ratio forwarded to every training task.
    svm_param_grid (dict): SVM hyperparameter grid.
    svm_fixed_params (dict): SVM fixed hyperparameters.
    lr_param_grid (dict): Logistic Regression hyperparameter grid.
    lr_fixed_params (dict): Logistic Regression fixed hyperparameters.
    knn_param_grid (dict): KNN hyperparameter grid.
    knn_fixed_params (dict): KNN fixed hyperparameters.
"""

import logging
from typing import Any

import pandas as pd
from airflow.exceptions import AirflowException
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import os

from python.tasks.metadata.constants import MLFlowConstants
from python.model_training.train import (
    load_feature_store,
    register_best_model,
    split_features_target,
    train_and_log_grid,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _train_model_task(
    model_name: str,
    model_class: type,
    fixed_params: dict[str, Any],
    param_grid: dict[str, list],
    test_size: float,
    random_state: int,
    **context: Any,
) -> dict[str, Any]:
    """Load the feature store and run a hyperparameter grid search.

    Internal helper shared by all model-specific public task callables.
    Pulls ``"dataset_version"`` from XCom (set by
    :func:`validate_feature_store_task`) and delegates the grid search and
    MLFlow logging to :func:`~python.model_training.train.train_and_log_grid`.

    Args:
        model_name: Identifier string used as the MLFlow parent run name
            and ``model_type`` tag (e.g. ``"svm"``).
        model_class: Scikit-learn estimator class (not an instance).
        fixed_params: Hyperparameters kept constant across all combinations.
        param_grid: Hyperparameter grid to sweep; each key maps to a list of
            candidate values.
        test_size: Fraction of data to reserve for testing. Sourced from the
            ``test_size`` DAG param set at trigger time.
        random_state: Random seed for the train/test split. Sourced from the
            ``random_state`` DAG param. Varying this seed across runs is a
            useful sanity check to confirm that high metrics reflect genuine
            dataset separability rather than a lucky split.
        **context: Airflow task context injected automatically by the
            PythonOperator.  The following key is consumed:

            - ``ti`` (:class:`airflow.models.TaskInstance`): used to pull
              ``"dataset_version"`` from the upstream validate task.

    Returns:
        Dictionary returned by :func:`~python.model_training.train.train_and_log_grid`.

    Raises:
        AirflowException: If the feature store cannot be loaded, the
            ``"dataset_version"`` XCom key is missing, or the MLFlow tracking
            server is unreachable.
    """
    ti = context["ti"]
    params = context.get("params", {})
    training_data: str = params.get("training_data", "all")

    dataset_version: str | None = ti.xcom_pull(
        key="dataset_version",
        task_ids="validate_feature_store",
    )
    if not dataset_version:
        raise AirflowException(
            "XCom key 'dataset_version' not found or is empty. "
            "Ensure 'validate_feature_store' completes successfully before "
            "any training task runs."
        )

    try:
        df = load_feature_store(training_data=training_data)
    except (FileNotFoundError, ValueError) as exc:
        raise AirflowException(str(exc)) from exc

    # Capture which processed_at dates were actually loaded for training.
    # Stored as sorted ISO date strings and propagated to the MLflow model version
    # tag so the drift detector can filter the feature store to the same rows.
    training_dates: list[str] = []
    if "processed_at" in df.columns:
        training_dates = sorted(
            pd.to_datetime(df["processed_at"]).dt.date.astype(str).unique().tolist()
        )

    X_train, X_test, y_train, y_test = split_features_target(
        df, test_size=test_size, random_state=random_state
    )

    try:
        result = train_and_log_grid(
            model_name=model_name,
            model_class=model_class,
            fixed_params=fixed_params,
            param_grid=param_grid,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            dataset_version=dataset_version,
        )
    except Exception as exc:  # noqa: BLE001
        raise AirflowException(
            f"MLFlow grid search for '{model_name}' failed: {exc}"
        ) from exc

    result["training_dates"] = training_dates

    logger.info(
        "Grid search for '%s' completed — run_id=%s, best_metrics=%s, training_dates=%s.",
        model_name,
        result["run_id"],
        result["best_metrics"],
        training_dates,
    )
    return result


# ---------------------------------------------------------------------------
# Airflow task callables
# ---------------------------------------------------------------------------


def validate_feature_store_task(**context: Any) -> dict[str, Any]:
    """Validate the Iris feature store and register dataset metadata in XCom.

    Loads ``/public/iris_features.parquet``, confirms required columns are
    present and the file is non-empty, extracts the ``processed_at`` UTC
    timestamp as a dataset-version identifier for MLFlow data lineage, and
    pushes it to XCom under the key ``"dataset_version"``.

    Example DAG usage::

        from airflow.operators.python import PythonOperator
        from python.model_training.tasks import validate_feature_store_task

        validate = PythonOperator(
            task_id="validate_feature_store",
            python_callable=validate_feature_store_task,
        )

    Args:
        **context: Airflow task context injected automatically by the
            PythonOperator.  The following key is consumed:

            - ``ti`` (:class:`airflow.models.TaskInstance`): used to push
              ``"dataset_version"`` via ``xcom_push``.

    Returns:
        Dictionary with ``rows``, ``columns``, and ``dataset_version`` keys.

    Raises:
        AirflowException: If the Parquet file is missing, empty, or lacks
            required feature or target columns.
    """
    ti = context["ti"]
    params = context.get("params", {})
    training_data: str = params.get("training_data", "all")

    try:
        df = load_feature_store(training_data=training_data)
    except (FileNotFoundError, ValueError) as exc:
        raise AirflowException(str(exc)) from exc

    dataset_version: str = (
        str(df["processed_at"].iloc[0])
        if "processed_at" in df.columns
        else "unknown"
    )
    ti.xcom_push(key="dataset_version", value=dataset_version)

    logger.info(
        "Feature store validated — %d rows (training_data='%s'), dataset_version: %s",
        len(df),
        training_data,
        dataset_version,
    )

    return {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "dataset_version": dataset_version,
    }


def train_svm_task(**context: Any) -> dict[str, Any]:
    """Train a Support Vector Classifier on the Iris feature store.

    Reads ``svm_param_grid``, ``svm_fixed_params``, and ``test_size`` from
    the DAG params resolved at trigger time, then delegates to
    :func:`_train_model_task` using a :class:`~sklearn.svm.SVC`.

    Example DAG usage::

        from airflow.operators.python import PythonOperator
        from python.model_training.tasks import train_svm_task

        train_svm = PythonOperator(
            task_id="train_svm",
            python_callable=train_svm_task,
        )

    Args:
        **context: Airflow task context injected automatically by the
            PythonOperator. The ``params`` key must contain ``svm_param_grid``,
            ``svm_fixed_params``, and ``test_size``.

    Returns:
        Dictionary returned by :func:`_train_model_task`.

    Raises:
        AirflowException: Propagated from :func:`_train_model_task`.
    """
    params = context["params"]
    return _train_model_task(
        model_name="svm",
        model_class=SVC,
        fixed_params=params["svm_fixed_params"],
        param_grid=params["svm_param_grid"],
        test_size=params["test_size"],
        random_state=params["random_state"],
        **context,
    )


def train_logistic_regression_task(**context: Any) -> dict[str, Any]:
    """Train a Logistic Regression classifier on the Iris feature store.

    Reads ``lr_param_grid``, ``lr_fixed_params``, and ``test_size`` from the
    DAG params resolved at trigger time, then delegates to
    :func:`_train_model_task` using a
    :class:`~sklearn.linear_model.LogisticRegression`.

    Example DAG usage::

        from airflow.operators.python import PythonOperator
        from python.model_training.tasks import train_logistic_regression_task

        train_lr = PythonOperator(
            task_id="train_logistic_regression",
            python_callable=train_logistic_regression_task,
        )

    Args:
        **context: Airflow task context injected automatically by the
            PythonOperator. The ``params`` key must contain ``lr_param_grid``,
            ``lr_fixed_params``, and ``test_size``.

    Returns:
        Dictionary returned by :func:`_train_model_task`.

    Raises:
        AirflowException: Propagated from :func:`_train_model_task`.
    """
    params = context["params"]
    return _train_model_task(
        model_name="logistic_regression",
        model_class=LogisticRegression,
        fixed_params=params["lr_fixed_params"],
        param_grid=params["lr_param_grid"],
        test_size=params["test_size"],
        random_state=params["random_state"],
        **context,
    )


def train_knn_task(**context: Any) -> dict[str, Any]:
    """Train a K-Nearest Neighbours classifier on the Iris feature store.

    Reads ``knn_param_grid``, ``knn_fixed_params``, and ``test_size`` from
    the DAG params resolved at trigger time, then delegates to
    :func:`_train_model_task` using a
    :class:`~sklearn.neighbors.KNeighborsClassifier`.

    Example DAG usage::

        from airflow.operators.python import PythonOperator
        from python.model_training.tasks import train_knn_task

        train_knn = PythonOperator(
            task_id="train_knn",
            python_callable=train_knn_task,
        )

    Args:
        **context: Airflow task context injected automatically by the
            PythonOperator. The ``params`` key must contain ``knn_param_grid``,
            ``knn_fixed_params``, and ``test_size``.

    Returns:
        Dictionary returned by :func:`_train_model_task`.

    Raises:
        AirflowException: Propagated from :func:`_train_model_task`.
    """
    params = context["params"]
    return _train_model_task(
        model_name="knn",
        model_class=KNeighborsClassifier,
        fixed_params=params["knn_fixed_params"],
        param_grid=params["knn_param_grid"],
        test_size=params["test_size"],
        random_state=params["random_state"],
        **context,
    )


def register_models_task(**context: Any) -> dict[str, Any]:
    """Select the overall best model and register it in MLflow.

    Pulls XCom return values from the three upstream training tasks
    (``train_svm``, ``train_logistic_regression``, ``train_knn``), selects
    the single best result by ``f1_macro``, and delegates to
    :func:`~python.model_training.train.register_best_model`. A new
    ``iris_classifier`` version is **always** created. The ``champion`` alias
    is moved to the new version only when it strictly outperforms the existing
    champion; otherwise the version is registered but the alias is left
    untouched.

    Example DAG usage::

        from airflow.operators.python import PythonOperator
        from python.model_training.tasks import register_models_task

        register = PythonOperator(
            task_id="register_models",
            python_callable=register_models_task,
        )

    Args:
        **context: Airflow task context injected automatically by the
            PythonOperator. The ``ti`` key is used to pull XCom values from
            the upstream training tasks.

    Returns:
        Dictionary with ``winner_model_name``, ``winner_f1_macro``,
        ``winner_run_id``, ``registered_model_name``, and
        ``registered_model_version``.

    Raises:
        AirflowException: If no training task produced a usable XCom result
            or if the MLflow tracking server is unreachable.
    """
    ti = context["ti"]
    tracking_uri = os.environ.get(
        MLFlowConstants.TRACKING_URI_ENV,
        MLFlowConstants.DEFAULT_TRACKING_URI,
    )
    dataset_version: str = (
        ti.xcom_pull(key="dataset_version", task_ids="validate_feature_store")
        or "unknown"
    )

    training_task_ids = [
        "train_svm",
        "train_logistic_regression",
        "train_knn",
    ]

    candidates: list[dict[str, Any]] = []
    for task_id in training_task_ids:
        result: dict[str, Any] | None = ti.xcom_pull(task_ids=task_id)
        if not result:
            logger.warning(
                "No XCom result from task '%s' — skipping candidate.",
                task_id,
            )
            continue
        candidates.append(result)

    if not candidates:
        raise AirflowException(
            "All upstream training tasks failed or produced empty XCom results."
        )

    winner = max(candidates, key=lambda r: r["best_metrics"]["f1_macro"])
    winner_model_name: str = winner["model_name"]
    winner_run_id: str = winner["best_run_id"]
    winner_f1: float = winner["best_metrics"]["f1_macro"]
    winner_training_dates: list[str] = winner.get("training_dates", [])

    logger.info(
        "Overall winner: model_type=%s, best_run_id=%s, f1_macro=%.4f, training_dates=%s.",
        winner_model_name,
        winner_run_id,
        winner_f1,
        winner_training_dates,
    )

    try:
        registered_name, registered_version = register_best_model(
            model_name=winner_model_name,
            best_run_id=winner_run_id,
            candidate_f1=winner_f1,
            dataset_version=dataset_version,
            tracking_uri=tracking_uri,
            training_dates=winner_training_dates,
        )
    except Exception as exc:  # noqa: BLE001
        raise AirflowException(
            f"Registration failed for model '{winner_model_name}': {exc}"
        ) from exc

    logger.info(
        "Registered '%s' version %d (f1_macro=%.4f).",
        registered_name,
        registered_version,
        winner_f1,
    )

    return {
        "winner_model_name": winner_model_name,
        "winner_f1_macro": winner_f1,
        "winner_run_id": winner_run_id,
        "registered_model_name": registered_name,
        "registered_model_version": registered_version,
    }
