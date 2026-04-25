"""DAG for training Iris classification models (standalone, no ingest step).

This DAG is the retraining entrypoint triggered automatically by the
``ingest_with_drift`` DAG when Kolmogorov-Smirnov drift is detected, and can
also be run manually via the Airflow UI.

It assumes that a valid feature store already exists at
``/public/iris_features.parquet`` (produced by any ingest DAG).  It trains
SVM, Logistic Regression, and KNN classifiers in parallel via grid search,
registers the best model from each family in the MLflow Model Registry, and
notifies the inference API to hot-reload the new champion.

Trigger:
    Manual or programmatic (``TriggerDagRunOperator`` from
    ``ingest_with_drift``).

Tasks:
    validate_feature_store: Load and validate the Parquet feature store, push
        ``dataset_version`` to XCom.
    train_svm: Grid-search SVC and log runs to MLflow.
    train_logistic_regression: Grid-search LogisticRegression and log to MLflow.
    train_knn: Grid-search KNeighborsClassifier and log to MLflow.
    register_models: Register best run per model family; assign ``@champion``.
    post_update_model: POST ``/update_model`` to hot-reload the champion.
    get_model_info: GET ``/info`` to confirm the updated model is serving.

Dependencies:
    ``validate_feature_store >> [train_svm, train_lr, train_knn]
    >> register_models >> post_update_model >> get_model_info``
"""

from datetime import datetime

from airflow.providers.http.operators.http import HttpOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import dag, Param

from python.model_training.tasks import (
    register_models_task,
    train_knn_task,
    train_logistic_regression_task,
    train_svm_task,
    validate_feature_store_task,
)

_DEFAULT_ARGS: dict = {
    "retries": 1,
}


@dag(
    dag_id="training_pipeline",
    description=(
        "Validate the Iris feature store and train SVM, Logistic Regression, "
        "and KNN classifiers in parallel, logging each run to MLflow.  "
        "Triggered automatically on data drift or manually via the UI."
    ),
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=_DEFAULT_ARGS,
    tags=["iris", "training", "mlflow"],
    params={
        "test_size": Param(
            0.2,
            type="number",
            minimum=0.05,
            maximum=0.5,
            description="Fraction of data reserved for testing (0.05–0.5).",
        ),
        "random_state": Param(
            42,
            type="integer",
            minimum=0,
            description="Random seed for the train/test split.",
        ),
        "svm_param_grid": Param(
            {"C": [0.1, 1.0, 10.0, 100.0], "kernel": ["linear", "rbf"]},
            type="object",
            description="SVM hyperparameter grid.",
        ),
        "svm_fixed_params": Param(
            {"random_state": 42},
            type="object",
            description="SVM fixed hyperparameters.",
        ),
        "lr_param_grid": Param(
            {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs", "saga"]},
            type="object",
            description="Logistic Regression hyperparameter grid.",
        ),
        "lr_fixed_params": Param(
            {"max_iter": 200, "random_state": 42},
            type="object",
            description="Logistic Regression fixed hyperparameters.",
        ),
        "knn_param_grid": Param(
            {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
            type="object",
            description="KNN hyperparameter grid.",
        ),
        "knn_fixed_params": Param(
            {},
            type="object",
            description="KNN fixed hyperparameters (empty by default).",
        ),
        "training_data": Param(
            "all",
            type="string",
            enum=["all", "latest"],
            description=(
                "'all' (default) trains on the entire feature store; "
                "'latest' restricts training to rows from the most recently "
                "ingested batch (identified by the maximum processed_at date)."
            ),
        ),
    },
)
def training_pipeline() -> None:
    """Iris standalone training pipeline (manual or drift-triggered).

    Validates the pre-built feature store, trains three classifier families
    in parallel via grid search, registers the best version of each in the
    MLflow Model Registry under the ``champion`` alias, and hot-reloads the
    inference API.

    Pipeline steps:
        1. ``validate_feature_store``: Confirm the Parquet feature store is
           present and valid; push ``dataset_version`` to XCom.
        2. ``train_svm`` / ``train_logistic_regression`` / ``train_knn``:
           Parallel grid-search training with MLflow experiment logging.
        3. ``register_models``: Register best run per family; assign
           ``@champion``.
        4. ``post_update_model``: POST to the API to reload the champion.
        5. ``get_model_info``: GET ``/info`` to confirm serving state.

    Note:
        Hyperparameter grids are configurable at trigger time via DAG params.
    """
    validate_fs = PythonOperator(
        task_id="validate_feature_store",
        python_callable=validate_feature_store_task,
    )

    train_svm = PythonOperator(
        task_id="train_svm",
        python_callable=train_svm_task,
    )

    train_lr = PythonOperator(
        task_id="train_logistic_regression",
        python_callable=train_logistic_regression_task,
    )

    train_knn = PythonOperator(
        task_id="train_knn",
        python_callable=train_knn_task,
    )

    register = PythonOperator(
        task_id="register_models",
        python_callable=register_models_task,
    )

    deploy_new_model = HttpOperator(
        task_id="post_update_model",
        http_conn_id="iris_api_default",
        method="POST",
        endpoint="/update_model",
        headers={"Content-Type": "application/json"},
        data='{"model_name": null}',
        response_check=lambda response: response.status_code == 200,
        log_response=True,
    )

    get_api_info = HttpOperator(
        task_id="get_model_info",
        http_conn_id="iris_api_default",
        method="GET",
        endpoint="/info",
        headers={"Content-Type": "application/json"},
        response_check=lambda response: response.status_code == 200,
        log_response=True,
    )

    (
        validate_fs
        >> [train_svm, train_lr, train_knn]
        >> register
        >> deploy_new_model
        >> get_api_info
    )


training_pipeline()
