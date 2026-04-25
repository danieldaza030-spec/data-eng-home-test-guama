"""DAG for training Iris classification models and logging experiments to MLFlow.

This DAG runs exclusively on manual trigger and orchestrates four steps:
validating the preprocessed feature store, then training three classifiers
(SVM, Logistic Regression, KNN) in parallel, logging each experiment run to
the MLFlow tracking server.

Trigger:
    Manual only ('schedule=None').

Tasks:
    validate_feature_store: Load and validate '/public/iris_features.parquet',
        then push '"dataset_version"' to XCom under that key.
    train_svm: Train a Support Vector Classifier via grid search and log runs
        to MLFlow.
    train_logistic_regression: Train a Logistic Regression classifier via grid
        search and log runs to MLFlow.
    train_knn: Train a K-Nearest Neighbours classifier via grid search and log
        runs to MLFlow.

Dependencies:
    'validate_feature_store >> [train_svm, train_logistic_regression, train_knn]'

DAG Params:
    test_size (float): Fraction of data reserved for testing. Default '0.2'.
    svm_param_grid (dict): SVM hyperparameter grid. Default sweeps C and kernel.
    svm_fixed_params (dict): SVM fixed hyperparameters. Default 'random_state=42'.
    lr_param_grid (dict): Logistic Regression hyperparameter grid.
    lr_fixed_params (dict): Logistic Regression fixed hyperparameters.
    knn_param_grid (dict): KNN hyperparameter grid.
    knn_fixed_params (dict): KNN fixed hyperparameters (empty by default).
"""

from datetime import datetime

from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.http.operators.http import HttpOperator
from airflow.sdk import dag, Param

from python.model_training.tasks import (
    register_models_task,
    train_knn_task,
    train_logistic_regression_task,
    train_svm_task,
    validate_feature_store_task,
)

from python.tasks.data_ingest.tasks import (
    download_iris_data_task,
    transform_iris_data_task,
    validate_iris_data_task,
)

_DEFAULT_ARGS: dict = {
    "retries": 1,
}


@dag(
    dag_id="complete_ml_pipeline",
    description=(
        "Download, validate, and transform the Iris dataset from Kaggle."
        "Then validate the Iris feature store and train SVM, Logistic Regression, "
        "and KNN classifiers in parallel, logging each run to MLFlow."
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
            description=(
                "Random seed for the train/test split. "
                "Change this to verify that perfect metrics are a property "
                "of the dataset and not of a lucky seed."
            ),
        ),
        "svm_param_grid": Param(
            {"C": [0.1, 1.0, 10.0, 100.0], "kernel": ["linear", "rbf"]},
            type="object",
            description=(
                "SVM hyperparameter grid. Keys must be valid SVC parameter names; "
                "values must be lists of candidate values."
            ),
        ),
        "svm_fixed_params": Param(
            {"random_state": 42},
            type="object",
            description="SVM fixed hyperparameters applied to every grid combination.",
        ),
        "lr_param_grid": Param(
            {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs", "saga"]},
            type="object",
            description=(
                "Logistic Regression hyperparameter grid. Keys must be valid "
                "LogisticRegression parameter names; values must be lists of candidates."
            ),
        ),
        "lr_fixed_params": Param(
            {"max_iter": 200, "random_state": 42},
            type="object",
            description="Logistic Regression fixed hyperparameters.",
        ),
        "knn_param_grid": Param(
            {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
            type="object",
            description=(
                "KNN hyperparameter grid. Keys must be valid "
                "KNeighborsClassifier parameter names; values must be lists of candidates."
            ),
        ),
        "knn_fixed_params": Param(
            {},
            type="object",
            description="KNN fixed hyperparameters (empty by default).",
        ),
        "write_mode": Param(
            "overwrite",
            type="string",
            enum=["overwrite", "append"],
            description=(
                "'overwrite' (default) replaces the feature store entirely with "
                "the current batch; 'append' merges incoming rows with the existing "
                "Parquet file and deduplicates on feature + target columns."
            ),
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
def complete_ml_pipeline() -> None:
    """Iris end-to-end ML pipeline (manual trigger only).

    Downloads, validates, and transforms the Iris dataset, then trains SVM,
    Logistic Regression, and KNN classifiers in parallel, registers the best
    models in the MLflow Model Registry, and notifies the inference API.

    Pipeline steps:
        1. 'download_iris_data': Download and extract the raw CSV from Kaggle.
        2. 'validate_iris_data': Validate the CSV against the Iris schema.
        3. 'transform_iris_data': Encode the target and write the raw feature
           values to '/public/feature_storage/features_iris.csv'.
        4. 'validate_feature_store': Validate the CSV feature store and
           push 'dataset_version' to XCom.
        5. 'train_svm' / 'train_logistic_regression' / 'train_knn':
           Run grid-search training in parallel and log each run to MLflow
           under the 'iris_classification' experiment.
        6. 'register_models': Register the best-performing version of each
           classifier in the MLflow Model Registry and assign the
           'champion' alias.
        7. 'post_update_model': POST to the inference API to reload the
           champion model.
        8. 'get_model_info': GET '/info' to confirm the API is serving the
           updated model.

    Note:
        Hyperparameter grids and the train/test split ratio are configurable
        via DAG params in the Airflow Trigger UI.
    """
    download = PythonOperator(
        task_id="download_iris_data",
        python_callable=download_iris_data_task,
    )

    validate = PythonOperator(
        task_id="validate_iris_data",
        python_callable=validate_iris_data_task,
    )

    transform = PythonOperator(
        task_id="transform_iris_data",
        python_callable=transform_iris_data_task,
    )

    validate_feature_store = PythonOperator(
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
        download
        >> validate
        >> transform
        >> validate_feature_store
    )
    (
        validate_feature_store
        >> [train_svm, train_lr, train_knn]
        >> register
        >> deploy_new_model
        >> get_api_info
    )


complete_ml_pipeline()
