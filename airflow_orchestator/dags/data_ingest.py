"""DAG for the Iris dataset ingestion pipeline.

Supports two data sources (Kaggle or a locally-generated synthetic CSV),
configurable validation error handling, and both overwrite and
incremental-append write modes for the feature store.

Supported data sources (``data_source`` param):
    * ``"kaggle"`` (default) — downloads the dataset from the Kaggle API.
    * ``"synthetic"`` — reads ``/public/source_data/simulated.csv``
      produced by the ``generate_synthetic_data`` DAG.

Trigger:
    Manual only (``schedule=None``).

Task graph::

    download_iris_data
        -> validate_iris_data
        -> transform_iris_data
"""

import logging
import urllib.error
from datetime import datetime
from typing import Any

from airflow.exceptions import AirflowException
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import Param, dag

from python.tasks.data_ingest.extract_data import download_and_extract_dataset
from python.tasks.data_ingest.tasks import (
    transform_iris_data_task,
    validate_iris_data_task,
)
from python.tasks.metadata.constants import IrisConstants

logger = logging.getLogger(__name__)

_DEFAULT_ARGS: dict = {
    "retries": 1,
}


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _download_data_task(**context: Any) -> None:
    """Download or locate the Iris CSV and push its path to XCom.

    Supports two data sources controlled by the ``data_source`` DAG param:

    * ``"kaggle"`` -- downloads from the Kaggle REST API and extracts
      ``Iris.csv`` to a temporary directory.
    * ``"synthetic"`` -- resolves ``/public/source_data/simulated.csv``.
      Raises :class:`~airflow.exceptions.AirflowException` if that file
      does not exist (run ``generate_synthetic_data`` first).

    The resulting path is pushed to XCom under the key ``"temp_file"``.

    Args:
        **context: Airflow task context injected by the ``PythonOperator``.

    Raises:
        AirflowException: For Kaggle API errors, network failures, missing
            archive contents, or a missing simulated CSV.
    """
    ti = context["ti"]
    params = context["params"]
    data_source: str = params.get("data_source", "kaggle")

    if data_source == "synthetic":
        simulated_path = (
            IrisConstants.SOURCE_DATA_DIR / IrisConstants.SIMULATED_CSV_FILENAME
        )
        if not simulated_path.exists():
            raise AirflowException(
                f"Simulated data not found at '{simulated_path}'. "
                "Run the 'generate_synthetic_data' DAG first."
            )
        ti.xcom_push(key="temp_file", value=str(simulated_path))
        logger.info("Using simulated data source: %s", simulated_path)
        return

    try:
        csv_path = download_and_extract_dataset(
            owner=IrisConstants.IRIS_OWNER,
            dataset=IrisConstants.IRIS_DATASET,
            target_filename=IrisConstants.IRIS_FILENAME,
        )
    except urllib.error.HTTPError as exc:
        raise AirflowException(
            f"Kaggle API returned HTTP {exc.code} ({exc.reason}). "
            "Check that KAGGLE_USERNAME and KAGGLE_KEY are valid."
        ) from exc
    except urllib.error.URLError as exc:
        raise AirflowException(
            f"Network error while contacting the Kaggle API: {exc.reason}"
        ) from exc
    except FileNotFoundError as exc:
        raise AirflowException(str(exc)) from exc

    ti.xcom_push(key="temp_file", value=csv_path)
    logger.info("Iris CSV registered in XCom 'temp_file': %s", csv_path)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------


@dag(
    dag_id="data_ingest",
    description=(
        "Iris ingest pipeline. Supports Kaggle and synthetic data sources. "
        "Validates, transforms, and writes to the feature store."
    ),
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=_DEFAULT_ARGS,
    tags=["iris", "ingestion"],
    params={
        "data_source": Param(
            "kaggle",
            type="string",
            enum=["kaggle", "synthetic"],
            description=(
                "'kaggle' downloads from the Kaggle API (requires credentials); "
                "'synthetic' reads /public/source_data/simulated.csv generated "
                "by the 'generate_synthetic_data' DAG."
            ),
        ),
        "write_mode": Param(
            "overwrite",
            type="string",
            enum=["overwrite", "append"],
            description=(
                "'overwrite' (default) replaces the feature store entirely; "
                "'append' merges incoming rows with the existing CSV and "
                "deduplicates on feature + target columns."
            ),
        ),
        "on_validation_error": Param(
            "fail",
            type="string",
            enum=["fail", "drop_invalid_rows"],
            description=(
                "'fail' (default) halts the pipeline on any schema violation; "
                "'drop_invalid_rows' removes offending rows and continues."
            ),
        ),
    },
)
def data_ingest() -> None:
    """Iris data ingestion pipeline (manual trigger only).

    Downloads or reads the Iris dataset, validates it against the Iris schema,
    and writes the cleaned feature values to the feature store CSV at
    ``/public/feature_storage/features_iris.csv``.

    Pipeline steps:
        1. ``download_iris_data``: Download from Kaggle or locate the
           simulated CSV; push path to XCom.
        2. ``validate_iris_data``: Validate against ``IrisSchema``; behaves
           according to ``on_validation_error``.
        3. ``transform_iris_data``: Encode target and write feature store.
    """
    download = PythonOperator(
        task_id="download_iris_data",
        python_callable=_download_data_task,
    )

    validate = PythonOperator(
        task_id="validate_iris_data",
        python_callable=validate_iris_data_task,
    )

    transform = PythonOperator(
        task_id="transform_iris_data",
        python_callable=transform_iris_data_task,
    )

    download >> validate >> transform


data_ingest()
