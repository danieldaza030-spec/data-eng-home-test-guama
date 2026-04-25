"""DAG for generating synthetic Iris data and simulating production predictions.

This DAG has two responsibilities:

1. **Generate** — produces ``/public/source_data/simulated.csv`` in the
   chosen mode, using the same Gaussian parameters as the real UCI dataset.
2. **Simulate** — sends all valid rows from that CSV to ``POST /predict``
   on the inference API, which logs each prediction to
   ``/public/prediction_storage/predicted.csv``.

This two-step flow is the primary way to populate the prediction store for
drift-monitoring tests without waiting for real user traffic:

* Run with ``mode="normal"`` → predictions follow the training distribution
  → ``monitor_prediction_drift`` will **not** trigger retraining.
* Run with ``mode="drifted"`` → predictions shift the petal feature
  distributions → ``monitor_prediction_drift`` **will** detect drift and
  trigger ``training_pipeline`` automatically.
* Run with ``mode="dirty"`` → generates data with intentional errors;
  only the valid rows are forwarded to the API (NaN / negative values are
  filtered out before sending).

Three generation modes are available:

* **normal** — balanced class proportions, standard UCI Gaussian parameters.
* **drifted** — petal features shifted ~2.5 std above normal means, class
  proportions skewed 10/20/70 % towards *Iris-virginica*.
* **dirty** — balanced proportions with 12 % nulls, 8 % negative values,
  8 % invalid Species labels, and 10 % duplicate rows.

Output: ``/public/source_data/simulated.csv`` (overwritten on every run).

Task graph::

    generate_and_save
        -> send_predictions_to_api
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from airflow.exceptions import AirflowException
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import Param, dag

from python.tasks.data_generation.generate import generate_synthetic_iris
from python.tasks.metadata.constants import IrisConstants

_FEATURE_COLS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
_API_URL = "http://iris-api:8000/predict"
_REQUEST_TIMEOUT_S: int = 10

logger = logging.getLogger(__name__)

_DEFAULT_ARGS: dict = {
    "retries": 1,
}


# ---------------------------------------------------------------------------
# Task callable
# ---------------------------------------------------------------------------


def _generate_and_save_task(**context: Any) -> dict[str, Any]:
    """Generate synthetic Iris rows and write them to the shared volume.

    Pulls the ``mode``, ``n_samples``, and ``random_state`` DAG params,
    generates the dataset via :func:`generate_synthetic_iris`, and writes the
    result as a CSV to ``/public/source_data/simulated.csv``.

    Args:
        **context: Airflow task context injected by the ``PythonOperator``.

    Returns:
        Dictionary with ``output_path``, ``n_rows``, and ``mode`` keys pushed
        to XCom as the task return value.
    """
    params = context["params"]
    mode: str = params["mode"]
    n_samples: int = params["n_samples"]
    random_state: int = params["random_state"]

    df = generate_synthetic_iris(
        n_samples=n_samples,
        mode=mode,
        random_state=random_state,
    )

    output_path = IrisConstants.SOURCE_DATA_DIR / IrisConstants.SIMULATED_CSV_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(
        "Synthetic data written to %s (%d rows, mode='%s').",
        output_path,
        len(df),
        mode,
    )
    return {"output_path": str(output_path), "n_rows": len(df), "mode": mode}


def _send_predictions_task(**context: Any) -> dict[str, Any]:
    """Send all valid rows from the generated CSV to the inference API.

    Reads ``/public/source_data/simulated.csv`` (written by
    ``generate_and_save``), filters out rows with NaN or non-positive feature
    values, and POSTs each remaining row to ``POST /predict``.  The API logs
    every accepted prediction to
    ``/public/prediction_storage/predicted.csv``.

    Dirty-mode data may produce skipped rows — this is expected and logged
    as a warning rather than an error.

    Args:
        **context: Airflow task context injected by the ``PythonOperator``.

    Returns:
        Dictionary with ``total``, ``sent``, ``skipped``, and ``failed``
        counts pushed to XCom as the task return value.

    Raises:
        AirflowException: When no valid rows are available or no request
            was accepted by the API.
    """
    output_path = str(
        IrisConstants.SOURCE_DATA_DIR / IrisConstants.SIMULATED_CSV_FILENAME
    )
    df = pd.read_csv(output_path)

    # Filter to rows where all four features are present and positive
    df_valid = df[_FEATURE_COLS].copy()
    df_valid = df_valid.dropna()
    df_valid = df_valid[(df_valid[_FEATURE_COLS] > 0).all(axis=1)].reset_index(drop=True)

    skipped = len(df) - len(df_valid)
    if skipped:
        logger.warning(
            "Skipped %d row(s) with NaN or non-positive values (expected for 'dirty' mode).",
            skipped,
        )

    if df_valid.empty:
        raise AirflowException(
            "No valid rows to send after filtering. "
            "Check the generated data or switch to 'normal'/'drifted' mode."
        )

    total = len(df_valid)
    sent = 0
    failed = 0

    logger.info("Sending %d prediction request(s) to %s …", total, _API_URL)

    for i, row in df_valid.iterrows():
        payload = {
            "sepal_length_cm": float(row["SepalLengthCm"]),
            "sepal_width_cm": float(row["SepalWidthCm"]),
            "petal_length_cm": float(row["PetalLengthCm"]),
            "petal_width_cm": float(row["PetalWidthCm"]),
        }
        try:
            resp = requests.post(_API_URL, json=payload, timeout=_REQUEST_TIMEOUT_S)
            if resp.status_code == 200:
                sent += 1
            else:
                failed += 1
                logger.warning(
                    "Row %d: API returned HTTP %d — %s",
                    i,
                    resp.status_code,
                    resp.text[:200],
                )
        except requests.exceptions.RequestException as exc:
            failed += 1
            logger.warning("Row %d: request failed — %s", i, exc)

        if (i + 1) % 100 == 0:
            logger.info("Progress: %d / %d rows sent.", sent, total)

    logger.info(
        "Simulation complete — total=%d  sent=%d  skipped=%d  failed=%d",
        total,
        sent,
        skipped,
        failed,
    )

    if sent == 0:
        raise AirflowException(
            f"No predictions were accepted by the API (failed={failed}). "
            "Check that 'iris-api' is running and a champion model is loaded."
        )

    return {"total": total, "sent": sent, "skipped": skipped, "failed": failed}


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------


@dag(
    dag_id="generate_synthetic_data",
    description=(
        "Generate synthetic Iris data and simulate production predictions by "
        "sending rows to the inference API. Supports normal, drifted, and dirty modes."
    ),
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=_DEFAULT_ARGS,
    tags=["iris", "data-generation", "testing"],
    params={
        "mode": Param(
            "normal",
            type="string",
            enum=["normal", "drifted", "dirty"],
            description=(
                "'normal' reproduces the standard Iris distribution — no drift expected; "
                "'drifted' shifts petal features and skews class proportions "
                "to trigger the KS drift detector; "
                "'dirty' injects nulls, negatives, invalid labels, and duplicates "
                "(valid rows are still forwarded to the API)."
            ),
        ),
        "n_samples": Param(
            1200,
            type="integer",
            minimum=100,
            maximum=100_000,
            description="Total number of synthetic rows to generate.",
        ),
        "random_state": Param(
            42,
            type="integer",
            minimum=0,
            description="Random seed for reproducibility.",
        ),
    },
)
def generate_synthetic_data() -> None:
    """Generate synthetic Iris data and simulate predictions (manual trigger only).

    Step 1 (``generate_and_save``) generates a CSV at
    ``/public/source_data/simulated.csv`` following the chosen distribution
    mode.  Step 2 (``send_predictions_to_api``) sends all valid rows to
    ``POST /predict``, populating ``/public/prediction_storage/predicted.csv``
    so that ``monitor_prediction_drift`` has real-looking data to analyse.
    """
    generate = PythonOperator(
        task_id="generate_and_save",
        python_callable=_generate_and_save_task,
    )

    send = PythonOperator(
        task_id="send_predictions_to_api",
        python_callable=_send_predictions_task,
    )

    generate >> send


generate_synthetic_data()
