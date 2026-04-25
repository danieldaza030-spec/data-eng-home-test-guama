"""DAG for monitoring production prediction drift and triggering retraining.

This DAG implements the Tipo B drift monitoring approach: it compares the
feature distributions of **real production requests** (logged by the inference
API to ``/public/prediction_storage/predicted.csv``) against the feature
distributions of the **training data** in the feature store
(``/public/feature_storage/features_iris.csv``).

This is the correct MLOps pattern because it measures what the deployed model
actually sees in production, rather than comparing arbitrary ingestion batches.

When drift is detected (any feature KS p-value < ``DriftConstants.KS_ALPHA``),
the ``training_pipeline`` DAG is automatically triggered so the model is
retrained on the current feature store.

Task graph::

    detect_prediction_drift
        -> should_retrain (ShortCircuitOperator)
        -> trigger_training_pipeline (TriggerDagRunOperator)

Schedule:
    Daily (``@daily``) — can be adjusted in the Airflow UI.
"""

import logging
from datetime import datetime
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
import os
import pandas as pd
from airflow.exceptions import AirflowSkipException
from airflow.providers.standard.operators.python import (
    PythonOperator,
    ShortCircuitOperator,
)
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import dag

from python.tasks.drift_detection.ks_test import run_ks_drift_test
from python.tasks.metadata.constants import DriftConstants, IrisConstants, MLFlowConstants
from python.tasks.metadata.iris_schema import IRIS_FEATURE_COLS

logger = logging.getLogger(__name__)

_DEFAULT_ARGS: dict = {
    "retries": 1,
}

_FEATURE_STORE_PATH = IrisConstants.FEATURE_STORAGE_DIR / IrisConstants.FEATURE_STORE_FILENAME
_PREDICTIONS_PATH = IrisConstants.PREDICTION_STORAGE_DIR / IrisConstants.PREDICTIONS_FILENAME


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _detect_prediction_drift_task(**context: Any) -> None:
    """Compare production prediction features against the champion's training distribution.

    Loads the feature store and the prediction log, then filters the feature
    store to only the ``processed_at`` dates the current ``@champion`` model
    was actually trained on (read from the ``"training_dates"`` tag on the
    MLflow model version). This ensures the KS test compares production traffic
    against the exact distribution the deployed model saw during training,
    preventing false positives or negatives caused by accumulated batches in
    the feature store that were not part of the last training run.

    Falls back to the full feature store when the ``"training_dates"`` tag is
    absent (backward compatibility with versions registered before this feature).

    Pushes a boolean ``drift_detected`` flag to XCom and skips gracefully
    (``AirflowSkipException``) when either file is absent or either set has
    fewer than 30 rows.

    Args:
        **context: Airflow task context injected by the ``PythonOperator``.

    Raises:
        AirflowSkipException: When required files are missing or have
            insufficient rows.
    """
    ti = context["ti"]

    if not _FEATURE_STORE_PATH.exists():
        logger.warning(
            "Feature store not found at '%s' — skipping drift check. "
            "Run an ingest DAG first.",
            _FEATURE_STORE_PATH,
        )
        raise AirflowSkipException("Feature store CSV missing.")

    if not _PREDICTIONS_PATH.exists():
        logger.warning(
            "Prediction log not found at '%s' — no production traffic yet.",
            _PREDICTIONS_PATH,
        )
        raise AirflowSkipException("Prediction log CSV missing.")

    reference = pd.read_csv(_FEATURE_STORE_PATH)
    current = pd.read_csv(_PREDICTIONS_PATH)

    # ------------------------------------------------------------------
    # Filter the feature store to the rows the champion was trained on.
    # The training pipeline saves the exact processed_at dates used as a
    # comma-separated tag ("training_dates") on the champion model version.
    # If the tag is absent (older versions, backward compat) fall back to
    # all rows with a warning so the check degrades gracefully.
    # ------------------------------------------------------------------
    tracking_uri = os.environ.get(
        MLFlowConstants.TRACKING_URI_ENV,
        MLFlowConstants.DEFAULT_TRACKING_URI,
    )
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        champion = client.get_model_version_by_alias(
            name=MLFlowConstants.REGISTERED_MODEL_NAME,
            alias=MLFlowConstants.CHAMPION_ALIAS,
        )
        training_dates_tag: str | None = champion.tags.get("training_dates")
        if training_dates_tag:
            training_dates = [d.strip() for d in training_dates_tag.split(",") if d.strip()]
            if "processed_at" in reference.columns:
                ref_dates = pd.to_datetime(reference["processed_at"]).dt.date.astype(str)
                reference = reference[ref_dates.isin(training_dates)].reset_index(drop=True)
                logger.info(
                    "Feature store filtered to champion training dates %s — %d rows remaining.",
                    training_dates,
                    len(reference),
                )
                if reference.empty:
                    logger.warning(
                        "No feature store rows match the champion training dates %s. "
                        "Falling back to full feature store.",
                        training_dates,
                    )
                    reference = pd.read_csv(_FEATURE_STORE_PATH)
        else:
            logger.warning(
                "Champion version has no 'training_dates' tag — "
                "using the full feature store as reference (backward compat)."
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not resolve champion training dates from MLflow (%s) — "
            "using full feature store as reference.",
            exc,
        )

    _MIN_ROWS = 30
    if len(reference) < _MIN_ROWS or len(current) < _MIN_ROWS:
        logger.warning(
            "Insufficient data for KS test (reference=%d, predictions=%d, min=%d). "
            "Skipping drift check.",
            len(reference),
            len(current),
            _MIN_ROWS,
        )
        raise AirflowSkipException(
            f"Not enough rows for KS test (need {_MIN_ROWS} in each set)."
        )

    # Only keep feature columns present in both DataFrames
    available_features = [c for c in IRIS_FEATURE_COLS if c in reference.columns and c in current.columns]
    if not available_features:
        raise AirflowSkipException(
            "No common feature columns found between feature store and predictions."
        )

    report = run_ks_drift_test(
        reference=reference,
        current=current,
        feature_cols=available_features,
        alpha=DriftConstants.KS_ALPHA,
    )

    logger.info(
        "Drift check complete — reference=%d rows, predictions=%d rows, drift=%s",
        report.n_reference,
        report.n_current,
        report.overall_drift,
    )
    for fr in report.feature_results:
        level = logging.WARNING if fr.drift_detected else logging.INFO
        logger.log(
            level,
            "  %s: KS=%.4f  p=%.4f  drift=%s",
            fr.feature,
            fr.ks_statistic,
            fr.p_value,
            fr.drift_detected,
        )

    ti.xcom_push(key="drift_detected", value=report.overall_drift)


def _should_retrain_task(**context: Any) -> bool:
    """Gate retraining based on the drift flag pushed by the detection task.

    Returns ``True`` to allow the ``TriggerDagRunOperator`` to proceed, or
    ``False`` to short-circuit and skip it.

    Args:
        **context: Airflow task context injected by the ``ShortCircuitOperator``.

    Returns:
        ``True`` when production drift has been detected, ``False`` otherwise.
    """
    ti = context["ti"]
    drift_detected: bool = ti.xcom_pull(
        task_ids="detect_prediction_drift",
        key="drift_detected",
    )
    if drift_detected:
        logger.warning(
            "Production drift detected — triggering '%s' DAG.",
            DriftConstants.TRAINING_DAG_ID,
        )
    else:
        logger.info(
            "No significant production drift detected. Skipping retraining."
        )
    return bool(drift_detected)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------


@dag(
    dag_id="monitor_prediction_drift",
    description=(
        "Daily Tipo B drift monitor: compares production prediction features "
        "against the training feature store using KS tests. "
        "Triggers 'training_pipeline' when drift is detected."
    ),
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=_DEFAULT_ARGS,
    tags=["iris", "drift", "monitoring"],
)
def monitor_prediction_drift() -> None:
    """Production drift monitoring pipeline (runs daily).

    Reads the prediction log written by the inference API and the feature
    store written by the ingest pipeline, runs two-sample KS tests on all
    four Iris feature columns, and triggers the ``training_pipeline`` DAG
    when overall drift is detected.

    Pipeline steps:
        1. ``detect_prediction_drift``: Load both CSVs, run KS tests,
           push ``drift_detected`` bool to XCom.
        2. ``should_retrain``: Return ``True`` if drift was detected;
           short-circuits (skips) the trigger step otherwise.
        3. ``trigger_training_pipeline``: Fire the ``training_pipeline`` DAG
           to retrain all classifiers on the current feature store.
    """
    detect = PythonOperator(
        task_id="detect_prediction_drift",
        python_callable=_detect_prediction_drift_task,
    )

    gate = ShortCircuitOperator(
        task_id="should_retrain",
        python_callable=_should_retrain_task,
    )

    trigger = TriggerDagRunOperator(
        task_id="trigger_training_pipeline",
        trigger_dag_id=DriftConstants.TRAINING_DAG_ID,
        wait_for_completion=False,
        reset_dag_run=True,
    )

    detect >> gate >> trigger


monitor_prediction_drift()
