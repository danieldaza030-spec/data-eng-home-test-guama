"""Airflow task callables for the Iris ingestion pipeline.

This module contains orchestration-only logic for Airflow 3.x. Domain
services such as extraction, validation, and transformation live in the
'python.data_ingest' package, while this module handles XCom interaction
and exception translation.
"""

import logging
import tempfile
import urllib.error
from pathlib import Path
from typing import Any

import pandas as pd
from airflow.exceptions import AirflowException

from python.tasks.data_ingest.extract_data import (
    download_and_extract_dataset,
)
from python.tasks.data_ingest.transform_data import (
    clean_iris_dataframe,
    prepare_universal_features,
    save_features_to_csv,
)
from python.tasks.data_ingest.validate import (
    format_validation_errors,
    load_iris_data,
    validate_iris_data,
)
from python.tasks.metadata.constants import IrisConstants
from python.tasks.metadata.iris_schema import IRIS_FEATURE_COLS, IRIS_TARGET_COL

logger = logging.getLogger(__name__)

_VALID_SPECIES: frozenset[str] = frozenset(
    {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}
)


# ---------------------------------------------------------------------------
# Airflow task callables
# ---------------------------------------------------------------------------


def download_iris_data_task(**context: Any) -> None:
    """Download the Iris dataset from Kaggle and register the file path in XCom.

    Uses the Kaggle public REST API to download the 'uciml/iris' dataset,
    extracts 'Iris.csv' to a temporary directory, and pushes the resulting
    absolute path to XCom under the key '"temp_file"' so that downstream
    tasks can consume it.

    Args:
        **context: Airflow task context injected automatically by the
            'PythonOperator'. The task instance is used for XCom writes.

    Raises:
        AirflowException: When the Kaggle API returns an error, a network
            failure occurs, or the target file is absent from the downloaded
            archive.
    """
    ti = context["ti"]

    try:
        csv_path = download_and_extract_dataset(
            owner=IrisConstants.IRIS_OWNER,
            dataset=IrisConstants.IRIS_DATASET,
            target_filename=IrisConstants.IRIS_FILENAME,
        )
    except urllib.error.HTTPError as exc:
        raise AirflowException(
            f"Kaggle API returned HTTP {exc.code} ({exc.reason}). "
            "Check that KAGGLE_USERNAME and KAGGLE_KEY are valid and that "
            f"the dataset '{IrisConstants.IRIS_OWNER}/{IrisConstants.IRIS_DATASET}' "
            "is accessible."
        ) from exc
    except urllib.error.URLError as exc:
        raise AirflowException(
            f"Network error while contacting the Kaggle API: {exc.reason}"
        ) from exc
    except FileNotFoundError as exc:
        raise AirflowException(str(exc)) from exc

    ti.xcom_push(key="temp_file", value=csv_path)
    logger.info("Iris CSV registered in XCom 'temp_file': %s", csv_path)


def validate_iris_data_task(**context: Any) -> dict[str, Any]:
    """Validate the Iris dataset and optionally recover from schema errors.

    Reads the CSV path from XCom (pushed by the ``download_iris_data`` task),
    validates it against ``IrisSchema``, and handles failures according to the
    ``on_validation_error`` DAG param:

    * **``"fail"``** (default) — raises :class:`~airflow.exceptions.AirflowException`
      on the first validation failure, halting the pipeline.
    * **``"drop_invalid_rows"``** — logs every validation error in detail,
      drops the offending rows using a multi-step sanitisation strategy
      (dtype coercion → index-based drops → species filtering), writes a
      cleaned CSV to a temporary file, and pushes its path to XCom under the
      key ``"cleaned_temp_file"``.  The ``transform_iris_data`` task will
      prefer this cleaned path over the original raw file.  The raw file
      remains unchanged so that ``detect_drift`` can still compare against
      the original distribution.

    Args:
        **context: Airflow task context injected automatically by the
            'PythonOperator'. The task instance is used for XCom reads and
            writes.

    Returns:
        Dictionary with ``valid``, ``errors``, ``stats``, and (when rows are
        dropped) ``rows_dropped`` and ``rows_remaining`` keys.

    Raises:
        AirflowException: When the ``"temp_file"`` XCom key is absent, the
            file cannot be read, no valid rows remain after sanitisation, or
            ``on_validation_error`` is ``"fail"`` and errors are present.
    """
    ti = context["ti"]
    params = context.get("params", {})
    on_validation_error: str = params.get("on_validation_error", "fail")

    temp_file: str | None = ti.xcom_pull(key="temp_file", task_ids="download_iris_data")
    if not temp_file:
        raise AirflowException(
            "XCom key 'temp_file' not found or is empty. "
            "Ensure an upstream task pushes the file path via "
            "xcom_push(key='temp_file', value=<path>)."
        )

    try:
        df = load_iris_data(temp_file)
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        raise AirflowException(str(exc)) from exc

    result = validate_iris_data(df)

    if not result.valid:
        error_count = len(result.errors)

        if on_validation_error == "drop_invalid_rows":
            logger.warning(
                "Schema validation failed with %d error(s). "
                "Strategy 'drop_invalid_rows' active — sanitising dataset.",
                error_count,
            )
            logger.warning(
                "Validation error details:\n%s",
                format_validation_errors(result.errors),
            )

            initial_count = len(df)
            df = df.copy()

            # Step 1: Coerce feature columns to numeric — converts strings and
            # other non-parseable values to NaN so they are dropped in step 3.
            for col in IRIS_FEATURE_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Step 2: Drop rows flagged by Pandera with a known row index
            # (covers range violations such as gt=0, isin failures, etc.)
            error_row_indices = {
                int(err.index) for err in result.errors if err.index is not None
            }
            if error_row_indices:
                df = df.drop(index=list(error_row_indices), errors="ignore")

            # Step 3: Drop rows with NaN in any feature column (nulls and
            # coercion failures from step 1).
            df = df.dropna(subset=list(IRIS_FEATURE_COLS)).reset_index(drop=True)

            # Step 4: Drop rows with invalid Species labels.
            if IRIS_TARGET_COL in df.columns:
                df = df[df[IRIS_TARGET_COL].isin(_VALID_SPECIES)].reset_index(drop=True)

            rows_dropped = initial_count - len(df)
            logger.warning(
                "Sanitisation complete — dropped %d row(s), %d valid row(s) remain.",
                rows_dropped,
                len(df),
            )

            if len(df) == 0:
                raise AirflowException(
                    "No valid rows remain after dropping invalid rows. "
                    "Check the data source and consider regenerating it."
                )

            # Write cleaned data to a temporary CSV so the transform task can
            # skip the dirty file without affecting detect_drift (which still
            # reads the original raw file for distribution comparison).
            tmp_dir = Path(tempfile.mkdtemp())
            cleaned_path = tmp_dir / "cleaned_iris.csv"
            df.to_csv(cleaned_path, index=False)
            ti.xcom_push(key="cleaned_temp_file", value=str(cleaned_path))
            logger.info(
                "Cleaned dataset written to %s (%d rows).", cleaned_path, len(df)
            )

            return {
                "valid": False,
                "errors": [error.model_dump(mode="json") for error in result.errors],
                "stats": result.stats.model_dump(mode="json"),
                "rows_dropped": rows_dropped,
                "rows_remaining": len(df),
            }

        # Default strategy: fail the task
        raise AirflowException(
            f"Iris data validation failed with {error_count} error(s):\n"
            f"{format_validation_errors(result.errors)}"
        )

    logger.info("Iris data validation passed. Stats: %s", result.stats)

    return {
        "valid": result.valid,
        "errors": [error.model_dump(mode="json") for error in result.errors],
        "stats": result.stats.model_dump(mode="json"),
    }


def transform_iris_data_task(**context: Any) -> dict[str, Any]:
    """Clean, encode, and persist the Iris dataset to the feature store.

    Pulls the raw CSV path from XCom, removes duplicate and null rows,
    label-encodes the ``Species`` target, and writes the raw (unscaled)
    feature values to ``/public/iris_features.parquet``.
    Numeric features are kept at their original centimetre values;
    :class:`~sklearn.preprocessing.StandardScaler` is applied later inside
    the training sklearn ``Pipeline``.

    When the upstream ``validate_iris_data`` task ran in
    ``"drop_invalid_rows"`` mode it pushes a pre-cleaned CSV path under the
    key ``"cleaned_temp_file"``.  This task prefers that cleaned path over the
    original raw file so that transformation operates on valid data only.

    The ``write_mode`` DAG param controls how the feature store is updated:

    * **``"overwrite"``** (default) — replaces the Parquet file entirely.
    * **``"append"``** — merges incoming rows with the existing Parquet file
      and deduplicates on feature + target columns before writing.

    Args:
        **context: Airflow task context with TaskInstance for XCom operations.

    Returns:
        Dictionary with ``status``, ``output_path``, ``n_features``,
        ``n_rows``, ``classes``, and ``write_mode`` keys.

    Raises:
        AirflowException: On missing XCom key, file read errors, or I/O failures.
    """
    ti = context["ti"]
    params = context.get("params", {})
    write_mode: str = params.get("write_mode", "overwrite")

    # Prefer a pre-cleaned file produced by validate_iris_data_task when the
    # "drop_invalid_rows" strategy was active.
    cleaned_temp: str | None = ti.xcom_pull(
        key="cleaned_temp_file", task_ids="validate_iris_data"
    )
    raw_temp: str | None = ti.xcom_pull(
        key="temp_file", task_ids="download_iris_data"
    )

    temp_file = cleaned_temp or raw_temp

    if not temp_file:
        raise AirflowException(
            "No usable CSV path found in XCom. "
            "Ensure 'download_iris_data' (and optionally 'validate_iris_data') "
            "ran successfully before this task."
        )

    if cleaned_temp:
        logger.info(
            "Using pre-cleaned CSV from validate_iris_data: %s", cleaned_temp
        )
    else:
        logger.info("Using raw CSV from download_iris_data: %s", raw_temp)

    try:
        df = load_iris_data(temp_file)
    except (FileNotFoundError, PermissionError, ValueError) as exc:
        raise AirflowException(str(exc)) from exc

    df_clean = clean_iris_dataframe(df)
    transformed = prepare_universal_features(df_clean)

    try:
        output_path = save_features_to_csv(transformed, write_mode=write_mode)
    except OSError as exc:
        raise AirflowException(
            f"Failed to write feature store: {exc}"
        ) from exc

    logger.info(
        "Transform complete — %d rows written to '%s' (write_mode='%s').",
        len(transformed.data),
        output_path,
        write_mode,
    )

    return {
        "status": "success",
        "output_path": output_path,
        "n_features": len(transformed.feature_names),
        "n_rows": len(transformed.data),
        "classes": transformed.classes,
        "write_mode": write_mode,
    }




