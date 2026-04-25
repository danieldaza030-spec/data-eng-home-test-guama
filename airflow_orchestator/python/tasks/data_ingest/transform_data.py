"""Feature engineering and persistence services for the Iris dataset.

This module is framework-agnostic and raises only standard Python exceptions,
so callers such as Airflow tasks can translate failures at the orchestration
layer. Feature payloads are represented with Pydantic models to replace
anonymous dictionaries with explicit contracts.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame as PaDataFrame
from sklearn.preprocessing import LabelEncoder

from python.tasks.metadata.constants import IrisConstants
from python.tasks.metadata.iris_schema import (
    IRIS_FEATURE_COLS,
    IRIS_TARGET_COL,
    IrisSchema,
    IrisTransformedSchema,
)
from python.tasks.metadata.pipeline_responses import PreparedFeatureSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------


def clean_iris_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and rows with null values.

    The input DataFrame is not mutated; a clean copy is returned with a
    reset integer index.

    Args:
        df: Raw Iris DataFrame, optionally containing an ``Id`` column.

    Returns:
        New dataframe without duplicate or null rows and with a reset index.
    """
    initial_rows = len(df)
    df_clean = df.drop_duplicates().dropna().reset_index(drop=True)
    removed = initial_rows - len(df_clean)
    if removed:
        logger.warning(
            "Removed %d row(s) (duplicates and/or nulls) from the dataset.",
            removed,
        )
    else:
        logger.info("No duplicate or null rows found in the dataset.")
    return df_clean


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def prepare_universal_features(
    validated_data: PaDataFrame[IrisSchema],
) -> PreparedFeatureSet:
    """Encode the target column and prepare raw numeric features for the feature store.

    Drops the ``Id`` column, label-encodes the ``Species`` target with
    :class:`~sklearn.preprocessing.LabelEncoder`, and appends a UTC
    ``processed_at`` timestamp.  Numeric feature columns are kept at their
    **original centimetre values** — no scaling is performed here.
    :class:`~sklearn.preprocessing.StandardScaler` is applied later inside
    the training sklearn ``Pipeline``, which is persisted with the model
    artefact so inference benefits from the same transformation without risk
    of data leakage.

    Args:
        validated_data: Validated and cleaned Iris DataFrame.

    Returns:
        Structured feature payload with raw (unscaled) numeric features,
        label-encoded target, and transformation metadata.
    """
    copy_validated_data = validated_data.drop(columns=["Id"], errors="ignore").copy()

    X = copy_validated_data[IRIS_FEATURE_COLS].copy()
    y = copy_validated_data[IRIS_TARGET_COL].copy()

    le = LabelEncoder()
    y_encoded = pd.Series(
        le.fit_transform(y), name=IRIS_TARGET_COL, index=y.index
    )

    result_df = pd.concat([X, y_encoded], axis=1)
    result_df["processed_at"] = datetime.now(timezone.utc)

    logger.info(
        "Feature preparation complete — rows: %d, features: %d, classes: %s.",
        len(result_df),
        len(IRIS_FEATURE_COLS),
        list(le.classes_),
    )

    return PreparedFeatureSet(
        data=result_df,
        feature_names=IRIS_FEATURE_COLS,
        target_name=IRIS_TARGET_COL,
        classes=list(le.classes_),
        encoder=le,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_features_to_csv(
    transformed: PreparedFeatureSet,
    output_dir: Path | None = None,
    filename: str = IrisConstants.FEATURE_STORE_FILENAME,
    write_mode: Literal["overwrite", "append"] = "overwrite",
) -> str:
    """Validate and write the transformed feature dataframe to a CSV file.

    In ``"overwrite"`` mode the feature store is replaced entirely by the
    current batch (after deduplication).  In ``"append"`` mode the new rows
    are merged with any existing CSV file and the combined dataset is
    deduplicated before writing, so the feature store grows incrementally
    without accumulating duplicate measurements.

    Deduplication is performed on the feature and target columns only
    (``SepalLengthCm``, ``SepalWidthCm``, ``PetalLengthCm``,
    ``PetalWidthCm``, ``Species``), keeping the first occurrence and ignoring
    the ``processed_at`` timestamp so that identical measurements ingested at
    different times are treated as duplicates.

    Args:
        transformed: Structured feature payload from
            :func:`prepare_universal_features`.
        output_dir: Destination directory.  Defaults to
            :attr:`~python.tasks.metadata.constants.IrisConstants.FEATURE_STORAGE_DIR`.
        filename: Output filename.  Defaults to
            :attr:`~python.tasks.metadata.constants.IrisConstants.FEATURE_STORE_FILENAME`.
        write_mode: ``"overwrite"`` (default) replaces the feature store
            entirely; ``"append"`` merges new rows with the existing store
            and deduplicates.

    Returns:
        Absolute path of the written CSV file as a string.

    Raises:
        pandera.errors.SchemaErrors: If the incoming batch does not conform
            to :class:`~python.tasks.metadata.iris_schema.IrisTransformedSchema`.
        OSError: If the directory cannot be created or the file cannot be
            written due to permission or disk-space issues.
    """
    destination = output_dir if output_dir is not None else IrisConstants.FEATURE_STORAGE_DIR
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / filename

    incoming = transformed.data.copy()

    try:
        IrisTransformedSchema.validate(incoming, lazy=True)
        logger.info("Transformed schema validation passed.")
    except pa.errors.SchemaErrors as exc:
        raise exc

    dedup_cols = list(IRIS_FEATURE_COLS) + [IRIS_TARGET_COL]

    if write_mode == "append" and output_path.exists():
        try:
            existing = pd.read_csv(output_path)
            combined = pd.concat([existing, incoming], ignore_index=True)
            before = len(combined)
            combined = combined.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
            removed = before - len(combined)
            if removed:
                logger.warning(
                    "Deduplication removed %d row(s) during append to feature store.",
                    removed,
                )
            data = combined
            logger.info(
                "Appended %d new rows; feature store now contains %d rows.",
                len(incoming),
                len(data),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not read existing feature store for append (%s). "
                "Falling back to overwrite.",
                exc,
            )
            data = incoming.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    else:
        before = len(incoming)
        data = incoming.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
        removed = before - len(data)
        if removed:
            logger.warning(
                "Deduplication removed %d row(s) from incoming batch before writing.",
                removed,
            )

    data.to_csv(output_path, index=False)

    logger.info(
        "Feature store written to: %s (%d rows, %d columns, mode='%s').",
        output_path,
        len(data),
        len(data.columns),
        write_mode,
    )
    return str(output_path)
