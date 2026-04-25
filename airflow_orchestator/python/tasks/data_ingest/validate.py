"""Domain services for loading and validating the Iris dataset.

This module is framework-agnostic and raises only standard Python exceptions,
so callers such as Airflow tasks can translate failures at the orchestration
layer. Validation responses are expressed with Pydantic models to keep the
contract explicit across the full pipeline.
"""

import logging

import pandera.pandas as pa
import pandas as pd

from python.tasks.metadata.pipeline_responses import (
    IrisValidationResult,
    NumericColumnStats,
    ValidationErrorItem,
)
from python.tasks.metadata.iris_schema import IrisSchema, IRIS_FEATURE_COLS, IRIS_TARGET_COL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_iris_data(file_path: str) -> pd.DataFrame:
    """Read an Iris CSV file from disk.

    Args:
        file_path: Absolute or relative path to the CSV file.

    Returns:
        DataFrame with the raw Iris records.

    Raises:
        FileNotFoundError: If no file exists at 'file_path'.
        PermissionError: If the process lacks read access to the file.
        ValueError: If the file cannot be parsed as a valid CSV.
    """
    logger.info("Reading Iris dataset from: %s", file_path)
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise
    except PermissionError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not parse file as CSV: {file_path!r}. Detail: {exc}"
        ) from exc

    logger.info("File loaded — rows: %d, columns: %d.", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------


def format_validation_errors(errors: list[ValidationErrorItem]) -> str:
    """Render validation failures as a human-readable string.

    Args:
        errors: Validation failure items produced by ``validate_iris_data``.

    Returns:
        Multiline string with one numbered failure per line. Returns
        ``"(no details available)"`` when the list is empty.
    """
    if not errors:
        return "(no details available)"

    lines: list[str] = []
    for idx, case in enumerate(errors, start=1):
        column = case.column or "unknown column"
        check = case.check or "unknown check"
        failure = case.failure_case if case.failure_case is not None else "N/A"
        row_index = case.index if case.index is not None else "N/A"
        lines.append(
            f"  [{idx}] column={column!r}, check={check!r}, "
            f"failure_case={failure!r}, row_index={row_index}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_iris_data(df: pd.DataFrame) -> IrisValidationResult:
    """Validate the Iris dataset and compute descriptive statistics.

    Args:
        df: Iris dataset with the original Kaggle column layout.

    Returns:
        Structured validation result with status, detailed failures, the
        validated dataframe when successful, and descriptive statistics.
    """
    result = IrisValidationResult()

    try:
        validated_df = IrisSchema.validate(df, lazy=True)
        result.valid = True
        result.data = validated_df
        logger.info("Schema validation passed.")
    except pa.errors.SchemaErrors as exc:
        result.errors = [
            ValidationErrorItem(**failure_case)
            for failure_case in exc.failure_cases.to_dict("records")
        ]
        validated_df = df
        logger.warning("%d schema error(s) found.", len(exc.failure_cases))

    # -- Summary header -------------------------------------------------------
    logger.info("=" * 60)
    logger.info("DATA SUMMARY")
    logger.info("=" * 60)

    # -- General stats --------------------------------------------------------
    total = len(validated_df)
    duplicates = int(validated_df.duplicated().sum())
    missing = int(validated_df.isnull().sum().sum())
    unique_ids = bool(validated_df["Id"].is_unique)

    logger.info("General:")
    logger.info("  Records   : %d", total)
    logger.info("  Columns   : %d", len(validated_df.columns))
    logger.info("  Duplicates: %d", duplicates)
    logger.info("  Missing   : %d", missing)
    logger.info("  Unique IDs: %s", unique_ids)

    result.stats.total_records = total
    result.stats.duplicates = duplicates
    result.stats.missing_values = {
        column_name: int(missing_count)
        for column_name, missing_count in validated_df.isnull().sum().to_dict().items()
    }
    result.stats.unique_ids = unique_ids

    # -- Species distribution -------------------------------------------------
    if "Species" in validated_df.columns:
        species_counts = validated_df["Species"].value_counts()
        result.stats.species_distribution = {
            species_name: int(species_total)
            for species_name, species_total in species_counts.to_dict().items()
        }
        logger.info("Species distribution:")
        for species, count in species_counts.items():
            pct = count / total * 100
            logger.info("  %s: %d (%.1f%%)", species, count, pct)

    # -- Numeric statistics ---------------------------------------------------
    numeric_cols = [
        col
        for col in validated_df.select_dtypes(include=["float64", "int64"]).columns
        if col != "Id"
    ]
    if numeric_cols:
        logger.info("Numeric statistics:")
        logger.info(
            "%-20s %-10s %-10s %-10s %-10s",
            "Column",
            "Mean",
            "Std",
            "Min",
            "Max",
        )
        logger.info("-" * 60)
        for col in numeric_cols:
            mean = validated_df[col].mean()
            std = validated_df[col].std()
            min_val = validated_df[col].min()
            max_val = validated_df[col].max()
            median = validated_df[col].median()
            logger.info(
                "%-20s %-10.2f %-10.2f %-10.2f %-10.2f",
                col,
                mean,
                std,
                min_val,
                max_val,
            )
            result.stats.numeric[col] = NumericColumnStats(
                mean=float(mean),
                std=float(std),
                min=float(min_val),
                max=float(max_val),
                median=float(median),
            )

    # -- Correlations ---------------------------------------------------------
    if all(col in validated_df.columns for col in IRIS_FEATURE_COLS):
        corr_matrix = validated_df[IRIS_FEATURE_COLS].corr()
        result.stats.correlations = {
            row_name: {
                column_name: float(correlation)
                for column_name, correlation in row_values.items()
            }
            for row_name, row_values in corr_matrix.to_dict().items()
        }
        logger.info("Strong correlations (|r| > 0.7):")
        found = False
        for i in range(len(IRIS_FEATURE_COLS)):
            for j in range(i + 1, len(IRIS_FEATURE_COLS)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    logger.info(
                        "  %s <-> %s: %.3f",
                        IRIS_FEATURE_COLS[i],
                        IRIS_FEATURE_COLS[j],
                        corr,
                    )
                    found = True
        if not found:
            logger.info("  No strong correlations found.")

    logger.info("=" * 60)
    return result