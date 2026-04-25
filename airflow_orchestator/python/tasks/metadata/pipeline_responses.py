"""Typed response models shared across the Iris pipelines.

This module centralizes the Pydantic models exchanged between ingestion,
validation, transformation, and training functions. The goal is to replace
anonymous dictionaries with explicit contracts that remain documented and
type-safe through the full Airflow 3.x pipeline flow.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.preprocessing import LabelEncoder


class PipelineResponseModel(BaseModel):
    """Base model for pipeline responses with arbitrary runtime objects enabled."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ValidationErrorItem(PipelineResponseModel):
    """Single validation failure produced by Pandera lazy validation."""

    schema_context: str | None = None
    column: str | None = None
    check: str | None = None
    check_number: int | None = None
    failure_case: Any = None
    index: int | str | None = None


class NumericColumnStats(PipelineResponseModel):
    """Descriptive statistics for one numeric feature column."""

    mean: float
    std: float
    min: float
    max: float
    median: float


class ValidationStats(PipelineResponseModel):
    """Descriptive summary generated while validating the Iris dataset."""

    total_records: int = 0
    duplicates: int = 0
    missing_values: dict[str, int] = Field(default_factory=dict)
    unique_ids: bool = False
    species_distribution: dict[str, int] = Field(default_factory=dict)
    numeric: dict[str, NumericColumnStats] = Field(default_factory=dict)
    correlations: dict[str, dict[str, float]] = Field(default_factory=dict)


class IrisValidationResult(PipelineResponseModel):
    """Validation result returned by the dataset validation domain service."""

    valid: bool = False
    errors: list[ValidationErrorItem] = Field(default_factory=list)
    data: pd.DataFrame | None = None
    stats: ValidationStats = Field(default_factory=ValidationStats)


class PreparedFeatureSet(PipelineResponseModel):
    """Feature-engineering payload shared between transformation steps."""

    data: pd.DataFrame
    feature_names: list[str]
    target_name: str
    classes: list[str]
    encoder: LabelEncoder


class TrainingMetrics(PipelineResponseModel):
    """Common evaluation metrics logged for a trained classification model."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float