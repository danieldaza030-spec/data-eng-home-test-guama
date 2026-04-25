"""Synthetic Iris data generation for pipeline testing.

Generates artificial samples that either mimic (normal mode) or diverge from
(drifted mode) the known Iris dataset class-conditional distributions.
Output is formatted as a raw Iris CSV compatible with ``IrisSchema``, so it
can be fed directly into the ``ingest_with_drift`` DAG.

Normal mode reproduces the standard UCI Iris Gaussian parameters per class.
Drifted mode applies significant additive shifts to the petal features and
skews the class proportions towards *Iris-virginica*, producing a batch that
the KS drift detector will flag with high confidence (p << 0.05).
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

_DIRTY_NULL_FRACTION: float = 0.12
_DIRTY_NEGATIVE_FRACTION: float = 0.08
_DIRTY_INVALID_SPECIES_FRACTION: float = 0.08
_DIRTY_DUPLICATE_FRACTION: float = 0.10
_INVALID_SPECIES_LABEL: str = "Iris-unknown"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference statistics — raw centimetre scale, per class
# Source: UCI Iris dataset descriptive statistics (n=50 per class)
# ---------------------------------------------------------------------------

_CLASS_PARAMS: dict[str, dict[str, tuple[float, float]]] = {
    "Iris-setosa": {
        "SepalLengthCm": (5.006, 0.352),
        "SepalWidthCm": (3.418, 0.381),
        "PetalLengthCm": (1.464, 0.174),
        "PetalWidthCm": (0.244, 0.107),
    },
    "Iris-versicolor": {
        "SepalLengthCm": (5.936, 0.516),
        "SepalWidthCm": (2.770, 0.314),
        "PetalLengthCm": (4.260, 0.470),
        "PetalWidthCm": (1.326, 0.198),
    },
    "Iris-virginica": {
        "SepalLengthCm": (6.588, 0.636),
        "SepalWidthCm": (2.974, 0.322),
        "PetalLengthCm": (5.552, 0.552),
        "PetalWidthCm": (2.026, 0.275),
    },
}

# Balanced proportions for normal mode (1/3 each)
_NORMAL_PROPORTIONS: dict[str, float] = {
    "Iris-setosa": 1 / 3,
    "Iris-versicolor": 1 / 3,
    "Iris-virginica": 1 / 3,
}

# Skewed proportions for drifted mode (bias towards virginica)
_DRIFTED_PROPORTIONS: dict[str, float] = {
    "Iris-setosa": 0.10,
    "Iris-versicolor": 0.20,
    "Iris-virginica": 0.70,
}

# Additive shifts applied in drifted mode (raw cm scale)
# Petal features are shifted by ~5x their within-class std to guarantee
# that scipy.stats.ks_2samp yields p << 0.05 even for moderate sample sizes.
_DRIFT_SHIFTS: dict[str, float] = {
    "SepalLengthCm": 0.0,
    "SepalWidthCm": -0.70,
    "PetalLengthCm": 2.50,
    "PetalWidthCm": 1.00,
}

_FEATURE_COLS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_class(
    species: str,
    n_samples: int,
    rng: np.random.Generator,
    shifts: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Sample synthetic records for a single Iris species.

    Args:
        species: Species label (e.g. ``"Iris-setosa"``).
        n_samples: Number of records to generate for this class.
        rng: NumPy random Generator used for reproducibility.
        shifts: Optional per-feature additive offsets applied after sampling.
            Missing keys default to zero shift.

    Returns:
        DataFrame with feature columns and a ``Species`` label column.
    """
    params = _CLASS_PARAMS[species]
    shifts = shifts or {}
    data: dict[str, np.ndarray] = {}

    for col in _FEATURE_COLS:
        mean, std = params[col]
        shift = shifts.get(col, 0.0)
        samples = rng.normal(loc=mean + shift, scale=std, size=n_samples)
        # Enforce gt=0 schema constraint; clip to a small positive value.
        data[col] = np.clip(samples, 0.01, None).round(2)

    data["Species"] = species  # type: ignore[assignment]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Internal helper — dirty data
# ---------------------------------------------------------------------------


def _generate_dirty_iris(n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a synthetic Iris dataset with intentional data-quality errors.

    Produces a base batch with balanced class proportions (same as ``"normal"``
    mode) and then injects four types of errors to exercise validation and
    error-handling paths in the ingest pipeline:

    * **Null values** — a random subset of rows has ``NaN`` injected into one
      randomly chosen feature column, triggering Pandera null checks.
    * **Negative values** — a random subset of rows has one feature column
      set to a negative value, violating the ``gt=0`` schema constraint.
    * **Invalid Species labels** — a random subset of rows has the
      ``Species`` column replaced with ``"Iris-unknown"``, failing the
      ``isin`` check.
    * **Duplicate rows** — a random subset of rows is repeated, to test the
      deduplication step downstream.

    Args:
        n_samples: Total number of rows (including injected errors) to produce.
        rng: NumPy random Generator for reproducibility.

    Returns:
        DataFrame with columns ``Id``, ``SepalLengthCm``, ``SepalWidthCm``,
        ``PetalLengthCm``, ``PetalWidthCm``, and ``Species``.
    """
    # 1. Generate clean base using normal proportions
    frames: list[pd.DataFrame] = []
    for species, proportion in _NORMAL_PROPORTIONS.items():
        n_class = max(1, round(n_samples * proportion))
        frames.append(_sample_class(species, n_class, rng))

    df = pd.concat(frames, ignore_index=True).reset_index(drop=True)

    total = len(df)

    # 2. Inject NaN in a random feature column for _DIRTY_NULL_FRACTION rows
    null_idx = rng.choice(total, size=max(1, round(total * _DIRTY_NULL_FRACTION)), replace=False)
    null_cols = rng.choice(_FEATURE_COLS, size=len(null_idx))
    for row_i, col in zip(null_idx, null_cols):
        df.at[row_i, col] = np.nan

    # 3. Inject negative values in a random feature column for _DIRTY_NEGATIVE_FRACTION rows
    neg_idx = rng.choice(total, size=max(1, round(total * _DIRTY_NEGATIVE_FRACTION)), replace=False)
    neg_cols = rng.choice(_FEATURE_COLS, size=len(neg_idx))
    for row_i, col in zip(neg_idx, neg_cols):
        df.at[row_i, col] = round(float(rng.uniform(-5.0, -0.01)), 2)

    # 4. Inject invalid Species label for _DIRTY_INVALID_SPECIES_FRACTION rows
    inv_idx = rng.choice(total, size=max(1, round(total * _DIRTY_INVALID_SPECIES_FRACTION)), replace=False)
    df.loc[inv_idx, "Species"] = _INVALID_SPECIES_LABEL

    # 5. Append duplicate rows (_DIRTY_DUPLICATE_FRACTION of the base)
    dup_idx = rng.choice(total, size=max(1, round(total * _DIRTY_DUPLICATE_FRACTION)), replace=False)
    duplicates = df.iloc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # Shuffle and assign sequential IDs
    df = df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    df.insert(0, "Id", range(1, len(df) + 1))

    null_count = int(df[_FEATURE_COLS].isnull().sum().sum())
    neg_count = int((df[_FEATURE_COLS].fillna(0) < 0).sum().sum())
    inv_species_count = int((df["Species"] == _INVALID_SPECIES_LABEL).sum())
    dup_count = int(df.duplicated(subset=_FEATURE_COLS + ["Species"]).sum())

    logger.warning(
        "Generated %d dirty rows. Injected errors — nulls: %d, negatives: %d, "
        "invalid species: %d, duplicates: %d.",
        len(df),
        null_count,
        neg_count,
        inv_species_count,
        dup_count,
    )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_synthetic_iris(
    n_samples: int,
    mode: Literal["normal", "drifted", "dirty"],
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic Iris dataset in raw CSV format.

    Produces ``n_samples`` rows using class-conditional Gaussian distributions
    derived from the original UCI Iris dataset.  In *drifted* mode significant
    additive shifts are applied to the petal features and the class proportions
    are skewed towards *Iris-virginica*, producing a batch that the two-sample
    KS test will detect with high statistical confidence (p << 0.05).
    In *dirty* mode the batch contains intentional data-quality errors (nulls,
    out-of-range negatives, invalid Species labels, and duplicates) to
    exercise validation and error-handling paths.

    The output schema matches the raw Iris CSV validated by ``IrisSchema``
    (``Id``, ``SepalLengthCm``, ``SepalWidthCm``, ``PetalLengthCm``,
    ``PetalWidthCm``, ``Species``), so it can be fed directly into the
    ``ingest_with_drift`` DAG.

    Args:
        n_samples: Total number of rows to generate.
        mode: ``"normal"`` to reproduce the standard Iris distribution;
            ``"drifted"`` to introduce detectable petal-feature shifts and
            class-proportion imbalance; ``"dirty"`` to inject validation
            errors for error-handling demonstrations.
        random_state: Seed for the NumPy random Generator; ensures
            reproducibility across runs.

    Returns:
        DataFrame with columns ``Id``, ``SepalLengthCm``, ``SepalWidthCm``,
        ``PetalLengthCm``, ``PetalWidthCm``, and ``Species``.  Rows are
        shuffled and assigned sequential integer IDs starting at 1.
    """
    rng = np.random.default_rng(random_state)

    if mode == "dirty":
        return _generate_dirty_iris(n_samples=n_samples, rng=rng)

    proportions = _NORMAL_PROPORTIONS if mode == "normal" else _DRIFTED_PROPORTIONS
    shifts = None if mode == "normal" else _DRIFT_SHIFTS

    frames: list[pd.DataFrame] = []
    for species, proportion in proportions.items():
        n_class = max(1, round(n_samples * proportion))
        frames.append(_sample_class(species, n_class, rng, shifts))

    df = pd.concat(frames, ignore_index=True)

    # Shuffle rows so classes are not grouped together
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Assign sequential IDs (schema requires Id > 0 and unique)
    df.insert(0, "Id", range(1, len(df) + 1))

    class_counts = df["Species"].value_counts().to_dict()
    logger.info(
        "Generated %d rows in '%s' mode. Class distribution: %s",
        len(df),
        mode,
        class_counts,
    )
    return df
