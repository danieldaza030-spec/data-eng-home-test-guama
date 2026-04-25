"""Kolmogorov-Smirnov data-drift detection for the Iris feature store.

Compares a reference (baseline) raw feature distribution against the most
recently ingested batch using a two-sample KS test on each numeric feature
column independently.

The KS test is non-parametric: it makes no assumptions about the underlying
distribution and is sensitive to any difference in the shape, location, or
scale of two empirical CDFs.  A feature is flagged as drifted when its
p-value falls below the configured significance level ``alpha``.

Drift is detected on the *raw* (pre-StandardScaler) feature values because
re-fitting the scaler on each new batch would normalise away any
location/scale shift, making post-scaling comparison unreliable.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class FeatureDriftResult:
    """KS test result for a single numeric feature.

    Attributes:
        feature: Column name tested.
        ks_statistic: Maximum absolute difference between the two empirical
            CDFs (D statistic).  Ranges from 0 (identical distributions)
            to 1 (completely separated distributions).
        p_value: Two-tailed p-value for the null hypothesis that both samples
            were drawn from the same continuous distribution.
        drift_detected: ``True`` when ``p_value < alpha``.
    """

    feature: str
    ks_statistic: float
    p_value: float
    drift_detected: bool


@dataclass
class DriftReport:
    """Aggregate KS drift report across all tested features.

    Attributes:
        feature_results: One :class:`FeatureDriftResult` per tested feature
            column, in the order they were evaluated.
        overall_drift: ``True`` when at least one feature is flagged as
            drifted.
        alpha: Significance level used for each individual feature test.
        n_reference: Number of rows in the reference (baseline) dataset.
        n_current: Number of rows in the newly ingested dataset.
    """

    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    overall_drift: bool = False
    alpha: float = 0.05
    n_reference: int = 0
    n_current: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_ks_drift_test(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
    alpha: float = 0.05,
) -> DriftReport:
    """Run per-feature two-sample KS tests to detect data drift.

    Applies :func:`scipy.stats.ks_2samp` to each feature column
    independently.  A feature is flagged as drifted when its p-value falls
    below ``alpha``.  Overall drift is declared when at least one feature is
    flagged.

    Results are logged at ``WARNING`` level for drifted features and
    ``INFO`` level for stable ones.  A summary ``WARNING`` is emitted when
    overall drift is detected.

    Args:
        reference: Baseline feature DataFrame from the last processed batch
            (raw, pre-scaling values).
        current: Newly ingested feature DataFrame to compare against the
            baseline (raw, pre-scaling values).
        feature_cols: Names of numeric columns to test.  Rows with ``NaN``
            in a column are dropped before running that column's test.
        alpha: Significance level for each individual KS test; default 0.05.

    Returns:
        :class:`DriftReport` with per-feature results and an overall drift
        flag.
    """
    report = DriftReport(
        alpha=alpha,
        n_reference=len(reference),
        n_current=len(current),
    )

    logger.info(
        "Running KS drift test: reference=%d rows, current=%d rows, alpha=%.2f",
        report.n_reference,
        report.n_current,
        alpha,
    )

    for col in feature_cols:
        ref_values = reference[col].dropna().to_numpy()
        cur_values = current[col].dropna().to_numpy()

        ks_result = stats.ks_2samp(ref_values, cur_values)
        feature_drift = FeatureDriftResult(
            feature=col,
            ks_statistic=float(ks_result.statistic),
            p_value=float(ks_result.pvalue),
            drift_detected=ks_result.pvalue < alpha,
        )
        report.feature_results.append(feature_drift)

        log_fn = logger.warning if feature_drift.drift_detected else logger.info
        log_fn(
            "[KS] %-18s  D=%.4f  p=%.6f  drift=%s",
            col,
            feature_drift.ks_statistic,
            feature_drift.p_value,
            "YES" if feature_drift.drift_detected else "NO",
        )

    report.overall_drift = any(r.drift_detected for r in report.feature_results)

    if report.overall_drift:
        drifted = [r.feature for r in report.feature_results if r.drift_detected]
        logger.warning(
            "DATA DRIFT DETECTED — %d/%d features flagged: %s",
            len(drifted),
            len(feature_cols),
            ", ".join(drifted),
        )
    else:
        logger.info(
            "No data drift detected across all %d features (alpha=%.2f).",
            len(feature_cols),
            alpha,
        )

    return report
