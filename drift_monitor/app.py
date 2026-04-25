"""Streamlit dashboard for Iris production drift monitoring.

Displays distribution comparisons between training features and production
prediction features, KS test results, class distribution charts, and a
recent predictions table.

Data sources (read from the shared ``/public`` volume):
    * ``/public/feature_storage/features_iris.csv`` — training feature store
    * ``/public/prediction_storage/predicted.csv`` — inference API prediction log

Sidebar controls:
    * Refresh — clears the 30-second data cache and reloads the page.
    * Trigger Retraining — POSTs a new DAG run to the Airflow REST API
      (``training_pipeline`` DAG, basic auth ``airflow / airflow``).
"""

import os

import requests
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FEATURE_STORE_PATH = "/public/feature_storage/features_iris.csv"
_PREDICTIONS_PATH = "/public/prediction_storage/predicted.csv"

_FEATURE_COLS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

_CLASS_LABELS = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

_MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
_REGISTERED_MODEL_NAME = "iris_classifier"
_CHAMPION_ALIAS = "champion"

_AIRFLOW_TRIGGER_URL = (
    "http://airflow-apiserver:8080/api/v2/dags/training_pipeline/dagRuns"
)
_AIRFLOW_AUTH = ("airflow", "airflow")

_KS_ALPHA = 0.05
_MIN_ROWS = 30


# ---------------------------------------------------------------------------
# Data loading (cached with 30-second TTL for auto-refresh feel)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30)
def _load_feature_store() -> pd.DataFrame | None:
    """Load the training feature store CSV.

    Returns:
        DataFrame with training features and target, or ``None`` if the file
        does not exist.
    """
    try:
        return pd.read_csv(_FEATURE_STORE_PATH)
    except FileNotFoundError:
        return None


@st.cache_data(ttl=60)
def _get_champion_training_dates() -> list[str] | None:
    """Fetch the ``training_dates`` tag from the current champion model version.

    Queries the MLflow Model Registry for the ``@champion`` alias of
    ``iris_classifier`` and returns the comma-separated ``training_dates`` tag
    parsed into a sorted list of ISO date strings.

    Returns:
        Sorted list of ``YYYY-MM-DD`` strings, or ``None`` when the tag is
        absent or the champion cannot be resolved.
    """
    try:
        client = MlflowClient(tracking_uri=_MLFLOW_TRACKING_URI)
        champion = client.get_model_version_by_alias(
            name=_REGISTERED_MODEL_NAME,
            alias=_CHAMPION_ALIAS,
        )
        raw = champion.tags.get("training_dates")
        if raw:
            return sorted(d.strip() for d in raw.split(",") if d.strip())
        return None
    except Exception:  # noqa: BLE001
        return None


def _filter_to_training_dates(
    df: pd.DataFrame,
    training_dates: list[str],
) -> pd.DataFrame:
    """Filter a feature-store DataFrame to only the rows used for training.

    Args:
        df: Full feature store DataFrame with a ``processed_at`` column.
        training_dates: List of ISO date strings (``YYYY-MM-DD``) to keep.

    Returns:
        Filtered DataFrame, or the original ``df`` when ``processed_at`` is
        absent or no rows match.
    """
    if "processed_at" not in df.columns:
        return df
    mask = pd.to_datetime(df["processed_at"]).dt.date.astype(str).isin(training_dates)
    filtered = df[mask].reset_index(drop=True)
    return filtered if not filtered.empty else df


@st.cache_data(ttl=30)
def _load_predictions() -> pd.DataFrame | None:
    """Load the prediction log written by the inference API.

    Returns:
        DataFrame with logged predictions, or ``None`` if the file does not
        exist.
    """
    try:
        return pd.read_csv(_PREDICTIONS_PATH)
    except FileNotFoundError:
        return None


def _run_ks_tests(
    reference: pd.DataFrame, current: pd.DataFrame
) -> list[dict]:
    """Run two-sample KS tests for each feature column.

    Args:
        reference: Training feature store DataFrame.
        current: Production predictions DataFrame.

    Returns:
        List of dicts with keys ``feature``, ``ks_statistic``, ``p_value``,
        ``drift_detected`` for each tested feature column.
    """
    results = []
    for col in _FEATURE_COLS:
        if col not in reference.columns or col not in current.columns:
            continue
        ref_vals = reference[col].dropna()
        cur_vals = current[col].dropna()
        if len(ref_vals) < _MIN_ROWS or len(cur_vals) < _MIN_ROWS:
            results.append(
                {
                    "feature": col,
                    "ks_statistic": None,
                    "p_value": None,
                    "drift_detected": None,
                    "note": f"Insufficient data (ref={len(ref_vals)}, cur={len(cur_vals)})",
                }
            )
            continue
        stat, p_val = stats.ks_2samp(ref_vals, cur_vals)
        results.append(
            {
                "feature": col,
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_val), 4),
                "drift_detected": bool(p_val < _KS_ALPHA),
                "note": "",
            }
        )
    return results


# ---------------------------------------------------------------------------
# Page layout helpers
# ---------------------------------------------------------------------------


def _render_metrics(
    training: pd.DataFrame,
    predictions: pd.DataFrame,
    ks_results: list[dict],
) -> None:
    """Render the top-level summary metric cards.

    Args:
        training: Training feature store DataFrame.
        predictions: Production predictions DataFrame.
        ks_results: List of per-feature KS test result dicts.
    """
    drifted = [r for r in ks_results if r.get("drift_detected") is True]
    drift_status = f"⚠️ {len(drifted)} feature(s)" if drifted else "✅ None"

    last_pred_time = "—"
    if "prediction_at" in predictions.columns and not predictions.empty:
        last_pred_time = predictions["prediction_at"].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training samples", f"{len(training):,}")
    col2.metric("Predictions logged", f"{len(predictions):,}")
    col3.metric("Drift detected", drift_status)
    col4.metric("Last prediction", last_pred_time)


def _render_distribution_plots(
    training: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    """Render 2×2 overlapping histogram grid (training vs production).

    Args:
        training: Training feature store DataFrame.
        predictions: Production predictions DataFrame.
    """
    st.subheader("Feature Distributions — Training vs Production")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for i, col in enumerate(_FEATURE_COLS):
        ax = axes_flat[i]
        if col in training.columns:
            ax.hist(
                training[col].dropna(),
                bins=25,
                alpha=0.6,
                label="Training",
                color="steelblue",
                density=True,
            )
        if col in predictions.columns:
            ax.hist(
                predictions[col].dropna(),
                bins=25,
                alpha=0.6,
                label="Production",
                color="tomato",
                density=True,
            )
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_ks_table(ks_results: list[dict]) -> None:
    """Render KS test results table with drift highlighting.

    Args:
        ks_results: List of per-feature KS test result dicts.
    """
    st.subheader("KS Drift Test Results")

    if not ks_results:
        st.info("No KS results available (insufficient data).")
        return

    df = pd.DataFrame(ks_results)[["feature", "ks_statistic", "p_value", "drift_detected", "note"]]
    df.columns = ["Feature", "KS Statistic", "P-Value", "Drift Detected", "Note"]

    def _highlight_drift(row: pd.Series) -> list[str]:
        color = "background-color: #ffe0e0" if row["Drift Detected"] is True else ""
        return [color] * len(row)

    st.dataframe(
        df.style.apply(_highlight_drift, axis=1),
        use_container_width=True,
        hide_index=True,
    )


def _render_class_distribution(
    training: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    """Render side-by-side class distribution bar charts.

    Args:
        training: Training feature store DataFrame.
        predictions: Production predictions DataFrame.
    """
    st.subheader("Class Distribution — Training vs Production")

    col_train, col_pred = st.columns(2)

    with col_train:
        st.markdown("**Training**")
        if "Species" in training.columns:
            counts = training["Species"].value_counts().sort_index()
            labels = [_CLASS_LABELS.get(k, str(k)) for k in counts.index]
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(labels, counts.values, color="steelblue")
            ax.set_ylabel("Count")
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("'Species' column not found in training data.")

    with col_pred:
        st.markdown("**Production**")
        if "predicted_class" in predictions.columns:
            counts = predictions["predicted_class"].value_counts().sort_index()
            labels = [_CLASS_LABELS.get(k, str(k)) for k in counts.index]
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(labels, counts.values, color="tomato")
            ax.set_ylabel("Count")
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("'predicted_class' column not found in predictions.")


def _render_recent_predictions(predictions: pd.DataFrame) -> None:
    """Render a table of the most recent 50 predictions.

    Args:
        predictions: Production predictions DataFrame.
    """
    st.subheader("Recent Predictions (last 50)")
    st.dataframe(
        predictions.tail(50).sort_index(ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def _render_sidebar() -> None:
    """Render the sidebar with refresh and retraining controls."""
    with st.sidebar:
        st.title("Controls")

        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.markdown("### Retraining")
        st.markdown(
            "Manually trigger the ``training_pipeline`` DAG in Airflow "
            "to retrain all classifiers on the current feature store."
        )

        if st.button("🚀 Trigger Retraining"):
            _trigger_retraining()


def _trigger_retraining() -> None:
    """POST a new DAG run to the Airflow REST API for ``training_pipeline``."""
    try:
        response = requests.post(
            _AIRFLOW_TRIGGER_URL,
            auth=_AIRFLOW_AUTH,
            json={},
            timeout=10,
        )
        if response.status_code in (200, 201):
            st.sidebar.success("Retraining triggered successfully.")
        else:
            st.sidebar.error(
                f"Airflow API returned HTTP {response.status_code}: {response.text}"
            )
    except requests.exceptions.ConnectionError:
        st.sidebar.error(
            "Could not reach the Airflow API. "
            "Make sure the 'airflow-apiserver' service is running."
        )
    except requests.exceptions.Timeout:
        st.sidebar.error("Request to Airflow API timed out.")


# ---------------------------------------------------------------------------
# Main app entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Streamlit drift monitoring dashboard."""
    st.set_page_config(
        page_title="Iris Drift Monitor",
        page_icon="🌸",
        layout="wide",
    )
    st.title("🌸 Iris Production Drift Monitor")
    st.caption(
        "Compares feature distributions of real production requests against "
        "the training feature store using two-sample KS tests."
    )

    _render_sidebar()

    training = _load_feature_store()
    predictions = _load_predictions()

    # Filter feature store to the rows the current champion was trained on.
    # Falls back to the full store when the tag is absent (backward compat).
    training_dates = _get_champion_training_dates()
    if training is not None and training_dates:
        training = _filter_to_training_dates(training, training_dates)
        st.caption(
            f"Reference distribution filtered to champion training dates: "
            f"``{', '.join(training_dates)}`` ({len(training):,} rows)"
        )
    elif training is not None:
        st.caption(
            "Reference distribution: full feature store "
            "(champion has no ``training_dates`` tag — backward compat mode)."
        )

    if training is None:
        st.error(
            "Training feature store not found at "
            f"``{_FEATURE_STORE_PATH}``. "
            "Run an ingest DAG first."
        )
        return

    if predictions is None:
        st.warning(
            "No prediction log found at "
            f"``{_PREDICTIONS_PATH}``. "
            "The model has not received any requests yet."
        )
        st.info(f"Training feature store loaded: **{len(training):,} rows**")
        return

    ks_results = _run_ks_tests(training, predictions)

    _render_metrics(training, predictions, ks_results)
    st.divider()
    _render_distribution_plots(training, predictions)
    st.divider()
    _render_ks_table(ks_results)
    st.divider()
    _render_class_distribution(training, predictions)
    st.divider()
    _render_recent_predictions(predictions)


if __name__ == "__main__":
    main()
