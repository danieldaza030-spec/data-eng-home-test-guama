# Iris ML Platform

End-to-end MLOps solution for Iris multi-class classification, orchestrated with Apache Airflow, tracked with MLflow, and served via a FastAPI inference API — all running in Docker Compose.

---

## Tech Stack & Component Roles

| Technology | Role in the project |
|---|---|
| **Apache Airflow 3** | Orchestration engine for the five DAGs in the repo: `data_ingest`, `complete_ml_pipeline`, `training_pipeline`, `generate_synthetic_data`, and `monitor_prediction_drift`. Handles scheduling, retries, and manual. |
| **MLflow** | Experiment tracking and model registry. Logs metrics, parameters, and artifacts for every training. |
| **FastAPI** | Inference server. Exposes `POST /predict`, `GET /info`, and `POST /update_model`. It also appends every prediction to `predicted.csv`. |
| **Streamlit** | Drift monitoring dashboard. Reads the training slice from the feature store using the `training_dates` tag when available, compares it against live prediction traffic, and offers a manual retraining trigger. I Used Streamlit because it is easy to run locally and lets us build a simple interface quickly. |

---

## Local Deployment

### Prerequisites

- Docker Desktop running
- Docker Compose v2+

### Start all services

```bash
docker compose up --build -d
```

### Service URLs

| Service | URL | Notes |
|---|---|---|
| Airflow UI / REST API | http://localhost:8080 | Login: `airflow` / `airflow123` |
| MLflow UI | http://localhost:5001 | Experiment tracking and model registry |
| FastAPI docs | http://localhost:8000/docs | Inference API documentation |
| Drift Monitor | http://localhost:8501 | Streamlit drift dashboard |

---

## DAGs Reference

### `data_ingest`
**Trigger:** Manual only.

Handles the full ingestion cycle for the Iris dataset without any training step. Supports two data sources controlled by the `data_source` param:
- `kaggle` (default) — downloads the dataset directly from the Kaggle API, extracts `Iris.csv`, and processes it.
- `synthetic` — reads `/public/source_data/simulated.csv` produced by `generate_synthetic_data`.

The `on_validation_error` param controls how Pandera schema violations are handled (`fail` stops the pipeline; `drop_invalid_rows` sanitises the dataset and continues). The `write_mode` param (`overwrite` or `append`) controls how the processed records are written to `features_iris.csv`.

Task graph: `download_iris_data → validate_iris_data → transform_iris_data`

---

### `complete_ml_pipeline`
**Trigger:** Manual only.

Bootstrap DAG for a full end-to-end run: ingests data from Kaggle, validates the resulting feature store, and then trains SVM, Logistic Regression, and KNN classifiers in parallel via `GridSearchCV`. Each training run is logged to MLflow, the best model per family is registered, the `@champion` alias is assigned, and the inference API is hot-reloaded via `POST /update_model`.

This is the entry point to use when starting from scratch with no existing feature store or champion model.

Task graph: `download_iris_data → validate_iris_data → transform_iris_data → validate_feature_store → [train_svm ‖ train_logistic_regression ‖ train_knn] → register_models → post_update_model → get_model_info`

---

### `training_pipeline`
**Trigger:** Manual or automatic (triggered by `monitor_prediction_drift` when drift is detected).

Train-only DAG that assumes a valid feature store already exists. Validates the feature store, trains the three classifiers in parallel, registers the best model per family in MLflow, updates the `@champion` alias only if the new model strictly improves `f1_macro` over the current champion, and notifies the inference API to hot-reload.

The `training_data` param (`all` or `latest`) controls whether the full feature store or only the most recent ingestion batch is used for training.

Task graph: `validate_feature_store → [train_svm ‖ train_logistic_regression ‖ train_knn] → register_models → post_update_model → get_model_info`

---

### `generate_synthetic_data`
**Trigger:** Manual only.

Generates a synthetic Iris dataset and populates the prediction store for drift-monitoring tests. Has two responsibilities:
1. **Generate** — produces `/public/source_data/simulated.csv` using UCI Gaussian parameters.
2. **Simulate** — sends all valid rows to `POST /predict` on the inference API, which logs each prediction to `predicted.csv`.

Three modes are available via the `mode` param:
- `normal` — balanced classes, standard distribution → predictions follow training distribution → no drift triggered.
- `drifted` — `PetalLengthCm` shifted +2.5 cm, `PetalWidthCm` +1.0 cm, class proportions 10/20/70 % → KS p-values << 0.05 → drift detected.
- `dirty` — balanced proportions with injected nulls, negative values, invalid species labels, and duplicates. Only valid rows are forwarded to the API.

Task graph: `generate_and_save → send_predictions_to_api`

---

### `monitor_prediction_drift`
**Trigger:** Daily (`@daily`).

Implements data drift monitoring: compares the feature distributions of real production requests (logged in `predicted.csv`) against the exact training distribution of the current `@champion` model. The reference distribution is built by filtering `features_iris.csv` to the `processed_at` dates stored in the `training_dates` tag on the champion model version in MLflow, preventing false positives from accumulated batches that were not part of the last training run.

A two-sample KS test is run per feature column. If any feature has `p < 0.05`, drift is flagged and `training_pipeline` is triggered automatically. The test is skipped gracefully when either CSV is missing or contains fewer than 30 rows.

Task graph: `detect_prediction_drift → should_retrain (ShortCircuitOperator) → trigger_training_pipeline (TriggerDagRunOperator)`

---

## Part 1 — Full Pipeline (First Run)

### Step 1 — Trigger `complete_ml_pipeline`

Open the Airflow UI at http://localhost:8080, find the **`complete_ml_pipeline`** DAG, and trigger it with the default values unless you want to tune the training run.

The DAG exposes the training split, random state, model-grid settings, `write_mode`, and `training_data`.


---

### Step 2 — Watch the bootstrap flow

The DAG downloads the Iris dataset from Kaggle, validates it, writes `public/feature_storage/features_iris.csv`, validates the feature store, trains SVM / Logistic Regression / KNN in parallel, registers the best model from each family in MLflow, and reloads the champion in the API.


---

### Step 3 — Verify the champion model in MLflow

Open http://localhost:5001 and navigate to:  
**Models → iris_classifier → champion alias**

You should see one version per model family and a `@champion` alias pointing to the best-performing version.


---

### Step 4 — Check the API is serving the champion

```bash
curl http://localhost:8000/info
```

Expected response:

```json
{
  "model_name": "iris_classifier",
  "model_version": "...",
  "run_id": "...",
  "metrics": {},
  "params": {},
  "tags": {}
}
```

`GET /info` returns the current model metadata. A `503` response before the first training run is expected and healthy.


---

### Step 5 — Run a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length_cm": 5.1, "sepal_width_cm": 3.5, "petal_length_cm": 1.4, "petal_width_cm": 0.2}'
```

Expected response:

```json
{
  "prediction": 0,
  "species": "Iris-setosa"
}
```

This request also appends a row to `public/prediction_storage/predicted.csv`.


---

## Data Drift Simulation — Design Rationale

The `generate_synthetic_data` DAG with `mode=drifted` simulates the kind of distributional shift that could realistically occur in production. It is meant to run after the API already has a champion model, because it posts the generated rows to `/predict`:

- **New measurement regions:** If Iris flowers are sampled from a different geographic region, petal dimensions may follow a different distribution than the UCI training set.
- **Seasonal or environmental changes:** Flowers growing under different temperature or humidity conditions can develop larger or smaller petals over time, gradually shifting the input distribution.

The simulation shifts `PetalLengthCm` by +2.5 cm and `PetalWidthCm` by +1.0 cm, and skews the class proportions, which is sufficient to push the KS test p-values well below 0.05 and trigger automatic retraining — demonstrating the full drift detection → retraining loop without requiring real labeled data from a new source.

---

## Part 2 — Emulating Input Drift

This section assumes the API is already serving a champion model. `generate_synthetic_data` posts to `/predict`, so it only works after the bootstrap flow has completed.

### Step 6 — Generate drifted predictions

Trigger the **`generate_synthetic_data`** DAG with `mode=drifted`.

The defaults are `n_samples=1200` and `random_state=42`. The DAG writes `public/source_data/simulated.csv` and sends all valid rows to `POST /predict`, filling `public/prediction_storage/predicted.csv`.

In `mode=drifted`, `PetalLengthCm` is shifted by +2.5 cm, `PetalWidthCm` is shifted by +1.0 cm, and the class mix is skewed so the KS test can flag drift.

<!-- SCREENSHOT: Airflow UI — generate_synthetic_data DAG with mode=drifted -->

---

### Step 7 — Open the Drift Monitor

Open http://localhost:8501.

The dashboard compares the feature store against the prediction log, filters the training slice by the champion's `training_dates` tag when available, and shows distribution plots plus the KS table.

<!-- SCREENSHOT: Drift Monitor — histograms showing distribution shift -->

<!-- SCREENSHOT: Drift Monitor — KS test table with p-values < 0.05 -->

---

### Step 8 — Trigger retraining

`monitor_prediction_drift` runs daily. When any feature has `p < 0.05`, it triggers `training_pipeline` automatically.

You can also use the manual Trigger Retraining button in the Airflow Interface.

<!-- SCREENSHOT: Drift Monitor — Trigger Retraining button -->

---

### Step 9 — Verify the new champion

After `training_pipeline` completes, go back to http://localhost:5001 and check that the `@champion` alias has moved to a new model version trained on the updated feature store.

<!-- SCREENSHOT: MLflow — new @champion version after retraining -->

---

## Use Case

<div align="center">
  <img src="mlops-Página-1.drawio.png" alt="End-to-end MLOps architecture overview" width="750" />
  <br/><sub><em>End-to-end MLOps architecture — services, DAGs, and data flows</em></sub>
</div>

### 1. Run the first flow

Open the Airflow UI at http://localhost:8080 and launch `complete_ml_pipeline` to walk through the end-to-end flow step by step.

<div align="center">
  <img src="docs/images/airflow_complete_pipeline.png" alt="Airflow — complete_ml_pipeline graph view" width="750" />
  <br/><sub><em>Airflow graph view — <code>complete_ml_pipeline</code> running with parallel SVM / LR / KNN training tasks</em></sub>
</div>


### 2. Review experiments in MLflow

Open the MLflow UI at http://localhost:5001 and inspect the `iris_classification` experiment.

In the experiment view you can:  
- Compare runs by metrics such as accuracy, precision, recall, and F1.

<div align="center">
  <img src="docs/images/mlflow_run_metrics.png" alt="MLflow run comparison — accuracy, precision, recall and F1" width="750" />
  <br/><sub><em>MLflow run comparison — accuracy, precision, recall and F1 for all three model families</em></sub>
</div>

- Open the artifact tab to inspect the confusion matrix image and any saved outputs.

<div align="center">
  <img src="docs/images/mlflow_confusion_matrix.png" alt="MLflow artifact tab — confusion matrix PNG" width="750" />
  <br/><sub><em>Confusion matrix logged as an artifact for each training run</em></sub>
</div>

- Check the registered model versions and verify which one is assigned to `@champion`.


### 3. Simulate data drift

Use `generate_synthetic_data` with `mode=drifted` to create shifted input data and send prediction traffic to the API at http://localhost:8000/predict.


### 4. Re-ingest the adapted batch and reactivate training

Run `data_ingest` with `data_source=synthetic` so the new batch is read from `simulated.csv` and written back to the feature store at `http://localhost:8080`.

Make sure `monitor_prediction_drift` is enabled and that `training_pipeline` can be triggered, so the training loop is reactivated when drift is detected.


### 5. See the new model upload

Open the MLflow UI at http://localhost:5001 and verify that the new model version was registered and the `@champion` alias moved to it.

### 6. View the drift results

Open the Streamlit drift monitor at http://localhost:8501 to inspect the feature distributions, KS test results, and retraining controls.

<div align="center">
  <img src="docs/images/streamlit_feature_distributions.png" alt="Streamlit — training vs production feature distributions" width="750" />
  <br/><sub><em>Streamlit drift monitor — training vs production distributions show clear petal feature shift after drifted data</em></sub>
</div>

<div align="center">
  <img src="docs/images/streamlit_ks_drift_results.png" alt="Streamlit — KS drift test results table" width="750" />
  <br/><sub><em>KS drift test results — all four features flagged as drifted (p-value = 0.000)</em></sub>
</div>

### 7. Handle corrupt data

Trigger `generate_synthetic_data` with `mode=dirty` to create a batch with nulls, negative values, invalid species labels, and duplicates.

Then run `data_ingest` from http://localhost:8080 with `data_source=synthetic` to read the dirty batch from `simulated.csv`.

If `on_validation_error=fail`, the pipeline stops at `validate_iris_data` and the validation errors are shown in the Airflow logs.

If `on_validation_error=drop_invalid_rows`, the invalid rows are removed, a cleaned file is pushed through XCom, and `transform_iris_data` writes only valid records to the feature store.

---

## Stopping Services

```bash
docker compose down
```

To also remove volumes (resets all data):

```bash
docker compose down -v
```
