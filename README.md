# Iris ML Platform

End-to-end MLOps solution for Iris multi-class classification, orchestrated with Apache Airflow, tracked with MLflow, and served via a FastAPI inference API — all running in Docker Compose.

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

## Part 1 — Full Pipeline (First Run)

### Step 1 — Trigger `complete_ml_pipeline`

Open the Airflow UI at http://localhost:8080, find the **`complete_ml_pipeline`** DAG, and trigger it with the default values unless you want to tune the training run.

The DAG exposes the training split, random state, model-grid settings, `write_mode`, and `training_data`.

<!-- SCREENSHOT: Airflow UI — complete_ml_pipeline trigger -->

---

### Step 2 — Watch the bootstrap flow

The DAG downloads the Iris dataset from Kaggle, validates it, writes `public/feature_storage/features_iris.csv`, validates the feature store, trains SVM / Logistic Regression / KNN in parallel, registers the best model from each family in MLflow, and reloads the champion in the API.

<!-- SCREENSHOT: Airflow task graph or logs for complete_ml_pipeline -->

---

### Step 3 — Verify the champion model in MLflow

Open http://localhost:5001 and navigate to:  
**Models → iris_classifier → champion alias**

You should see one version per model family and a `@champion` alias pointing to the best-performing version.

<!-- SCREENSHOT: MLflow Model Registry showing @champion alias -->

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

<!-- SCREENSHOT: /info response or Swagger UI at localhost:8000/docs -->

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

<!-- SCREENSHOT: POST /predict response -->

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

## Stopping Services

```bash
docker compose down
```

To also remove volumes (resets all data):

```bash
docker compose down -v
```
