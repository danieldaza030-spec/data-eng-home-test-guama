# Iris ML Platform — Agent Instructions

Todo el código debe seguir estándares PEP8 y buenas prácticas, la documentación de las funciones la vas a realizar con la documentación de tipo docstring de google tanto de funciones como de modulos. usa Airflow 3.x. Tanto código como funciones y variables los vas a redactar en inglés.

## Project Goal
End-to-end MLOps solution for Iris multi-class classification (setosa / versicolor / virginica).
Orchestrated with Apache Airflow, tracked with MLflow, served via a FastAPI inference API — all running in Docker Compose.

---

## Architecture at a Glance

- `data_ingest` DAG: descarga desde Kaggle o lee `simulated.csv`, valida el CSV y transforma los datos hasta `features_iris.csv`.
- `complete_ml_pipeline` DAG: flujo E2E completo para bootstrap (ingesta + validación de feature store + entrenamiento + registro + despliegue).
- `generate_synthetic_data` DAG: genera `simulated.csv` en modo `normal`, `drifted` o `dirty` y, cuando la API ya tiene un modelo cargado, envía filas válidas a `/predict`.
- `monitor_prediction_drift` DAG: monitor diario KS entre `predicted.csv` y la feature store filtrada por `training_dates`; si detecta drift, dispara `training_pipeline`.

## Services (docker-compose.yaml)

| Service | Port | Role |
|---|---|---|
| `airflow-apiserver` | 8080 | Airflow Web UI + REST API |
| `mlflow` | 5001→5000 | Experiment tracking + Model Registry |
| `iris-api` | 8000 | FastAPI inference server |
| `drift-monitor` | 8501 | Streamlit production drift dashboard |
| `postgres` | — | Airflow metadata DB |
| `mlflow-postgres` | — | MLflow backend DB |
| `redis` | 6379 | Celery broker |
| `flower` | 5555 | Celery monitoring (profile: `flower`) |

Airflow workers have `MLFLOW_TRACKING_URI=http://mlflow:5000` and `AIRFLOW_CONN_IRIS_API_DEFAULT=http://iris-api:8000` injected.
The `drift-monitor` service also has `MLFLOW_TRACKING_URI=http://mlflow:5000` injected (reads champion training dates).
The `/public` bind-mount is shared between all Airflow workers and the API.

---

## Key Directories and Files

```
airflow_orchestator/
  dags/
    data_ingest.py              # DAG: download → validate → transform (Kaggle or synthetic, no training)
    complete_pipeline.py        # DAG: full E2E ingest + train/register/deploy
    training_pipeline.py        # DAG: train-only (triggered by drift or manually)
    generate_synthetic_data.py  # DAG: generate /public/source_data/simulated.csv + send to API
    monitor_prediction_drift.py # DAG: Tipo B KS drift monitor — champion training slice vs predictions → retrain
  python/
    tasks/
      data_ingest/
        extract_data.py         # Kaggle download + zip extraction
        validate.py             # Pandera schema validation + descriptive stats logging
        transform_data.py       # clean → encode → save_features_to_csv (overwrite|append + dedup)
        tasks.py                # Airflow callables: download / validate / transform
      data_generation/
        generate.py             # generate_synthetic_iris(n, mode, seed) → DataFrame
      drift_detection/
        ks_test.py              # run_ks_drift_test() → DriftReport (scipy KS two-sample)
      metadata/
        constants.py            # All shared constants (paths, URIs, model names, drift config)
        iris_schema.py          # Pandera schemas for raw CSV and feature store
        pipeline_responses.py   # Pydantic response models (IrisValidationResult, PreparedFeatureSet…)
    model_training/
      tasks.py                  # Airflow callables: validate_feature_store / train_* / register_models
      train.py                  # load_feature_store / split / train_and_log_grid / register_best_model

iris_classifier_api/
  main.py                   # FastAPI app factory + lifespan (loads @champion)
  endpoints/
    predict.py              # POST /predict — logs features + prediction to /public/prediction_storage/predicted.csv
    info.py                 # GET /info
    update_model.py         # POST /update_model (hot-reload)
  metadata/
    state.py                # ModelState singleton — loads models:/iris_classifier@champion
    models.py               # Pydantic schemas: PredictRequest, ModelInfo
    constants.py            # FEATURE_COLUMNS, CLASS_LABELS

drift_monitor/
  app.py                    # Streamlit dashboard: distribution plots + KS table + retraining trigger
                            #   reads champion's training_dates tag from MLflow to filter feature store
  requirements.txt          # streamlit, pandas, matplotlib, scipy, requests, mlflow-skinny

deployment/
  Dockerfile.airflow        # Custom Airflow 3.2.1 image with project deps pre-installed at build time
  Dockerfile.api            # Python 3.11 + FastAPI image
  Dockerfile.mlflow         # MLflow tracking server image
  Dockerfile.monitor        # Python 3.11 + Streamlit drift dashboard image

notebooks/
  data_ingest.ipynb         # Exploratory notebook (not the main demo)
  datasets/Iris.csv         # Local copy of raw dataset
```

---

## Constants (source of truth: airflow_orchestator/python/tasks/metadata/constants.py)

| Constant | Class | Value |
|---|---|---|
| Feature store dir | `IrisConstants` | `/public/feature_storage/` |
| Feature store file | `IrisConstants` | `features_iris.csv` |
| Source data dir | `IrisConstants` | `/public/source_data/` |
| Simulated CSV | `IrisConstants` | `simulated.csv` |
| Prediction store dir | `IrisConstants` | `/public/prediction_storage/` |
| Predictions file | `IrisConstants` | `predicted.csv` |
| MLflow experiment | `MLFlowConstants` | `iris_classification` |
| Registered model | `MLFlowConstants` | `iris_classifier` |
| Champion alias | `MLFlowConstants` | `champion` |
| MLflow URI (internal) | `MLFlowConstants` | `http://mlflow:5000` |
| Kaggle dataset | `IrisConstants` | `uciml/iris` |
| Feature columns | `IrisSchema` | `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm` |
| Label column | `IrisSchema` | `Species` |
| KS alpha | `DriftConstants` | `0.05` |
| Training DAG ID | `DriftConstants` | `training_pipeline` |

---

## DAGs Summary

| DAG ID | Trigger | Purpose |
|---|---|---|
| `data_ingest` | Manual | Ingest (Kaggle or synthetic): download → validate → transform |
| `complete_ml_pipeline` | Manual | Full E2E: ingest + train + register + deploy (Kaggle bootstrap) |
| `training_pipeline` | Manual or drift-triggered | Train only: validate feature store → [SVM‖LR‖KNN] → register → deploy |
| `generate_synthetic_data` | Manual | Generate `/public/source_data/simulated.csv` + send rows to `/predict` |
| `monitor_prediction_drift` | `@daily` | Tipo B KS drift: feature store vs prediction log → triggers `training_pipeline` if p < 0.05 |

### Key DAG Params

| DAG | Params |
|---|---|
| `data_ingest` | `data_source`, `on_validation_error`, `write_mode` |
| `complete_ml_pipeline` | `test_size`, `random_state`, `svm_param_grid`, `svm_fixed_params`, `lr_param_grid`, `lr_fixed_params`, `knn_param_grid`, `knn_fixed_params`, `write_mode`, `training_data` |
| `training_pipeline` | `test_size`, `random_state`, `svm_param_grid`, `svm_fixed_params`, `lr_param_grid`, `lr_fixed_params`, `knn_param_grid`, `knn_fixed_params`, `training_data` |
| `generate_synthetic_data` | `mode`, `n_samples`, `random_state` |

Los valores por defecto viven en cada DAG. Si necesitas ajustar un flujo, toma la definición del propio archivo antes de cambiar la documentación.

### Drift Detection Flow (monitor_prediction_drift — Tipo B)
- Runs **daily**. Compares the feature distributions of **real production requests** (logged by the API to `predicted.csv`) against the **exact training distribution** of the current `@champion` model.
- The reference distribution is built by filtering `features_iris.csv` to the `processed_at` dates stored in the `training_dates` tag on the champion model version in MLflow. Falls back to the full feature store when the tag is absent (backward compatibility).
- Uses the same two-sample KS test (`run_ks_drift_test`) — if `p < 0.05` for any feature column, triggers `training_pipeline`.
- Skips gracefully when either CSV is missing or has fewer than 30 rows (insufficient data for a reliable test).
- `ShortCircuitOperator` gates the `TriggerDagRunOperator`; if no drift, the trigger is skipped.
- The Streamlit drift monitor applies the same `training_dates` filter so its histograms and KS table always reflect the champion's actual training distribution.

### Validation Error Strategy (on_validation_error)
- **`fail`** (default): raises `AirflowException` on first Pandera error, halts the pipeline.
- **`drop_invalid_rows`**: logs all Pandera errors in detail, sanitises the dataset (dtype coercion → index-based drops → NaN drop → species filter), writes a cleaned CSV to a temp file pushed to XCom as `cleaned_temp_file`. Raises if no valid rows remain.

### Feature Store (features_iris.csv)
- `write_mode=overwrite`: replaces the CSV entirely with the current batch.
- `write_mode=append`: merges with existing CSV and deduplicates on feature + target columns (keeping first occurrence, ignoring `processed_at` timestamp).
- `training_data=all`: loads every row in the feature store for training.
- `training_data=latest`: filters to rows whose `processed_at` date matches the most recent batch.
- Implicit versioning via `processed_at` UTC timestamp; no explicit batch_id.

### Prediction Logging (predicted.csv)
- Every `POST /predict` call appends a row to `/public/prediction_storage/predicted.csv`.
- Columns logged: `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `predicted_class` (int), `predicted_species` (str), `prediction_at` (ISO UTC timestamp).
- Thread-safe via a module-level `threading.Lock` — safe with uvicorn's single-worker default.
- Prediction logging is wrapped in try/except and never blocks the inference response.

### Synthetic Data Modes
- `normal`: Balanced 1/3 per class, standard UCI Gaussian parameters — API predictions show no drift.
- `drifted`: PetalLengthCm +2.5 cm, PetalWidthCm +1.0 cm, class proportions 10/20/70 % — API predictions trigger KS drift (p << 0.05).
- `dirty`: Balanced proportions + injected errors: 12 % nulls, 8 % negative values, 8 % invalid `Species` labels, 10 % duplicate rows. Only valid rows are forwarded to the API. Designed to test `drop_invalid_rows` in the ingest DAGs.

### End-to-end Drift Simulation Workflow
1. Run `complete_ml_pipeline` with real data to build the feature store and deploy a champion model. If the feature store already exists, `training_pipeline` can be run independently.
2. Run `generate_synthetic_data` with `mode="drifted"` — this generates the CSV **and** sends ~1200 drifted predictions to the API.
3. Run (or wait for) `monitor_prediction_drift` — it detects the petal feature shift and triggers `training_pipeline`.

There is no standalone replay DAG in the current codebase; `generate_synthetic_data` is the supported way to create synthetic production traffic for drift testing.

---

## ML Approach

- **Models:** SVM (`SVC`), Logistic Regression, KNN — trained in parallel
- **Tuning:** `GridSearchCV` with configurable grids via Airflow DAG Params
- **Preprocessing:** `StandardScaler` inside a `sklearn.Pipeline` (fitted on train split only)
- **Metrics logged:** accuracy, precision, recall, F1, confusion matrix PNG
- **Best model per family** is registered in MLflow as `@champion` and hot-reloaded by the API.
- Every training run **always** registers a new model version. The `@champion` alias is only moved to the new version when its `f1_macro` strictly exceeds the current champion's metric.
- Each model version carries a `training_dates` tag (comma-separated sorted `YYYY-MM-DD` strings) identifying the exact `processed_at` dates present in the training set.

---

## Known Gotchas

- `shutil.copy2` fails on `/public` bind-mount (PermissionError on `copystat`). Use `Path.write_bytes(Path.read_bytes())` for file copies on that volume.
- `airflow.models.param.Param` is deprecated in Airflow 3.x — import from `airflow.sdk` instead.
- `ShortCircuitOperator` skips all downstream tasks (including `TriggerDagRunOperator`) when it returns `False`. This is correct behaviour — it means no drift was detected.
- When `on_validation_error=drop_invalid_rows`, `transform_iris_data` must pull `cleaned_temp_file` from `validate_iris_data` XCom — if it falls back to `temp_file` the dirty rows will re-enter the feature store.
- `monitor_prediction_drift` requires at least 30 rows in **both** CSVs to run the KS test; it raises `AirflowSkipException` otherwise.
- The feature store is now a CSV (`features_iris.csv`), not a Parquet file — `pyarrow` is not a dependency.
- Do **not** use `_PIP_ADDITIONAL_REQUIREMENTS` to install project dependencies at container startup — this downgrades `packaging` and `cachetools` at runtime and breaks Airflow 3.x core. All dependencies must be baked into `deployment/Dockerfile.airflow` at build time.
- The `iris-api` healthcheck accepts both HTTP 200 (model loaded) and HTTP 503 (API running, no champion model yet). A 503 at startup is expected before the first `training_pipeline` run.

## MLOps Gaps (not yet implemented)

| Gap | Impact | Priority |
|---|---|---|
| Concept drift monitoring (label feedback) | Cannot detect that predictions are wrong without ground-truth labels | Medium |

---

## Run Commands

```bash
# Build and start all services (first time or after Dockerfile changes)
docker compose up --build -d

# Trigger the full pipeline (Airflow UI at http://localhost:8080)
# DAG: complete_ml_pipeline

# Run inference
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length_cm": 5.1, "sepal_width_cm": 3.5, "petal_length_cm": 1.4, "petal_width_cm": 0.2}'

# Check serving model
curl http://localhost:8000/info
```

---

## Conventions

- All shared constants live in `airflow_orchestator/python/tasks/metadata/constants.py` — do not hardcode paths or URIs elsewhere.
- API responses use Pydantic v2 models defined in `iris_classifier_api/metadata/models.py`.
- The API returns HTTP 503 when no model is loaded (before the first pipeline run). This is expected and healthy — do not treat it as a container failure.
- DAG tasks use XCom only for lightweight values (file paths, dataset versions, training date lists). The feature store CSV is passed via the shared `/public` volume.
- Model training is always wrapped in an MLflow `start_run` context; artifacts (confusion matrix PNG) are logged via `mlflow.log_artifact`.
- Every registered model version carries a `training_dates` tag (comma-separated `YYYY-MM-DD` strings). Drift detection components must read this tag to reconstruct the exact training distribution; do not compare against the full feature store when this tag is present.
