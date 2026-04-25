# Project Notes — Iris ML Platform

## Assumptions

- The `/public` bind-mount acts as a lightweight shared filesystem. This is fine for local development, but a production deployment would use object storage or a managed feature store.
- The local stack uses Airflow with CeleryExecutor and a dedicated `airflow-worker`; the `flower` service is optional and only starts with its profile.
- `complete_ml_pipeline` is the bootstrap path and expects Kaggle access. `generate_synthetic_data` is a post-bootstrap traffic simulator because it POSTs to `/predict`.
- The dataset is small enough that a CSV feature store is sufficient. No Parquet/Delta dependency is assumed.
- Drift is detected at the feature distribution level only (covariate drift). Label feedback is not available, so concept drift cannot be measured yet.
- The API can legitimately return HTTP 503 before the first training run because no champion model exists yet.
- The `@champion` alias is updated only when the new model strictly improves `f1_macro` over the current champion, preventing regressions from scheduled retraining runs triggered by drift.

---

## What would I do with more time, and how could the system be scaled as the project grows?

### 1. Proper Feature Store

Replace the CSV-based feature store with a dedicated tool such as **Feast**. A real feature store provides:

- Point-in-time correct feature retrieval (prevents data leakage between training and serving).
- Feature versioning and lineage tracking.
- Online/offline store separation — the online store serves low-latency features to the API; the offline store feeds batch training jobs.
- Feature sharing across multiple models and teams.

### 2. Data Versioning with DVC

Integrate **DVC (Data Version Control)** to version the datasets and feature store alongside the model artifacts. Each training run would reference an explicit DVC data version, making it possible to:

- Reproduce any past training run exactly (same data + same code + same hyperparameters).
- Audit which data was used to produce a specific model version registered in MLflow.
- Roll back to a previous feature store snapshot if a bad data batch corrupts the store.

DVC remote storage would be backed by S3 in a cloud deployment.

### 3. Distributed Ingestion and Scalable Training

As data volume grows or model complexity increases, the current single-machine approach hits limits:

- **Apache Spark** for data ingestion and feature engineering — distributed data validation, transformation, and deduplication across large amounts of data, replacing the pandas-based pipeline.
- **On-demand training jobs** with AWS SageMaker Training Jobs to launch training on dedicated GPU/CPU instances and shut them down when done, paying only for actual compute time.
- The Airflow DAGs would remain as the orchestration layer, calling Spark jobs and cloud training APIs rather than running computation directly in the worker containers.

### 4. Concept Drift Monitoring

The current system detects **data drift** (input feature distribution shifts). A complete monitoring solution would also track **concept drift**.

### 5. Better Data Drift Visualization

For drift analysis, I would complement or replace the custom Streamlit charts with a dedicated tool such as **Evidently AI**. It is better suited for Data Drift because it can generate richer batch comparison reports, feature distribution visualizations, and shareable HTML dashboards for both technical and non-technical reviewers.
