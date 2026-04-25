"""Iris classification model training package.

This package provides training utilities and Airflow-compatible task callables
for the Iris classification pipeline.  It reads the preprocessed feature store
produced by the ``data_ingest`` DAG and trains multiple scikit-learn
classifiers, logging every experiment run to an MLFlow tracking server.

Modules:
    train: Core training logic (feature loading, train/test splitting, model
        training, and MLFlow run logging).
    tasks: Airflow PythonOperator-compatible callable functions for use in
        the ``model_training`` DAG.
"""
