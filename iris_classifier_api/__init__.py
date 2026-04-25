"""Iris Classifier REST API package.

Exposes a FastAPI application that serves the champion Iris classification
model registered in MLflow.  The model is loaded once during application
startup and kept in memory for the lifetime of the process.

Endpoints:
    GET  /info          – Current model name and MLflow run metadata.
    POST /update_model  – Hot-reload the in-memory model from the registry.
    POST /predict       – Classify a single Iris feature vector.
"""
