"""Feature and label constants for the Iris classifier API.

This module centralises every constant that describes the shape and class
vocabulary of the Iris dataset so that all API layers share a single source
of truth.

Constants:
    FEATURE_COLUMNS: Ordered list of input feature column names expected by
        every trained pipeline.
    CLASS_LABELS: Human-readable species labels indexed by the integer class
        predicted by the model.
"""

FEATURE_COLUMNS: list[str] = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

CLASS_LABELS: list[str] = [
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica",
]
