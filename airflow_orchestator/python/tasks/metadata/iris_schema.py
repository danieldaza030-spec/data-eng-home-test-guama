import pandera.pandas as pa
from pandera.typing import Series
import pandas as pd


IRIS_FEATURE_COLS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]
IRIS_TARGET_COL = "Species"


class IrisSchema(pa.DataFrameModel):
    """Validation schema for the Iris dataset."""

    Id: Series[int] = pa.Field(
        gt=0,
        description="Unique record ID.",
        nullable=False
    )

    SepalLengthCm: Series[float] = pa.Field(
        gt=0,
        description="Sepal length in centimetres.",
        nullable=False
    )

    SepalWidthCm: Series[float] = pa.Field(
        gt=0,
        description="Sepal width in centimetres.",
        nullable=False
    )

    PetalLengthCm: Series[float] = pa.Field(
        gt=0,
        description="Petal length in centimetres.",
        nullable=False
    )

    PetalWidthCm: Series[float] = pa.Field(
        gt=0,
        description="Petal width in centimetres.",
        nullable=False
    )

    Species: Series[str] = pa.Field(
        isin=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
        description="Iris species.",
        nullable=False
    )

    class Config:
        strict = True
        coerce = True
        ordered = False

    @pa.dataframe_check
    def check_no_duplicates(cls, df: pd.DataFrame) -> bool:
        """Checks that there are no duplicate rows."""
        return ~df.duplicated().any()

    @pa.dataframe_check
    def check_unique_ids(cls, df: pd.DataFrame) -> bool:
        """Checks that all IDs are unique."""
        return df['Id'].is_unique


class IrisTransformedSchema(pa.DataFrameModel):
    """Validation schema for the transformed Iris feature store.

    Describes the output produced by
    :func:`~python.data_ingest.transform_data.prepare_universal_features` and
    persisted by
    :func:`~python.data_ingest.transform_data.save_features_to_parquet`.

    The four numeric feature columns retain their **raw centimetre values**
    from the original Iris CSV.  No scaling is applied at this stage;
    :class:`~sklearn.preprocessing.StandardScaler` is fitted inside the
    training sklearn ``Pipeline`` so the scaler is persisted alongside the
    model artefact and applied automatically at inference time without data
    leakage.
    The ``Species`` column is label-encoded as an integer
    (0 = Iris-setosa, 1 = Iris-versicolor, 2 = Iris-virginica).
    The ``processed_at`` column records the UTC timestamp of the
    transformation run that produced each row.
    """

    SepalLengthCm: Series[float] = pa.Field(
        gt=0,
        description="Raw sepal length in centimetres (unscaled).",
        nullable=False,
    )

    SepalWidthCm: Series[float] = pa.Field(
        gt=0,
        description="Raw sepal width in centimetres (unscaled).",
        nullable=False,
    )

    PetalLengthCm: Series[float] = pa.Field(
        gt=0,
        description="Raw petal length in centimetres (unscaled).",
        nullable=False,
    )

    PetalWidthCm: Series[float] = pa.Field(
        gt=0,
        description="Raw petal width in centimetres (unscaled).",
        nullable=False,
    )

    Species: Series[int] = pa.Field(
        ge=0,
        le=2,
        description=(
            "Label-encoded species. "
            "0 = Iris-setosa, 1 = Iris-versicolor, 2 = Iris-virginica."
        ),
        nullable=False,
    )

    processed_at: Series[pa.DateTime] = pa.Field(
        description=(
            "UTC timestamp of the transformation run that produced this record. "
            "All rows in a single pipeline execution share the same value."
        ),
        nullable=False,
    )

    class Config:
        strict = True
        coerce = True
        ordered = False
