"""Reusable Palmer Penguins modeling helpers for the demo and notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_portfolio.data_sources import assert_min_bytes, assert_sha256, download_bytes_with_retry


DATA_URL = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
    "8957207b78d6ccd1b4654a9dd9c9041b657478ab/inst/extdata/penguins.csv"
)
DATA_SHA256 = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"
TARGET_COLUMN = "species"
NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
CATEGORICAL_FEATURES = ["island", "sex", "year"]
FEATURE_COLUMNS = ["island", *NUMERIC_FEATURES, "sex", "year"]
RANDOM_STATE = 42


@dataclass(frozen=True)
class PenguinSample:
    """One demo input sample."""

    island: str
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: str
    year: str

    def to_frame(self) -> pd.DataFrame:
        """Return the sample as a one-row modeling frame."""

        return pd.DataFrame(
            [
                {
                    "island": self.island,
                    "bill_length_mm": self.bill_length_mm,
                    "bill_depth_mm": self.bill_depth_mm,
                    "flipper_length_mm": self.flipper_length_mm,
                    "body_mass_g": self.body_mass_g,
                    "sex": self.sex,
                    "year": str(self.year),
                }
            ],
            columns=FEATURE_COLUMNS,
        )


@dataclass(frozen=True)
class PenguinPrediction:
    """Serializable prediction result for the demo."""

    predicted_species: str
    probabilities: dict[str, float]


def default_penguin_sample() -> PenguinSample:
    """Return a plausible example used by the CLI and Streamlit app."""

    return PenguinSample(
        island="Dream",
        bill_length_mm=45.2,
        bill_depth_mm=16.4,
        flipper_length_mm=196.0,
        body_mass_g=4150.0,
        sex="female",
        year="2008",
    )


def load_penguin_data(url: str = DATA_URL, *, min_bytes: int = 10_000) -> pd.DataFrame:
    """Load the pinned public Palmer Penguins CSV and verify its payload hash."""

    content = download_bytes_with_retry(url, attempts=3, timeout=30)
    assert_min_bytes(content, min_bytes=min_bytes, label="Palmer Penguins CSV")

    if url == DATA_URL:
        assert_sha256(
            content,
            expected_sha256=DATA_SHA256,
            label="Palmer Penguins CSV",
        )

    return pd.read_csv(BytesIO(content))


def validate_penguin_columns(data: pd.DataFrame) -> None:
    """Raise a clear error if required Palmer Penguins columns are missing."""

    required_columns = [TARGET_COLUMN, *FEATURE_COLUMNS]
    missing = sorted(set(required_columns) - set(data.columns))
    if missing:
        raise ValueError(f"Penguins data is missing required columns: {', '.join(missing)}")


def prepare_penguin_modeling_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return clean feature and target frames for the demo model."""

    validate_penguin_columns(data)
    modeling_data = data.dropna(subset=[TARGET_COLUMN, *FEATURE_COLUMNS]).copy()
    modeling_data["year"] = modeling_data["year"].astype(str)
    return modeling_data[FEATURE_COLUMNS], modeling_data[TARGET_COLUMN]


def build_penguin_pipeline() -> Pipeline:
    """Build the fold-safe preprocessing and logistic-regression pipeline."""

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def fit_penguin_model(data: pd.DataFrame | None = None) -> Pipeline:
    """Fit the demo model from public data or a supplied frame."""

    raw_data = load_penguin_data() if data is None else data
    X, y = prepare_penguin_modeling_data(raw_data)
    model = build_penguin_pipeline()
    model.fit(X, y)
    return model


def predict_penguin_sample(model: Pipeline, sample: PenguinSample) -> PenguinPrediction:
    """Predict one sample and return class probabilities."""

    sample_frame = sample.to_frame()
    predicted_species = str(model.predict(sample_frame)[0])
    probabilities = model.predict_proba(sample_frame)[0]
    probability_by_species = {
        str(species): float(probability)
        for species, probability in zip(model.classes_, probabilities, strict=True)
    }
    return PenguinPrediction(
        predicted_species=predicted_species,
        probabilities=probability_by_species,
    )


def explain_penguin_prediction(
    model: Pipeline,
    sample: PenguinSample,
    *,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return top feature contributions to the predicted species logit.

    These are model-internal logit contributions from a fitted linear classifier,
    not causal explanations about penguin biology or collection conditions.
    """

    sample_frame = sample.to_frame()
    prediction = str(model.predict(sample_frame)[0])
    class_index = list(model.classes_).index(prediction)

    preprocess = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    transformed_sample = preprocess.transform(sample_frame)[0]
    feature_names = preprocess.get_feature_names_out()
    coefficients = classifier.coef_[class_index]

    contributions = transformed_sample * coefficients
    rows = []
    for feature, contribution in zip(feature_names, contributions, strict=True):
        cleaned_feature = (
            str(feature)
            .replace("numeric__", "")
            .replace("categorical__", "")
            .replace("_", " ")
        )
        rows.append(
            {
                "feature": cleaned_feature,
                "contribution": float(contribution),
                "abs_contribution": float(abs(contribution)),
                "direction": "supports" if contribution >= 0 else "pulls against",
            }
        )

    return sorted(rows, key=lambda row: row["abs_contribution"], reverse=True)[:top_n]


def feature_ranges(data: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Return min/median/max values for numeric demo controls."""

    X, _ = prepare_penguin_modeling_data(data)
    ranges = {}
    for column in NUMERIC_FEATURES:
        ranges[column] = (
            float(X[column].min()),
            float(X[column].median()),
            float(X[column].max()),
        )
    return ranges


def categorical_options(data: pd.DataFrame) -> dict[str, list[str]]:
    """Return sorted categorical options for the local interactive demo."""

    X, _ = prepare_penguin_modeling_data(data)
    return {
        column: sorted(str(value) for value in X[column].dropna().unique())
        for column in CATEGORICAL_FEATURES
    }
