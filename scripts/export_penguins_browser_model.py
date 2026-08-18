"""Export the fitted Penguins pipeline for deterministic browser inference."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from ml_portfolio.penguins import (
    CATEGORICAL_FEATURES,
    DATA_SHA256,
    DATA_URL,
    NUMERIC_FEATURES,
    PenguinSample,
    default_penguin_sample,
    fit_penguin_model,
    load_penguin_data,
    prepare_penguin_modeling_data,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "site" / "model" / "penguins-logistic-v1.json"
DEFAULT_FIXTURE_PATH = (
    ROOT / "tests" / "browser" / "fixtures" / "penguins_reference_cases.json"
)
DEFAULT_PROVENANCE_PATH = (
    ROOT / "reports" / "evidence" / "penguins_browser_model_provenance.json"
)
SOURCE_COMMIT = "8957207b78d6ccd1b4654a9dd9c9041b657478ab"

NUMERIC_METADATA = {
    "bill_length_mm": ("Bill length", "mm", 0.1),
    "bill_depth_mm": ("Bill depth", "mm", 0.1),
    "flipper_length_mm": ("Flipper length", "mm", 1.0),
    "body_mass_g": ("Body mass", "g", 50.0),
}
CATEGORICAL_METADATA = {
    "island": ("Island", "Collection context"),
    "sex": ("Sex", "Recorded attribute"),
    "year": ("Year", "Collection context"),
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_dict(sample: PenguinSample) -> dict[str, str | float]:
    return {
        "island": sample.island,
        "bill_length_mm": sample.bill_length_mm,
        "bill_depth_mm": sample.bill_depth_mm,
        "flipper_length_mm": sample.flipper_length_mm,
        "body_mass_g": sample.body_mass_g,
        "sex": sample.sex,
        "year": sample.year,
    }


def _sample_from_dict(values: dict[str, str | float]) -> PenguinSample:
    return PenguinSample(
        island=str(values["island"]),
        bill_length_mm=float(values["bill_length_mm"]),
        bill_depth_mm=float(values["bill_depth_mm"]),
        flipper_length_mm=float(values["flipper_length_mm"]),
        body_mass_g=float(values["body_mass_g"]),
        sex=str(values["sex"]),
        year=str(values["year"]),
    )


def _warning_messages(
    values: dict[str, str | float], numeric_inputs: list[dict[str, Any]]
) -> list[str]:
    warnings = []
    for feature in numeric_inputs:
        value = float(values[feature["name"]])
        if value < feature["observed_min"] or value > feature["observed_max"]:
            warnings.append(
                f"{feature['label']} ({feature['unit']}) is outside the observed training "
                f"range of {feature['observed_min']:.1f} to {feature['observed_max']:.1f}."
            )
    return warnings


def build_artifact(data: pd.DataFrame) -> tuple[dict[str, Any], Any, pd.DataFrame]:
    """Fit the reference model and return its browser-safe numeric representation."""

    X, _ = prepare_penguin_modeling_data(data)
    model = fit_penguin_model(data)
    preprocess = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    numeric_pipeline = preprocess.named_transformers_["numeric"]
    categorical_pipeline = preprocess.named_transformers_["categorical"]
    imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]
    one_hot = categorical_pipeline.named_steps["one_hot"]
    defaults = _sample_dict(default_penguin_sample())

    numeric_inputs = []
    for name in NUMERIC_FEATURES:
        label, unit, step = NUMERIC_METADATA[name]
        observed_min = float(X[name].min())
        observed_median = float(X[name].median())
        observed_max = float(X[name].max())
        padding = 0.12 * (observed_max - observed_min)
        numeric_inputs.append(
            {
                "name": name,
                "label": label,
                "unit": unit,
                "step": step,
                "default": float(defaults[name]),
                "observed_min": observed_min,
                "observed_median": observed_median,
                "observed_max": observed_max,
                "control_min": observed_min - padding,
                "control_max": observed_max + padding,
            }
        )

    categorical_inputs = []
    for name, categories in zip(CATEGORICAL_FEATURES, one_hot.categories_, strict=True):
        label, role = CATEGORICAL_METADATA[name]
        categorical_inputs.append(
            {
                "name": name,
                "label": label,
                "role": role,
                "default": str(defaults[name]),
                "options": [str(value) for value in categories],
            }
        )

    artifact = {
        "schema_version": 1,
        "artifact_id": "penguins-logistic-v1",
        "model_family": "multinomial logistic regression",
        "training_rows": int(len(X)),
        "inputs": {
            "numeric": numeric_inputs,
            "categorical": categorical_inputs,
        },
        "preprocessing": {
            "numeric_features": list(NUMERIC_FEATURES),
            "imputer_statistics": [float(value) for value in imputer.statistics_],
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) for value in scaler.scale_],
            "categorical_features": list(CATEGORICAL_FEATURES),
            "categorical_options": [
                [str(value) for value in categories] for categories in one_hot.categories_
            ],
        },
        "transformed_features": [
            str(feature) for feature in preprocess.get_feature_names_out()
        ],
        "classes": [str(value) for value in classifier.classes_],
        "classifier": {
            "coefficients": classifier.coef_.astype(float).tolist(),
            "intercepts": classifier.intercept_.astype(float).tolist(),
            "probability": "stable_softmax",
        },
        "explanation": {
            "method": "signed transformed-feature contribution to the predicted-class logit",
            "limit": 5,
            "tie_break": "transformed feature order",
            "causal": False,
        },
        "provenance": {
            "source_dataset": "Palmer Penguins",
            "source_url": DATA_URL,
            "source_commit": SOURCE_COMMIT,
            "source_sha256": DATA_SHA256,
            "source_license": "CC0-1.0",
            "exporter": "scripts/export_penguins_browser_model.py",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    return artifact, model, X


def _fixture_inputs(artifact: dict[str, Any]) -> list[tuple[str, dict[str, str | float]]]:
    defaults = _sample_dict(default_penguin_sample())
    medians = {
        feature["name"]: feature["observed_median"]
        for feature in artifact["inputs"]["numeric"]
    }
    cases: list[tuple[str, dict[str, str | float]]] = [("default", defaults.copy())]

    categorical = artifact["inputs"]["categorical"]
    for island, sex, year in itertools.product(
        categorical[0]["options"], categorical[1]["options"], categorical[2]["options"]
    ):
        values = {**defaults, **medians, "island": island, "sex": sex, "year": year}
        cases.append((f"categories-{island.lower()}-{sex}-{year}", values))

    for feature in artifact["inputs"]["numeric"]:
        name = feature["name"]
        step = feature["step"]
        boundaries = (
            ("min", feature["observed_min"]),
            ("median", feature["observed_median"]),
            ("max", feature["observed_max"]),
            ("below", feature["observed_min"] - step),
            ("above", feature["observed_max"] + step),
        )
        for label, value in boundaries:
            cases.append((f"{name}-{label}", {**defaults, name: value}))

    cases.extend(
        [
            ("small-morphology", {**defaults, **{f["name"]: f["observed_min"] for f in artifact["inputs"]["numeric"]}}),
            ("large-morphology", {**defaults, **{f["name"]: f["observed_max"] for f in artifact["inputs"]["numeric"]}}),
            (
                "chinstrap-like",
                {
                    "island": "Dream",
                    "bill_length_mm": 50.1,
                    "bill_depth_mm": 18.4,
                    "flipper_length_mm": 198.0,
                    "body_mass_g": 3700.0,
                    "sex": "male",
                    "year": "2009",
                },
            ),
        ]
    )
    return cases


def build_fixtures(artifact: dict[str, Any], model: Any) -> dict[str, Any]:
    """Generate frozen expectations directly from the fitted Python pipeline."""

    preprocess = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    feature_names = [str(value) for value in preprocess.get_feature_names_out()]
    classes = [str(value) for value in classifier.classes_]
    fixtures = []

    for fixture_id, values in _fixture_inputs(artifact):
        sample = _sample_from_dict(values)
        frame = sample.to_frame()
        probabilities = model.predict_proba(frame)[0]
        logits = classifier.decision_function(preprocess.transform(frame))[0]
        predicted_species = str(model.predict(frame)[0])
        class_index = classes.index(predicted_species)
        transformed = preprocess.transform(frame)[0]
        contributions = transformed * classifier.coef_[class_index]
        ranked = sorted(
            enumerate(contributions),
            key=lambda item: (-abs(float(item[1])), item[0]),
        )[: artifact["explanation"]["limit"]]
        top_contributions = [
            {
                "feature": feature_names[index]
                .replace("numeric__", "")
                .replace("categorical__", ""),
                "contribution": float(value),
                "direction": "supports" if value >= 0 else "pulls against",
            }
            for index, value in ranked
        ]
        fixtures.append(
            {
                "id": fixture_id,
                "input": values,
                "expected": {
                    "predicted_species": predicted_species,
                    "probabilities": {
                        species: float(probability)
                        for species, probability in zip(classes, probabilities, strict=True)
                    },
                    "logits": [float(value) for value in logits],
                    "warnings": _warning_messages(values, artifact["inputs"]["numeric"]),
                    "top_contributions": top_contributions,
                },
            }
        )

    return {
        "schema_version": 1,
        "reference": "fitted Python scikit-learn pipeline",
        "model_artifact_id": artifact["artifact_id"],
        "tolerance": 1e-10,
        "cases": fixtures,
    }


def export(
    data: pd.DataFrame,
    *,
    model_path: Path,
    fixture_path: Path,
    provenance_path: Path,
) -> None:
    artifact, model, X = build_artifact(data)
    fixtures = build_fixtures(artifact, model)
    _write_json(model_path, artifact)
    _write_json(fixture_path, fixtures)

    exporter_path = Path(__file__).resolve()
    provenance = {
        "schema_version": 1,
        "artifact_id": artifact["artifact_id"],
        "training_rows": int(len(X)),
        "source_url": DATA_URL,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": DATA_SHA256,
        "source_license": "CC0-1.0",
        "model_path": model_path.relative_to(ROOT).as_posix(),
        "model_sha256": _sha256(model_path),
        "fixtures_path": fixture_path.relative_to(ROOT).as_posix(),
        "fixtures_sha256": _sha256(fixture_path),
        "exporter_path": exporter_path.relative_to(ROOT).as_posix(),
        "exporter_sha256": _sha256(exporter_path),
        "raw_data_committed": False,
        "boundary": "Educational browser inference, not a production or field-use system.",
    }
    _write_json(provenance_path, provenance)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-url", default=DATA_URL)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--fixture-path", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--provenance-path", type=Path, default=DEFAULT_PROVENANCE_PATH)
    args = parser.parse_args()

    data = load_penguin_data(args.data_url)
    export(
        data,
        model_path=args.model_path.resolve(),
        fixture_path=args.fixture_path.resolve(),
        provenance_path=args.provenance_path.resolve(),
    )
    print(f"Exported {DEFAULT_MODEL_PATH.relative_to(ROOT)} from {len(data)} source rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
