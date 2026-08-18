"""Verify committed Penguins browser-model evidence, with optional refitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_penguins_browser_model import (  # noqa: E402
    DEFAULT_FIXTURE_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROVENANCE_PATH,
    build_artifact,
    build_fixtures,
)
from ml_portfolio.penguins import DATA_SHA256, DATA_URL, load_penguin_data  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _assert_close(actual: Any, expected: Any, *, path: str, tolerance: float) -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise ValueError(f"{path} changed: expected {expected!r}, received {actual!r}")
        return
    if isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), float(expected), rel_tol=0, abs_tol=tolerance
        ):
            raise ValueError(f"{path} changed: expected {expected!r}, received {actual!r}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path} changed shape")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_close(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
                tolerance=tolerance,
            )
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{path} changed keys")
        for key in expected:
            _assert_close(
                actual[key], expected[key], path=f"{path}.{key}", tolerance=tolerance
            )
        return
    raise TypeError(f"Unsupported value at {path}: {type(expected)!r}")


def verify_offline() -> tuple[dict[str, Any], dict[str, Any]]:
    provenance = _load_json(DEFAULT_PROVENANCE_PATH)
    artifact = _load_json(DEFAULT_MODEL_PATH)
    fixtures = _load_json(DEFAULT_FIXTURE_PATH)

    expected_hashes = {
        "model_sha256": _sha256(DEFAULT_MODEL_PATH),
        "fixtures_sha256": _sha256(DEFAULT_FIXTURE_PATH),
        "exporter_sha256": _sha256(ROOT / provenance["exporter_path"]),
    }
    for key, actual in expected_hashes.items():
        if provenance.get(key) != actual:
            raise ValueError(f"{key} does not match the committed file")

    if artifact.get("schema_version") != 1 or fixtures.get("schema_version") != 1:
        raise ValueError("Unsupported browser-model evidence schema")
    if artifact.get("artifact_id") != provenance.get("artifact_id"):
        raise ValueError("Artifact identity does not match provenance")
    if fixtures.get("model_artifact_id") != artifact.get("artifact_id"):
        raise ValueError("Fixture identity does not match model artifact")
    if len(fixtures.get("cases", [])) < 42:
        raise ValueError("Reference fixture coverage is incomplete")
    if artifact.get("provenance", {}).get("source_url") != DATA_URL:
        raise ValueError("Model artifact does not name the pinned source URL")
    if artifact.get("provenance", {}).get("source_sha256") != DATA_SHA256:
        raise ValueError("Model artifact does not name the pinned source hash")
    return artifact, fixtures


def verify_refit(artifact: dict[str, Any], fixtures: dict[str, Any]) -> None:
    data = load_penguin_data()
    rebuilt_artifact, model, _ = build_artifact(data)
    rebuilt_fixtures = build_fixtures(rebuilt_artifact, model)

    comparable_artifact_keys = (
        "schema_version",
        "artifact_id",
        "model_family",
        "training_rows",
        "inputs",
        "preprocessing",
        "transformed_features",
        "classes",
        "classifier",
        "explanation",
    )
    _assert_close(
        {key: artifact[key] for key in comparable_artifact_keys},
        {key: rebuilt_artifact[key] for key in comparable_artifact_keys},
        path="artifact",
        tolerance=1e-8,
    )
    _assert_close(fixtures, rebuilt_fixtures, path="fixtures", tolerance=1e-8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Verify committed identities without downloading and refitting.",
    )
    args = parser.parse_args()

    artifact, fixtures = verify_offline()
    if not args.offline:
        verify_refit(artifact, fixtures)
    mode = "offline identities" if args.offline else "identities and fresh Python refit"
    print(f"Browser model evidence verified: {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
