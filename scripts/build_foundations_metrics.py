#!/usr/bin/env python3
"""Build and check the compact public foundations-metrics snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "foundations_metrics.json"
NOTEBOOKS = {
    "kmeans": Path("notebooks/kmeans_clustering.ipynb"),
    "regression": Path("notebooks/regression_modeling_project.ipynb"),
}

KMEANS_PROBE = r"""
import json

selected_row = cluster_scores.loc[cluster_scores["k"] == best_k].iloc[0]
print("__FOUNDATIONS_METRICS__=" + json.dumps({
    "selected_k": int(best_k),
    "silhouette": round(float(selected_row["silhouette"]), 3),
    "adjusted_rand_index": round(float(ari), 3),
    "random_state": int(RANDOM_STATE),
}, sort_keys=True))
"""

REGRESSION_PROBE = r"""
import json

print("__FOUNDATIONS_METRICS__=" + json.dumps({
    "selected_model": str(best_name),
    "test_rmse": round(float(test_rmse), 2),
    "test_r2": round(float(test_r2), 3),
    "random_state": int(RANDOM_STATE),
}, sort_keys=True))
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _execute_notebook(path: Path, probe: str, temporary_root: Path) -> dict[str, Any]:
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell(probe))

    figure_dir = temporary_root / "figures"
    mpl_config_dir = temporary_root / "mplconfig"
    figure_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    old_figure_dir = os.environ.get("ML_PORTFOLIO_FIGURE_DIR")
    old_mpl_config_dir = os.environ.get("MPLCONFIGDIR")
    os.environ["ML_PORTFOLIO_FIGURE_DIR"] = str(figure_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    try:
        client = NotebookClient(
            notebook,
            kernel_name="python3",
            timeout=900,
            record_timing=False,
            resources={"metadata": {"path": str(ROOT)}},
        )
        executed = client.execute()
    finally:
        if old_figure_dir is None:
            os.environ.pop("ML_PORTFOLIO_FIGURE_DIR", None)
        else:
            os.environ["ML_PORTFOLIO_FIGURE_DIR"] = old_figure_dir
        if old_mpl_config_dir is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = old_mpl_config_dir

    output_text = ""
    for output in executed.cells[-1].get("outputs", []):
        if output.get("output_type") == "stream":
            output_text += output.get("text", "")
        elif output.get("output_type") == "execute_result":
            output_text += output.get("data", {}).get("text/plain", "")

    match = re.search(r"__FOUNDATIONS_METRICS__=(\{.*\})", output_text)
    if match is None:
        raise RuntimeError(f"Foundations probe output missing from {path}")
    return json.loads(match.group(1))


def _build_manifest() -> dict[str, Any]:
    notebook_paths = {
        name: ROOT / relative_path for name, relative_path in NOTEBOOKS.items()
    }
    missing = [str(path) for path in notebook_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required source notebook missing: {', '.join(missing)}"
        )

    with tempfile.TemporaryDirectory(prefix="foundations-metrics-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        kmeans = _execute_notebook(
            notebook_paths["kmeans"], KMEANS_PROBE, temporary_root / "kmeans"
        )
        regression = _execute_notebook(
            notebook_paths["regression"],
            REGRESSION_PROBE,
            temporary_root / "regression",
        )

    return {
        "schema_version": 1,
        "boundary": (
            "raw data is not redistributed. This snapshot records compact educational "
            "benchmark results from the two foundations notebooks only."
        ),
        "regeneration_command": "python scripts/build_foundations_metrics.py --write",
        "environment": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit_learn": _package_version("scikit-learn"),
            "nbclient": _package_version("nbclient"),
            "nbformat": _package_version("nbformat"),
        },
        "source_notebooks": {
            name: {
                "path": relative_path.as_posix(),
                "sha256": _sha256(notebook_paths[name]),
            }
            for name, relative_path in NOTEBOOKS.items()
        },
        "datasets": {
            "wine": {
                "loader": "sklearn.datasets.load_wine",
                "source": "UCI Machine Learning Repository Wine dataset",
                "attribution": "UCI Wine Dataset, CC BY 4.0; attribution required.",
            },
            "diabetes": {
                "loader": "sklearn.datasets.load_diabetes",
                "source": "scikit-learn bundled toy dataset",
                "attribution": "Loaded from the scikit-learn runtime; raw data is not redistributed.",
            },
        },
        "results": {
            "kmeans": kmeans,
            "regression": regression,
        },
    }


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _output_path(value: str | None) -> Path:
    if value is None:
        return DEFAULT_OUTPUT
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_snapshot(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(_canonical_json(manifest), encoding="utf-8")
    temporary_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the snapshot")
    mode.add_argument(
        "--check", action="store_true", help="rebuild and compare the snapshot"
    )
    parser.add_argument(
        "--output", help="snapshot path, relative to the repository root"
    )
    args = parser.parse_args()

    output_path = _output_path(args.output)
    manifest = _build_manifest()
    if args.write:
        _write_snapshot(output_path, manifest)
        print(f"Wrote foundations metrics: {output_path}")
        return 0

    if not output_path.is_file():
        raise SystemExit(f"Foundations metrics snapshot missing: {output_path}")
    actual = json.loads(output_path.read_text(encoding="utf-8"))
    if _canonical_json(actual) != _canonical_json(manifest):
        raise SystemExit(f"Foundations metrics snapshot is stale: {output_path}")
    print(f"Foundations metrics verified: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
