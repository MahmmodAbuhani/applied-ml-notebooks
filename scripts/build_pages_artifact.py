#!/usr/bin/env python3
"""Assemble the verified static files published through GitHub Pages."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANK_MANIFEST_PATH = Path("reports/evidence/bank_marketing_provenance.json")
BROWSER_PROVENANCE_PATH = Path(
    "reports/evidence/penguins_browser_model_provenance.json"
)


def _repository_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a repository-relative path: {path}")
    return ROOT / path


def _copy_repository_file(relative_path: str | Path, output_dir: Path) -> None:
    source = _repository_path(relative_path)
    if not source.is_file():
        raise FileNotFoundError(f"Required Pages source is missing: {source}")
    destination = output_dir / Path(relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_pages_artifact(
    *, output_dir: Path, commit: str, deploy: bool
) -> Path:
    """Build the static Pages tree without copying source notebooks."""
    output_dir = output_dir.resolve()
    repository_root = ROOT.resolve()
    if output_dir == repository_root or repository_root in output_dir.parents:
        raise ValueError("Pages output must be outside the repository worktree")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "site", output_dir, dirs_exist_ok=True)
    shutil.copy2(
        _repository_path(BROWSER_PROVENANCE_PATH),
        output_dir / "model-provenance.json",
    )

    bank_manifest = json.loads(
        _repository_path(BANK_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    bank_report_path = bank_manifest["artifact"]["path"]
    _copy_repository_file(bank_report_path, output_dir)
    _copy_repository_file(BANK_MANIFEST_PATH, output_dir)
    for asset in bank_manifest["artifact"]["assets"]:
        _copy_repository_file(asset["path"], output_dir)

    build_provenance = {
        "schema_version": 1,
        "artifact_id": "applied-ml-notebooks-pages",
        "commit": commit,
        "deploy": deploy,
        "model_provenance": "model-provenance.json",
        "bank_marketing_report": bank_report_path,
        "bank_marketing_provenance": BANK_MANIFEST_PATH.as_posix(),
    }
    (output_dir / "build-provenance.json").write_text(
        json.dumps(build_provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    notebooks = sorted(output_dir.rglob("*.ipynb"))
    if notebooks:
        paths = ", ".join(path.relative_to(output_dir).as_posix() for path in notebooks)
        raise RuntimeError(f"Pages artifact must not contain source notebooks: {paths}")

    return output_dir


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return normalized == "true"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--deploy", required=True, type=_boolean)
    args = parser.parse_args()

    output_dir = build_pages_artifact(
        output_dir=args.output_dir,
        commit=args.commit,
        deploy=args.deploy,
    )
    print(f"Built Pages artifact at {output_dir}")


if __name__ == "__main__":
    main()
