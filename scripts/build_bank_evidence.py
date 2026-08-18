#!/usr/bin/env python3
"""Execute the Bank notebook and build a committed HTML evidence snapshot."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

from nbclient import NotebookClient
from nbconvert import HTMLExporter
import nbformat

from ml_portfolio.bank_marketing import BANK_DATA_SHA256, BANK_DATA_URL
from ml_portfolio.evidence import (
    BANK_EVIDENCE_BOUNDARY,
    build_bank_evidence_manifest,
    externalize_embedded_images,
    sha256_file,
    verify_bank_evidence_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = Path("notebooks/bank_marketing_response_model.ipynb")
DEFAULT_OUTPUT_DIR = Path("reports/evidence")
HTML_NAME = "bank_marketing_executed.html"
MANIFEST_NAME = "bank_marketing_provenance.json"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_clean_source() -> tuple[str, str]:
    tracked_status = _git("status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise SystemExit("Refusing to build evidence from a dirty tracked worktree")

    source_commit = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    committed_notebook = subprocess.run(
        ["git", "show", f"{source_commit}:{NOTEBOOK_PATH.as_posix()}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    working_notebook = (ROOT / NOTEBOOK_PATH).read_bytes()
    if committed_notebook != working_notebook:
        raise SystemExit("The Bank source notebook does not match the source commit")
    return source_commit, source_tree


def _environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "matplotlib": metadata.version("matplotlib"),
        "nbclient": metadata.version("nbclient"),
        "nbconvert": metadata.version("nbconvert"),
        "numpy": metadata.version("numpy"),
        "pandas": metadata.version("pandas"),
        "scikit-learn": metadata.version("scikit-learn"),
    }


def _provenance_cell(source_commit: str, notebook_sha256: str) -> nbformat.NotebookNode:
    source = f"""# Executed Evidence Snapshot

This static HTML was executed from source commit `{source_commit}`.

- Source notebook SHA-256: `{notebook_sha256}`
- Input: [UCI Bank Marketing archive]({BANK_DATA_URL})
- Input archive SHA-256: `{BANK_DATA_SHA256}`
- Regeneration command: `python scripts/build_bank_evidence.py`

**Boundary:** {BANK_EVIDENCE_BOUNDARY}
"""
    cell = nbformat.v4.new_markdown_cell(source)
    cell["id"] = "bank-execution-provenance"
    return cell


def _execute_notebook(source_commit: str, notebook_sha256: str, temp_dir: Path) -> str:
    notebook = nbformat.read(ROOT / NOTEBOOK_PATH, as_version=4)
    notebook.cells.insert(0, _provenance_cell(source_commit, notebook_sha256))

    figure_dir = temp_dir / "figures"
    mpl_config_dir = temp_dir / "matplotlib"
    previous_figure_dir = os.environ.get("ML_PORTFOLIO_FIGURE_DIR")
    previous_mpl_config = os.environ.get("MPLCONFIGDIR")
    os.environ["ML_PORTFOLIO_FIGURE_DIR"] = str(figure_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    try:
        NotebookClient(
            notebook,
            timeout=900,
            kernel_name="python3",
            record_timing=False,
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
    finally:
        if previous_figure_dir is None:
            os.environ.pop("ML_PORTFOLIO_FIGURE_DIR", None)
        else:
            os.environ["ML_PORTFOLIO_FIGURE_DIR"] = previous_figure_dir
        if previous_mpl_config is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_mpl_config

    for cell in notebook.cells:
        cell.get("metadata", {}).pop("execution", None)
    notebook.metadata.pop("widgets", None)

    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input = False
    body, _ = exporter.from_notebook_node(
        notebook,
        resources={"metadata": {"name": "Bank Marketing Executed Evidence"}},
    )
    return body


def build(output_dir: Path) -> tuple[Path, Path]:
    source_commit, source_tree = _require_clean_source()
    notebook_path = ROOT / NOTEBOOK_PATH
    notebook_sha256 = sha256_file(notebook_path)
    output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / HTML_NAME
    manifest_path = output_dir / MANIFEST_NAME
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for stale_asset in asset_dir.glob("figure-*.png"):
        stale_asset.unlink()

    with tempfile.TemporaryDirectory(prefix="ml-bank-evidence-") as tmp:
        html = _execute_notebook(source_commit, notebook_sha256, Path(tmp))
    sanitized_html, asset_records = externalize_embedded_images(html, asset_dir)

    temporary_html = html_path.with_suffix(".html.tmp")
    temporary_html.write_text(sanitized_html, encoding="utf-8")
    temporary_html.replace(html_path)

    artifact_root = output_dir.relative_to(ROOT).as_posix()
    manifest_assets = [
        {**asset, "path": f"{artifact_root}/{asset['path']}"}
        for asset in asset_records
    ]

    manifest = build_bank_evidence_manifest(
        source_commit=source_commit,
        source_tree=source_tree,
        notebook_path=NOTEBOOK_PATH.as_posix(),
        notebook_sha256=notebook_sha256,
        input_url=BANK_DATA_URL,
        input_sha256=BANK_DATA_SHA256,
        artifact_path=html_path.relative_to(ROOT).as_posix(),
        artifact_sha256=sha256_file(html_path),
        artifact_bytes=html_path.stat().st_size,
        environment=_environment_versions(),
        artifact_assets=manifest_assets,
    )
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    verify_bank_evidence_manifest(manifest, root=ROOT)

    print(f"Source commit: {source_commit}")
    print(f"HTML: {html_path.relative_to(ROOT)} ({sha256_file(html_path)})")
    print(f"Manifest: {manifest_path.relative_to(ROOT)} ({sha256_file(manifest_path)})")
    return html_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Repository-relative output directory (default: reports/evidence)",
    )
    args = parser.parse_args()
    if args.output_dir.is_absolute() or ".." in args.output_dir.parts:
        raise SystemExit("--output-dir must stay inside the repository")
    build(args.output_dir)


if __name__ == "__main__":
    main()
