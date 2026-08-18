#!/usr/bin/env python3
"""Verify the committed Bank execution snapshot without network access."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess

from ml_portfolio.evidence import verify_bank_evidence_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "reports/evidence/bank_marketing_provenance.json"


def _require_full_git_checkout() -> None:
    git_tree = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if git_tree.returncode != 0 or git_tree.stdout.strip() != "true":
        raise SystemExit(
            "Bank evidence ancestry verification requires a full Git checkout. "
            "A downloaded ZIP or source archive does not include the required commit history; "
            "clone the repository and run this command from that checkout."
        )


def main() -> None:
    _require_full_git_checkout()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    verify_bank_evidence_manifest(manifest, root=ROOT)

    source_commit = manifest["source"]["commit"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=ROOT,
        check=True,
    )
    committed_notebook = subprocess.run(
        ["git", "show", f"{source_commit}:{manifest['source']['notebook']['path']}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if sha256(committed_notebook).hexdigest() != manifest["source"]["notebook"]["sha256"]:
        raise SystemExit("The source commit does not contain the attested notebook bytes")

    print(f"Verified Bank execution evidence from source commit {source_commit}")


if __name__ == "__main__":
    main()
