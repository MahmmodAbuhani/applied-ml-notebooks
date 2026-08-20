from __future__ import annotations

import hashlib
import json
import base64
import inspect
import re
from pathlib import Path
import tempfile
import unittest

import nbformat

from ml_portfolio import evidence
from ml_portfolio.evidence import (
    build_bank_evidence_manifest,
    canonical_json_bytes,
    sha256_file,
    verify_bank_evidence_manifest,
)
from scripts.build_bank_evidence import _static_html_exporter


class EvidenceTests(unittest.TestCase):
    def test_static_html_exporter_has_no_external_runtime_code(self) -> None:
        notebook = nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_markdown_cell("Static evidence")]
        )
        html, _ = _static_html_exporter().from_notebook_node(notebook)
        external_runtime_references = re.findall(
            r'<script\b[^>]*\bsrc=["\']https?://[^"\']+'
            r'|\bimport\(\s*["\']https?://[^"\']+',
            html,
            flags=re.IGNORECASE,
        )

        self.assertEqual(external_runtime_references, [])

    def test_externalize_embedded_images_removes_data_uris_and_generic_alt_text(self) -> None:
        helper = getattr(evidence, "externalize_embedded_images", None)
        self.assertIsNotNone(helper, "evidence image externalizer is missing")
        png_payload = base64.b64encode(b"png-bytes").decode("ascii")
        svg_payload = base64.b64encode(b"<svg></svg>").decode("ascii")
        source = (
            '<style>:root{--jp-icon-add:url(data:image/svg+xml;base64,'
            f"{svg_payload});--jp-trail:url(data:image/png;base64,{png_payload})}}</style>"
            '<img alt="No description has been provided for this image" '
            f'src="data:image/png;base64,{png_payload}">'
        )

        with tempfile.TemporaryDirectory() as tmp:
            output, assets = helper(source, Path(tmp) / "assets")
            self.assertNotIn("data:image", output)
            self.assertIn(
                'alt="Late-period precision-recall curve showing weak ranking under source-order validation."',
                output,
            )
            self.assertEqual(len(assets), 1)
            self.assertEqual(assets[0]["path"], "assets/figure-01.png")
            self.assertEqual((Path(tmp) / "assets" / "figure-01.png").read_bytes(), b"png-bytes")

    def test_canonical_json_is_order_invariant(self) -> None:
        first = {"b": 2, "a": {"d": 4, "c": 3}}
        second = {"a": {"c": 3, "d": 4}, "b": 2}

        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))

    def test_manifest_binds_source_input_environment_and_html(self) -> None:
        manifest = build_bank_evidence_manifest(
            source_commit="a" * 40,
            source_tree="b" * 40,
            notebook_path="notebooks/bank_marketing_response_model.ipynb",
            notebook_sha256="c" * 64,
            input_url="https://example.test/bank.zip",
            input_sha256="d" * 64,
            artifact_path="reports/evidence/bank_marketing_executed.html",
            artifact_sha256="e" * 64,
            artifact_bytes=123,
            environment={"python": "3.12.0", "nbconvert": "7.17.1"},
        )

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["source"]["commit"], "a" * 40)
        self.assertEqual(manifest["input"]["sha256"], "d" * 64)
        self.assertEqual(manifest["artifact"]["bytes"], 123)
        self.assertIn("historical execution snapshot", manifest["boundary"])

    def test_verifier_detects_modified_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notebook = root / "notebooks" / "bank.ipynb"
            artifact = root / "reports" / "evidence" / "bank.html"
            notebook.parent.mkdir(parents=True)
            artifact.parent.mkdir(parents=True)
            notebook.write_bytes(b"source notebook")
            artifact.write_bytes(b"executed html")
            manifest = build_bank_evidence_manifest(
                source_commit="a" * 40,
                source_tree="b" * 40,
                notebook_path="notebooks/bank.ipynb",
                notebook_sha256=sha256_file(notebook),
                input_url="https://example.test/bank.zip",
                input_sha256=hashlib.sha256(b"input").hexdigest(),
                artifact_path="reports/evidence/bank.html",
                artifact_sha256=sha256_file(artifact),
                artifact_bytes=artifact.stat().st_size,
                environment={"python": "3.12.0"},
            )

            verify_bank_evidence_manifest(manifest, root=root)
            artifact.write_bytes(b"changed html")

            with self.assertRaisesRegex(ValueError, "artifact SHA-256"):
                verify_bank_evidence_manifest(manifest, root=root)

    def test_manifest_verifier_checks_external_evidence_assets(self) -> None:
        self.assertIn(
            "artifact_assets",
            inspect.signature(build_bank_evidence_manifest).parameters,
            "manifest builder must accept external asset records",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notebook = root / "notebooks" / "bank.ipynb"
            artifact = root / "reports" / "evidence" / "bank.html"
            asset = root / "reports" / "evidence" / "assets" / "figure-01.png"
            notebook.parent.mkdir(parents=True)
            artifact.parent.mkdir(parents=True)
            asset.parent.mkdir(parents=True)
            notebook.write_bytes(b"source notebook")
            artifact.write_bytes(b"executed html")
            asset.write_bytes(b"figure bytes")
            manifest = build_bank_evidence_manifest(
                source_commit="a" * 40,
                source_tree="b" * 40,
                notebook_path="notebooks/bank.ipynb",
                notebook_sha256=sha256_file(notebook),
                input_url="https://example.test/bank.zip",
                input_sha256=hashlib.sha256(b"input").hexdigest(),
                artifact_path="reports/evidence/bank.html",
                artifact_sha256=sha256_file(artifact),
                artifact_bytes=artifact.stat().st_size,
                environment={"python": "3.12.0"},
                artifact_assets=[
                    {
                        "path": "reports/evidence/assets/figure-01.png",
                        "sha256": sha256_file(asset),
                        "bytes": asset.stat().st_size,
                        "alt": "Bank Marketing execution figure 01",
                    }
                ],
            )

            verify_bank_evidence_manifest(manifest, root=root)
            asset.write_bytes(b"changed figure")

            with self.assertRaisesRegex(ValueError, "evidence asset"):
                verify_bank_evidence_manifest(manifest, root=root)

    def test_manifest_serialization_is_stable(self) -> None:
        manifest = build_bank_evidence_manifest(
            source_commit="a" * 40,
            source_tree="b" * 40,
            notebook_path="notebooks/bank.ipynb",
            notebook_sha256="c" * 64,
            input_url="https://example.test/bank.zip",
            input_sha256="d" * 64,
            artifact_path="reports/evidence/bank.html",
            artifact_sha256="e" * 64,
            artifact_bytes=10,
            environment={"python": "3.12.0"},
        )

        decoded = json.loads(canonical_json_bytes(manifest))
        self.assertEqual(decoded, manifest)


if __name__ == "__main__":
    unittest.main()
