from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
MODEL = SITE / "model" / "penguins-logistic-v1.json"
FIXTURES = ROOT / "tests" / "browser" / "fixtures" / "penguins_reference_cases.json"
PROVENANCE = ROOT / "reports" / "evidence" / "penguins_browser_model_provenance.json"


class _SiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))


class PenguinsBrowserExplorerTests(unittest.TestCase):
    def test_node_runtime_matches_python_reference_fixtures(self) -> None:
        result = subprocess.run(
            ["node", "--test", "tests/browser/model-parity.test.mjs", "tests/browser/reflow-contract.test.mjs"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_offline_verifier_binds_model_fixtures_and_provenance(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/verify_penguins_browser_model.py", "--offline"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Browser model evidence verified", result.stdout)

        provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        for key, path in (("model_sha256", MODEL), ("fixtures_sha256", FIXTURES)):
            with self.subTest(key=key):
                self.assertEqual(provenance[key], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_static_page_has_semantic_controls_status_and_no_external_runtime(self) -> None:
        html = (SITE / "index.html").read_text(encoding="utf-8")
        parser = _SiteParser()
        parser.feed(html)

        tags = parser.tags
        self.assertTrue(any(tag == "main" for tag, _ in tags))
        self.assertTrue(any(tag == "fieldset" for tag, _ in tags))
        self.assertTrue(any(tag == "noscript" for tag, _ in tags))
        self.assertTrue(
            any(attrs.get("aria-live") == "polite" for _, attrs in tags),
            "prediction changes need a polite live region",
        )
        labels = [attrs.get("for") for tag, attrs in tags if tag == "label"]
        controls = [attrs.get("id") for tag, attrs in tags if tag in {"input", "select"}]
        self.assertGreaterEqual(len(controls), 7)
        self.assertTrue(set(controls).issubset(labels))

        csp = [
            attrs.get("content", "")
            for tag, attrs in tags
            if tag == "meta" and attrs.get("http-equiv", "").lower() == "content-security-policy"
        ]
        self.assertEqual(len(csp), 1)
        self.assertIn("default-src 'self'", csp[0])
        self.assertIn("object-src 'none'", csp[0])

        external_assets = []
        for tag, attrs in tags:
            if tag not in {"img", "link", "script"}:
                continue
            candidate = attrs.get("src") or attrs.get("href")
            if candidate and candidate.startswith(("http://", "https://", "//")):
                external_assets.append((tag, candidate))
        self.assertEqual(external_assets, [])

    def test_site_contains_no_tracking_storage_or_unsafe_dom_sinks(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(SITE.rglob("*"))
            if path.suffix in {".html", ".css", ".js", ".mjs"}
        )
        forbidden = (
            "document.cookie",
            "eval(",
            "innerHTML",
            "localStorage",
            "sessionStorage",
            "XMLHttpRequest",
        )
        self.assertEqual([term for term in forbidden if term in source], [])


if __name__ == "__main__":
    unittest.main()
