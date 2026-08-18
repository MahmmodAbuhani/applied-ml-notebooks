from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "reports" / "foundations_metrics.json"
SCRIPT = ROOT / "scripts" / "build_foundations_metrics.py"


class FoundationsMetricsTests(unittest.TestCase):
    def _run_builder(self, *args: str) -> subprocess.CompletedProcess[str]:
        environment = {"PYTHONPATH": str(ROOT / "src")}
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=ROOT,
            env={**os.environ, **environment},
            capture_output=True,
            text=True,
            check=False,
        )

    def test_snapshot_contains_only_the_two_foundations_and_expected_results(
        self,
    ) -> None:
        self.assertTrue(SNAPSHOT.is_file(), "expected committed foundations snapshot")
        manifest = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            manifest["environment"]["python"],
            f"{sys.version_info.major}.{sys.version_info.minor}",
        )
        self.assertIn("raw data is not redistributed", manifest["boundary"].lower())
        self.assertEqual(
            set(manifest["results"]),
            {"kmeans", "regression"},
        )
        self.assertEqual(manifest["results"]["kmeans"]["selected_k"], 3)
        self.assertEqual(manifest["results"]["kmeans"]["silhouette"], 0.285)
        self.assertEqual(manifest["results"]["kmeans"]["adjusted_rand_index"], 0.897)
        self.assertEqual(
            manifest["results"]["regression"]["selected_model"], "Ridge Regression"
        )
        self.assertEqual(manifest["results"]["regression"]["test_rmse"], 53.63)
        self.assertEqual(manifest["results"]["regression"]["test_r2"], 0.457)

        serialized = json.dumps(manifest, sort_keys=True)
        for residue in (
            "/Users/",
            "worktrees",
            "owner",
            "agent",
            "gate",
            "Bank",
            "Penguins",
        ):
            self.assertNotIn(residue.lower(), serialized.lower())

    def test_check_mode_rebuilds_the_snapshot(self) -> None:
        result = self._run_builder("--check")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_readme_links_the_snapshot(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("(reports/foundations_metrics.json)", readme)


if __name__ == "__main__":
    unittest.main()
