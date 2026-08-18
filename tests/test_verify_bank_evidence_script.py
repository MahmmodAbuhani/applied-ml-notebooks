from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class VerifyBankEvidenceScriptTests(unittest.TestCase):
    def test_non_git_source_tree_exits_with_actionable_message_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive_root = Path(tmp)
            shutil.copytree(ROOT / "src" / "ml_portfolio", archive_root / "src" / "ml_portfolio")
            for relative_path in (
                "scripts/verify_bank_evidence.py",
                "notebooks/bank_marketing_response_model.ipynb",
                "reports/evidence/bank_marketing_executed.html",
                "reports/evidence/bank_marketing_provenance.json",
            ):
                destination = archive_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative_path, destination)

            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(archive_root / "src")
            result = subprocess.run(
                [sys.executable, str(archive_root / "scripts/verify_bank_evidence.py")],
                cwd=archive_root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("full Git checkout", output)
        self.assertIn("clone", output.lower())
        self.assertNotIn("Traceback", output)


if __name__ == "__main__":
    unittest.main()
