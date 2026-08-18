from __future__ import annotations

import re
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATHS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
IMMUTABLE_ACTION_REF = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
PUBLIC_PROSE_EXCLUDED_DIRS = frozenset(
    {
        ".direnv",
        ".git",
        ".nox",
        ".tox",
        ".venv",
        ".virtualenv",
        "env",
        "site-packages",
        "venv",
        "virtualenv",
    }
)


def _iter_public_markdown_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if not PUBLIC_PROSE_EXCLUDED_DIRS.intersection(path.relative_to(root).parts)
    )


def _load_workflow(path: Path) -> dict[str, object]:
    workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    if not isinstance(workflow, dict):
        raise AssertionError(f"{path} must contain a YAML mapping")
    return workflow


def _exact_pins(path: Path) -> dict[str, Version]:
    pins: dict[str, Version] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-r ")):
            continue
        if "==" not in line:
            continue
        name, version = line.split("==", maxsplit=1)
        pins[name.lower()] = Version(version)
    return pins


class WorkflowContractTests(unittest.TestCase):
    def test_workflows_use_minimal_permissions_and_bounded_jobs(self) -> None:
        self.assertTrue(WORKFLOW_PATHS, "Expected at least one GitHub Actions workflow")

        for path in WORKFLOW_PATHS:
            with self.subTest(workflow=path.name):
                workflow = _load_workflow(path)
                self.assertEqual(workflow.get("permissions"), {"contents": "read"})
                self.assertIsInstance(workflow.get("on"), dict)

                jobs = workflow.get("jobs")
                self.assertIsInstance(jobs, dict)
                for job_name, job in jobs.items():
                    self.assertIsInstance(job, dict, job_name)
                    self.assertRegex(job.get("timeout-minutes", ""), r"^[1-9][0-9]*$")

    def test_third_party_actions_use_full_commit_shas(self) -> None:
        for path in WORKFLOW_PATHS:
            workflow = _load_workflow(path)
            for job_name, job in workflow["jobs"].items():
                for index, step in enumerate(job.get("steps", []), start=1):
                    action_ref = step.get("uses")
                    if action_ref is None or action_ref.startswith("./"):
                        continue
                    with self.subTest(workflow=path.name, job=job_name, step=index):
                        self.assertRegex(action_ref, IMMUTABLE_ACTION_REF)

    def test_notebook_artifacts_fail_closed_when_outputs_are_missing(self) -> None:
        workflow = _load_workflow(ROOT / ".github" / "workflows" / "notebooks.yml")
        steps = workflow["jobs"]["execute-notebooks"]["steps"]
        upload_steps = [step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@")]

        self.assertEqual(len(upload_steps), 2)
        for step in upload_steps:
            self.assertEqual(step.get("with", {}).get("if-no-files-found"), "error")
            self.assertRegex(step.get("with", {}).get("path", ""), r"^/tmp/")

    def test_fast_ci_verifies_the_committed_bank_evidence(self) -> None:
        workflow = _load_workflow(ROOT / ".github" / "workflows" / "ci.yml")
        steps = workflow["jobs"]["ci"]["steps"]
        run_commands = "\n".join(step.get("run", "") for step in steps)

        self.assertIn("python scripts/verify_bank_evidence.py", run_commands)

        checkout_steps = [
            step
            for step in steps
            if step.get("uses", "").startswith("actions/checkout@")
        ]
        self.assertEqual(len(checkout_steps), 1)
        self.assertEqual(
            checkout_steps[0].get("with", {}).get("fetch-depth"),
            "0",
            "CI verifies evidence tied to an earlier commit, so checkout history must include it.",
        )


class DependencyContractTests(unittest.TestCase):
    def test_test_and_lint_dependencies_are_declared_directly(self) -> None:
        direct = _exact_pins(ROOT / "requirements-dev.txt")

        for dependency in ("packaging", "pyyaml", "ruff"):
            with self.subTest(dependency=dependency):
                self.assertIn(dependency, direct)

    def test_notebook_security_updates_are_coordinated_with_jupyterlab(self) -> None:
        direct = _exact_pins(ROOT / "requirements-dev.txt")
        constraints = _exact_pins(ROOT / "constraints-py312.txt")

        self.assertGreaterEqual(direct["notebook"], Version("7.5.6"))
        self.assertGreaterEqual(direct["nbconvert"], Version("7.17.1"))
        self.assertGreaterEqual(constraints["jupyterlab"], Version("4.5.7"))
        self.assertLess(constraints["jupyterlab"], Version("4.6"))
        self.assertEqual(constraints["notebook"], direct["notebook"])
        self.assertEqual(constraints["nbconvert"], direct["nbconvert"])


class PreCommitContractTests(unittest.TestCase):
    def test_plain_python_and_notebook_code_both_run_ruff(self) -> None:
        config = yaml.load(
            (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        hooks = [
            (repository.get("repo"), hook)
            for repository in config["repos"]
            for hook in repository.get("hooks", [])
        ]
        plain_ruff = [
            (repository, hook)
            for repository, hook in hooks
            if hook.get("id") == "ruff"
        ]
        notebook_ruff = [hook for _, hook in hooks if hook.get("id") == "nbqa-ruff"]

        self.assertEqual(len(plain_ruff), 1)
        self.assertEqual(plain_ruff[0][0], "local")
        self.assertEqual(plain_ruff[0][1].get("entry"), "ruff check")
        self.assertEqual(len(notebook_ruff), 1)


class PublicProseContractTests(unittest.TestCase):
    def test_public_tree_has_no_internal_metadata_document(self) -> None:
        self.assertFalse(
            (ROOT / "docs" / ("GITHUB_" + "METADATA.md")).exists(),
            "repository metadata planning notes must stay outside the public tree",
        )
        self.assertTrue((ROOT / "docs" / "REPRODUCING.md").is_file())

    def test_public_prose_walk_ignores_virtual_environment_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("Repository prose\n", encoding="utf-8")
            for directory in (".venv", "venv", "env", ".tox", ".nox"):
                dependency_readme = root / directory / "lib" / "README.md"
                dependency_readme.parent.mkdir(parents=True)
                dependency_readme.write_text("Third-party prose\n", encoding="utf-8")

            walked = [
                path.relative_to(root).as_posix()
                for path in _iter_public_markdown_paths(root)
            ]

        self.assertEqual(walked, ["README.md"])

    def test_public_prose_avoids_unqualified_portfolio_claims(self) -> None:
        forbidden = (
            "\N{EM DASH}",
            "balanced random " + "forest",
            "estimated " + "likelihood",
            "hosted" + "-ready",
            "performs " + "strongly",
            "performs " + "well",
            "production" + "-ready",
            "quality " + "upgrade",
            "RMSE/MAE/R2",
            "separate " + "cleanly",
            "state" + "-of-the-art",
            "trustworthy, reviewable applied " + "ml",
            "usually " + "improve",
            "world" + "-class",
        )
        public_text: dict[str, str] = {
            path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
            for path in _iter_public_markdown_paths(ROOT)
        }
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            public_text[path.relative_to(ROOT).as_posix()] = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if cell.get("cell_type") in {"code", "markdown"}
            )

        failures = []
        for path, text in public_text.items():
            lowered = text.lower()
            for phrase in forbidden:
                if phrase.lower() in lowered:
                    failures.append(f"{path}: {phrase}")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
