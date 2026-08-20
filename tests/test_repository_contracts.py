from __future__ import annotations

import re
import json
import subprocess
import sys
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

    def test_pages_build_verifies_and_assembles_published_evidence(self) -> None:
        workflow = _load_workflow(ROOT / ".github" / "workflows" / "pages.yml")
        steps = workflow["jobs"]["build"]["steps"]
        run_commands = "\n".join(step.get("run", "") for step in steps)

        self.assertIn("python scripts/verify_bank_evidence.py", run_commands)
        self.assertIn("python scripts/build_pages_artifact.py", run_commands)
        self.assertIn("/tmp/applied-ml-notebooks-pages", run_commands)

        checkout_steps = [
            step
            for step in steps
            if step.get("uses", "").startswith("actions/checkout@")
        ]
        self.assertEqual(len(checkout_steps), 1)
        self.assertEqual(
            checkout_steps[0].get("with", {}).get("fetch-depth"),
            "0",
            "Pages verifies evidence tied to an earlier commit, so checkout must include history.",
        )

        deploy_job = workflow["jobs"]["deploy"]
        self.assertEqual(deploy_job.get("if"), "${{ inputs.deploy }}")
        self.assertEqual(
            deploy_job.get("permissions"),
            {"pages": "write", "id-token": "write"},
        )


class PagesArtifactContractTests(unittest.TestCase):
    def test_published_bank_html_has_no_external_runtime_code(self) -> None:
        report_html = (
            ROOT / "reports" / "evidence" / "bank_marketing_executed.html"
        ).read_text(encoding="utf-8")
        external_runtime_references = re.findall(
            r'<script\b[^>]*\bsrc=["\']https?://[^"\']+'
            r'|\bimport\(\s*["\']https?://[^"\']+',
            report_html,
            flags=re.IGNORECASE,
        )

        self.assertEqual(external_runtime_references, [])

    def test_pages_artifact_contains_complete_bank_report_bundle_without_notebooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "pages"
            candidate_sha = "0123456789abcdef0123456789abcdef01234567"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_pages_artifact.py",
                    "--output-dir",
                    str(output_dir),
                    "--commit",
                    candidate_sha,
                    "--deploy",
                    "false",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            expected_paths = (
                "index.html",
                "model-provenance.json",
                "build-provenance.json",
                "reports/evidence/bank_marketing_executed.html",
                "reports/evidence/bank_marketing_provenance.json",
                "reports/evidence/assets/figure-01.png",
                "reports/evidence/assets/figure-02.png",
                "reports/evidence/assets/figure-03.png",
                "reports/evidence/assets/figure-04.png",
                "reports/evidence/assets/figure-05.png",
            )
            for relative_path in expected_paths:
                with self.subTest(relative_path=relative_path):
                    self.assertTrue((output_dir / relative_path).is_file())

            report_path = output_dir / "reports/evidence/bank_marketing_executed.html"
            report_html = report_path.read_text(encoding="utf-8")
            figure_sources = sorted(
                set(re.findall(r'src="(assets/figure-[0-9]{2}\.png)"', report_html))
            )
            self.assertEqual(
                figure_sources,
                [
                    "assets/figure-01.png",
                    "assets/figure-02.png",
                    "assets/figure-03.png",
                    "assets/figure-04.png",
                    "assets/figure-05.png",
                ],
            )
            for figure_source in figure_sources:
                with self.subTest(figure_source=figure_source):
                    self.assertTrue((report_path.parent / figure_source).is_file())

            self.assertEqual(list(output_dir.rglob("*.ipynb")), [])
            provenance = json.loads(
                (output_dir / "build-provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["artifact_id"], "applied-ml-notebooks-pages")
            self.assertEqual(provenance["commit"], candidate_sha)
            self.assertFalse(provenance["deploy"])
            self.assertEqual(
                provenance["bank_marketing_report"],
                "reports/evidence/bank_marketing_executed.html",
            )

    def test_generated_evidence_html_is_classified_as_generated(self) -> None:
        result = subprocess.run(
            [
                "git",
                "check-attr",
                "linguist-generated",
                "--",
                "reports/evidence/bank_marketing_executed.html",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertEqual(
            result.stdout.strip(),
            "reports/evidence/bank_marketing_executed.html: linguist-generated: true",
        )

    def test_pages_builder_rejects_a_nonempty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "pages"
            output_dir.mkdir()
            stale_file = output_dir / "stale.txt"
            stale_file.write_text("must not ship\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_pages_artifact.py",
                    "--output-dir",
                    str(output_dir),
                    "--commit",
                    "0123456789abcdef0123456789abcdef01234567",
                    "--deploy",
                    "false",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Pages output directory must be empty", result.stderr)
            self.assertEqual(stale_file.read_text(encoding="utf-8"), "must not ship\n")

    def test_pages_builder_requires_a_full_commit_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "pages"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/build_pages_artifact.py",
                    "--output-dir",
                    str(output_dir),
                    "--commit",
                    "local-review",
                    "--deploy",
                    "false",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("40-character lowercase commit SHA", result.stderr)
            self.assertFalse(output_dir.exists())


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

    def test_public_indexes_route_reviewers_to_the_rendered_bank_report(self) -> None:
        rendered_report_url = (
            "https://mahmmodabuhani.github.io/applied-ml-notebooks/"
            "reports/evidence/bank_marketing_executed.html"
        )
        index_paths = (
            ROOT / "README.md",
            ROOT / "reports" / "README.md",
            ROOT / "reports" / "evidence" / "README.md",
            ROOT / "docs" / "REPRODUCING.md",
        )

        for path in index_paths:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                self.assertIn(rendered_report_url, path.read_text(encoding="utf-8"))

    def test_public_surfaces_share_the_current_hosting_boundary(self) -> None:
        app_text = (ROOT / "demo" / "penguin_streamlit_app.py").read_text(encoding="utf-8")
        demo_readme = (ROOT / "demo" / "README.md").read_text(encoding="utf-8")
        reproduction = (ROOT / "docs" / "REPRODUCING.md").read_text(encoding="utf-8")
        model_card = (ROOT / "reports" / "palmer_penguins_model_card.md").read_text(
            encoding="utf-8"
        )
        root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        public_docs = (root_readme, demo_readme, reproduction, model_card)
        pages_url = "https://mahmmodabuhani.github.io/applied-ml-notebooks/"
        current_streamlit_sha = "94de46d03def0ffa60386d2d3e4d3b14e310335a"

        self.assertIn("Hosted on Streamlit Community Cloud", app_text)
        self.assertNotIn("the checked-in app runs locally", app_text)
        self.assertNotIn("A public URL is only claimed", app_text)
        self.assertIn(current_streamlit_sha, demo_readme)
        self.assertNotIn("09cc8d26d365560915b423cd98f1abf5158f1b53", demo_readme)
        self.assertIn("2026-08-19", demo_readme)
        self.assertIn("demo/README.md", model_card)
        self.assertNotIn("The hosted Streamlit demo is verified at", model_card)
        self.assertIn("Average Precision (AP)", root_readme)
        self.assertIn("versioned Bank execution snapshot", root_readme)
        self.assertNotIn("externally linked Bank execution snapshot", root_readme)
        for document in public_docs:
            with self.subTest(document=document[:40]):
                self.assertIn(pages_url, document)
                lowered = document.lower()
                self.assertNotIn("no live pages url is claimed", lowered)
                self.assertNotIn("currently a checked-in static artifact", lowered)
                self.assertNotIn("pages workflow is prepared", lowered)

    def test_demo_surfaces_offer_reviewer_navigation(self) -> None:
        app_text = (ROOT / "demo" / "penguin_streamlit_app.py").read_text(encoding="utf-8")
        static_explorer = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
        pages_url = "https://mahmmodabuhani.github.io/applied-ml-notebooks/"

        for text in (app_text, static_explorer):
            with self.subTest(surface="app" if text is app_text else "static"):
                self.assertIn("https://github.com/MahmmodAbuhani/applied-ml-notebooks", text)
                self.assertIn("palmer_penguins_end_to_end.ipynb", text)
                self.assertIn("palmer_penguins_model_card.md", text)

        self.assertIn(pages_url, app_text)
        self.assertIn("STATIC_EXPLORER_SOURCE_URL", app_text)
        self.assertIn("/blob/main/site/index.html", app_text)
        self.assertIn("Start with a plausible profile", app_text)
        self.assertIn("ml-notebooks-portfolio-public.streamlit.app", static_explorer)

    def test_static_explorer_offers_a_primary_explorer_action(self) -> None:
        static_explorer = (ROOT / "site" / "index.html").read_text(encoding="utf-8")

        self.assertIn('class="hero-cta"', static_explorer)
        self.assertIn('href="#explorer"', static_explorer)
        self.assertIn("Try the explorer", static_explorer)


if __name__ == "__main__":
    unittest.main()
