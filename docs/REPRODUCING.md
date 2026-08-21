# Reproducing And Verifying

This guide shows how to install the declared Python environment and reproduce the repository's reviewable checks.

## Python 3.12 Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt -c constraints-py312.txt
```

Run the repository checks:

```bash
python -c "import ml_portfolio; print('import ok')"
python -m unittest discover -s tests -v
python demo/predict_penguin_species.py
python scripts/verify_penguins_browser_model.py --offline
node --test tests/browser/model-parity.test.mjs tests/browser/reflow-contract.test.mjs
python -m py_compile \
  streamlit_app.py \
  demo/penguin_streamlit_app.py \
  demo/predict_penguin_species.py \
  src/ml_portfolio/*.py
pre-commit run --all-files
git diff --check
```

GitHub Pages is the canonical no-account route. It publishes both the [browser-native explorer](https://mahmmodabuhani.github.io/applied-ml-notebooks/) and the [rendered Bank Marketing report](https://mahmmodabuhani.github.io/applied-ml-notebooks/reports/evidence/bank_marketing_executed.html) as static files with no backend. The [hosted Streamlit companion](https://ml-notebooks-portfolio-public.streamlit.app/) is secondary and availability-dependent: it may sleep, show a wake-up or hosting prompt, or be unavailable. The dated checks are recorded in [`demo/README.md`](../demo/README.md). To inspect the same combined artifact locally, build and serve the Pages tree:

```bash
python scripts/build_pages_artifact.py \
  --output-dir /tmp/applied-ml-notebooks-pages \
  --commit "$(git rev-parse HEAD)" \
  --deploy false
python3 -m http.server 8000 --directory /tmp/applied-ml-notebooks-pages
```

The browser parity tests compare the exported JavaScript model with Python reference fixtures across representative inputs and observed-range boundaries. The Pages builder copies the verified Bank HTML, manifest, and external figures without publishing source notebooks.

Validate stripped notebooks:

```bash
python - <<'PY'
from pathlib import Path
import nbformat

for path in sorted(Path("notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type == "code":
            assert cell.execution_count is None, f"{path}: cell {index} has execution count"
            assert not cell.outputs, f"{path}: cell {index} has outputs"
            assert not cell.get("metadata", {}).get("execution"), (
                f"{path}: cell {index} has execution metadata"
            )
    print(f"{path}: valid and stripped")
PY
```

Execute notebooks into a temporary directory:

```bash
mkdir -p /tmp/ml-notebook-checks
jupyter nbconvert \
  --to notebook \
  --execute notebooks/*.ipynb \
  --output-dir /tmp/ml-notebook-checks \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=900
```

Build and verify the durable Bank execution snapshot from a clean committed source tree:

```bash
python scripts/build_bank_evidence.py
python scripts/verify_bank_evidence.py
```

The Bank evidence builder writes a static HTML snapshot, external figure assets, and a provenance manifest. The snapshot is historical review evidence, not a live report, deployed model, or policy recommendation. The source notebooks remain stripped.

The ancestry check requires a full Git checkout because the manifest names the source commit that produced the snapshot.

## Public Surface Checks

Before sharing a checkout, confirm that generated local residue is ignored or absent:

```bash
find . -name ".DS_Store" -o -name "__pycache__" -o -name "*.pyc" -o -name ".ipynb_checkpoints"
```

Review public prose for credentials, personal machine paths, stale hosting claims, em dashes, and generated residue.

## Reviewable Changes

Inspect the exact local changes before committing:

```bash
git status --short
git diff --stat
git diff --check
```

The fast CI workflow checks package import, unit tests, the network-free CLI smoke test, notebook structure, stripped-source policy, and the versioned Bank evidence. The separate Notebook Execution workflow runs every notebook against its declared public inputs and retains temporary executed outputs for 30 days.
