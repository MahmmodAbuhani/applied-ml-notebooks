# Demo: Palmer Penguins Species Classifier

This folder contains the Palmer Penguins species-classification demo used by the portfolio. It runs through the root `streamlit_app.py` entrypoint.

The demo is intentionally modest. It is not a wildlife field tool or a production decision system. It gives a reviewer a concrete way to inspect the inference flow: inputs, predicted species, class probabilities, the training-data geometry, simple model-internal feature contributions, and the public-data boundary.

## Streamlit App

Run from the repository root:

```bash
streamlit run streamlit_app.py
```

What it shows:

- plausible morphology controls: bill length, bill depth, flipper length, and body mass
- optional collection-context inputs: island, sex, and year
- predicted species and class probabilities
- a training-distribution scatter plot with the current input marked
- warnings when numeric inputs are outside the observed training range
- top logistic-regression logit contributions for the predicted species
- an explicit educational-demo boundary and source-data note

## CLI Demo

Run from the repository root:

```bash
python demo/predict_penguin_species.py
```

Override the sample:

```bash
python demo/predict_penguin_species.py \
  --island Dream \
  --bill-length-mm 45.2 \
  --bill-depth-mm 16.4 \
  --flipper-length-mm 196 \
  --body-mass-g 4150 \
  --sex female \
  --year 2008
```

The CLI uses the same reusable modeling helpers as the Streamlit app.

## Hosted Demo Boundary

The root entrypoint and pinned dependencies are structured for a hosted Streamlit deployment. This repository does not claim a live URL until a signed-out browser check confirms that the deployed app loads and matches the reviewed source. GitHub Pages cannot run this Python app; it can host a static project page that links to the hosted app.

## Reproducibility Boundary

- Data source: pinned public Palmer Penguins CSV from `allisonhorst/palmerpenguins`.
- Training: the demo model is fit from the public CSV when the script or app starts.
- Artifacts: no hidden pre-trained model is required.
- Privacy: the app code does not persist inputs, write them to a database, or send them to a model API.
- Interpretation: contribution values are model-internal logit contributions, not causal explanations.

For the full methodology, holdout metrics, feature ablation, and error review, read [`../notebooks/palmer_penguins_end_to_end.ipynb`](../notebooks/palmer_penguins_end_to_end.ipynb) and [`../reports/palmer_penguins_model_card.md`](../reports/palmer_penguins_model_card.md).
