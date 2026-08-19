# Model Card: Palmer Penguins Species Classifier

## Intended Use

Demonstrate an end-to-end public-CSV classification workflow using the Palmer Penguins dataset. The model predicts penguin species from island, bill measurements, flipper length, body mass, sex, and year.

The repository also includes a local Streamlit demo and a CLI demo built from the same public-data boundary. The demo shows inputs, class probabilities, training-data context, feature contributions, and model limits; it is not a wildlife field-identification tool or a production decision system.

## Data

- Source: `palmerpenguins` public CSV.
- License: CC0-1.0 in the source repository.
- Rows used after dropping missing modeling values: `333`.
- Target classes: Adelie, Chinstrap, Gentoo.

## Model And Evaluation

- Selected model: Logistic Regression.
- Selection method: 5-fold stratified cross-validation on the training split using macro F1.
- Holdout sample size: `84`.
- Holdout accuracy: `0.988`.
- Holdout balanced accuracy: `0.991`.
- Holdout macro F1: `0.986`.
- Holdout weighted F1: `0.988`.

### Repeated Training-Only Check

The selected pipeline was evaluated with 5-fold stratified cross-validation repeated 3 times on the 249-row training split. This readout describes metric spread across training-only resamples; the untouched 84-row holdout above remains the primary result.

| Metric | Mean | Standard deviation |
| --- | ---: | ---: |
| Accuracy | `0.993` | `0.010` |
| Balanced accuracy | `0.989` | `0.016` |
| Macro F1 | `0.992` | `0.012` |

## Key Interpretation

Permutation importance on the holdout split identifies `bill_length_mm` as the strongest feature, followed by `bill_depth_mm`. These are descriptive model signals for this curated dataset, not evidence of biological causality.

## Robustness Check

The feature-ablation section compares the full feature set with morphology-only and island-only variants on the same holdout split. This makes the shortcut-risk question explicit, but it is a diagnostic comparison rather than an independent generalization estimate. `island` and `year` are collection-context features, while bill, flipper, and body-mass measurements are the morphology variables used in this dataset.

## Limitations

- The dataset is small and curated for teaching.
- The model is not intended for biological field deployment.
- The demo fits a model from the public CSV at runtime and does not represent a production service.
- The hosted Streamlit demo is documented with a dated public-commit check in [`demo/README.md`](../demo/README.md). It remains an educational interface, not a production service or field tool. The repository also includes a browser-local static explorer.
- The public CSV is loaded from GitHub at runtime, so execution depends on network availability.
- Island and year are useful for this dataset, but they should be treated cautiously outside the original Palmer Archipelago sampling context.
