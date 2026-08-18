# Model Card: Handwritten Digits kNN Classifier

## Intended Use

Demonstrate validation-safe preprocessing for a multi-class image-feature classification task using `scikit-learn`'s handwritten digits dataset.

## Data

- Source: `sklearn.datasets.load_digits()`.
- Rows: `1,797`.
- Features: `64` flattened 8x8 grayscale pixel values.
- Target classes: digits `0` through `9`.

## Model And Evaluation

- Selected model: k-nearest neighbors with `k=5`.
- Preprocessing: `StandardScaler` inside a `Pipeline`.
- Selection method: 5-fold stratified cross-validation on the training split.
- Holdout sample size: `450`.
- Holdout accuracy: `0.964`.
- Holdout macro F1: `0.964`.

## Key Interpretation

The notebook is primarily a validation-hygiene artifact: scaling is learned inside each validation fold rather than before cross-validation. This avoids fold-level data leakage.

## Limitations

- The dataset is a compact benchmark, not a modern computer-vision task.
- PCA visualizations are interpretive two-dimensional views, not the final 64-feature decision surface.
