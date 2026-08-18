# Model Card: Breast Cancer Tree-Based Classifier

## Intended Use

Demonstrate tree-based classifier comparison, train-only model selection, final holdout evaluation, and feature-importance interpretation on a compact tabular benchmark.

## Data

- Source: `sklearn.datasets.load_breast_cancer()`.
- Rows: `569`.
- Features: `30` tumor measurement features.
- Target classes: malignant and benign.

## Model And Evaluation

- Selected model: Random Forest.
- Selection method: 5-fold stratified cross-validation on the training split using balanced accuracy.
- Holdout sample size: `114`.
- Holdout accuracy: `0.947`.
- Holdout balanced accuracy: `0.943`.
- Holdout macro F1: `0.943`.

## Key Interpretation

Top random-forest feature importances include perimeter, area, concave points, and radius measurements. The interpretation is useful for model understanding, not a substitute for clinical validation.

## Limitations

- This is a benchmark modeling exercise, not a medical diagnostic system.
- Reported metrics come from one fixed holdout split.
- Feature importances describe the fitted model, not causal clinical relationships.
