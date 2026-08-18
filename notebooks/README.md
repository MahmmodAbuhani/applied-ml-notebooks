# Notebook Reading Guide

This folder contains seven stripped Jupyter notebooks. They are meant to be read on GitHub first and executed locally when a reviewer wants to reproduce the results.

## Recommended Order

| Role | Notebook | Why read it |
| --- | --- | --- |
| Flagship | [`bank_marketing_response_model.ipynb`](bank_marketing_response_model.ipynb) | Real public marketing data, checksum verification, missing-like values, class imbalance, leakage exclusion, source-order model selection, fixed budget readouts, calibration diagnosis, and an honest weak stress-test result. |
| Interactive companion | [`palmer_penguins_end_to_end.ipynb`](palmer_penguins_end_to_end.ipynb) | Compact public-CSV workflow that powers the local Streamlit and CLI demo. |
| Educational foundation | [`knn_classification_project.ipynb`](knn_classification_project.ipynb) | Validation-safe preprocessing with `Pipeline(StandardScaler(), KNeighborsClassifier())`, error review, and PCA caveats. |
| Educational foundation | [`decision_tree_classifier.ipynb`](decision_tree_classifier.ipynb) | Tree-versus-forest comparison with train-only model selection and feature-importance interpretation. |
| Educational foundation | [`random_forest_tree_models.ipynb`](random_forest_tree_models.ipynb) | Tree-path inspection, voting, and educational out-of-fold stacking examples. |
| Educational foundation | [`regression_modeling_project.ipynb`](regression_modeling_project.ipynb) | Validation-based model choice, a single final holdout evaluation, residual review, and coefficient interpretation on a real regression benchmark. |
| Educational foundation | [`kmeans_clustering.ipynb`](kmeans_clustering.ipynb) | Unsupervised model selection, cluster profiling, PCA visualization, and labels reserved for post-hoc benchmarking. |

## What To Look For

- Train/test or early/late separation before model selection.
- Leakage boundaries stated where they matter, especially `duration` in Bank Marketing.
- Preprocessing inside validation folds.
- Reusable helpers imported from `src/ml_portfolio` when logic becomes shared code.
- Metrics matched to the task: average precision and lift for response ranking, macro F1 for classification, silhouette and adjusted Rand index for clustering, RMSE/MAE/R² for regression.
- Conclusions that state both the result and the limitation.
- Stripped notebook outputs in source control, with execution handled by local commands and CI.

## Accessibility Notes

Each notebook starts with purpose, dataset, method, metric, and headline takeaway. Code is split into narrated sections so a reviewer can follow the workflow without running the notebook.
