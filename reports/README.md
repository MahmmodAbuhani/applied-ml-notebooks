# Reports And Model Cards

These short reports summarize the main notebook conclusions for reviewers who want the result, boundary, and limitation before reading every notebook cell.

| Report | Notebook | Why it matters |
| --- | --- | --- |
| [`bank_marketing_case_study.md`](bank_marketing_case_study.md) | `bank_marketing_response_model.ipynb` | Flagship case-study summary of the business question, leakage boundary, source-order validation, fixed-budget policy readout, and stress-test limitation. |
| [`bank_marketing_response_model_card.md`](bank_marketing_response_model_card.md) | `bank_marketing_response_model.ipynb` | Messy real-data response modeling with class imbalance, missing-like values, checksum verification, leakage exclusion, early-only selection, and lift analysis. |
| [`palmer_penguins_model_card.md`](palmer_penguins_model_card.md) | `palmer_penguins_end_to_end.ipynb` | Full public-CSV workflow from source ingestion to holdout evaluation, interpretation, and feature ablation. |
| [`digits_knn_model_card.md`](digits_knn_model_card.md) | `knn_classification_project.ipynb` | Clear validation-safe preprocessing with `Pipeline`. |
| [`breast_cancer_tree_model_card.md`](breast_cancer_tree_model_card.md) | `decision_tree_classifier.ipynb` | Train-only model selection and tree-based interpretation. |

These are portfolio model cards, not production governance documents.

The flagship also has a versioned [`executed HTML snapshot`](evidence/bank_marketing_executed.html), external figure assets, and a [`provenance manifest`](evidence/bank_marketing_provenance.json). Download or open the HTML locally. It is a historical execution witness for the attested source commit and pinned public input, not a live report or deployed model.
