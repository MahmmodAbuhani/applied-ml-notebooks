# Model Card: Bank Marketing Response Model

## Intended Use

Demonstrate leakage-aware response modeling on a real public marketing dataset. The model ranks contacts by a pre-call score for the observed term-deposit subscription label.

This is a portfolio model card, not a production governance document.

## Data

- Source: UCI Bank Marketing, `bank-additional-full.csv`.
- Runtime archive: <https://archive.ics.uci.edu/static/public/222/bank+marketing.zip>
- Verified ZIP SHA-256: `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4`.
- License: CC BY 4.0 as listed by UCI.
- Rows: `41,188`.
- Target: `y == "yes"` means the client subscribed to a term deposit.
- Overall positive rate: approximately `0.113`.

## Leakage Boundary

`duration` is excluded because call duration is known only after the marketing call ends. Including it would answer a post-call explanation question, not the pre-call ranking question.

The notebook includes a diagnostic showing that early-fold AP rises from `0.084` without `duration` to `0.427` with `duration`.

## Model And Validation

- Selected recipe: class-weighted random forest.
- Selected parameters: `min_samples_leaf=50`, `max_features=0.5`, `n_estimators=50`.
- Outer split: earliest 75% of source-order rows for development, latest 25% for one stress-test evaluation.
- Inner selection: four expanding-window folds inside the early segment only.
- Primary selection metric: mean fold average precision.
- Policy readout: fixed budget shares at 1%, 5%, 10%, 20%, and 30%.
- Calibration: diagnostic only. No calibration transformation is fitted.

## Metrics

| Metric | Value |
| --- | ---: |
| Early selection mean AP | `0.108` |
| Early selection mean AP lift over fold base rate | `1.09x` |
| Early selection mean ROC AUC | `0.519` |
| Early selected-model fold AP range | `0.046` to `0.266` |
| Early selected-model fold prevalence range | `0.055` to `0.176` |
| Early pooled OOF AP | `0.078` |
| Early pooled OOF base prevalence | `0.087` |
| Early pooled OOF ROC AUC | `0.470` |
| Late stress AP | `0.238` |
| Late base response rate | `0.258` |
| Late stress ROC AUC | `0.457` |
| Late exploratory top-1% concentration | Lift `1.69x`; not policy evidence |
| Late top 10% lift | `0.79x` |
| Late top 10% response rate | `0.203` |

## Interpretation

The selected forest ranks first under the predeclared mean-fold AP rule, but the fold prevalence and AP ranges make that comparison unstable. It should not be read as meaningful superiority, especially because pooled early OOF AP is below pooled prevalence and pooled ROC AUC is below `0.50`. The late top 10% budget also underperforms the late base rate. The `1.69x` value is an exploratory top-1% concentration, not policy evidence without campaign economics, stability analysis, and a further untouched evaluation.

## Limitations And Non-Use

- Do not use this model for live customer targeting.
- Do not use it for credit eligibility, customer exclusion, or automated adverse decisions.
- The public dataset covers campaigns from 2008 to 2010 and may not represent current customers or current economic conditions.
- The stress test uses source-file order because the public CSV does not include complete row-level timestamps.
- A production system would need verified timestamps, decision costs, fairness review, monitoring, data contracts, and compliance approval.
