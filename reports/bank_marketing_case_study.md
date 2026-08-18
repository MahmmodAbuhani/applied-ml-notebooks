# Case Study: Bank Marketing Response Modeling

## Question

Can a bank rank prospective marketing contacts before a call using only information available before that call?

This case study is not a campaign recommendation. It is a validation exercise on messy public data. The workflow defines the data boundary, blocks post-call leakage, uses source-order validation, reports fixed capacity readouts, and does not turn a weak stress test into a business claim.

## Data Boundary

The notebook uses UCI Bank Marketing `bank-additional-full.csv`, collected from Portuguese direct-marketing campaigns between 2008 and 2010. The target is whether the client subscribed to a term deposit.

The key excluded feature is `duration`, the length of the marketing call. It is known only after contact, so it cannot support pre-call ranking. A diagnostic inside the early segment shows why the exclusion matters: balanced logistic regression has mean early-fold AP `0.084` without `duration` and `0.427` with `duration`.

## Validation Design

The notebook uses a deterministic forward protocol:

1. Freeze the earliest 75% of source-order rows as the development segment.
2. Hold out the latest 25% as one untouched order-based temporal stress test.
3. Inside the early segment, use four expanding-window folds.
4. Fit preprocessing inside each fold through scikit-learn pipelines.
5. Select the model recipe by mean fold average precision.
6. Refit the selected recipe on all early rows.
7. Score the late segment once.

The public CSV has row order but no complete row-level timestamp, so this is not a formal timestamped deployment validation.

## Result

| Evidence | Result | Interpretation |
| --- | ---: | --- |
| Selected recipe | Class-weighted random forest, `min_samples_leaf=50`, `max_features=0.5` | Chosen by the predeclared mean-fold AP rule; not meaningful superiority |
| Early mean-fold AP and lift | AP `0.108`; AP lift `1.09x` | Unstable across folds: AP `0.046` to `0.266`, prevalence `0.055` to `0.176` |
| Early pooled OOF ranking | AP `0.078`; ROC AUC `0.470` | Below pooled base prevalence `0.087` |
| Late stress AP | `0.238` | Below late base response rate `0.258` |
| Late stress ROC AUC | `0.457` | Weak late-period ranking |
| Late top 1% lift | `1.69x` | Exploratory top-1% concentration, not policy evidence |
| Late top 10% lift | `0.79x` | Not strong enough for a broad contact policy |

The selected forest's fold AP values are `0.061`, `0.056`, `0.046`, and `0.266`, with corresponding fold prevalences of `0.061`, `0.055`, `0.056`, and `0.176`. Its AP lift over each fold's base rate is `1.01x`, `1.02x`, `0.83x`, and `1.51x`. The late holdout remains untouched by this selection analysis.

## Policy Layer

The decision layer is budget-only. It asks how fixed contact budgets behave at 1%, 5%, 10%, 20%, and 30% of scored records. It does not invent contact costs, revenue, value per conversion, or campaign returns.

The default reviewer readout is the top 10% budget. Under the late stress test, that budget has response rate `0.203` versus a late base response rate of `0.258`, so the model is not a defensible broad contact rule. The `1.69x` value is an exploratory top-1% concentration. It is not policy evidence without campaign economics, stability analysis, and a further untouched evaluation.

## What A Reviewer Should Notice

- `duration` is excluded from the pre-call model.
- The late rows are not used for preprocessing, model family selection, hyperparameter selection, policy setup, or calibration treatment.
- F1 is not used as the business policy.
- Calibration is reported as a diagnostic, not as an automatically fitted transformation.
- The conclusion follows the negative broad-policy evidence rather than the selected model label.

## Limitations

This is a portfolio case study, not a production targeting system. A real deployment would need verified timestamps, campaign economics, fairness review, monitoring, operational constraints, compliance review, and a policy owner. None of those are claimed here.
