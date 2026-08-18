"""Reusable Bank Marketing data and forward-validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
import zipfile

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_portfolio.data_sources import assert_min_bytes, assert_sha256, download_bytes_with_retry
from ml_portfolio.ranking import (
    DEFAULT_BUDGET_SHARES,
    SourceOrderSplit,
    average_precision_lift,
    expanding_window_splits,
    fixed_budget_table,
    source_order_split_indices,
)


RANDOM_STATE = 42
BANK_DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
BANK_DATA_SHA256 = "e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4"
BANK_MIN_BYTES = 1_000_000
INNER_ZIP = "bank-additional.zip"
CSV_PATH = "bank-additional/bank-additional-full.csv"
LEAKY_FEATURE = "duration"
TARGET_COLUMN = "target"

EXPECTED_COLUMNS = {
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
}

NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays_clean",
    "pdays_was_999",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]


@dataclass(frozen=True)
class BankCandidateSpec:
    """A named model recipe considered during early-only selection."""

    name: str
    estimator: BaseEstimator
    params: dict[str, object] | None = None


@dataclass(frozen=True)
class BankSelectionResult:
    """Selected model recipe and early-fold diagnostics."""

    selected_name: str
    selected_params: dict[str, object]
    cv_results: pd.DataFrame
    fold_results: pd.DataFrame
    early_oof_scores: np.ndarray
    early_oof_labels: np.ndarray
    early_oof_base_positive_rate: float
    early_oof_average_precision: float
    early_oof_roc_auc: float


@dataclass(frozen=True)
class BankForwardValidationResult:
    """Forward-validation output with one untouched late-segment evaluation."""

    outer_split: SourceOrderSplit
    selection: BankSelectionResult
    policy_config: dict[str, object]
    late_metrics: dict[str, float | int | str]
    late_budget_table: pd.DataFrame
    late_scores: np.ndarray
    late_labels: np.ndarray
    fitted_model: Pipeline


def load_bank_marketing_data(url: str = BANK_DATA_URL) -> pd.DataFrame:
    """Load and verify the public UCI Bank Marketing CSV from its nested ZIP."""

    archive_bytes = download_bytes_with_retry(url, attempts=3, timeout=30)
    assert_min_bytes(archive_bytes, min_bytes=BANK_MIN_BYTES, label="UCI Bank Marketing ZIP")
    if url == BANK_DATA_URL:
        assert_sha256(
            archive_bytes,
            expected_sha256=BANK_DATA_SHA256,
            label="UCI Bank Marketing ZIP",
        )

    outer_zip = zipfile.ZipFile(BytesIO(archive_bytes))
    inner_zip = zipfile.ZipFile(BytesIO(outer_zip.read(INNER_ZIP)))
    with inner_zip.open(CSV_PATH) as csv_file:
        data = pd.read_csv(csv_file, sep=";")
    validate_bank_columns(data)
    return data


def validate_bank_columns(data: pd.DataFrame) -> None:
    """Raise a clear error if the Bank Marketing file shape changed."""

    missing = sorted(EXPECTED_COLUMNS.difference(data.columns))
    if missing:
        raise ValueError(f"Bank Marketing data is missing required columns: {', '.join(missing)}")


def prepare_bank_modeling_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    """Return pre-call feature matrix, target vector, and feature role metadata."""

    validate_bank_columns(data)
    bank = data.replace("unknown", np.nan).copy()
    bank["pdays_was_999"] = (bank["pdays"] == 999).astype(int)
    bank["pdays_clean"] = bank["pdays"].mask(bank["pdays"] == 999, np.nan)
    bank[TARGET_COLUMN] = (bank["y"] == "yes").astype(int)

    feature_columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    feature_roles = {
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "excluded_features": [LEAKY_FEATURE],
    }
    return bank[feature_columns], bank[TARGET_COLUMN], feature_roles


def make_bank_preprocessor(
    *,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Build preprocessing for Bank Marketing pre-call features."""

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(categorical_features),
            ),
        ],
        verbose_feature_names_out=True,
    )


def build_bank_pipeline(
    estimator: BaseEstimator,
    *,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> Pipeline:
    """Build a leakage-safe pre-call Bank Marketing modeling pipeline."""

    forbidden = set(numeric_features).union(categorical_features).intersection({LEAKY_FEATURE})
    if forbidden:
        raise ValueError(f"Pre-call pipeline cannot include leakage features: {sorted(forbidden)}")

    return Pipeline(
        steps=[
            (
                "preprocess",
                make_bank_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                ),
            ),
            ("model", clone(estimator)),
        ]
    )


def default_bank_candidate_specs(random_state: int = RANDOM_STATE) -> list[BankCandidateSpec]:
    """Return small, reviewable Bank Marketing candidate recipes."""

    return [
        BankCandidateSpec("Dummy majority", DummyClassifier(strategy="most_frequent")),
        BankCandidateSpec(
            "Balanced logistic regression C=0.3",
            LogisticRegression(max_iter=1000, class_weight="balanced", C=0.3, random_state=random_state),
            {"family": "logistic_regression", "C": 0.3},
        ),
        BankCandidateSpec(
            "Balanced logistic regression C=1.0",
            LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=random_state),
            {"family": "logistic_regression", "C": 1.0},
        ),
        BankCandidateSpec(
            "Balanced logistic regression C=3.0",
            LogisticRegression(max_iter=1000, class_weight="balanced", C=3.0, random_state=random_state),
            {"family": "logistic_regression", "C": 3.0},
        ),
        BankCandidateSpec(
            "Class-weighted random forest leaf=20 features=sqrt",
            RandomForestClassifier(
                n_estimators=50,
                min_samples_leaf=20,
                max_features="sqrt",
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
            {"family": "random_forest", "min_samples_leaf": 20, "max_features": "sqrt"},
        ),
        BankCandidateSpec(
            "Class-weighted random forest leaf=50 features=sqrt",
            RandomForestClassifier(
                n_estimators=50,
                min_samples_leaf=50,
                max_features="sqrt",
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
            {"family": "random_forest", "min_samples_leaf": 50, "max_features": "sqrt"},
        ),
        BankCandidateSpec(
            "Class-weighted random forest leaf=20 features=0.5",
            RandomForestClassifier(
                n_estimators=50,
                min_samples_leaf=20,
                max_features=0.5,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
            {"family": "random_forest", "min_samples_leaf": 20, "max_features": 0.5},
        ),
        BankCandidateSpec(
            "Class-weighted random forest leaf=50 features=0.5",
            RandomForestClassifier(
                n_estimators=50,
                min_samples_leaf=50,
                max_features=0.5,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
            {"family": "random_forest", "min_samples_leaf": 50, "max_features": 0.5},
        ),
    ]


def default_policy_config() -> dict[str, object]:
    """Return fixed contact-budget policy configuration."""

    return {
        "policy_type": "fixed_contact_budget",
        "primary_budget_share": 0.10,
        "budget_shares": DEFAULT_BUDGET_SHARES,
    }


def select_bank_recipe(
    X_early: pd.DataFrame,
    y_early: pd.Series,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    candidate_specs: Sequence[BankCandidateSpec] | None = None,
    n_splits: int = 4,
    min_train_size: int | None = None,
) -> BankSelectionResult:
    """Select the model recipe using only early source-order rows."""

    specs = list(candidate_specs) if candidate_specs is not None else default_bank_candidate_specs()
    splits = list(
        expanding_window_splits(
            len(X_early),
            n_splits=n_splits,
            min_train_size=min_train_size,
        )
    )
    rows = []
    all_fold_metric_rows = []
    score_by_name: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for spec in specs:
        fold_scores = np.full(len(X_early), np.nan)
        fold_labels = np.full(len(X_early), np.nan)
        fold_metric_rows = []
        for fold, (train_index, validation_index) in enumerate(splits, start=1):
            pipeline = build_bank_pipeline(
                spec.estimator,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            )
            pipeline.fit(X_early.iloc[train_index], y_early.iloc[train_index])
            probabilities = pipeline.predict_proba(X_early.iloc[validation_index])[:, 1]
            validation_labels = y_early.iloc[validation_index].to_numpy()
            fold_scores[validation_index] = probabilities
            fold_labels[validation_index] = validation_labels
            fold_metric_rows.append(
                {
                    "model": spec.name,
                    "fold": fold,
                    "train_rows": int(len(train_index)),
                    "validation_rows": int(len(validation_index)),
                    "validation_positive_rate": float(np.mean(validation_labels)),
                    "average_precision": average_precision_score(validation_labels, probabilities),
                    "ap_lift_over_base_rate": average_precision_lift(
                        validation_labels,
                        probabilities,
                    ),
                    "roc_auc": roc_auc_score(validation_labels, probabilities)
                    if len(np.unique(validation_labels)) == 2
                    else np.nan,
                }
            )

        fold_metrics = pd.DataFrame(fold_metric_rows)
        all_fold_metric_rows.extend(fold_metric_rows)
        rows.append(
            {
                "model": spec.name,
                "params": spec.params or {},
                "cv_average_precision_mean": float(fold_metrics["average_precision"].mean()),
                "cv_average_precision_std": float(fold_metrics["average_precision"].std(ddof=0)),
                "cv_ap_lift_over_base_mean": float(
                    fold_metrics["ap_lift_over_base_rate"].mean(skipna=True)
                ),
                "cv_ap_lift_over_base_std": float(
                    fold_metrics["ap_lift_over_base_rate"].std(ddof=0, skipna=True)
                ),
                "cv_roc_auc_mean": float(fold_metrics["roc_auc"].mean(skipna=True)),
            }
        )
        observed = ~np.isnan(fold_scores)
        score_by_name[spec.name] = (
            fold_scores[observed],
            fold_labels[observed].astype(int),
        )

    cv_results = pd.DataFrame(rows).sort_values(
        ["cv_average_precision_mean", "model"],
        ascending=[False, True],
    ).reset_index(drop=True)
    selected_name = str(cv_results.iloc[0]["model"])
    selected_spec = next(spec for spec in specs if spec.name == selected_name)
    early_oof_scores, early_oof_labels = score_by_name[selected_name]
    fold_results = pd.DataFrame(all_fold_metric_rows)
    early_oof_base_positive_rate = float(np.mean(early_oof_labels))
    early_oof_average_precision = float(
        average_precision_score(early_oof_labels, early_oof_scores)
    )
    early_oof_roc_auc = (
        float(roc_auc_score(early_oof_labels, early_oof_scores))
        if len(np.unique(early_oof_labels)) == 2
        else float("nan")
    )

    return BankSelectionResult(
        selected_name=selected_name,
        selected_params=dict(selected_spec.params or {}),
        cv_results=cv_results,
        fold_results=fold_results,
        early_oof_scores=early_oof_scores,
        early_oof_labels=early_oof_labels,
        early_oof_base_positive_rate=early_oof_base_positive_rate,
        early_oof_average_precision=early_oof_average_precision,
        early_oof_roc_auc=early_oof_roc_auc,
    )


def run_forward_bank_validation(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    candidate_specs: Sequence[BankCandidateSpec] | None = None,
    train_fraction: float = 0.75,
    n_splits: int = 4,
    min_train_size: int | None = None,
) -> BankForwardValidationResult:
    """Select on early rows, then evaluate once on the untouched late segment."""

    early_indices, late_indices = source_order_split_indices(len(X), train_fraction=train_fraction)
    split = SourceOrderSplit(early_indices=early_indices, late_indices=late_indices)
    X_early = X.iloc[early_indices]
    y_early = y.iloc[early_indices]
    X_late = X.iloc[late_indices]
    y_late = y.iloc[late_indices]

    selection = select_bank_recipe(
        X_early,
        y_early,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        candidate_specs=candidate_specs,
        n_splits=n_splits,
        min_train_size=min_train_size,
    )
    specs = list(candidate_specs) if candidate_specs is not None else default_bank_candidate_specs()
    selected_spec = next(spec for spec in specs if spec.name == selection.selected_name)
    fitted_model = build_bank_pipeline(
        selected_spec.estimator,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    fitted_model.fit(X_early, y_early)
    late_scores = fitted_model.predict_proba(X_late)[:, 1]
    late_predictions = (late_scores >= 0.5).astype(int)
    late_labels = y_late.to_numpy()
    policy = default_policy_config()
    late_budget_table = fixed_budget_table(
        late_labels,
        late_scores,
        budget_shares=policy["budget_shares"],
    )

    late_metrics = {
        "validation_design": "Order-based temporal stress test",
        "model": selection.selected_name,
        "n_samples": int(len(late_labels)),
        "average_precision": float(average_precision_score(late_labels, late_scores)),
        "roc_auc": float(roc_auc_score(late_labels, late_scores))
        if len(np.unique(late_labels)) == 2
        else float("nan"),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(late_labels, late_predictions))
        if len(np.unique(late_labels)) == 2
        else float("nan"),
        "base_positive_rate": float(np.mean(late_labels)),
    }

    return BankForwardValidationResult(
        outer_split=split,
        selection=selection,
        policy_config=policy,
        late_metrics=late_metrics,
        late_budget_table=late_budget_table,
        late_scores=late_scores,
        late_labels=late_labels,
        fitted_model=fitted_model,
    )
