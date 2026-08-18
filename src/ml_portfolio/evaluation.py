"""Small evaluation helpers shared by portfolio notebooks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate


def classification_summary(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    model_name: str,
) -> dict[str, float | int | str]:
    """Return standard classification metrics as a serializable dictionary."""

    return {
        "model": model_name,
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def repeated_classification_summary(
    estimator: object,
    X: object,
    y: Sequence[int] | np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Summarize repeated stratified cross-validation with mean and spread.

    This is a bounded uncertainty readout for small educational benchmarks.
    It does not replace a final untouched holdout evaluation.
    """

    if n_splits < 2 or n_repeats < 1:
        raise ValueError("n_splits must be at least 2 and n_repeats at least 1")

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=splitter,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "f1_macro",
        },
        n_jobs=None,
    )

    result: dict[str, float | int] = {
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "n_folds": n_splits * n_repeats,
    }
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        values = scores[f"test_{metric}"]
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values, ddof=1))
    return result


def regression_summary(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    model_name: str,
) -> dict[str, float | int | str]:
    """Return standard regression metrics as a serializable dictionary."""

    return {
        "model": model_name,
        "n_samples": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
