"""Ranking, lift, and source-order validation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


DEFAULT_BUDGET_SHARES = (0.01, 0.05, 0.10, 0.20, 0.30)


@dataclass(frozen=True)
class SourceOrderSplit:
    """Frozen source-order outer split indices."""

    early_indices: np.ndarray
    late_indices: np.ndarray


def source_order_split_indices(
    n_samples: int,
    *,
    train_fraction: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Return early and late row indices using a deterministic source-order cutoff."""

    if n_samples < 2:
        raise ValueError("source-order split requires at least two rows")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    cutoff = int(n_samples * train_fraction)
    if cutoff <= 0 or cutoff >= n_samples:
        raise ValueError("source-order split must leave rows on both sides of the cutoff")

    indices = np.arange(n_samples)
    return indices[:cutoff], indices[cutoff:]


def expanding_window_splits(
    n_samples: int,
    *,
    n_splits: int = 4,
    min_train_size: int | None = None,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield expanding-window train/validation splits over source-ordered rows."""

    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if n_samples < 2:
        raise ValueError("expanding-window splits require at least two rows")

    resolved_min_train_size = n_samples // 2 if min_train_size is None else min_train_size
    if resolved_min_train_size < 1:
        raise ValueError("min_train_size must be at least 1")
    if resolved_min_train_size >= n_samples:
        raise ValueError("min_train_size must be smaller than n_samples")

    validation_indices = np.arange(resolved_min_train_size, n_samples)
    if len(validation_indices) < n_splits:
        raise ValueError("expanding-window splits need at least one validation row per split")

    for validation_index in np.array_split(validation_indices, n_splits):
        if len(validation_index) == 0:
            raise ValueError("expanding-window splits need at least one validation row per split")
        train_index = np.arange(0, int(validation_index[0]))
        if len(train_index) < resolved_min_train_size:
            raise ValueError("train split is smaller than min_train_size")
        yield train_index, validation_index


def _ranked_frame(y_true: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> pd.DataFrame:
    y_array = np.asarray(y_true)
    score_array = np.asarray(scores)
    if y_array.shape[0] != score_array.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    if y_array.shape[0] == 0:
        raise ValueError("ranking metrics require at least one row")

    return pd.DataFrame({"target": y_array, "score": score_array}).sort_values(
        "score",
        ascending=False,
        kind="mergesort",
    )


def average_precision_lift(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float:
    """Return average precision divided by the target prevalence."""

    ranked = _ranked_frame(y_true, scores)
    base_rate = float(ranked["target"].mean())
    if base_rate <= 0:
        return float("nan")
    return float(average_precision_score(ranked["target"], ranked["score"]) / base_rate)


def budget_contact_count(n_samples: int, budget_share: float) -> int:
    """Return the deterministic contact count for a fixed capacity share."""

    if n_samples < 1:
        raise ValueError("budget policy requires at least one scored row")
    if not 0 < budget_share <= 1:
        raise ValueError("budget_share must be greater than 0 and at most 1")
    return min(n_samples, max(1, int(ceil(n_samples * budget_share))))


def fixed_budget_table(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    budget_shares: Sequence[float] = DEFAULT_BUDGET_SHARES,
) -> pd.DataFrame:
    """Summarize response concentration at fixed contact-budget shares."""

    ranked = _ranked_frame(y_true, scores)
    total_responders = float(ranked["target"].sum())
    base_rate = float(ranked["target"].mean())
    rows = []

    for budget_share in budget_shares:
        contacts = budget_contact_count(len(ranked), float(budget_share))
        contacted = ranked.head(contacts)
        responders = int(contacted["target"].sum())
        response_rate = float(contacted["target"].mean())
        rows.append(
            {
                "budget_share": float(budget_share),
                "contacts": int(contacts),
                "responders": responders,
                "response_rate": response_rate,
                "lift_vs_base": response_rate / base_rate if base_rate > 0 else float("nan"),
                "responders_captured_share": responders / total_responders
                if total_responders > 0
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


def cumulative_gains_frame(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    label: str,
) -> pd.DataFrame:
    """Return a cumulative gains frame for a scored binary ranking."""

    ranked = _ranked_frame(y_true, scores).reset_index(drop=True)
    total_responders = float(ranked["target"].sum())
    ranked["share_contacted"] = (np.arange(1, len(ranked) + 1)) / len(ranked)
    if total_responders > 0:
        ranked["responders_captured_share"] = ranked["target"].cumsum() / total_responders
    else:
        ranked["responders_captured_share"] = 0.0
    ranked["validation_design"] = label
    return ranked


def calibration_by_score_band(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return deterministic score-ordered calibration bands."""

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    ranked = _ranked_frame(y_true, scores).reset_index(drop=True)
    chunks = [chunk for chunk in np.array_split(ranked.index.to_numpy(), min(n_bins, len(ranked))) if len(chunk)]
    rows = []
    for band, chunk in enumerate(chunks, start=1):
        band_frame = ranked.iloc[chunk]
        rows.append(
            {
                "score_band": band,
                "rows": int(len(band_frame)),
                "mean_score": float(band_frame["score"].mean()),
                "response_rate": float(band_frame["target"].mean()),
                "min_score": float(band_frame["score"].min()),
                "max_score": float(band_frame["score"].max()),
            }
        )
    return pd.DataFrame(rows)
