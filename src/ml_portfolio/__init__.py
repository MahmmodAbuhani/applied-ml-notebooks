"""Reusable helpers for the machine-learning notebook portfolio."""

from .data_sources import assert_min_bytes, download_bytes, download_bytes_with_retry
from .evaluation import (
    classification_summary,
    regression_summary,
    repeated_classification_summary,
)
from .penguins import PenguinSample, fit_penguin_model, predict_penguin_sample
from .plotting import ACCENT, HIGHLIGHT, apply_portfolio_style, save_figure

__all__ = [
    "ACCENT",
    "HIGHLIGHT",
    "assert_min_bytes",
    "apply_portfolio_style",
    "classification_summary",
    "download_bytes",
    "download_bytes_with_retry",
    "fit_penguin_model",
    "PenguinSample",
    "predict_penguin_sample",
    "regression_summary",
    "repeated_classification_summary",
    "save_figure",
]
