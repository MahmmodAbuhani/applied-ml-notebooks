"""Shared figure styling for the notebook portfolio."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt

ACCENT = "#2F6F73"
HIGHLIGHT = "#C2703D"
MUTED = "#5C6B8A"
CAPTION = "#6B7280"


def apply_portfolio_style() -> None:
    """Apply the portfolio matplotlib style bundled with the package."""

    style_path = Path(__file__).with_name("portfolio.mplstyle")
    plt.style.use(style_path)


def save_figure(fig, name: str, *, project_root: Path | None = None) -> Path:
    """Save a curated preview figure under assets/figures."""

    evidence_output_dir = os.environ.get("ML_PORTFOLIO_FIGURE_DIR")
    if evidence_output_dir:
        output_dir = Path(evidence_output_dir)
    else:
        root = Path.cwd() if project_root is None else Path(project_root)
        output_dir = root / "assets" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    return output_path
