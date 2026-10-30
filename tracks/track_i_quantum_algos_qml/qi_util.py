"""Utility functions for Track I: Quantum Algorithms and QML.

Provides CSV persistence, markdown table formatting without external dependencies,
matplotlib figure saving, and a wall-clock execution Timer context manager.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root and track root are on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import numpy as np


class Timer:
    """Context manager measuring wall-clock execution time with high precision."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    """Write a list of row dictionaries to a CSV file.

    Creates parent directories automatically if they do not exist.

    Args:
        path: Destination file path.
        rows: List of row dictionaries.
        columns: Optional list of fieldnames. Defaults to keys of the first row.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(p, "w", newline="", encoding="utf-8") as f:
            if columns:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
        return

    fieldnames = columns if columns is not None else list(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_markdown(
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
    float_fmt: str = "{:.4g}",
) -> str:
    """Format a list of dictionaries as a Markdown table without external libraries.

    Args:
        rows: List of row dictionaries.
        columns: Optional list of column keys to render.
        float_fmt: Python format string applied to floating point values.

    Returns:
        Formatted GitHub-flavored Markdown table string.
    """
    if not rows:
        return "*Empty table.*"

    cols = columns if columns is not None else list(rows[0].keys())
    header_line = "| " + " | ".join(cols) + " |"
    sep_line = "| " + " | ".join(["---"] * len(cols)) + " |"
    data_lines: list[str] = []

    for r in rows:
        formatted_row: list[str] = []
        for c in cols:
            val = r.get(c, "")
            if isinstance(val, (float, np.floating)):
                formatted_row.append(float_fmt.format(float(val)))
            else:
                formatted_row.append(str(val))
        data_lines.append("| " + " | ".join(formatted_row) + " |")

    return "\n".join([header_line, sep_line] + data_lines)


def save_fig(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save a matplotlib figure at standard DPI, ensuring parent directories exist.

    Args:
        fig: Matplotlib Figure instance.
        path: Destination image path.
        dpi: Dots-per-inch resolution (default: 150).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=dpi)
    plt.close(fig)

