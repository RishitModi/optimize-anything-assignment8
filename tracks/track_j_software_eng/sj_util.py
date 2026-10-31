"""Utility functions for Track J: CSV persistence, Markdown table rendering, and statistics."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

# Ensure repository root and track root are placed on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def write_csv(
    path: str | Path,
    rows: Sequence[dict[str, Any] | Sequence[Any]],
    columns: Sequence[str],
) -> None:
    """Write rows to a CSV file at the specified path.

    Parameters
    ----------
    path : str | Path
        Target filesystem path for the output CSV file.
    rows : Sequence[dict[str, Any] | Sequence[Any]]
        Row data represented either as dictionaries or sequences matching columns.
    columns : Sequence[str]
        Ordered column header names.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(columns))
        cap = len(rows) + 5
        steps = 0
        for row in rows:
            steps += 1
            if steps > cap:
                break
            if isinstance(row, dict):
                writer.writerow([row.get(col, "") for col in columns])
            else:
                writer.writerow(list(row))


def to_markdown(
    rows: Sequence[dict[str, Any] | Sequence[Any]],
    columns: Sequence[str],
    float_fmt: str = "{:.4g}",
) -> str:
    """Format structured rows into a GitHub-flavored Markdown table.

    Parameters
    ----------
    rows : Sequence[dict[str, Any] | Sequence[Any]]
        Row data to render in the table body.
    columns : Sequence[str]
        Column headers for the table.
    float_fmt : str
        Format specifier for floating-point values.

    Returns
    -------
    str
        Markdown string containing table headers, alignment dividers, and data rows.
    """
    headers = [str(c) for c in columns]
    divider = ["---"] * len(headers)
    lines: list[str] = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]

    cap = len(rows) + 5
    steps = 0
    for row in rows:
        steps += 1
        if steps > cap:
            break
        cells: list[str] = []
        if isinstance(row, dict):
            raw_vals = [row.get(col, "") for col in columns]
        else:
            raw_vals = list(row)

        for v in raw_vals:
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def median(values: Sequence[float | int]) -> float:
    """Compute the statistical median of a non-empty sequence of numeric values.

    Parameters
    ----------
    values : Sequence[float | int]
        Numeric measurements.

    Returns
    -------
    float
        Median value, or 0.0 if the sequence is empty.
    """
    if not values:
        return 0.0
    return float(statistics.median(values))

