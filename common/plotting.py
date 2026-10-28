"""Standardized plotting helpers for benchmark results and convergence curves."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union


def _resolve_track_output_dir(track: str, base_results_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve and create output directory under results/<track>/.

    Handles track aliases such as 'track_a' or 'track_a_continuous'.

    Args:
        track: Track identifier or directory name.
        base_results_dir: Optional base path. Defaults to 'results' at repo root.

    Returns:
        Path to the resolved track results folder.
    """
    if base_results_dir is None:
        base_path = Path("results")
    else:
        base_path = Path(base_results_dir)

    target_dir = base_path / track
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def save_line_plot(
    data: Dict[str, Sequence[float]],
    track: str,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    x_values: Optional[Sequence[float]] = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    grid: bool = True,
    base_results_dir: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Path:
    """Save a multi-series line plot (e.g., convergence curve) as a PNG.

    Args:
        data: Mapping of series label to sequence of y-values.
        track: Subfolder name under results/ (e.g. 'track_a' or 'track_a_continuous').
        filename: Destination PNG filename (e.g. 'convergence_loss.png').
        title: Plot title.
        xlabel: Label for x-axis including units (e.g., 'Iteration (step count)').
        ylabel: Label for y-axis including units (e.g., 'Loss (L2 objective)').
        x_values: Optional sequence of x coordinates shared across series.
        x_scale: Scale for x-axis ('linear', 'log', etc.). Defaults to 'linear'.
        y_scale: Scale for y-axis ('linear', 'log', etc.). Defaults to 'linear'.
        grid: Whether to display background grid lines. Defaults to True.
        base_results_dir: Optional custom results base directory.
        dpi: Output image resolution. Defaults to 300.

    Returns:
        Path to the saved PNG image.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, y_vals in data.items():
        if x_values is not None:
            ax.plot(x_values, y_vals, label=label, linewidth=1.75)
        else:
            ax.plot(y_vals, label=label, linewidth=1.75)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    if grid:
        ax.grid(True, linestyle="--", alpha=0.6)

    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    out_dir = _resolve_track_output_dir(track, base_results_dir)
    if not filename.endswith(".png"):
        filename = f"{filename}.png"
    out_path = out_dir / filename

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_bar_chart(
    categories: Sequence[str],
    values: Union[Sequence[float], Dict[str, Sequence[float]]],
    track: str,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    yerr: Optional[Union[Sequence[float], Dict[str, Sequence[float]]]] = None,
    base_results_dir: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> Path:
    """Save a single or grouped bar chart as a PNG.

    Args:
        categories: Sequence of category labels on the x-axis.
        values: Sequence of numeric values (for single series) or dictionary of
            series label to values sequence (for grouped bar chart).
        track: Subfolder name under results/ (e.g. 'track_b' or 'track_b_discrete').
        filename: Destination PNG filename.
        title: Plot title.
        xlabel: Label for x-axis including units.
        ylabel: Label for y-axis including units (e.g., 'Runtime (seconds)').
        yerr: Optional error bar magnitudes (std dev or 95% CI).
        base_results_dir: Optional custom results base directory.
        dpi: Output image resolution. Defaults to 300.

    Returns:
        Path to the saved PNG image.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5))
    x_indices = np.arange(len(categories))

    if isinstance(values, dict):
        # Grouped bar chart
        n_series = len(values)
        width = 0.8 / max(n_series, 1)

        for idx, (label, series_vals) in enumerate(values.items()):
            offset = (idx - (n_series - 1) / 2) * width
            err = yerr[label] if (isinstance(yerr, dict) and label in yerr) else None
            ax.bar(
                x_indices + offset,
                series_vals,
                width=width,
                label=label,
                yerr=err,
                capsize=4 if err is not None else 0,
                alpha=0.9,
            )
        ax.legend(frameon=True, fontsize=9)
    else:
        # Single series bar chart
        err = yerr if (isinstance(yerr, (list, tuple, np.ndarray))) else None
        ax.bar(
            x_indices,
            values,
            width=0.55,
            yerr=err,
            capsize=4 if err is not None else 0,
            color="#2b5c8f",
            alpha=0.9,
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(categories, rotation=25 if len(categories) > 5 else 0, ha="right" if len(categories) > 5 else "center")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    fig.tight_layout()

    out_dir = _resolve_track_output_dir(track, base_results_dir)
    if not filename.endswith(".png"):
        filename = f"{filename}.png"
    out_path = out_dir / filename

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
