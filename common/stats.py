"""Multi-seed stochastic experiment execution and descriptive statistics."""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable, Dict, List, Sequence, Union

from common.seeding import set_global_seed


def calculate_summary_stats(values: Sequence[float]) -> Dict[str, float]:
    """Calculate mean, sample standard deviation, minimum, and maximum for numeric results.

    Args:
        values: Sequence of float values obtained across runs.

    Returns:
        Dictionary containing 'mean', 'std', 'min', and 'max'.

    Raises:
        ValueError: If values sequence is empty.
    """
    if not values:
        raise ValueError("Cannot calculate summary statistics on an empty sequence.")

    val_list = [float(v) for v in values]
    n = len(val_list)

    mean_val = statistics.fmean(val_list) if hasattr(statistics, "fmean") else sum(val_list) / n
    std_val = statistics.stdev(val_list) if n > 1 else 0.0
    min_val = min(val_list)
    max_val = max(val_list)

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
    }


def run_multiseed_experiment(
    experiment_fn: Callable[[int], Union[float, int]],
    n_runs: int = 5,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Execute a stochastic experiment across multiple random seeds and aggregate metrics.

    For each run `i` in `[0, n_runs - 1]`, sets the global seed to `base_seed + i`,
    executes `experiment_fn(seed)`, and aggregates the scalar outcomes.

    Args:
        experiment_fn: Callable taking an integer seed and returning a numeric outcome metric.
        n_runs: Number of independent repetitions. Defaults to 5.
        base_seed: Starting seed value. Defaults to 42.

    Returns:
        Dictionary containing:
            - 'mean': float
            - 'std': float
            - 'min': float
            - 'max': float
            - 'values': list of float results from each run
            - 'seeds': list of integer seeds evaluated
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")

    seeds: List[int] = [base_seed + i for i in range(n_runs)]
    values: List[float] = []

    for seed in seeds:
        set_global_seed(seed)
        result = float(experiment_fn(seed))
        values.append(result)

    summary = calculate_summary_stats(values)
    summary["values"] = values
    summary["seeds"] = seeds
    return summary
