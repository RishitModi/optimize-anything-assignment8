"""Test suite minimization (Greedy and ILP Weighted Set Cover) and prioritization."""

from __future__ import annotations

from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence, Set

# Ensure repository root and track root are placed on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import scipy.optimize
import scipy.sparse


def select_greedy(
    cover: Mapping[str, Set[Any]],
    cost: Mapping[str, float],
    universe: Set[Any],
) -> list[str]:
    """Perform greedy weighted set cover to cover the universe of elements.

    Repeatedly picks the test offering the largest ratio of newly covered elements to test cost.

    Parameters
    ----------
    cover : Mapping[str, Set[Any]]
        Mapping from test identifier to set of elements covered.
    cost : Mapping[str, float]
        Mapping from test identifier to execution cost.
    universe : Set[Any]
        Target set of elements to cover.

    Returns
    -------
    list[str]
        Ordered list of selected test identifiers.
    """
    uncovered = set(universe)
    selected: list[str] = []
    selected_set: set[str] = set()

    max_iters = len(cover) + 5
    iters = 0

    while uncovered and iters < max_iters:
        iters += 1
        best_test: str | None = None
        best_ratio = -1.0

        for t, elements in cover.items():
            if t in selected_set:
                continue
            newly = len(elements & uncovered)
            if newly == 0:
                continue
            c = max(cost.get(t, 1e-6), 1e-9)
            ratio = newly / c
            if ratio > best_ratio:
                best_ratio = ratio
                best_test = t

        if best_test is None:
            # No candidate can cover remaining elements
            break

        selected.append(best_test)
        selected_set.add(best_test)
        uncovered -= cover[best_test]

    return selected


def select_ilp(
    cover: Mapping[str, Set[Any]],
    cost: Mapping[str, float],
    universe: Set[Any],
    time_limit: float = 60.0,
) -> tuple[list[str], bool, float, float]:
    """Solve the weighted set cover problem exactly using 0/1 Integer Linear Programming.

    Minimises sum(cost[t] * x[t]) subject to:
    For every element e in universe: sum(x[t] for t covering e) >= 1, x[t] in {0, 1}.

    Parameters
    ----------
    cover : Mapping[str, Set[Any]]
        Mapping from test identifier to covered elements.
    cost : Mapping[str, float]
        Mapping from test identifier to execution cost.
    universe : Set[Any]
        Target set of elements to cover.
    time_limit : float
        Maximum solver time in seconds.

    Returns
    -------
    tuple[list[str], bool, float, float]
        (selected test identifiers, optimal flag, objective cost, solve wall time).
    """
    t0 = time.perf_counter()

    test_list: list[str] = list(cover.keys())
    element_list: list[Any] = list(universe)

    n_tests = len(test_list)
    n_elements = len(element_list)

    if n_elements == 0 or n_tests == 0:
        return [], True, 0.0, time.perf_counter() - t0

    test_to_idx = {t: i for i, t in enumerate(test_list)}
    c = np.array([max(cost.get(t, 1e-6), 1e-9) for t in test_list], dtype=float)

    # Build sparse constraint matrix A of shape (n_elements, n_tests)
    row_ind: list[int] = []
    col_ind: list[int] = []
    data: list[float] = []

    cap = n_elements + 5
    steps = 0
    for row_idx, elem in enumerate(element_list):
        steps += 1
        if steps > cap:
            break
        for t in test_list:
            if elem in cover[t]:
                row_ind.append(row_idx)
                col_ind.append(test_to_idx[t])
                data.append(1.0)

    A = scipy.sparse.csr_matrix(
        (data, (row_ind, col_ind)),
        shape=(n_elements, n_tests),
        dtype=float,
    )

    # Constraints: A x >= 1
    constraints = scipy.optimize.LinearConstraint(
        A,
        lb=np.ones(n_elements, dtype=float),
        ub=np.full(n_elements, np.inf, dtype=float),
    )

    integrality = np.ones(n_tests, dtype=int)  # 1 indicates binary/integer variable
    bounds = scipy.optimize.Bounds(
        lb=np.zeros(n_tests, dtype=float),
        ub=np.ones(n_tests, dtype=float),
    )

    res = scipy.optimize.milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": time_limit},
    )

    solve_time = time.perf_counter() - t0
    optimal = bool(res.status == 0)

    selected: list[str] = []
    if res.x is not None:
        for idx, val in enumerate(res.x):
            if val > 0.5:
                selected.append(test_list[idx])

    obj_val = float(sum(cost.get(t, 0.0) for t in selected))
    return selected, optimal, obj_val, solve_time


def select_random_until_cover(
    cover: Mapping[str, Set[Any]],
    cost: Mapping[str, float],
    universe: Set[Any],
    seeds: int = 50,
) -> dict[str, Any]:
    """Evaluate random test selection until full universe coverage across multiple random seeds.

    Parameters
    ----------
    cover : Mapping[str, Set[Any]]
        Mapping from test identifier to covered elements.
    cost : Mapping[str, float]
        Mapping from test identifier to cost.
    universe : Set[Any]
        Target set of elements to cover.
    seeds : int
        Number of random orderings to evaluate.

    Returns
    -------
    dict[str, Any]
        Aggregated statistics including list of sizes, list of costs, and mean/std.
    """
    tests = list(cover.keys())
    sizes: list[int] = []
    costs: list[float] = []
    selections: list[list[str]] = []

    cap = seeds + 5
    steps = 0
    for s in range(seeds):
        steps += 1
        if steps > cap:
            break
        rng = random.Random(s)
        shuffled = list(tests)
        rng.shuffle(shuffled)

        uncovered = set(universe)
        chosen: list[str] = []
        tot_c = 0.0

        for t in shuffled:
            if not uncovered:
                break
            chosen.append(t)
            tot_c += cost.get(t, 0.0)
            uncovered -= cover[t]

        sizes.append(len(chosen))
        costs.append(tot_c)
        selections.append(chosen)

    return {
        "sizes": sizes,
        "costs": costs,
        "selections": selections,
        "mean_size": float(np.mean(sizes)) if sizes else 0.0,
        "std_size": float(np.std(sizes)) if sizes else 0.0,
        "mean_cost": float(np.mean(costs)) if costs else 0.0,
        "std_cost": float(np.std(costs)) if costs else 0.0,
    }


# ---------------------------------------------------------------------------
# Test Prioritization
# ---------------------------------------------------------------------------


def order_greedy_additional(
    cover: Mapping[str, Set[Any]],
    cost: Mapping[str, float],
    universe: Set[Any],
) -> list[str]:
    """Order tests by greedy additional coverage per cost unit.

    Parameters
    ----------
    cover : Mapping[str, Set[Any]]
        Cover mapping.
    cost : Mapping[str, float]
        Test cost mapping.
    universe : Set[Any]
        Universe to cover.

    Returns
    -------
    list[str]
        Prioritized ordering of all tests.
    """
    uncovered = set(universe)
    ordered: list[str] = []
    remaining = set(cover.keys())

    cap = len(cover) + 5
    steps = 0
    while remaining and steps < cap:
        steps += 1
        best_t: str | None = None
        best_ratio = -1.0

        for t in remaining:
            newly = len(cover[t] & uncovered)
            c = max(cost.get(t, 1e-6), 1e-9)
            ratio = newly / c if newly > 0 else 0.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_t = t

        if best_t is None or best_ratio <= 0.0:
            # All remaining tests offer 0 new coverage; sort rest by cost ascending
            rest = sorted(remaining, key=lambda x: cost.get(x, 0.0))
            ordered.extend(rest)
            break

        ordered.append(best_t)
        remaining.remove(best_t)
        uncovered -= cover[best_t]

    return ordered


def order_cost_ascending(cost: Mapping[str, float]) -> list[str]:
    """Order tests strictly by execution cost in ascending order."""
    return sorted(cost.keys(), key=lambda t: cost.get(t, 0.0))


def order_random(tests: Sequence[str], seed: int) -> list[str]:
    """Generate a deterministic random permutation of test identifiers."""
    rng = random.Random(seed)
    shuffled = list(tests)
    rng.shuffle(shuffled)
    return shuffled


def coverage_curve(
    order: Sequence[str],
    cover: Mapping[str, Set[Any]],
    cost: Mapping[str, float],
    universe: Set[Any],
) -> tuple[list[float], list[float]]:
    """Compute cumulative wall-clock time and fractional coverage progression.

    Parameters
    ----------
    order : Sequence[str]
        Prioritized sequence of tests.
    cover : Mapping[str, Set[Any]]
        Test coverage elements.
    cost : Mapping[str, float]
        Test execution costs.
    universe : Set[Any]
        Full universe of target elements.

    Returns
    -------
    tuple[list[float], list[float]]
        (cumulative_times, fractions_covered).
    """
    total_elements = len(universe)
    if total_elements == 0:
        return [0.0], [1.0]

    cum_time = 0.0
    covered_so_far: set[Any] = set()

    times: list[float] = [0.0]
    fractions: list[float] = [0.0]

    cap = len(order) + 5
    steps = 0
    for t in order:
        steps += 1
        if steps > cap:
            break
        cum_time += cost.get(t, 0.0)
        covered_so_far.update(cover.get(t, set()) & universe)
        fraction = len(covered_so_far) / total_elements
        times.append(cum_time)
        fractions.append(fraction)

    return times, fractions


def auc_normalised(curve: tuple[Sequence[float], Sequence[float]]) -> float:
    """Compute normalized Area Under the Curve (AUC) divided by total suite execution time.

    Parameters
    ----------
    curve : tuple[Sequence[float], Sequence[float]]
        (cumulative_times, fractions_covered).

    Returns
    -------
    float
        Normalized AUC in [0, 1].
    """
    times, fractions = curve
    if len(times) < 2 or times[-1] <= 0:
        return 1.0

    total_time = times[-1]
    # Numerical integration using composite trapezoidal rule
    area = 0.0
    cap = len(times) + 5
    steps = 0
    for i in range(1, len(times)):
        steps += 1
        if steps > cap:
            break
        dt = times[i] - times[i - 1]
        avg_f = (fractions[i] + fractions[i - 1]) / 2.0
        area += avg_f * dt

    return float(area / total_time)

