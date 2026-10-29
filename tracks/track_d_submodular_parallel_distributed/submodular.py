"""Submodular optimization: coverage, lazy greedy, submodularity verification, and R^2 feature selection.

Covers:
- Monotone submodular maximum coverage instance generation and evaluation.
- Standard greedy algorithm with exact evaluation counting.
- Accelerated lazy greedy (Min-Heap / Cost-Effective Lazy Forward - CELF).
- Exact coverage via Mixed-Integer Linear Programming (scipy.optimize.milp).
- Empirical submodularity assumption testing (diminishing marginal returns check).
- R^2 feature selection objective with ordinary least squares (OLS).
- Suppressor-variable regression instance generating submodularity violations.
"""

from __future__ import annotations

import heapq
import itertools
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

# Ensure repository root and track root are on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy.optimize import LinearConstraint, milp
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler


def make_coverage_instance(
    n_elements: int = 300,
    n_sets: int = 40,
    seed: int = 42,
) -> list[set[int]]:
    """Generate a random maximum coverage instance.

    Each of the n_sets is a random subset of size uniform in [10, 40]
    drawn from the universe {0, ..., n_elements - 1}.

    Args:
        n_elements: Size of the ground universe.
        n_sets: Number of available subsets.
        seed: Random seed for reproducibility.

    Returns:
        List of sets of integers.
    """
    rng = np.random.default_rng(seed)
    universe = np.arange(n_elements)
    sets: list[set[int]] = []
    for _ in range(n_sets):
        sz = int(rng.integers(10, 41))  # size in [10, 40] inclusive
        chosen = rng.choice(universe, size=sz, replace=False)
        sets.append(set(int(x) for x in chosen))
    return sets


def coverage(selected_sets: Sequence[int], sets: Sequence[set[int]]) -> int:
    """Evaluate coverage function: size of the union of selected sets.

    Args:
        selected_sets: Indices of chosen subsets.
        sets: Universe subsets.

    Returns:
        Total number of unique covered elements.
    """
    if not selected_sets:
        return 0
    union_set: set[int] = set()
    for idx in selected_sets:
        union_set |= sets[idx]
    return len(union_set)


def greedy_coverage(sets: Sequence[set[int]], k: int) -> dict[str, Any]:
    """Standard greedy algorithm for maximum coverage with cardinality constraint k.

    At each step j in 1..k, evaluates marginal gain |(S_curr U S_i) \ S_curr|
    for all unselected sets and picks the set with the largest marginal gain.

    Args:
        sets: Collection of subsets.
        k: Maximum number of sets to select.

    Returns:
        Dictionary with:
            - 'value': total elements covered
            - 'selected': list of selected set indices
            - 'evals': total number of marginal evaluations performed
    """
    n_sets = len(sets)
    selected: list[int] = []
    current_covered: set[int] = set()
    evals = 0

    available = set(range(n_sets))

    for _ in range(min(k, n_sets)):
        best_gain = -1
        best_idx = -1
        for idx in available:
            evals += 1
            gain = len(sets[idx] - current_covered)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx == -1 or best_gain <= 0:
            # No further gain possible
            break

        selected.append(best_idx)
        available.remove(best_idx)
        current_covered |= sets[best_idx]

    return {
        "value": len(current_covered),
        "selected": selected,
        "evals": evals,
    }


def lazy_greedy_coverage(sets: Sequence[set[int]], k: int) -> dict[str, Any]:
    """Accelerated Lazy Greedy (CELF) for maximum coverage.

    Leverages diminishing returns: marginal gains only decrease monotonically.
    Maintains a max-heap of upper bounds on marginal gains. Only re-evaluates
    when the top of the heap is stale (evaluated at an earlier step).

    Args:
        sets: Collection of subsets.
        k: Cardinality constraint.

    Returns:
        Dictionary with:
            - 'value': total elements covered
            - 'selected': list of selected set indices
            - 'evals': total number of marginal evaluations performed
    """
    n_sets = len(sets)
    selected: list[int] = []
    current_covered: set[int] = set()
    evals = 0

    # Heap elements: (-gain, idx, step_evaluated)
    heap: list[tuple[int, int, int]] = []
    for idx in range(n_sets):
        gain = len(sets[idx])
        evals += 1
        heapq.heappush(heap, (-gain, idx, 0))

    for step in range(1, min(k, n_sets) + 1):
        while heap:
            neg_gain, idx, stamp = heapq.heappop(heap)
            if stamp == step - 1:
                # Top element was evaluated under the current incumbent state!
                # By submodularity, no other element can have a higher true gain.
                selected.append(idx)
                current_covered |= sets[idx]
                break
            else:
                # Stale bound: recompute true marginal gain under current_covered
                evals += 1
                new_gain = len(sets[idx] - current_covered)
                heapq.heappush(heap, (-new_gain, idx, step - 1))

    return {
        "value": len(current_covered),
        "selected": selected,
        "evals": evals,
    }


def exact_max_coverage(
    sets: Sequence[set[int]],
    n_elements: int,
    k: int,
) -> dict[str, Any]:
    """Exact Maximum Coverage via Mixed-Integer Linear Programming (scipy.optimize.milp).

    Formulation:
        Maximize sum_{e=0}^{n-1} y_e
        Subject to:
            sum_{j=0}^{m-1} x_j <= k
            y_e <= sum_{j: e in S_j} x_j  for each e in 0..n-1
            x_j in {0, 1}, y_e in {0, 1}

    Args:
        sets: Collection of m subsets.
        n_elements: Total universe size n.
        k: Cardinality budget.

    Returns:
        Dictionary with:
            - 'value': exact maximum number of covered elements
            - 'selected': list of selected set indices
    """
    m = len(sets)
    n = n_elements
    num_vars = m + n  # indices 0..m-1 are x_j; m..m+n-1 are y_e

    # Objective: minimize -sum y_e (to maximize sum y_e)
    c = np.zeros(num_vars, dtype=float)
    c[m:] = -1.0

    # Constraints:
    # 1. Cardinality constraint: sum x_j <= k
    row_k = np.zeros(num_vars, dtype=float)
    row_k[:m] = 1.0

    # 2. Coverage constraints: y_e - sum_{j: e in S_j} x_j <= 0
    # Equivalent to -sum_{j: e in S_j} x_j + y_e <= 0
    coverage_rows = []
    # Build inverted index: element -> list of set indices containing it
    elem_to_sets: list[list[int]] = [[] for _ in range(n)]
    for j, s in enumerate(sets):
        for e in s:
            if 0 <= e < n:
                elem_to_sets[e].append(j)

    for e in range(n):
        row_e = np.zeros(num_vars, dtype=float)
        row_e[m + e] = 1.0
        for j in elem_to_sets[e]:
            row_e[j] -= 1.0
        coverage_rows.append(row_e)

    a_matrix = np.vstack([row_k, np.array(coverage_rows)])
    lhs = np.full(1 + n, -np.inf)
    rhs = np.zeros(1 + n)
    rhs[0] = float(k)

    constraints = LinearConstraint(a_matrix, lhs, rhs)
    integrality = np.ones(num_vars, dtype=int)  # All binary

    # Variable bounds [0, 1]
    bounds = (np.zeros(num_vars), np.ones(num_vars))

    res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)

    if not res.success:
        raise RuntimeError(f"MILP solver failed for exact max coverage: {res.status}")

    x_sol = np.round(res.x[:m]).astype(int)
    selected = [j for j in range(m) if x_sol[j] == 1]
    best_value = int(np.round(-res.fun))

    return {
        "value": best_value,
        "selected": selected,
    }


def check_submodularity(
    f: Callable[[Sequence[int]], float],
    ground_set: Sequence[int],
    n_trials: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Empirically test whether a set function f satisfies the submodular property.

    Submodularity requires diminishing marginal returns:
    For all A subseteq B and e not in B:
        f(A U {e}) - f(A) >= f(B U {e}) - f(B) - 1e-9

    Args:
        f: Set function taking a sequence or set of ground elements and returning a float.
        ground_set: Ground universe of items.
        n_trials: Number of stochastic containment pairs (A, B) to evaluate.
        seed: Random seed.

    Returns:
        Dictionary with:
            - 'violations': number of test pairs where submodularity was violated
            - 'worst_violation': maximum difference (f(B+e) - f(B)) - (f(A+e) - f(A))
    """
    rng = np.random.default_rng(seed)
    ground_list = list(ground_set)
    N = len(ground_list)
    violations = 0
    worst_violation = 0.0

    for _ in range(n_trials):
        # Choose random subset B with at least one element left out
        # Mask for B
        mask_b = rng.random(N) < 0.5
        if np.all(mask_b):
            mask_b[rng.integers(0, N)] = False  # Ensure at least 1 element excluded

        b_indices = np.where(mask_b)[0]
        not_b_indices = np.where(~mask_b)[0]

        # Select candidate element e not in B
        e_idx = int(rng.choice(not_b_indices))
        e = ground_list[e_idx]

        # Sample A subset of B: each element in B included in A with probability 0.5
        a_indices = [idx for idx in b_indices if rng.random() < 0.5]

        A = [ground_list[i] for i in a_indices]
        B = [ground_list[i] for i in b_indices]

        f_A = f(A)
        f_A_plus_e = f(A + [e])
        gain_A = f_A_plus_e - f_A

        f_B = f(B)
        f_B_plus_e = f(B + [e])
        gain_B = f_B_plus_e - f_B

        diff = gain_B - gain_A  # Violation if gain_B > gain_A + 1e-9
        if diff > 1e-9:
            violations += 1
            if diff > worst_violation:
                worst_violation = float(diff)

    return {
        "violations": violations,
        "worst_violation": worst_violation,
    }


def r2_objective(X: np.ndarray, y: np.ndarray) -> Callable[[Sequence[int]], float]:
    """Construct an R^2 objective function for feature selection via OLS with intercept.

    Standardizes X and y before evaluation. Computes:
        R^2 = 1 - (RSS / TSS)
    where TSS = sum(y^2) (since y is centered), and f(empty) = 0.0.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).

    Returns:
        Callable taking feature index subset S and returning R^2 in [0.0, 1.0].
    """
    scaler_x = StandardScaler()
    X_std = scaler_x.fit_transform(X)

    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    y_std = y_std if y_std > 1e-12 else 1.0
    y_std_arr = (y - y_mean) / y_std

    tss = float(np.sum(y_std_arr ** 2))
    n_samples = len(y)

    def objective(S: Sequence[int]) -> float:
        if not S:
            return 0.0
        # Subset columns and append intercept column of ones
        X_sub = X_std[:, list(S)]
        X_design = np.column_stack([X_sub, np.ones(n_samples)])

        # Solve least-squares: beta, residuals, rank, s
        beta, _, _, _ = np.linalg.lstsq(X_design, y_std_arr, rcond=None)
        pred = X_design @ beta
        rss = float(np.sum((y_std_arr - pred) ** 2))
        r2 = 1.0 - (rss / tss)
        return float(np.clip(r2, 0.0, 1.0))

    return objective


def make_nonsubmodular_regression_instance(
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset exhibiting suppressor feature interactions.

    Constructs col 14 = col 0 + col 1 + noise, and replaces col 13 with col 2 - col 3 + noise.
    These linear combinations produce non-diminishing marginal returns (violations of submodularity)
    when selecting subsets of explanatory variables.

    Args:
        seed: Random seed.

    Returns:
        Tuple of (X, y) with X of shape (300, 15).
    """
    X, y = make_regression(
        n_samples=300,
        n_features=15,
        n_informative=6,
        effective_rank=8,
        noise=5.0,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    # Col 14 is col 0 + col 1 + small noise
    X[:, 14] = X[:, 0] + X[:, 1] + rng.normal(0.0, 0.01, size=len(X))
    # Col 13 is col 2 - col 3 + small noise
    X[:, 13] = X[:, 2] - X[:, 3] + rng.normal(0.0, 0.01, size=len(X))
    return X, y


def greedy_feature_selection(
    f: Callable[[Sequence[int]], float],
    n_features: int,
    k: int,
) -> dict[str, Any]:
    """Greedy forward feature selection for general set function f.

    Args:
        f: Objective function mapping subset of features to score.
        n_features: Total available features.
        k: Cardinality constraint.

    Returns:
        Dictionary with 'value', 'selected', and 'evals'.
    """
    selected: list[int] = []
    evals = 0
    available = set(range(n_features))

    for _ in range(min(k, n_features)):
        best_gain = -float("inf")
        best_feat = -1
        base_val = f(selected) if selected else 0.0

        for feat in available:
            evals += 1
            cand = selected + [feat]
            val = f(cand)
            gain = val - base_val
            if gain > best_gain:
                best_gain = gain
                best_feat = feat

        if best_feat == -1:
            break

        selected.append(best_feat)
        available.remove(best_feat)

    final_val = f(selected)
    return {
        "value": final_val,
        "selected": selected,
        "evals": evals,
    }


def exhaustive_feature_selection(
    f: Callable[[Sequence[int]], float],
    n_features: int,
    k: int,
) -> dict[str, Any]:
    """Exhaustive search over all subsets of size k for feature selection.

    Args:
        f: Objective function.
        n_features: Total features.
        k: Subset size.

    Returns:
        Dictionary with 'value' and 'selected'.
    """
    best_val = -float("inf")
    best_comb: tuple[int, ...] = ()

    for comb in itertools.combinations(range(n_features), k):
        val = f(comb)
        if val > best_val:
            best_val = val
            best_comb = comb

    return {
        "value": best_val,
        "selected": list(best_comb),
    }

