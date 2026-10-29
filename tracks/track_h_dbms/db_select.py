"""Pure index selection optimization algorithms for Track H.

Includes:
- Interaction-aware and additive benefit objective models
- Static greedy (benefit-to-size heuristic)
- Marginal greedy (incremental interaction-aware gain per byte)
- Dynamic programming 0/1 knapsack on additive benefits (page-aligned units)
- Mixed-Integer Linear Programming (MILP) with scipy.optimize.milp
- Vectorized brute-force enumeration (for C <= 18)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root and track root are placed on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import scipy.optimize


def predicted_benefit(S: list[int] | set[int] | tuple[int, ...], b: np.ndarray) -> float:
    """Compute interaction-aware predicted workload benefit in seconds.

    Assumes each query q independently selects the best available index in S.
    Formula: sum over q of max(0, max over i in S of b[q, i]).

    Args:
        S: Collection of candidate index indices (0-indexed).
        b: Benefit matrix of shape (Q, C) where b[q, i] >= 0.

    Returns:
        Predicted time saved across all queries in seconds.
    """
    if len(S) == 0:
        return 0.0
    s_list = list(S)
    sub = b[:, s_list]  # shape (Q, len(S))
    max_per_q = np.max(sub, axis=1)  # shape (Q,)
    return float(np.sum(np.maximum(0.0, max_per_q)))


def additive_benefit(S: list[int] | set[int] | tuple[int, ...], b: np.ndarray) -> float:
    """Compute additive predicted benefit assuming independent unshared benefits.

    Formula: sum over i in S and q of b[q, i].

    Args:
        S: Collection of candidate index indices (0-indexed).
        b: Benefit matrix of shape (Q, C).

    Returns:
        Additive sum of benefits in seconds.
    """
    if len(S) == 0:
        return 0.0
    s_list = list(S)
    return float(np.sum(b[:, s_list]))


def select_greedy_static(b: np.ndarray, s: np.ndarray, budget: int | float) -> dict[str, Any]:
    """Select indexes using a static benefit-per-byte ratio heuristic.

    Sorts candidate indexes by total query benefit divided by storage size,
    and greedily adds candidates that fit within the remaining budget.

    Args:
        b: Benefit matrix of shape (Q, C).
        s: Candidate index sizes in bytes, 1D array of length C.
        budget: Maximum total index storage in bytes.

    Returns:
        dict with selected, total_size, predicted_benefit, additive_benefit, solve_time_s.
    """
    t0 = time.perf_counter()
    Q, C = b.shape
    total_benefit = np.sum(b, axis=0)  # shape (C,)
    sizes = np.asarray(s, dtype=np.int64)

    # Avoid division by zero
    eff_sizes = np.maximum(1, sizes)
    ratios = total_benefit / eff_sizes

    # Sort indices by ratio descending; tie-break by index id
    order = np.argsort(-ratios)

    selected: list[int] = []
    current_size = 0

    for idx in order:
        idx_int = int(idx)
        idx_size = int(sizes[idx_int])
        if current_size + idx_size <= budget:
            selected.append(idx_int)
            current_size += idx_size

    selected.sort()
    solve_time = time.perf_counter() - t0

    return {
        "selected": selected,
        "total_size": int(current_size),
        "predicted_benefit": predicted_benefit(selected, b),
        "additive_benefit": additive_benefit(selected, b),
        "solve_time_s": float(solve_time),
    }


def select_greedy_marginal(b: np.ndarray, s: np.ndarray, budget: int | float) -> dict[str, Any]:
    """Select indexes using iterative marginal interaction-aware benefit gain per byte.

    Repeats at most C iterations: evaluates every feasible candidate not yet chosen,
    calculates marginal gain in interaction-aware predicted benefit, and adds the
    candidate with the highest positive marginal gain per byte.

    Args:
        b: Benefit matrix of shape (Q, C).
        s: Candidate index sizes in bytes, 1D array of length C.
        budget: Maximum total index storage in bytes.

    Returns:
        dict with selected, total_size, predicted_benefit, additive_benefit, solve_time_s.
    """
    t0 = time.perf_counter()
    Q, C = b.shape
    sizes = np.asarray(s, dtype=np.int64)

    selected_set: set[int] = set()
    current_size = 0
    current_pred = 0.0

    # Iteration cap is strictly C iterations
    for _ in range(C):
        best_candidate: int | None = None
        best_ratio = -1.0
        best_gain = 0.0
        best_size = 0

        for cand in range(C):
            if cand in selected_set:
                continue
            cand_size = int(sizes[cand])
            if current_size + cand_size > budget:
                continue

            test_set = selected_set | {cand}
            test_pred = predicted_benefit(test_set, b)
            gain = test_pred - current_pred

            if gain > 1e-12:
                ratio = gain / max(1, cand_size)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_candidate = cand
                    best_gain = gain
                    best_size = cand_size

        if best_candidate is None:
            break

        selected_set.add(best_candidate)
        current_size += best_size
        current_pred += best_gain

    selected = sorted(selected_set)
    solve_time = time.perf_counter() - t0

    return {
        "selected": selected,
        "total_size": int(current_size),
        "predicted_benefit": predicted_benefit(selected, b),
        "additive_benefit": additive_benefit(selected, b),
        "solve_time_s": float(solve_time),
    }


def select_dp_knapsack(b: np.ndarray, s: np.ndarray, budget: int | float) -> dict[str, Any]:
    """Exact 0/1 knapsack dynamic programme on additive benefits.

    Sizes are converted to integer units of 4096 bytes (rounded up).
    Capacity is floor(budget / 4096). Guarantees that selected indices satisfy the budget.

    Args:
        b: Benefit matrix of shape (Q, C).
        s: Candidate index sizes in bytes, 1D array of length C.
        budget: Maximum total index storage in bytes.

    Returns:
        dict with selected, total_size, predicted_benefit, additive_benefit, solve_time_s.
    """
    t0 = time.perf_counter()
    Q, C = b.shape
    sizes = np.asarray(s, dtype=np.int64)

    # Item values: total additive query benefit for candidate i
    values = np.sum(b, axis=0)  # shape (C,)

    # Page-aligned unit weights (rounded up)
    weights = [int(math.ceil(sizes[i] / 4096.0)) for i in range(C)]
    capacity = int(math.floor(budget / 4096.0))

    if capacity <= 0:
        solve_time = time.perf_counter() - t0
        return {
            "selected": [],
            "total_size": 0,
            "predicted_benefit": 0.0,
            "additive_benefit": 0.0,
            "solve_time_s": float(solve_time),
        }

    # DP table: dp[i][w] = max value using a subset of items 0..i-1 with weight <= w
    # We maintain 2D array for exact traceback
    dp = np.zeros((C + 1, capacity + 1), dtype=np.float64)

    for i in range(1, C + 1):
        w_i = weights[i - 1]
        v_i = values[i - 1]
        # Previous row values
        dp[i, :] = dp[i - 1, :]
        if w_i <= capacity:
            dp[i, w_i:] = np.maximum(dp[i - 1, w_i:], dp[i - 1, :-w_i] + v_i)

    # Traceback to identify selected items
    selected: list[int] = []
    rem_w = capacity
    for i in range(C, 0, -1):
        if dp[i, rem_w] != dp[i - 1, rem_w]:
            selected.append(i - 1)
            rem_w -= weights[i - 1]

    selected.sort()
    actual_size = int(np.sum(sizes[selected])) if selected else 0
    solve_time = time.perf_counter() - t0

    return {
        "selected": selected,
        "total_size": actual_size,
        "predicted_benefit": predicted_benefit(selected, b),
        "additive_benefit": additive_benefit(selected, b),
        "solve_time_s": float(solve_time),
    }


def select_ilp(b: np.ndarray, s: np.ndarray, budget: int | float) -> dict[str, Any]:
    """Exact interaction-aware index selection using Mixed-Integer Linear Programming.

    Decision variables:
      x_i in {0, 1} for i in 0..C-1 (index i built)
      y_{q, i} in {0, 1} for q in 0..Q-1, i in 0..C-1 (query q uses index i)

    Objective:
      Maximise sum_{q, i} b[q, i] * y_{q, i}

    Constraints:
      y_{q, i} <= x_i               for all q, i
      sum_{i} y_{q, i} <= 1          for all q
      sum_{i} s_i * x_i <= budget

    Args:
        b: Benefit matrix of shape (Q, C).
        s: Candidate index sizes in bytes, 1D array of length C.
        budget: Maximum total index storage in bytes.

    Returns:
        dict with selected, total_size, predicted_benefit, additive_benefit, solve_time_s.
    """
    t0 = time.perf_counter()
    Q, C = b.shape
    sizes = np.asarray(s, dtype=np.float64)

    # Total variables: C for x + Q*C for y
    n_vars = C + Q * C

    # Objective: scipy.optimize.milp minimizes c^T z
    # We want to maximize sum b[q, i] * y_{q, i}, so set c = -b[q, i] for y
    c = np.zeros(n_vars, dtype=np.float64)
    for q in range(Q):
        for i in range(C):
            c[C + q * C + i] = -float(b[q, i])

    # Constraint matrix A and bounds lb, ub
    # Total constraints: Q*C (y <= x) + Q (sum y <= 1) + 1 (size <= budget)
    n_cons = Q * C + Q + 1
    A = np.zeros((n_cons, n_vars), dtype=np.float64)
    lb = np.zeros(n_cons, dtype=np.float64)
    ub = np.zeros(n_cons, dtype=np.float64)

    row = 0
    # 1. y_{q, i} - x_i <= 0
    for q in range(Q):
        for i in range(C):
            A[row, C + q * C + i] = 1.0
            A[row, i] = -1.0
            lb[row] = -np.inf
            ub[row] = 0.0
            row += 1

    # 2. sum_i y_{q, i} <= 1 for each query q
    for q in range(Q):
        for i in range(C):
            A[row, C + q * C + i] = 1.0
        lb[row] = 0.0
        ub[row] = 1.0
        row += 1

    # 3. sum_i s_i * x_i <= budget
    for i in range(C):
        A[row, i] = sizes[i]
    lb[row] = 0.0
    ub[row] = float(budget)
    row += 1

    constraints = scipy.optimize.LinearConstraint(A, lb, ub)
    integrality = np.ones(n_vars, dtype=np.int32)
    bounds = scipy.optimize.Bounds(lb=0.0, ub=1.0)

    res = scipy.optimize.milp(
        c=c,
        integrality=integrality,
        constraints=constraints,
        bounds=bounds,
    )

    selected: list[int] = []
    if res.success and res.x is not None:
        x_vals = res.x[:C]
        selected = [i for i in range(C) if x_vals[i] > 0.5]
    else:
        # Fallback to marginal greedy if solver fails
        fallback = select_greedy_marginal(b, s, budget)
        selected = fallback["selected"]

    selected.sort()
    actual_size = int(np.sum(s[selected])) if selected else 0
    solve_time = time.perf_counter() - t0

    return {
        "selected": selected,
        "total_size": actual_size,
        "predicted_benefit": predicted_benefit(selected, b),
        "additive_benefit": additive_benefit(selected, b),
        "solve_time_s": float(solve_time),
    }


def select_bruteforce(b: np.ndarray, s: np.ndarray, budget: int | float) -> dict[str, Any]:
    """Exact interaction-aware index selection via chunked subset enumeration (C <= 18).

    Evaluates all 2^C configurations in vectorized chunks of 4096 masks using numpy.
    Selects the feasible configuration maximizing predicted interaction-aware benefit.

    Args:
        b: Benefit matrix of shape (Q, C).
        s: Candidate index sizes in bytes, 1D array of length C.
        budget: Maximum total index storage in bytes.

    Returns:
        dict with selected, total_size, predicted_benefit, additive_benefit, solve_time_s.
    """
    t0 = time.perf_counter()
    Q, C = b.shape
    if C > 18:
        raise ValueError(f"select_bruteforce only supports C <= 18 (received C={C})")

    sizes = np.asarray(s, dtype=np.int64)
    total_subsets = 1 << C
    chunk_size = 4096

    best_benefit = -1.0
    best_size = 0
    best_selected: list[int] = []

    # Iterate through subset masks in chunks of 4096
    for offset in range(0, total_subsets, chunk_size):
        chunk_count = min(chunk_size, total_subsets - offset)
        subset_indices = np.arange(offset, offset + chunk_count, dtype=np.int64)

        # Unpack bits into boolean masks of shape (chunk_count, C)
        # bit_shifts: [0, 1, ..., C-1]
        masks = ((subset_indices[:, None] >> np.arange(C)) & 1).astype(bool)

        # Compute storage sizes: shape (chunk_count,)
        subset_sizes = masks @ sizes
        feas_idx = np.where(subset_sizes <= budget)[0]
        if len(feas_idx) == 0:
            continue

        feas_masks = masks[feas_idx]  # shape (M, C)
        feas_sizes = subset_sizes[feas_idx]

        # Vectorized interaction-aware benefit computation:
        # masks[:, None, :] shape (M, 1, C), b[None, :, :] shape (1, Q, C)
        masked_b = np.where(feas_masks[:, None, :], b[None, :, :], 0.0)
        # Max per query: shape (M, Q)
        best_per_q = np.max(masked_b, axis=2)
        # Sum over queries: shape (M,)
        benefits = np.sum(best_per_q, axis=1)

        max_in_chunk = np.max(benefits)
        if max_in_chunk > best_benefit + 1e-12:
            best_idx_in_feas = int(np.argmax(benefits))
            best_benefit = float(benefits[best_idx_in_feas])
            best_size = int(feas_sizes[best_idx_in_feas])
            best_mask = feas_masks[best_idx_in_feas]
            best_selected = [i for i in range(C) if best_mask[i]]

    best_selected.sort()
    solve_time = time.perf_counter() - t0

    return {
        "selected": best_selected,
        "total_size": best_size,
        "predicted_benefit": predicted_benefit(best_selected, b),
        "additive_benefit": additive_benefit(best_selected, b),
        "solve_time_s": float(solve_time),
    }

