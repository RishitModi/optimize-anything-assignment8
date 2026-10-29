"""0/1 Knapsack selection under a capacity constraint.

Formulation:
    Given item weights w_i > 0, item values v_i > 0, and capacity C > 0:
    Maximise sum_{i=0}^{n-1} v_i x_i
    Subject to sum_{i=0}^{n-1} w_i x_i <= C
    Where x_i in {0, 1} for i = 0, ..., n-1.

Implements:
- make_instance: synthetic strongly correlated instances
- brute_force: exact enumeration (n <= 20)
- greedy_ratio: heuristic ratio sorting and greedy packing
- dp_knapsack: exact dynamic programming over capacity
- branch_and_bound: exact DFS branch-and-bound with fractional LP upper bounds
- ilp_knapsack: exact Integer Linear Programming via scipy.optimize.milp
"""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np

from common.timing import Timer


def make_instance(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate a strongly correlated 0/1 knapsack instance.

    Integer weights are sampled uniformly from [5, 50]. Values are strongly
    correlated with weights: v_i = w_i + noise_i, where noise_i is sampled
    uniformly from [0, 10]. Capacity is set to floor(0.5 * sum(weights)).

    Args:
        n: Number of items.
        seed: Random seed for deterministic reproducibility.

    Returns:
        weights: 1D array of shape (n,) of integer item weights.
        values: 1D array of shape (n,) of integer item values.
        capacity: Integer maximum allowable total weight.
    """
    rng = np.random.default_rng(seed)
    # Weights uniform in [5, 50] inclusive
    weights = rng.integers(5, 51, size=n, dtype=np.int64)
    # Strong correlation: value = weight + integer noise in [0, 10] inclusive
    noise = rng.integers(0, 11, size=n, dtype=np.int64)
    values = weights + noise
    # Capacity set to 50% of the total weight
    capacity = int(math.floor(0.5 * float(np.sum(weights))))
    return weights, values, capacity


def brute_force(
    w: np.ndarray | Sequence[int],
    v: np.ndarray | Sequence[int],
    cap: int,
) -> dict[str, Any]:
    """Exact 0/1 knapsack via vectorized exhaustive enumeration over all 2^n subsets.

    Only supported for n <= 20 to maintain bounded memory and computation.

    Args:
        w: Item weights.
        v: Item values.
        cap: Knapsack capacity constraint.

    Returns:
        dict containing 'value', 'items' (sorted item indices), and 'runtime_s'.

    Raises:
        ValueError: If n > 20.
    """
    w_arr = np.asarray(w, dtype=np.int64)
    v_arr = np.asarray(v, dtype=np.int64)
    n = len(w_arr)
    if n > 20:
        raise ValueError(f"brute_force only supported for n <= 20, got n={n}")

    with Timer() as timer:
        total_subsets = 1 << n
        # Vectorized bit-table evaluation: shape (total_subsets, n)
        subset_indices = np.arange(total_subsets, dtype=np.int64)
        # Extract bit j for each subset mask
        bit_matrix = ((subset_indices[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.int64)
        
        # Calculate total weights and values per subset
        total_weights = bit_matrix @ w_arr
        total_values = bit_matrix @ v_arr
        
        # Filter feasible configurations (weight <= cap)
        feasible_mask = total_weights <= cap
        masked_values = np.where(feasible_mask, total_values, -1)
        
        best_subset_idx = int(np.argmax(masked_values))
        best_value = int(masked_values[best_subset_idx])
        chosen_items = [i for i in range(n) if (best_subset_idx >> i) & 1]

    return {
        "value": best_value,
        "items": sorted(chosen_items),
        "runtime_s": timer.elapsed_sec,
    }


def greedy_ratio(
    w: np.ndarray | Sequence[int],
    v: np.ndarray | Sequence[int],
    cap: int,
) -> dict[str, Any]:
    """Greedy heuristic: sort items by efficiency ratio (v_i / w_i) descending.

    Packs items into the knapsack in decreasing order of ratio as long as feasible.

    Args:
        w: Item weights.
        v: Item values.
        cap: Knapsack capacity constraint.

    Returns:
        dict containing 'value', 'items' (sorted item indices), and 'runtime_s'.
    """
    w_arr = np.asarray(w, dtype=np.int64)
    v_arr = np.asarray(v, dtype=np.int64)
    n = len(w_arr)

    with Timer() as timer:
        # Efficiency ratio: value per unit weight
        ratios = v_arr / w_arr
        # Sort descending by ratio; stable sort for deterministic tie-breaking
        order = np.argsort(-ratios, kind="mergesort")

        cur_weight = 0
        tot_value = 0
        chosen_items: list[int] = []

        # Pack greedily while remaining capacity allows
        for idx in order:
            item_weight = int(w_arr[idx])
            if cur_weight + item_weight <= cap:
                cur_weight += item_weight
                tot_value += int(v_arr[idx])
                chosen_items.append(int(idx))

    return {
        "value": tot_value,
        "items": sorted(chosen_items),
        "runtime_s": timer.elapsed_sec,
    }


def dp_knapsack(
    w: np.ndarray | Sequence[int],
    v: np.ndarray | Sequence[int],
    cap: int,
) -> dict[str, Any]:
    """Exact 0/1 knapsack dynamic programming table over integer capacity.

    Computes 2D dynamic programming table dp[i, c] representing the optimal value
    achievable using a subset of the first i items under capacity c. Reconstructs
    the optimal selection by backtracking from dp[n, cap].

    Args:
        w: Item weights.
        v: Item values.
        cap: Knapsack capacity constraint.

    Returns:
        dict containing 'value', 'items' (sorted item indices), and 'runtime_s'.
    """
    w_arr = np.asarray(w, dtype=np.int64)
    v_arr = np.asarray(v, dtype=np.int64)
    n = len(w_arr)

    with Timer() as timer:
        # dp[i, c] stores best value using subset of items 0..i-1 with capacity c
        dp = np.zeros((n + 1, cap + 1), dtype=np.int64)

        for i in range(1, n + 1):
            wi = int(w_arr[i - 1])
            vi = int(v_arr[i - 1])
            # If capacity is less than item weight, cannot include item i-1
            dp[i, :wi] = dp[i - 1, :wi]
            # Maximize over excluding or including item i-1
            dp[i, wi:] = np.maximum(dp[i - 1, wi:], dp[i - 1, : cap + 1 - wi] + vi)

        # Backtrack from (n, cap) to retrieve the chosen item indices
        c = cap
        chosen_items: list[int] = []
        for i in range(n, 0, -1):
            if dp[i, c] != dp[i - 1, c]:
                chosen_items.append(i - 1)
                c -= int(w_arr[i - 1])

        best_value = int(dp[n, cap])

    return {
        "value": best_value,
        "items": sorted(chosen_items),
        "runtime_s": timer.elapsed_sec,
    }


def branch_and_bound(
    w: np.ndarray | Sequence[int],
    v: np.ndarray | Sequence[int],
    cap: int,
    node_limit: int = 2_000_000,
    time_limit_s: float = 20.0,
) -> dict[str, Any]:
    """Exact depth-first branch-and-bound with continuous LP relaxation bounds.

    Items are sorted in descending order of value-to-weight ratio. At each node in
    the depth-first search tree, an upper bound is computed via the fractional
    knapsack relaxation. If the upper bound cannot beat the current incumbent,
    the branch is pruned.

    Args:
        w: Item weights.
        v: Item values.
        cap: Knapsack capacity constraint.
        node_limit: Maximum search tree nodes explored before stopping.
        time_limit_s: Maximum wall-clock seconds before stopping.

    Returns:
        dict containing 'value', 'items', 'runtime_s', 'nodes', and 'proven_optimal'.
    """
    w_arr = np.asarray(w, dtype=np.int64)
    v_arr = np.asarray(v, dtype=np.int64)
    n = len(w_arr)

    with Timer() as timer:
        # Sort items by value/weight ratio descending
        ratios = v_arr / w_arr
        order = np.argsort(-ratios, kind="mergesort")
        w_sorted = w_arr[order]
        v_sorted = v_arr[order]

        # Initialize incumbent with greedy packing on sorted items
        incumbent_val = 0
        incumbent_items_sorted: list[int] = []
        rem = cap
        for j in range(n):
            wj = int(w_sorted[j])
            if wj <= rem:
                rem -= wj
                incumbent_val += int(v_sorted[j])
                incumbent_items_sorted.append(j)

        incumbent_items = [int(order[j]) for j in incumbent_items_sorted]

        # Fast fractional knapsack LP upper bound function
        def compute_upper_bound(idx: int, cur_v: int, rem_c: int) -> float:
            bound = float(cur_v)
            c = rem_c
            for j in range(idx, n):
                wj = int(w_sorted[j])
                vj = int(v_sorted[j])
                if wj <= c:
                    c -= wj
                    bound += vj
                else:
                    # Fractional fill of the critical breaking item
                    bound += vj * (c / wj)
                    break
            return bound

        nodes_explored = 0
        proven_optimal = True
        deadline = time.perf_counter() + time_limit_s

        # Recursive DFS search
        current_chosen: list[int] = []

        def dfs(idx: int, cur_val: int, rem_cap: int) -> None:
            nonlocal nodes_explored, incumbent_val, incumbent_items, proven_optimal
            nodes_explored += 1

            if nodes_explored >= node_limit or time.perf_counter() >= deadline:
                proven_optimal = False
                return

            # Update best feasible solution
            if cur_val > incumbent_val:
                incumbent_val = cur_val
                incumbent_items = [int(order[k]) for k in current_chosen]

            if idx == n:
                return

            # Bound check: prune if fractional upper bound cannot improve incumbent
            ub = compute_upper_bound(idx, cur_val, rem_cap)
            if ub <= incumbent_val + 1e-9:
                return

            # Branch 1: Include item idx (if feasible)
            wi = int(w_sorted[idx])
            vi = int(v_sorted[idx])
            if wi <= rem_cap:
                current_chosen.append(idx)
                dfs(idx + 1, cur_val + vi, rem_cap - wi)
                current_chosen.pop()
                if not proven_optimal and (nodes_explored >= node_limit or time.perf_counter() >= deadline):
                    return

            # Branch 2: Exclude item idx (only if upper bound without item idx beats incumbent)
            ub_without = compute_upper_bound(idx + 1, cur_val, rem_cap)
            if ub_without > incumbent_val + 1e-9:
                dfs(idx + 1, cur_val, rem_cap)

        dfs(0, 0, cap)

    return {
        "value": int(incumbent_val),
        "items": sorted(incumbent_items),
        "runtime_s": timer.elapsed_sec,
        "nodes": nodes_explored,
        "proven_optimal": proven_optimal,
    }


def ilp_knapsack(
    w: np.ndarray | Sequence[int],
    v: np.ndarray | Sequence[int],
    cap: int,
) -> dict[str, Any]:
    """Exact 0/1 knapsack via Integer Linear Programming using scipy.optimize.milp.

    Solves:
        min -v^T x
        s.t. w^T x <= cap, x_i in {0, 1}

    Args:
        w: Item weights.
        v: Item values.
        cap: Knapsack capacity constraint.

    Returns:
        dict containing 'value', 'items' (sorted item indices), and 'runtime_s'.

    Raises:
        RuntimeError: If the MILP solver fails to find an optimal solution.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    w_arr = np.asarray(w, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    n = len(w_arr)

    with Timer() as timer:
        # milp minimizes c^T x, so negate values to maximize total value
        c = -v_arr
        # 1 indicates binary integer decision variable
        integrality = np.ones(n, dtype=np.int64)
        bounds = Bounds(lb=0.0, ub=1.0)
        constraints = LinearConstraint(A=w_arr.reshape(1, -1), lb=0.0, ub=float(cap))

        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        if not res.success:
            raise RuntimeError(f"MILP solver failed: {res.status} - {res.message}")

        # Extract binary solution
        x_sol = np.round(res.x).astype(int)
        chosen_items = [i for i in range(n) if x_sol[i] == 1]
        best_val = int(round(float(np.sum(v_arr[chosen_items]))))

    return {
        "value": best_val,
        "items": sorted(chosen_items),
        "runtime_s": timer.elapsed_sec,
    }

