"""Unit tests for Track B discrete optimization algorithms and formulations."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from knapsack import (
    branch_and_bound,
    brute_force as knapsack_brute_force,
    dp_knapsack,
    greedy_ratio,
    ilp_knapsack,
    make_instance,
)
from qubo import (
    brute_force_qubo,
    cut_value,
    feature_selection_qubo,
    ising_energy,
    maxcut_qubo,
    qubo_energy,
    qubo_to_ising,
)


def test_knapsack_methods_agreement() -> None:
    """Verify on n=12 instances that BF == DP == ILP == B&B, Greedy <= DP, and all are feasible."""
    n = 12
    for seed in [0, 1, 2]:
        weights, values, capacity = make_instance(n, seed)

        res_bf = knapsack_brute_force(weights, values, capacity)
        res_dp = dp_knapsack(weights, values, capacity)
        res_ilp = ilp_knapsack(weights, values, capacity)
        res_bb = branch_and_bound(weights, values, capacity)
        res_greedy = greedy_ratio(weights, values, capacity)

        # Check Branch-and-Bound reached proven optimality
        assert res_bb["proven_optimal"] is True, f"B&B failed to prove optimality on seed {seed}"

        opt_val = res_dp["value"]
        # All exact methods must produce identical optimal values
        assert res_bf["value"] == opt_val, f"BF != DP on seed {seed}"
        assert res_ilp["value"] == opt_val, f"ILP != DP on seed {seed}"
        assert res_bb["value"] == opt_val, f"B&B != DP on seed {seed}"

        # Greedy heuristic value must not exceed optimum
        assert res_greedy["value"] <= opt_val, f"Greedy exceeded DP on seed {seed}"

        # Invariant: Every method must return a feasible solution and match recomputed value
        for method_name, result in [
            ("BF", res_bf),
            ("DP", res_dp),
            ("ILP", res_ilp),
            ("B&B", res_bb),
            ("Greedy", res_greedy),
        ]:
            items = result["items"]
            tot_w = int(np.sum(weights[items])) if items else 0
            tot_v = int(np.sum(values[items])) if items else 0
            assert tot_w <= capacity, f"{method_name} exceeded capacity: {tot_w} > {capacity}"
            assert tot_v == result["value"], f"{method_name} value mismatch: {tot_v} != {result['value']}"


def test_knapsack_tiny_handmade() -> None:
    """Verify that DP knapsack finds the exact known optimum on a small hand-crafted instance."""
    # Problem: 4 items with weights [2, 3, 4, 5], values [3, 4, 5, 6], capacity 5
    # Item 0 (w=2, v=3) + Item 1 (w=3, v=4) => total weight = 5, total value = 7 (optimum)
    weights = np.array([2, 3, 4, 5])
    values = np.array([3, 4, 5, 6])
    capacity = 5

    res = dp_knapsack(weights, values, capacity)
    assert res["value"] == 7
    assert res["items"] == [0, 1]


def test_qubo_ising_and_maxcut_equivalence() -> None:
    """Verify QUBO-to-Ising mapping, Max-Cut formulation, and brute force search on 6 nodes."""
    graph = nx.random_regular_graph(3, 6, seed=42)

    # 1. Energy equivalence between random QUBO and its Ising conversion
    rng = np.random.default_rng(123)
    random_mat = rng.normal(0, 1, size=(6, 6))
    Q_rand = np.triu(random_mat)
    h, J, offset = qubo_to_ising(Q_rand)

    num_assignments = 1 << 6
    for state_idx in range(num_assignments):
        x = np.array([(state_idx >> bit) & 1 for bit in range(6)], dtype=int)
        # s_i = 1 - 2 * x_i ensures x=0 => s=+1 and x=1 => s=-1
        s = 1 - 2 * x

        e_qubo = qubo_energy(Q_rand, x)
        e_ising = ising_energy(h, J, offset, s)
        assert math.isclose(e_qubo, e_ising, abs_tol=1e-9), f"Discrepancy at state {state_idx}"

    # 2. Max-Cut QUBO energy matches -cut_value for all 64 assignments
    Q_mc = maxcut_qubo(graph)
    max_cut_direct = 0
    for state_idx in range(num_assignments):
        x = np.array([(state_idx >> bit) & 1 for bit in range(6)], dtype=int)
        e_mc = qubo_energy(Q_mc, x)
        cv = cut_value(graph, x)
        assert math.isclose(e_mc, -float(cv), abs_tol=1e-9), f"Max-Cut QUBO != -Cut at state {state_idx}"
        if cv > max_cut_direct:
            max_cut_direct = cv

    # 3. Brute force QUBO minimum matches direct max cut enumeration
    bf_res = brute_force_qubo(Q_mc, offset=0.0)
    bf_cut = int(round(-float(bf_res["best_energy"])))
    assert bf_cut == max_cut_direct, f"Brute force cut ({bf_cut}) != direct max cut ({max_cut_direct})"


def test_feature_selection_qubo_equivalence() -> None:
    """Verify that feature selection QUBO energy plus offset matches direct objective for all 256 states."""
    n = 8
    rng = np.random.default_rng(999)
    r = rng.uniform(0.1, 1.0, size=n)
    R_raw = rng.uniform(0.0, 0.5, size=(n, n))
    R = (R_raw + R_raw.T) / 2.0
    np.fill_diagonal(R, 1.0)

    k = 3
    alpha = 0.4
    lam = 1.5

    Q, offset = feature_selection_qubo(r, R, k=k, alpha=alpha, lam=lam)

    num_assignments = 1 << n
    for state_idx in range(num_assignments):
        x = np.array([(state_idx >> bit) & 1 for bit in range(n)], dtype=np.float64)

        qubo_total = qubo_energy(Q, x) + offset

        # Direct unexpanded formula: -sum r x + alpha * sum_{i<j} R_ij x_i x_j + lam * (sum x - k)^2
        linear_term = -float(np.sum(r * x))
        quad_term = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                quad_term += R[i, j] * x[i] * x[j]
        penalty_term = float(lam) * float((np.sum(x) - k) ** 2)

        direct_total = linear_term + alpha * quad_term + penalty_term

        assert math.isclose(qubo_total, direct_total, abs_tol=1e-9), (
            f"State {state_idx}: QUBO {qubo_total} != Direct {direct_total}"
        )

