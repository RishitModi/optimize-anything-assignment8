"""Unit and integration test suite for Track H (DBMS Index Selection Optimization)."""

from __future__ import annotations

import importlib.util
import math
import sqlite3
import sys
from pathlib import Path

# Ensure repository root and track root are placed on sys.path
_TRACK_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

from db_data import build_database, copy_database
from db_eval import evaluate_config, measure_index_sizes
from db_queries import CANDIDATES, Query, make_workload, result_hash, run_query
from db_select import (
    additive_benefit,
    predicted_benefit,
    select_bruteforce,
    select_dp_knapsack,
    select_greedy_marginal,
    select_greedy_static,
    select_ilp,
)


def _plain_bruteforce_knapsack_additive(
    values: np.ndarray,
    sizes: np.ndarray,
    budget: int | float,
) -> float:
    """Exact brute-force 0/1 knapsack on additive benefits using page units."""
    C = len(sizes)
    weights = [int(math.ceil(sizes[i] / 4096.0)) for i in range(C)]
    capacity = int(math.floor(budget / 4096.0))
    if capacity <= 0:
        return 0.0

    best_val = 0.0
    for mask in range(1 << C):
        cur_w = 0
        cur_v = 0.0
        for i in range(C):
            if (mask >> i) & 1:
                cur_w += weights[i]
                cur_v += values[i]
        if cur_w <= capacity and cur_v > best_val:
            best_val = cur_v
    return best_val


def test_database_generation_and_equivalence(tmp_path: Path) -> None:
    """Verify data scaling formulas, index sizing, and execution hash invariance."""
    sf = 0.001
    seed = 42
    db_file = tmp_path / "test_base.db"
    dataset_csv = tmp_path / "dataset.csv"

    # 1. Build database with scale factor 0.001
    info = build_database(db_file, sf=sf, seed=seed, out_csv=dataset_csv)
    counts = info["row_counts"]

    # Verify expected formula values:
    # customer = max(100, int(150000*0.001)) = 150
    # orders = max(1000, int(1500000*0.001)) = 1500
    # part = max(100, int(200000*0.001)) = 200
    # supplier = max(10, int(10000*0.001)) = 10
    # nation = 25
    assert counts["customer"] == 150
    assert counts["orders"] == 1500
    assert counts["part"] == 200
    assert counts["supplier"] == 10
    assert counts["nation"] == 25
    assert counts["lineitem"] >= 1500  # at least 1 lineitem per order

    # Verify dataset.csv was written
    assert dataset_csv.is_file()

    # 2. Verify candidate index sizing returns positive sizes
    cand_sizes = measure_index_sizes(db_file, CANDIDATES, work_dir=tmp_path)
    assert len(cand_sizes) == len(CANDIDATES)
    for cs in cand_sizes:
        assert cs["size_bytes"] > 0
        assert cs["build_time_s"] >= 0.0

    # 3. Workload execution and hash equivalence: baseline vs all 16 indexes
    conn = sqlite3.connect(str(db_file))
    queries = make_workload(conn, seed=seed, sf=sf)
    conn.close()
    assert len(queries) == 12

    # Baseline evaluation (no indexes)
    base_eval = evaluate_config(
        base_db=db_file,
        index_ids=[],
        candidates=CANDIDATES,
        queries=queries,
        repeats=1,
        work_dir=tmp_path,
    )

    # All-16-indexes evaluation
    all_cand_ids = [c[0] for c in CANDIDATES]
    all_eval = evaluate_config(
        base_db=db_file,
        index_ids=all_cand_ids,
        candidates=CANDIDATES,
        queries=queries,
        repeats=1,
        work_dir=tmp_path,
    )

    # Invariant: Every query must produce identical result rows under both configurations
    for q in queries:
        base_h = base_eval["result_hashes"][q.id]
        all_h = all_eval["result_hashes"][q.id]
        assert base_h == all_h, f"Result mismatch on {q.id}: baseline={base_h} vs all_indexes={all_h}"


def test_selection_algorithms_on_random_instances() -> None:
    """Verify optimality and feasibility invariants on 30 random problem instances."""
    rng = np.random.default_rng(2026)

    for trial in range(30):
        C = int(rng.integers(3, 11))
        Q = int(rng.integers(2, 7))

        # Benefit matrix with non-negative entries
        b = rng.uniform(0.0, 10.0, size=(Q, C))
        # Sizes in bytes (e.g. 4096 .. 65536)
        sizes = rng.integers(4096, 65536, size=C, dtype=np.int64)
        total_size = int(np.sum(sizes))

        # Random budget fraction between 10% and 90%
        budget = int(rng.uniform(0.10, 0.90) * total_size)

        # Run selection algorithms
        res_ilp = select_ilp(b, sizes, budget)
        res_bf = select_bruteforce(b, sizes, budget)
        res_gm = select_greedy_marginal(b, sizes, budget)
        res_gs = select_greedy_static(b, sizes, budget)
        res_dp = select_dp_knapsack(b, sizes, budget)

        # Invariant 1: Budget satisfaction across all methods
        assert res_ilp["total_size"] <= budget, f"ILP exceeded budget: {res_ilp['total_size']} > {budget}"
        assert res_bf["total_size"] <= budget, f"BF exceeded budget: {res_bf['total_size']} > {budget}"
        assert res_gm["total_size"] <= budget, f"GM exceeded budget: {res_gm['total_size']} > {budget}"
        assert res_gs["total_size"] <= budget, f"GS exceeded budget: {res_gs['total_size']} > {budget}"
        assert res_dp["total_size"] <= budget, f"DP exceeded budget: {res_dp['total_size']} > {budget}"

        # Invariant 2: ILP predicted benefit matches Brute Force (tolerance 1e-7)
        assert math.isclose(
            res_ilp["predicted_benefit"],
            res_bf["predicted_benefit"],
            abs_tol=1e-7,
        ), f"Trial {trial}: ILP ({res_ilp['predicted_benefit']}) != BF ({res_bf['predicted_benefit']})"

        # Invariant 3: ILP predicted benefit >= Greedy Marginal and >= Greedy Static
        assert res_ilp["predicted_benefit"] >= res_gm["predicted_benefit"] - 1e-7
        assert res_ilp["predicted_benefit"] >= res_gs["predicted_benefit"] - 1e-7

        # Invariant 4: DP Knapsack on additive values matches plain brute-force knapsack
        additive_vals = np.sum(b, axis=0)
        expected_knapsack_val = _plain_bruteforce_knapsack_additive(additive_vals, sizes, budget)
        assert math.isclose(
            res_dp["additive_benefit"],
            expected_knapsack_val,
            abs_tol=1e-7,
        ), f"Trial {trial}: DP additive ({res_dp['additive_benefit']}) != BF knapsack ({expected_knapsack_val})"


def test_smoke_run_all(tmp_path: Path) -> None:
    """Smoke test running quick mode via dynamic importlib specification."""
    run_all_path = _TRACK_ROOT / "run_all.py"
    assert run_all_path.is_file(), f"run_all.py missing at {run_all_path}"

    spec = importlib.util.spec_from_file_location("track_h_run_all", str(run_all_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    smoke_out = tmp_path / "smoke_results"
    ret = mod.main([
        "--quick",
        "--sf", "0.002",
        "--out", str(smoke_out),
        "--max-minutes", "3.0",
    ])

    assert ret == 0, f"run_all returned non-zero code {ret}"
    assert (smoke_out / "SUMMARY.md").is_file(), "SUMMARY.md was not generated"
    assert (smoke_out / "candidates.csv").is_file(), "candidates.csv was not generated"
    assert (smoke_out / "selections.csv").is_file(), "selections.csv was not generated"

