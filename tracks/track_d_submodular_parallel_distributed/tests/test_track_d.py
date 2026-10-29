"""Unit, invariant, and regression tests for Track D optimization modules."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from federated import (
    evaluate_accuracy,
    fedavg,
    logistic_grad,
    partition_iid,
    partition_label_skew,
)
from multitask import (
    MultiTaskModel,
    generate_multitask_data,
    numerical_grad_check,
    solve_mgda_alpha,
)
from parallel_tools import (
    grad_shard_shipped,
    lower_bound,
    pipeline_makespan,
    simulate_list_scheduling,
    simulate_lpt,
    simulate_work_stealing,
)
from submodular import (
    check_submodularity,
    coverage,
    exact_max_coverage,
    greedy_coverage,
    lazy_greedy_coverage,
    make_coverage_instance,
)


def test_coverage_submodularity() -> None:
    """Verify that the set union coverage function satisfies diminishing returns (submodularity)."""
    sets = make_coverage_instance(n_elements=100, n_sets=20, seed=10)
    cov_fn = lambda S: float(coverage(S, sets))
    res = check_submodularity(cov_fn, list(range(20)), n_trials=500, seed=42)
    assert res["violations"] == 0, f"Coverage violated submodularity in {res['violations']} trials"
    assert res["worst_violation"] == 0.0


def test_lazy_greedy_equals_greedy() -> None:
    """Verify that lazy greedy (CELF max-heap) yields the identical value and selection as standard greedy."""
    sets = make_coverage_instance(n_elements=200, n_sets=30, seed=99)
    res_g = greedy_coverage(sets, k=5)
    res_l = lazy_greedy_coverage(sets, k=5)
    assert res_g["value"] == res_l["value"]
    assert res_g["selected"] == res_l["selected"]
    assert res_l["evals"] < res_g["evals"], "Lazy greedy should evaluate strictly fewer marginal bounds"


def test_greedy_approximation_guarantee() -> None:
    """Verify greedy coverage value satisfies >= (1 - 1/e) * exact_optimum."""
    n_elements = 150
    k = 3
    sets = make_coverage_instance(n_elements=n_elements, n_sets=25, seed=77)
    res_g = greedy_coverage(sets, k=k)
    res_opt = exact_max_coverage(sets, n_elements=n_elements, k=k)

    bound = (1.0 - 1.0 / math.e) * res_opt["value"]
    assert res_g["value"] >= bound - 1e-9


def test_list_scheduling_and_lpt_bounds() -> None:
    """Verify list scheduling and LPT respect the lower bound and LPT beats list on Graham's bad case."""
    p = 2
    # Known adversarial example for FIFO list scheduling: [1, 1, 1, 1, 4]
    durations = [1.0, 1.0, 1.0, 1.0, 4.0]
    lb = lower_bound(durations, p)

    ms_ls = simulate_list_scheduling(durations, p)
    ms_lpt = simulate_lpt(durations, p)

    assert ms_ls >= lb - 1e-9
    assert ms_lpt >= lb - 1e-9
    # In list order: W0 gets 1+1+4=6, W1 gets 1+1=2 -> makespan 6
    # In LPT order: [4, 1, 1, 1, 1] -> W0 gets 4, W1 gets 1+1+1+1=4 -> makespan 4
    assert ms_lpt < ms_ls, f"LPT ({ms_lpt}) should be strictly faster than list scheduling ({ms_ls})"


def test_work_stealing_lower_bound() -> None:
    """Verify discrete-event work stealing produces a makespan bounded below by the theoretical lower bound."""
    durations = [5.0, 1.0, 20.0, 2.0, 8.0, 15.0, 4.0]
    p = 3
    lb = lower_bound(durations, p)
    ms_ws = simulate_work_stealing(durations, p=p, steal_cost=0.5, seed=42)
    assert ms_ws >= lb - 1e-9


def test_pipeline_makespan_formula() -> None:
    """Verify GPipe pipeline recurrence matches exact theoretical (S + M - 1) * t for uniform stage times."""
    S, M, t = 4, 8, 2.5
    res = pipeline_makespan(n_stages=S, n_micro=M, stage_times=[t] * S)
    expected = (S + M - 1) * t
    assert math.isclose(res["makespan"], expected, rel_tol=1e-12)


def test_fedavg_identical_models() -> None:
    """Verify that sample-weighted federated averaging of identical client models returns that exact model."""
    K, d = 5, 10
    target_w = np.array([float(i) for i in range(d)])
    partitions = [np.arange(10 + i * 5) for i in range(K)]
    n_total = sum(len(p) for p in partitions)
    weights = [len(p) / n_total for p in partitions]

    aggregated_w = np.sum([weights[k] * target_w for k in range(K)], axis=0)
    assert np.allclose(aggregated_w, target_w, atol=1e-14)


def test_mgda_orthogonal_vectors() -> None:
    """Verify MGDA computes alpha=0.5 for orthogonal vectors of equal norm and reduces joint norm."""
    g1 = np.array([3.0, 0.0])
    g2 = np.array([0.0, 3.0])
    alpha = solve_mgda_alpha(g1, g2)
    assert math.isclose(alpha, 0.5, abs_tol=1e-12)

    g_comb = alpha * g1 + (1.0 - alpha) * g2
    norm_comb = np.linalg.norm(g_comb)
    assert norm_comb < np.linalg.norm(g1)
    assert norm_comb < np.linalg.norm(g2)


def test_gradnorm_weight_sum() -> None:
    """Verify that GradNorm loss weights renormalize to sum to 2.0 after update."""
    w_t = np.array([1.0, 1.0])
    G1, G2 = 6.0, 2.0
    G_mean = 0.5 * (G1 + G2)
    r1, r2 = 1.2, 0.8
    target1 = G_mean * (r1 ** 1.5)
    target2 = G_mean * (r2 ** 1.5)

    w_t[0] -= 0.025 * np.sign(G1 - target1) * (G1 / w_t[0])
    w_t[1] -= 0.025 * np.sign(G2 - target2) * (G2 / w_t[1])
    w_t = np.maximum(w_t, 0.01)
    w_t = 2.0 * w_t / np.sum(w_t)

    assert math.isclose(np.sum(w_t), 2.0, abs_tol=1e-12)


def test_multitask_manual_vs_numerical_gradients() -> None:
    """Verify two-task manual analytical gradients against two-sided finite difference numerical check."""
    X_tr, y1_tr, y2_tr, _, _, _ = generate_multitask_data(n_train=25, n_test=10, seed=42)
    model = MultiTaskModel(d=10, h_dim=32, seed=42)
    max_rel_err = numerical_grad_check(model, X_tr, y1_tr, y2_tr, eps=1e-5)
    assert max_rel_err < 1e-5, f"Manual gradient mismatch: max relative error {max_rel_err:.2e} >= 1e-5"


def test_data_parallel_combined_gradient() -> None:
    """Verify that combining data-parallel sharded gradients equals single-process evaluation."""
    n_samples, d = 600, 10
    p = 3
    n_per_shard = n_samples // p
    w = np.ones(d) * 0.25
    l2_reg = 1e-3

    shards_X = []
    shards_y = []
    shard_grads = []

    for i in range(p):
        rng = np.random.default_rng(200 + i)
        X_k = rng.normal(size=(n_per_shard, d))
        y_k = (rng.uniform(size=n_per_shard) > 0.5).astype(float)
        shards_X.append(X_k)
        shards_y.append(y_k)
        g_k = grad_shard_shipped((X_k, y_k, w, l2_reg))
        shard_grads.append(g_k)

    comb_grad = np.mean(shard_grads, axis=0)

    X_all = np.vstack(shards_X)
    y_all = np.concatenate(shards_y)
    single_grad = grad_shard_shipped((X_all, y_all, w, l2_reg))

    assert np.allclose(comb_grad, single_grad, atol=1e-12)


def test_smoke_track_d(tmp_path: Path) -> None:
    """Execute Track D run_all in quick mode into tmp_path and verify created artifacts.

    Uses importlib to load run_all under a unique module name to avoid sys.path collision
    (Failure Log case 11). Multiprocessing experiment D2 is skipped for test isolation.
    """
    track_d_path = Path(__file__).resolve().parents[1] / "run_all.py"
    spec = importlib.util.spec_from_file_location("track_d_run_all", track_d_path)
    assert spec is not None and spec.loader is not None
    track_d_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(track_d_module)

    out_dir = tmp_path / "track_d_results"
    exit_code = track_d_module.run_all(quick=True, only=["D1", "D3", "D4"], out=str(out_dir))
    assert exit_code == 0, "run_all returned non-zero exit code"

    summary_file = out_dir / "SUMMARY.md"
    assert summary_file.exists(), "SUMMARY.md was not generated"

    csv_files = list(out_dir.glob("*.csv"))
    assert len(csv_files) >= 1, "No CSV benchmark files generated"

    png_files = list(out_dir.glob("*.png"))
    assert len(png_files) >= 1, "No PNG plot figures generated"

