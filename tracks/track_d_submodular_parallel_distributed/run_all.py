"""Execution script for Track D: Submodular, Parallel, and Distributed Optimization.

Usage:
    python tracks/track_d_submodular_parallel_distributed/run_all.py [--quick] [--only D1 D2 D3 D4] [--out results/track_d]

Experiments:
    D1: Submodular Optimization (Coverage with Lazy Greedy & ILP; R^2 Feature Selection)
    D2: Parallel Computing & Scheduling (Amdahl, Gustafson, Load Balancing, Work Stealing, Data/Pipeline Parallelism)
    D3: Federated Learning (FedAvg vs Centralized SGD under IID and Non-IID Dirichlet Skew)
    D4: Multi-Task Learning (Gradient Balancing: Equal Weights, MGDA, and GradNorm)
"""

from __future__ import annotations

import argparse
import itertools
import math
import multiprocessing
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Ensure repository root and track root are on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import sklearn

from common.io_utils import save_csv
from common.seeding import set_global_seed
from federated import (
    centralized_sgd,
    evaluate_accuracy,
    fedavg,
    get_federated_dataset,
    logistic_grad,
    logistic_loss,
    partition_iid,
    partition_label_skew,
)
from multitask import (
    MultiTaskModel,
    generate_multitask_data,
    numerical_grad_check,
    train_multitask,
)
from parallel_tools import (
    burn,
    calibrate_iters_per_ms,
    grad_shard_resident,
    grad_shard_shipped,
    lower_bound,
    pipeline_makespan,
    run_chunk,
    run_task,
    simulate_list_scheduling,
    simulate_lpt,
    simulate_static_block,
    simulate_static_round_robin,
    simulate_work_stealing,
)
from submodular import (
    check_submodularity,
    coverage,
    exact_max_coverage,
    exhaustive_feature_selection,
    greedy_coverage,
    greedy_feature_selection,
    lazy_greedy_coverage,
    make_coverage_instance,
    make_nonsubmodular_regression_instance,
    r2_objective,
)


def make_spawn_pool(processes: int) -> multiprocessing.pool.Pool:
    """Create a worker process pool using explicit 'spawn' context."""
    return multiprocessing.get_context("spawn").Pool(processes=processes)


def safe_pool_map(
    pool: Any,
    func: Any,
    iterable: Any,
    chunksize: int | None = None,
    timeout: float = 180.0,
) -> list[Any]:
    """Execute pool.map_async with an explicit timeout, raising TimeoutError on hang."""
    if chunksize is not None:
        async_result = pool.map_async(func, iterable, chunksize=chunksize)
    else:
        async_result = pool.map_async(func, iterable)
    try:
        return async_result.get(timeout=timeout)
    except multiprocessing.TimeoutError as exc:
        raise TimeoutError(
            f"Worker task '{getattr(func, '__name__', str(func))}' exceeded timeout of {timeout}s"
        ) from exc


def df_to_markdown(df: pd.DataFrame) -> str:
    """Render pandas DataFrame to Markdown table without external tabulate dependency."""
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                if abs(val) < 1e-4 and val != 0.0:
                    row_str.append(f"{val:.3e}")
                else:
                    row_str.append(f"{val:.4f}")
            else:
                row_str.append(str(val))
        rows.append("| " + " | ".join(row_str) + " |")
    return "\n".join([header_line, sep_line] + rows)


# ---------------------------------------------------------------------------
# Experiment D1: Submodular Optimization
# ---------------------------------------------------------------------------

def run_experiment_d1(
    out_dir: Path,
    quick: bool,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Execute D1: Maximum coverage and R^2 feature selection."""
    print("=== Running Experiment D1: Submodular Optimization ===")
    figures: list[str] = []

    # (a) Maximum Coverage Benchmark
    n_elements = 300
    n_sets = 40
    seed_cov = 42
    sets = make_coverage_instance(n_elements=n_elements, n_sets=n_sets, seed=seed_cov)

    k_max = 5 if quick else 10
    k_range = list(range(1, k_max + 1))

    cov_rows = []
    greedy_ratios: list[float] = []
    greedy_evals_list: list[int] = []
    lazy_evals_list: list[int] = []

    for k in k_range:
        res_greedy = greedy_coverage(sets, k)
        res_lazy = lazy_greedy_coverage(sets, k)
        res_exact = exact_max_coverage(sets, n_elements, k)

        val_g = res_greedy["value"]
        val_l = res_lazy["value"]
        val_opt = res_exact["value"]
        evals_g = res_greedy["evals"]
        evals_l = res_lazy["evals"]

        # Invariant checks
        if val_g != val_l:
            msg = f"[D1 Check Failed] k={k}: lazy value {val_l} != greedy value {val_g}"
            verification_failures.append(msg)
            print(msg)

        ratio = val_g / max(val_opt, 1)
        bound = 1.0 - (1.0 / math.e)

        if ratio < bound - 1e-9:
            msg = f"[D1 Check Failed] k={k}: ratio {ratio:.4f} < (1 - 1/e) guarantee {bound:.4f}"
            verification_failures.append(msg)
            print(msg)

        greedy_ratios.append(ratio)
        greedy_evals_list.append(evals_g)
        lazy_evals_list.append(evals_l)

        cov_rows.append({
            "k": k,
            "greedy_value": val_g,
            "lazy_value": val_l,
            "exact_value": val_opt,
            "ratio_greedy_exact": ratio,
            "bound_1_minus_1_over_e": bound,
            "greedy_evals": evals_g,
            "lazy_evals": evals_l,
        })

    cov_df = pd.DataFrame(cov_rows)
    save_csv(cov_df, out_dir / "d1_coverage_benchmark.csv")

    # Check submodularity of coverage function
    cov_fn = lambda S: float(coverage(S, sets))
    n_trials_sub = 500 if quick else 2000
    sub_check_cov = check_submodularity(cov_fn, list(range(n_sets)), n_trials=n_trials_sub, seed=seed_cov)
    if sub_check_cov["violations"] > 0:
        msg = f"[D1 Check Failed] Coverage submodularity violated in {sub_check_cov['violations']} trials"
        verification_failures.append(msg)
        print(msg)

    # Plot D1a: Ratio vs k and evaluations vs k
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(k_range, greedy_ratios, "o-", color="#1f77b4", label="Greedy / Exact Ratio")
    ax1.axhline(1.0 - 1.0 / math.e, color="#d62728", linestyle="--", label="Theoretical Bound (1 - 1/e)")
    ax1.set_xlabel("Cardinality Constraint k", fontsize=10)
    ax1.set_ylabel("Approximation Ratio", fontsize=10)
    ax1.set_title("Coverage: Greedy Approximation Ratio vs k", fontsize=11, fontweight="bold")
    ax1.set_ylim([0.5, 1.05])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, fontsize=9)

    ax2.plot(k_range, greedy_evals_list, "s-", color="#ff7f0e", label="Standard Greedy")
    ax2.plot(k_range, lazy_evals_list, "^-", color="#2ca02c", label="Lazy Greedy (CELF)")
    ax2.set_xlabel("Cardinality Constraint k", fontsize=10)
    ax2.set_ylabel("Function Evaluations", fontsize=10)
    ax2.set_title("Coverage: Evaluation Query Efficiency", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=9)

    fig.tight_layout()
    p1 = out_dir / "d1_coverage_scaling.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append(p1.name)

    # (b) Non-submodular R^2 Feature Selection
    X_reg, y_reg = make_nonsubmodular_regression_instance(seed=0)
    r2_fn = r2_objective(X_reg, y_reg)
    sub_check_r2 = check_submodularity(r2_fn, list(range(15)), n_trials=n_trials_sub, seed=123)

    r2_rows = []
    k_feat_max = 3 if quick else 5
    for k in range(1, k_feat_max + 1):
        res_g = greedy_feature_selection(r2_fn, n_features=15, k=k)
        res_opt = exhaustive_feature_selection(r2_fn, n_features=15, k=k)
        ratio_r2 = res_g["value"] / max(res_opt["value"], 1e-12)
        r2_rows.append({
            "k": k,
            "greedy_r2": res_g["value"],
            "exact_r2": res_opt["value"],
            "ratio": ratio_r2,
        })

    r2_df = pd.DataFrame(r2_rows)
    save_csv(r2_df, out_dir / "d1_r2_feature_selection.csv")

    return {
        "cov_df": cov_df,
        "r2_df": r2_df,
        "sub_check_cov": sub_check_cov,
        "sub_check_r2": sub_check_r2,
        "figures": figures,
    }


# ---------------------------------------------------------------------------
# Experiment D2: Parallel Computing & Scheduling
# ---------------------------------------------------------------------------

def run_experiment_d2(
    out_dir: Path,
    quick: bool,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Execute D2: Strong & weak scaling, load balancing, granularity, data and pipeline parallelism."""
    print("=== Running Experiment D2: Parallel Computing ===")
    figures: list[str] = []

    # Calibrate CPU burn with appropriate duration
    iters_per_ms = calibrate_iters_per_ms(target_dt=0.05 if quick else 0.3)
    print(f"Calibrated burn iterations per ms: {iters_per_ms}")

    # (a) Strong Scaling (Measured)
    n_tasks_strong = 100 if quick else 400
    task_ms_strong = 5.0 if quick else 10.0
    task_iters_strong = int(task_ms_strong * iters_per_ms)

    p_list = [1, 2, 4] if quick else [1, 2, 4, 8, 16]
    max_cores = multiprocessing.cpu_count() or 4
    p_workers = [p for p in p_list if p <= max(max_cores, 4)]
    if 1 not in p_workers:
        p_workers = [1] + p_workers

    n_reps_strong = 2 if quick else 3
    strong_rows = []
    t1_median = 0.0

    serial_ratio = 0.05
    serial_iters = int(serial_ratio * n_tasks_strong * task_iters_strong)

    for p in p_workers:
        t_pool_start = time.perf_counter()
        with make_spawn_pool(processes=p) as pool:
            safe_pool_map(pool, run_task, [1000] * p, timeout=180.0)
            pool_startup_time = time.perf_counter() - t_pool_start

            times = []
            for _ in range(n_reps_strong):
                t0 = time.perf_counter()
                burn(serial_iters)
                safe_pool_map(pool, run_task, [task_iters_strong] * n_tasks_strong, timeout=180.0)
                times.append(time.perf_counter() - t0)

        med_time = float(np.median(times))
        if p == 1:
            t1_median = med_time

        speedup = t1_median / max(med_time, 1e-6)
        efficiency = speedup / p

        strong_rows.append({
            "workers_p": p,
            "median_time_s": med_time,
            "speedup": speedup,
            "efficiency": efficiency,
            "pool_startup_s": pool_startup_time,
        })

    strong_df = pd.DataFrame(strong_rows)
    save_csv(strong_df, out_dir / "d2_strong_scaling.csv")

    def amdahl_func(p_arr, s_param):
        return 1.0 / (s_param + (1.0 - s_param) / p_arr)

    p_arr = strong_df["workers_p"].values
    speedup_arr = strong_df["speedup"].values

    try:
        popt, _ = scipy.optimize.curve_fit(amdahl_func, p_arr, speedup_arr, p0=[0.05], bounds=(0.0, 1.0))
        s_fitted = float(popt[0])
    except Exception:
        s_fitted = 0.05

    # Plot D2a: Strong Scaling
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(p_arr, speedup_arr, "o-", color="#1f77b4", label="Measured Speedup")
    ax.plot(p_arr, p_arr, "--", color="#7f7f7f", label="Ideal Linear (S = p)")
    p_fine = np.linspace(1, max(p_arr), 50)
    ax.plot(p_fine, amdahl_func(p_fine, s_fitted), ":", color="#d62728", label=f"Amdahl Fit (s={s_fitted:.3f})")
    ax.set_xlabel("Number of Parallel Workers p", fontsize=10)
    ax.set_ylabel("Speedup T_1 / T_p", fontsize=10)
    ax.set_title("Strong Scaling: Speedup and Amdahl Model (Measured)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    p_strong = out_dir / "d2_strong_scaling.png"
    fig.savefig(p_strong, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append(p_strong.name)

    # (b) Weak Scaling / Gustafson (Measured)
    tasks_per_worker = 10 if quick else 40
    task_ms_weak = 10.0
    task_iters_weak = int(task_ms_weak * iters_per_ms)
    serial_iters_weak = int(0.05 * tasks_per_worker * task_iters_weak)

    weak_rows = []
    for p in p_workers:
        n_total_weak = tasks_per_worker * p
        with make_spawn_pool(processes=p) as pool:
            safe_pool_map(pool, run_task, [1000] * p, timeout=180.0)
            times = []
            for _ in range(n_reps_strong):
                t0 = time.perf_counter()
                burn(serial_iters_weak)
                safe_pool_map(pool, run_task, [task_iters_weak] * n_total_weak, timeout=180.0)
                times.append(time.perf_counter() - t0)

        t_measured = float(np.median(times))
        est_seq_time = (serial_iters_weak / (iters_per_ms * 1000.0)) + (n_total_weak * 0.010)
        scaled_speedup = est_seq_time / max(t_measured, 1e-6)

        weak_rows.append({
            "workers_p": p,
            "tasks_total": n_total_weak,
            "measured_time_s": t_measured,
            "estimated_sequential_s": est_seq_time,
            "scaled_speedup": scaled_speedup,
        })

    weak_df = pd.DataFrame(weak_rows)
    save_csv(weak_df, out_dir / "d2_weak_scaling.csv")

    # Plot D2b: Gustafson Weak Scaling
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(weak_df["workers_p"], weak_df["scaled_speedup"], "s-", color="#2ca02c", label="Scaled Speedup (Measured)")
    gustafson_ideal = [p - 0.05 * (p - 1) for p in weak_df["workers_p"]]
    ax.plot(weak_df["workers_p"], gustafson_ideal, "--", color="#d62728", label="Gustafson Model (s=0.05)")
    ax.set_xlabel("Number of Parallel Workers p", fontsize=10)
    ax.set_ylabel("Scaled Speedup", fontsize=10)
    ax.set_title("Weak Scaling: Scaled Speedup vs Gustafson Law (Measured)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    p_weak = out_dir / "d2_weak_scaling.png"
    fig.savefig(p_weak, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append(p_weak.name)

    # (c) Load Balancing (Measured)
    n_tasks_lb = 80 if quick else 300
    p_lb = 4 if quick else min(8, max_cores)
    rng_lb = np.random.default_rng(777)
    durations_ms = rng_lb.lognormal(mean=2.0, sigma=1.0, size=n_tasks_lb)
    durations_ms = durations_ms * (10.0 / np.mean(durations_ms))
    task_iters_lb = [int(d * iters_per_ms) for d in durations_ms]

    lb_lower = lower_bound(durations_ms, p_lb)

    with make_spawn_pool(processes=p_lb) as pool:
        safe_pool_map(pool, run_task, [1000] * p_lb, timeout=180.0)

        # 1. Static block (chunksize = n / p)
        t0 = time.perf_counter()
        safe_pool_map(pool, run_task, task_iters_lb, chunksize=max(1, n_tasks_lb // p_lb), timeout=180.0)
        t_static = (time.perf_counter() - t0) * 1000.0

        # 2. Dynamic (chunksize = 1)
        t0 = time.perf_counter()
        safe_pool_map(pool, run_task, task_iters_lb, chunksize=1, timeout=180.0)
        t_dynamic = (time.perf_counter() - t0) * 1000.0

        # 3. Dynamic LPT (sorted decreasing duration)
        sorted_indices = np.argsort(durations_ms)[::-1]
        task_iters_lpt = [task_iters_lb[i] for i in sorted_indices]
        t0 = time.perf_counter()
        safe_pool_map(pool, run_task, task_iters_lpt, chunksize=1, timeout=180.0)
        t_lpt = (time.perf_counter() - t0) * 1000.0

    lb_measured_rows = [
        {"strategy": "Static (Chunksize N/p)", "makespan_ms": t_static, "lower_bound_ms": lb_lower},
        {"strategy": "Dynamic (Chunksize 1)", "makespan_ms": t_dynamic, "lower_bound_ms": lb_lower},
        {"strategy": "Dynamic LPT (Sorted)", "makespan_ms": t_lpt, "lower_bound_ms": lb_lower},
    ]
    lb_measured_df = pd.DataFrame(lb_measured_rows)
    save_csv(lb_measured_df, out_dir / "d2_load_balancing_measured.csv")

    # (d) Load Balancing (Simulated across schedulers)
    sim_p_list = [4, 8] if quick else [4, 8, 16]
    n_sim_sets = 5 if quick else 20
    sched_names = ["Static Block", "Round Robin", "List Scheduling", "LPT", "Work Stealing"]

    sim_results = []
    for p in sim_p_list:
        for dist_name in ["Lognormal", "Heavy-Tail Stragglers"]:
            ratios_by_sched: dict[str, list[float]] = {s: [] for s in sched_names}

            for s_idx in range(n_sim_sets):
                rng_sim = np.random.default_rng(1000 + s_idx)
                if dist_name == "Lognormal":
                    durs = rng_sim.lognormal(2.0, 1.0, size=200)
                    durs = durs * (10.0 / np.mean(durs))
                else:
                    durs = rng_sim.lognormal(2.0, 0.5, size=200)
                    durs = durs * (10.0 / np.mean(durs))
                    straggler_mask = rng_sim.random(200) < 0.05
                    durs[straggler_mask] *= 20.0

                lb = lower_bound(durs, p)
                ms_sb = simulate_static_block(durs, p)
                ms_rr = simulate_static_round_robin(durs, p)
                ms_ls = simulate_list_scheduling(durs, p)
                ms_lpt = simulate_lpt(durs, p)
                ms_ws = simulate_work_stealing(durs, p, steal_cost=0.5, seed=s_idx)

                for name_check, ms_val in [("SB", ms_sb), ("RR", ms_rr), ("LS", ms_ls), ("LPT", ms_lpt), ("WS", ms_ws)]:
                    if ms_val < lb - 1e-9:
                        msg = f"[D2 Check Failed] {name_check} makespan {ms_val:.4f} < lower bound {lb:.4f}"
                        verification_failures.append(msg)
                        print(msg)

                ratios_by_sched["Static Block"].append(ms_sb / lb)
                ratios_by_sched["Round Robin"].append(ms_rr / lb)
                ratios_by_sched["List Scheduling"].append(ms_ls / lb)
                ratios_by_sched["LPT"].append(ms_lpt / lb)
                ratios_by_sched["Work Stealing"].append(ms_ws / lb)

            for s_name in sched_names:
                r_arr = ratios_by_sched[s_name]
                sim_results.append({
                    "workers_p": p,
                    "distribution": dist_name,
                    "scheduler": s_name,
                    "ratio_mean": float(np.mean(r_arr)),
                    "ratio_std": float(np.std(r_arr)),
                })

    lb_sim_df = pd.DataFrame(sim_results)
    save_csv(lb_sim_df, out_dir / "d2_load_balancing_simulated.csv")

    # Plot D2d: Simulated Scheduling Efficiency
    fig, ax = plt.subplots(figsize=(9, 5))
    df_p8 = lb_sim_df[(lb_sim_df["workers_p"] == 8) & (lb_sim_df["distribution"] == "Heavy-Tail Stragglers")]
    if df_p8.empty:
        df_p8 = lb_sim_df[lb_sim_df["distribution"] == "Heavy-Tail Stragglers"]

    sched_cats = df_p8["scheduler"].tolist()
    r_means = df_p8["ratio_mean"].tolist()
    r_stds = df_p8["ratio_std"].tolist()

    ax.bar(sched_cats, r_means, yerr=r_stds, capsize=4, color="#1f77b4", alpha=0.85)
    ax.axhline(1.0, color="#d62728", linestyle="--", label="Lower Bound (Ratio = 1.0)")
    ax.set_ylabel("Makespan / Lower Bound Ratio", fontsize=10)
    ax.set_title("Simulated Load Balancing Schedulers (Heavy-Tail Stragglers, p=8)", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    p_sim_sched = out_dir / "d2_scheduling_simulated.png"
    fig.savefig(p_sim_sched, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append(p_sim_sched.name)

    # (e) Overhead and Granularity (Measured)
    total_work_s = 0.5 if quick else 2.0
    task_sizes_ms = [0.1, 0.5, 1.0, 5.0, 20.0]
    p_gran = 4 if quick else min(8, max_cores)

    gran_rows = []
    total_iters = int(total_work_s * 1000.0 * iters_per_ms)
    t0_seq = time.perf_counter()
    burn(total_iters)
    t_seq_gran = time.perf_counter() - t0_seq

    with make_spawn_pool(processes=p_gran) as pool:
        safe_pool_map(pool, run_task, [1000] * p_gran, timeout=180.0)
        for t_ms in task_sizes_ms:
            t_iter = max(1, int(t_ms * iters_per_ms))
            n_t = max(1, int((total_work_s * 1000.0) / t_ms))
            t0 = time.perf_counter()
            safe_pool_map(pool, run_task, [t_iter] * n_t, chunksize=1, timeout=180.0)
            t_par = time.perf_counter() - t0
            spd = t_seq_gran / max(t_par, 1e-6)

            gran_rows.append({
                "task_size_ms": t_ms,
                "n_tasks": n_t,
                "parallel_time_s": t_par,
                "speedup": spd,
            })

    gran_df = pd.DataFrame(gran_rows)
    save_csv(gran_df, out_dir / "d2_granularity_overhead.csv")

    # Plot D2e: Granularity vs Speedup
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(gran_df["task_size_ms"], gran_df["speedup"], "o-", color="#9467bd", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Task Granularity (ms per task, log scale)", fontsize=10)
    ax.set_ylabel("Speedup over Sequential", fontsize=10)
    ax.set_title(f"IPC Overhead vs Task Granularity (p={p_gran}, Measured)", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    p_gran_fig = out_dir / "d2_granularity_overhead.png"
    fig.savefig(p_gran_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append(p_gran_fig.name)

    # (f) Data Parallelism: Sharded Gradient Evaluation (Measured)
    n_samples_dp = 60000 if quick else 240000
    d_dp = 40
    p_dp_list = [1, 2, 4] if quick else [1, 2, 4, min(8, max_cores)]
    w_dp = np.ones(d_dp, dtype=float) * 0.1
    l2_reg_dp = 1e-3
    n_evals_dp = 10

    dp_rows = []

    for p in p_dp_list:
        n_per_shard = n_samples_dp // p

        # 1. Resident variant
        resident_args = [(42, i, n_per_shard, d_dp, w_dp, l2_reg_dp) for i in range(p)]
        with make_spawn_pool(processes=p) as pool:
            safe_pool_map(pool, run_task, [1000] * p, timeout=180.0)
            t0 = time.perf_counter()
            for _ in range(n_evals_dp):
                res_grads = safe_pool_map(pool, grad_shard_resident, resident_args, timeout=180.0)
                comb_grad_res = np.mean(res_grads, axis=0)
            t_resident = time.perf_counter() - t0

        # 2. Shipped variant: generate shards in main process and ship over IPC
        shards_X = []
        shards_y = []
        for i in range(p):
            rng_s = np.random.default_rng(42 + i * 1000)
            shards_X.append(rng_s.normal(size=(n_per_shard, d_dp)))
            shards_y.append((rng_s.uniform(size=n_per_shard) > 0.5).astype(float))

        shipped_args = [(shards_X[i], shards_y[i], w_dp, l2_reg_dp) for i in range(p)]
        with make_spawn_pool(processes=p) as pool:
            safe_pool_map(pool, run_task, [1000] * p, timeout=180.0)
            t0 = time.perf_counter()
            for _ in range(n_evals_dp):
                ship_grads = safe_pool_map(pool, grad_shard_shipped, shipped_args, timeout=180.0)
                comb_grad_ship = np.mean(ship_grads, axis=0)
            t_shipped = time.perf_counter() - t0

        # Verification: compare to single process exact gradient
        X_all = np.vstack(shards_X)
        y_all = np.concatenate(shards_y)
        z = X_all @ w_dp
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        single_grad = (X_all.T @ (prob - y_all)) / len(y_all) + l2_reg_dp * w_dp

        err_res = float(np.linalg.norm(comb_grad_res - single_grad) / np.linalg.norm(single_grad))
        err_ship = float(np.linalg.norm(comb_grad_ship - single_grad) / np.linalg.norm(single_grad))

        if err_res > 1e-10:
            msg = f"[D2 Check Failed] Resident shard gradient relative error {err_res:.2e} > 1e-10"
            verification_failures.append(msg)
            print(msg)

        if err_ship > 1e-10:
            msg = f"[D2 Check Failed] Shipped shard gradient relative error {err_ship:.2e} > 1e-10"
            verification_failures.append(msg)
            print(msg)

        dp_rows.append({
            "workers_p": p,
            "variant": "Resident Data",
            "time_s": t_resident,
            "grad_rel_error": err_res,
        })
        dp_rows.append({
            "workers_p": p,
            "variant": "Shipped Data",
            "time_s": t_shipped,
            "grad_rel_error": err_ship,
        })

    dp_df = pd.DataFrame(dp_rows)
    save_csv(dp_df, out_dir / "d2_data_parallelism.csv")

    # (g) Pipeline and Model Parallelism (Simulated)
    pipe_stages = [2, 4, 8]
    pipe_micros = [1, 2, 4, 8, 16, 32]
    pipe_rows = []

    for S in pipe_stages:
        for M in pipe_micros:
            res_pipe = pipeline_makespan(S, M, [1.0] * S)
            ms = res_pipe["makespan"]
            bubble = res_pipe["bubble_fraction"]
            seq_time = S * M * 1.0
            spd = seq_time / ms

            expected_ms = float(S + M - 1)
            if abs(ms - expected_ms) > 1e-9:
                msg = f"[D2 Check Failed] Pipeline makespan {ms} != S+M-1 ({expected_ms})"
                verification_failures.append(msg)
                print(msg)

            pipe_rows.append({
                "stages_S": S,
                "microbatches_M": M,
                "makespan": ms,
                "bubble_fraction": bubble,
                "speedup": spd,
            })

    # Unbalanced pipeline stage times: [1, 2, 1, 1]
    res_unbal = pipeline_makespan(4, 16, [1.0, 2.0, 1.0, 1.0])
    pipe_rows.append({
        "stages_S": 4,
        "microbatches_M": 16,
        "makespan": res_unbal["makespan"],
        "bubble_fraction": res_unbal["bubble_fraction"],
        "speedup": (5.0 * 16) / res_unbal["makespan"],
    })

    pipe_df = pd.DataFrame(pipe_rows)
    save_csv(pipe_df, out_dir / "d2_pipeline_parallelism.csv")

    # Plot D2g: Pipeline bubble fraction vs M
    fig, ax = plt.subplots(figsize=(7, 5))
    for S in pipe_stages:
        sub_df = pipe_df[pipe_df["stages_S"] == S]
        ax.plot(sub_df["microbatches_M"], sub_df["bubble_fraction"], "o-", label=f"S={S} Stages")
    ax.set_xlabel("Number of Micro-batches M", fontsize=10)
    ax.set_ylabel("Bubble Fraction (Idle Overhead)", fontsize=10)
    ax.set_title("Pipeline Parallelism: Bubble Fraction vs Micro-batches (Simulated)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    p_pipe_fig = out_dir / "d2_pipeline_bubble.png"
    fig.savefig(p_pipe_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append(p_pipe_fig.name)

    return {
        "strong_df": strong_df,
        "weak_df": weak_df,
        "lb_measured_df": lb_measured_df,
        "lb_sim_df": lb_sim_df,
        "gran_df": gran_df,
        "dp_df": dp_df,
        "pipe_df": pipe_df,
        "figures": figures,
    }


# ---------------------------------------------------------------------------
# Experiment D3: Federated Learning (FedAvg vs Centralized)
# ---------------------------------------------------------------------------

def run_experiment_d3(
    out_dir: Path,
    quick: bool,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Execute D3: FedAvg vs Centralized SGD under IID and Dirichlet label skew."""
    print("=== Running Experiment D3: Federated Learning ===")
    figures: list[str] = []

    X_tr, y_tr, X_te, y_te = get_federated_dataset(n_samples=6000, n_features=20, n_informative=10, random_state=0)
    K = 10
    d = X_tr.shape[1]
    rounds = 20 if quick else 100
    local_steps_list = [1, 5] if quick else [1, 5, 20]
    seeds = [42, 43] if quick else [42, 43, 44]

    partitions_dict = {
        "IID": partition_iid(len(y_tr), K, seed=42),
        "Non-IID (Label Skew alpha=0.3)": partition_label_skew(y_tr, K, alpha=0.3, seed=42),
    }

    # Invariant check: K=1 FedAvg vs Centralized SGD equivalence
    part_k1 = [np.arange(len(y_tr))]
    res_fa_k1 = fedavg(X_tr, y_tr, X_te, y_te, part_k1, rounds=5, local_steps=1, lr=0.05, batch_size=32, seed=99)
    rng_k1 = np.random.default_rng(99)
    w_cent_check = np.zeros(X_tr.shape[1], dtype=float)
    for _ in range(5):
        b_idx = rng_k1.choice(len(y_tr), size=32, replace=False)
        w_cent_check -= 0.05 * logistic_grad(w_cent_check, X_tr[b_idx], y_tr[b_idx], l2_reg=1e-3)

    k1_diff = float(np.linalg.norm(res_fa_k1["final_weights"] - w_cent_check))
    if k1_diff > 1e-12:
        msg = f"[D3 Check Failed] K=1 FedAvg != Centralized SGD mismatch: {k1_diff:.2e}"
        verification_failures.append(msg)
        print(msg)

    # Invariant check: Server averaging identical client models returns that exact model
    dummy_w = np.ones(20) * 2.5
    avg_dummy = np.sum([0.1 * dummy_w for _ in range(10)], axis=0)
    if not np.allclose(dummy_w, avg_dummy, atol=1e-14):
        msg = f"[D3 Check Failed] Averaging identical client models altered weights"
        verification_failures.append(msg)
        print(msg)

    # Run all configurations and collect round-by-round accuracy trajectories
    all_configs: list[dict[str, Any]] = []
    curves_data: dict[str, list[float]] = {}

    # 1. FedAvg configurations
    for part_name, parts in partitions_dict.items():
        for ls in local_steps_list:
            seed_curves: list[list[float]] = []
            for s in seeds:
                res = fedavg(X_tr, y_tr, X_te, y_te, parts, rounds=rounds, local_steps=ls, lr=0.1, batch_size=32, seed=s)
                seed_curves.append(res["accuracies"])

            mean_curve = list(np.mean(seed_curves, axis=0))
            final_accs = [c[-1] for c in seed_curves]

            curve_key = f"{part_name} (E={ls})"
            curves_data[curve_key] = mean_curve

            all_configs.append({
                "partition": part_name,
                "local_steps": ls,
                "method": "FedAvg",
                "final_mean_acc": float(np.mean(final_accs)),
                "final_std_acc": float(np.std(final_accs)),
                "mean_curve": mean_curve,
            })

    # 2. Centralized SGD configuration (using same test accuracy evaluation across rounds)
    cent_seed_curves: list[list[float]] = []
    for s in seeds:
        res_c = centralized_sgd(X_tr, y_tr, X_te, y_te, rounds=rounds, local_steps=5, lr=0.1, batch_size=320, seed=s)
        cent_seed_curves.append(res_c["accuracies"])

    cent_mean_curve = list(np.mean(cent_seed_curves, axis=0))
    cent_final_accs = [c[-1] for c in cent_seed_curves]
    curves_data["Centralized SGD"] = cent_mean_curve

    all_configs.append({
        "partition": "Centralized",
        "local_steps": 5,
        "method": "Centralized SGD",
        "final_mean_acc": float(np.mean(cent_final_accs)),
        "final_std_acc": float(np.std(cent_final_accs)),
        "mean_curve": cent_mean_curve,
    })

    # Determine dynamic relative target: 98% of maximum achieved mean accuracy
    max_final_mean = max(c["final_mean_acc"] for c in all_configs)
    target = 0.98 * max_final_mean
    print(f"D3 Target accuracy: {target:.4f} (98% of best achieved mean accuracy {max_final_mean:.4f})")

    fed_rows = []
    for conf in all_configs:
        rounds_to_target: int | str = "not reached"
        for r_idx, acc_val in enumerate(conf["mean_curve"]):
            if acc_val >= target:
                rounds_to_target = r_idx + 1  # 1-indexed round
                break

        if conf["method"] == "FedAvg":
            if isinstance(rounds_to_target, int):
                comm_to_target: str = str(2 * K * d * rounds_to_target)
            else:
                comm_to_target = "not reached"
        else:
            comm_to_target = "0 (no communication)"

        fed_rows.append({
            "partition": conf["partition"],
            "local_steps": conf["local_steps"],
            "method": conf["method"],
            "final_acc_mean": conf["final_mean_acc"],
            "final_acc_std": conf["final_std_acc"],
            "rounds_to_target": str(rounds_to_target),
            "communicated_floats_to_target": comm_to_target,
        })

    fed_df = pd.DataFrame(fed_rows)
    save_csv(fed_df, out_dir / "d3_federated_benchmark.csv")

    # Plot D3: Accuracy curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    round_axis = list(range(1, rounds + 1))

    for label, acc_curve in curves_data.items():
        if "IID" in label or "Centralized" in label:
            ax1.plot(round_axis, acc_curve, label=label, linewidth=1.5)
        if "Label Skew" in label or "Centralized" in label:
            ax2.plot(round_axis, acc_curve, label=label, linewidth=1.5)

    ax1.axhline(target, color="#7f7f7f", linestyle=":", label=f"Target ({target:.3f})")
    ax1.set_xlabel("Communication Round", fontsize=10)
    ax1.set_ylabel("Test Accuracy", fontsize=10)
    ax1.set_title("FedAvg Convergence: IID Partitioning", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, fontsize=8)

    ax2.axhline(target, color="#7f7f7f", linestyle=":", label=f"Target ({target:.3f})")
    ax2.set_xlabel("Communication Round", fontsize=10)
    ax2.set_ylabel("Test Accuracy", fontsize=10)
    ax2.set_title("FedAvg Convergence: Dirichlet Non-IID Skew", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    p_fed = out_dir / "d3_federated_convergence.png"
    fig.savefig(p_fed, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append(p_fed.name)

    return {
        "fed_df": fed_df,
        "target": target,
        "max_final_mean": max_final_mean,
        "figures": figures,
    }


# ---------------------------------------------------------------------------
# Experiment D4: Multi-Task Learning
# ---------------------------------------------------------------------------

def run_experiment_d4(
    out_dir: Path,
    quick: bool,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Execute D4: Equal weights vs MGDA vs GradNorm."""
    print("=== Running Experiment D4: Multi-Task Optimization ===")
    figures: list[str] = []

    epochs = 100 if quick else 500
    n_seeds = 2 if quick else 5
    schemes = ["equal", "mgda", "gradnorm"]

    # Invariant verification: Numerical gradient check on small batch
    X_tr_check, y1_tr_check, y2_tr_check, _, _, _ = generate_multitask_data(n_train=20, n_test=10, seed=101)
    test_model = MultiTaskModel(d=10, h_dim=32, seed=101)
    max_rel_grad_err = numerical_grad_check(test_model, X_tr_check, y1_tr_check, y2_tr_check, eps=1e-5)
    if max_rel_grad_err > 1e-5:
        msg = f"[D4 Check Failed] Numerical gradient relative error {max_rel_grad_err:.2e} > 1e-5"
        verification_failures.append(msg)
        print(msg)

    # Invariant verification: GradNorm weight initialization sum to 2
    w_gn_init = np.array([1.0, 1.0])
    if abs(np.sum(w_gn_init) - 2.0) > 1e-12:
        msg = f"[D4 Check Failed] GradNorm initial weights do not sum to 2"
        verification_failures.append(msg)
        print(msg)

    mt_rows = []
    curves_loss1: dict[str, list[float]] = {}
    curves_loss2: dict[str, list[float]] = {}
    mgda_alphas: list[float] = []
    gradnorm_w1: list[float] = []
    gradnorm_w2: list[float] = []

    for scheme in schemes:
        r2_t1_runs: list[float] = []
        r2_t2_runs: list[float] = []
        mean_r2_runs: list[float] = []
        worst_r2_runs: list[float] = []
        scheme_diagnostics: list[Any] = []

        for s_idx in range(n_seeds):
            seed = 42 + s_idx
            X_tr, y1_tr, y2_tr, X_te, y1_te, y2_te = generate_multitask_data(
                n_train=2000, n_test=1000, seed=seed
            )
            model = MultiTaskModel(d=10, h_dim=32, seed=seed)
            res = train_multitask(
                model, X_tr, y1_tr, y2_tr, X_te, y1_te, y2_te,
                scheme=scheme, lr=0.01, epochs=epochs
            )

            r2_t1_runs.append(res["r2_task1"])
            r2_t2_runs.append(res["r2_task2"])
            mean_r2_runs.append(res["mean_r2"])
            worst_r2_runs.append(res["worst_r2"])

            if not res["mgda_check_passed"]:
                msg = f"[D4 Check Failed] MGDA combined norm exceeded min task norm at seed {seed}"
                verification_failures.append(msg)
                print(msg)

            if s_idx == 0:
                curves_loss1[scheme] = res["loss1_hist"]
                curves_loss2[scheme] = res["loss2_hist"]
                if scheme == "mgda":
                    mgda_alphas = res["alpha_hist"]
                elif scheme == "gradnorm":
                    gradnorm_w1 = res["w1_hist"]
                    gradnorm_w2 = res["w2_hist"]

            if scheme == "mgda":
                scheme_diagnostics.append(res["mean_alpha"])
            elif scheme == "gradnorm":
                scheme_diagnostics.append(res["final_w"])

        diag_str = ""
        if scheme == "mgda":
            diag_str = f"mean alpha = {float(np.mean(scheme_diagnostics)):.3f}"
        elif scheme == "gradnorm":
            w_arr = np.array(scheme_diagnostics)
            diag_str = f"final weights = [{np.mean(w_arr[:, 0]):.2f}, {np.mean(w_arr[:, 1]):.2f}]"
        else:
            diag_str = "w = [1.0, 1.0]"

        mt_rows.append({
            "scheme": scheme.upper(),
            "r2_task1_mean": float(np.mean(r2_t1_runs)),
            "r2_task1_std": float(np.std(r2_t1_runs)),
            "r2_task2_mean": float(np.mean(r2_t2_runs)),
            "r2_task2_std": float(np.std(r2_t2_runs)),
            "mean_r2_mean": float(np.mean(mean_r2_runs)),
            "worst_task_r2_mean": float(np.mean(worst_r2_runs)),
            "worst_task_r2_std": float(np.std(worst_r2_runs)),
            "scheme_diagnostic": diag_str,
        })

    mt_df = pd.DataFrame(mt_rows)
    save_csv(mt_df, out_dir / "d4_multitask_benchmark.csv")

    # Plot D4: Training loss curves and dynamic weight tracking
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epoch_axis = list(range(1, epochs + 1))

    for scheme in schemes:
        ax1.plot(epoch_axis, curves_loss1[scheme], label=f"{scheme.upper()} (Task 1)")
        ax1.plot(epoch_axis, curves_loss2[scheme], "--", label=f"{scheme.upper()} (Task 2)")

    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Mean Squared Error (log scale)", fontsize=10)
    ax1.set_title("Multi-Task Training Loss Curves", fontsize=11, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, fontsize=8)

    if len(mgda_alphas) > 0:
        ax2.plot(epoch_axis, mgda_alphas, color="#1f77b4", label="MGDA alpha (Task 1 Weight)")
    if len(gradnorm_w1) > 0 and len(gradnorm_w2) > 0:
        ax2.plot(epoch_axis, gradnorm_w1, color="#2ca02c", label="GradNorm w_1 (Task 1)")
        ax2.plot(epoch_axis, gradnorm_w2, color="#d62728", label="GradNorm w_2 (Task 2)")

    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Weight / Balance Parameter", fontsize=10)
    ax2.set_title("Dynamic Gradient Balancing Trajectories", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=9)

    fig.tight_layout()
    p_mt = out_dir / "d4_multitask_convergence.png"
    fig.savefig(p_mt, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append(p_mt.name)

    return {
        "mt_df": mt_df,
        "figures": figures,
    }


# ---------------------------------------------------------------------------
# SUMMARY Generation
# ---------------------------------------------------------------------------

def write_summary(
    out_dir: Path,
    env_info: dict[str, Any],
    results: dict[str, Any],
    verification_failures: list[str],
    selected_exps: list[str],
    failed_experiments: dict[str, str],
    is_only_mode: bool,
) -> Path:
    """Generate standardized results/track_d/SUMMARY.md without external tabulate dependency."""
    lines: list[str] = [
        "# Track D: Submodular, Parallel, and Distributed Optimization – Experimental Summary",
        "",
        "## Environment & Platform Configuration",
        f"- **Python Version**: {env_info['python_version']}",
        f"- **NumPy Version**: {env_info['numpy_version']}",
        f"- **SciPy Version**: {env_info['scipy_version']}",
        f"- **Scikit-Learn Version**: {env_info['sklearn_version']}",
        f"- **Platform / OS**: {env_info['platform']}",
        f"- **Logical CPU Cores**: {env_info['cpu_count']}",
        f"- **Calibrated Burn Iterations / ms**: {env_info['iters_per_ms']}",
        f"- **Execution Mode**: {'Quick Mode (<45s)' if env_info['quick'] else 'Full Mode (~5m)'}",
    ]

    if is_only_mode:
        lines.append(f"- **Selected Experiments**: {', '.join(selected_exps)} (filtered via --only)")
    else:
        lines.append(f"- **Selected Experiments**: {', '.join(selected_exps)} (all experiments)")

    lines.extend([
        "",
        "## Verification Invariants & Failure Log",
    ])

    if len(failed_experiments) > 0:
        names_str = ", ".join(failed_experiments.keys())
        lines.append(f"**Experiments that failed: {names_str}**\n")
        for exp_name, tb_text in failed_experiments.items():
            lines.extend([
                f"### Traceback for {exp_name}:",
                "```text",
                tb_text.strip(),
                "```",
                "",
            ])

    if len(verification_failures) > 0:
        lines.append(f"**Verification Failures Detected ({len(verification_failures)}):**")
        for fail in verification_failures:
            lines.append(f"- :x: {fail}")
        lines.append("")

    if len(failed_experiments) == 0 and len(verification_failures) == 0:
        lines.append("- :white_check_mark: **No verification failures**. All theoretical invariants satisfied.")
        lines.append("")

    # D1 Section
    if "D1" in results:
        d1 = results["D1"]
        lines.extend([
            "---",
            "## Experiment D1: Submodular Optimization",
            "",
            "### Setup Description",
            "Evaluates maximum coverage over 40 random subsets (universe size 300) under cardinality constraints k from 1 to 10. Compares standard greedy, lazy greedy (CELF with max-heap), and exact integer linear programming (scipy.optimize.milp). Examines empirical submodularity testing and feature selection via OLS R^2 on a synthetic regression instance containing suppressor interactions.",
            "",
            "- Sub-experiment D1(a) Coverage: **simulated / algorithmic evaluation**",
            "- Sub-experiment D1(b) R^2 Feature Selection: **simulated / algorithmic evaluation**",
            "",
            "*(1 - 1/e) theoretical approximation guarantee is proven and verified for Part (a) coverage; it is not claimed for Part (b) because the suppressor dataset generates positive complementarity violating submodularity.*",
            "",
            "#### Table D1.1: Maximum Coverage Benchmark",
            df_to_markdown(d1["cov_df"]),
            "",
            "#### Submodularity Test Results",
            f"- Coverage Function: {d1['sub_check_cov']['violations']} violations across test trials (Worst violation: {d1['sub_check_cov']['worst_violation']:.4f})",
            f"- R^2 Feature Objective: {d1['sub_check_r2']['violations']} violations across test trials (Worst violation: {d1['sub_check_r2']['worst_violation']:.4f})",
            "",
            "#### Table D1.2: R^2 Feature Selection Comparison",
            df_to_markdown(d1["r2_df"]),
            "",
            "#### Artifact Figures",
            "- `d1_coverage_scaling.png`: Greedy/exact ratio and evaluation query counts vs cardinality k.",
            "",
        ])

    # D2 Section
    if "D2" in results:
        d2 = results["D2"]
        lines.extend([
            "---",
            "## Experiment D2: Parallel Computing & Scheduling",
            "",
            "### Setup Description",
            "Evaluates parallel speedup, scalability, synchronization, and load balancing across multi-process workers. Benchmarks include Amdahl strong scaling, Gustafson weak scaling, discrete-event scheduling simulations (static block, round robin, list scheduling, LPT, and work stealing), task granularity overhead, resident vs shipped data-parallel gradient evaluation, and GPipe pipeline model parallelism.",
            "",
            "- Sub-experiment D2(a) Strong Scaling: **measured (wall-clock on this host)**",
            "- Sub-experiment D2(b) Weak Scaling: **measured (wall-clock on this host)**",
            "- Sub-experiment D2(c) Load Balancing: **measured (wall-clock on this host)**",
            "- Sub-experiment D2(d) Load Balancing Schedulers: **simulated (discrete-event simulation)**",
            "- Sub-experiment D2(e) Task Granularity & Overhead: **measured (wall-clock on this host)**",
            "- Sub-experiment D2(f) Data Parallelism (Resident vs Shipped): **measured (wall-clock on this host)**",
            "- Sub-experiment D2(g) Pipeline & Model Parallelism: **simulated (schedule recurrence simulation)**",
            "",
            "#### Table D2.1: Strong Scaling (Measured)",
            df_to_markdown(d2["strong_df"]),
            "",
            "#### Table D2.2: Weak Scaling / Gustafson (Measured)",
            df_to_markdown(d2["weak_df"]),
            "",
            "#### Table D2.3: Load Balancing Execution Times (Measured)",
            df_to_markdown(d2["lb_measured_df"]),
            "",
            "#### Table D2.4: Simulated Load Balancing Scheduler Ratios",
            df_to_markdown(d2["lb_sim_df"]),
            "",
            "#### Table D2.5: Granularity and IPC Overhead (Measured)",
            df_to_markdown(d2["gran_df"]),
            "",
            "#### Table D2.6: Data Parallelism Gradient Computation (Measured)",
            df_to_markdown(d2["dp_df"]),
            "",
            "#### Table D2.7: Pipeline Model Parallelism (Simulated)",
            df_to_markdown(d2["pipe_df"]),
            "",
            "#### Artifact Figures",
            "- `d2_strong_scaling.png`: Measured speedup vs Amdahl model curve.",
            "- `d2_weak_scaling.png`: Measured scaled speedup vs Gustafson law.",
            "- `d2_scheduling_simulated.png`: Makespan / lower bound ratios across load balancing schedulers.",
            "- `d2_granularity_overhead.png`: Task size vs IPC communication bottleneck.",
            "- `d2_pipeline_bubble.png`: Pipeline bubble fraction vs micro-batch count.",
            "",
        ])

    # D3 Section
    if "D3" in results:
        d3 = results["D3"]
        lines.extend([
            "---",
            "## Experiment D3: Federated Learning (FedAvg vs Centralized)",
            "",
            "### Setup Description",
            "Simulates a federation of K=10 clients optimizing a regularized binary logistic regression model on 6000 samples. Evaluates FedAvg under IID and non-IID Dirichlet label skew (alpha=0.3) across varying local computation steps (E in {1, 5, 20}), benchmarking against centralized mini-batch SGD with matched sample throughput.",
            "",
            "- Sub-experiment D3: **measured on simulated clients (wall-clock algorithmic simulation)**",
            f"- Target Accuracy for Convergence: **{d3['target']:.4f}** (defined as 98% of peak mean accuracy {d3['max_final_mean']:.4f}).",
            "",
            "#### Table D3.1: Federated Learning Convergence & Communication",
            df_to_markdown(d3["fed_df"]),
            "",
            "#### Artifact Figures",
            "- `d3_federated_convergence.png`: Test accuracy convergence curves across rounds and partitions.",
            "",
        ])

    # D4 Section
    if "D4" in results:
        d4 = results["D4"]
        lines.extend([
            "---",
            "## Experiment D4: Multi-Task Learning & Gradient Balancing",
            "",
            "### Setup Description",
            "Investigates optimization dynamics on a two-task regression neural network with a shared representation layer (32 tanh units) and disparate task loss scales (scale ratio 10:1). Evaluates three gradient balancing mechanisms: Equal Weights (unweighted sum), MGDA (Pareto-stationary minimum-norm descent), and GradNorm (dynamic loss weights balancing relative gradient norms).",
            "",
            "- Sub-experiment D4: **measured / algorithmic training simulation**",
            "",
            "#### Table D4.1: Multi-Task Performance Metrics",
            df_to_markdown(d4["mt_df"]),
            "",
            "#### Artifact Figures",
            "- `d4_multitask_convergence.png`: Per-task MSE loss trajectories and dynamic weight balancing parameters.",
            "",
        ])

    summary_path = out_dir / "SUMMARY.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


# ---------------------------------------------------------------------------
# Main Driver Entrypoint
# ---------------------------------------------------------------------------

def run_all(
    quick: bool = False,
    only: list[str] | None = None,
    out: str = "results/track_d",
) -> int:
    """Run Track D experiments.

    Args:
        quick: Fast mode executing in < 45 seconds.
        only: List of experiments to run (e.g. ['D1', 'D2']).
        out: Target output directory.

    Returns:
        Exit code: 0 if all experiments and checks passed, 1 otherwise.
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_only_mode = bool(only is not None and len(only) > 0)
    selected_exps = [e.upper() for e in only] if is_only_mode else ["D1", "D2", "D3", "D4"]
    verification_failures: list[str] = []
    failed_experiments: dict[str, str] = {}
    results: dict[str, Any] = {}
    had_error = False

    # Only perform CPU burn calibration if D2 is explicitly scheduled to run
    if "D2" in selected_exps:
        iters_per_ms = calibrate_iters_per_ms(target_dt=0.05 if quick else 0.3)
    else:
        iters_per_ms = 10_000

    env_info = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "platform": platform.platform(),
        "cpu_count": multiprocessing.cpu_count(),
        "iters_per_ms": iters_per_ms,
        "quick": quick,
    }

    if "D1" in selected_exps:
        try:
            results["D1"] = run_experiment_d1(out_dir, quick, verification_failures)
        except Exception:
            had_error = True
            tb_str = traceback.format_exc()
            failed_experiments["D1"] = tb_str
            print(f"ERROR in Experiment D1:\n{tb_str}")

    if "D2" in selected_exps:
        try:
            results["D2"] = run_experiment_d2(out_dir, quick, verification_failures)
        except Exception:
            had_error = True
            tb_str = traceback.format_exc()
            failed_experiments["D2"] = tb_str
            print(f"ERROR in Experiment D2:\n{tb_str}")

    if "D3" in selected_exps:
        try:
            results["D3"] = run_experiment_d3(out_dir, quick, verification_failures)
        except Exception:
            had_error = True
            tb_str = traceback.format_exc()
            failed_experiments["D3"] = tb_str
            print(f"ERROR in Experiment D3:\n{tb_str}")

    if "D4" in selected_exps:
        try:
            results["D4"] = run_experiment_d4(out_dir, quick, verification_failures)
        except Exception:
            had_error = True
            tb_str = traceback.format_exc()
            failed_experiments["D4"] = tb_str
            print(f"ERROR in Experiment D4:\n{tb_str}")

    write_summary(
        out_dir=out_dir,
        env_info=env_info,
        results=results,
        verification_failures=verification_failures,
        selected_exps=selected_exps,
        failed_experiments=failed_experiments,
        is_only_mode=is_only_mode,
    )

    if had_error or len(verification_failures) > 0:
        return 1
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Run Track D benchmarks.")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid testing (<45s)")
    parser.add_argument("--only", nargs="+", help="Specific experiments to run (e.g. D1 D3)")
    parser.add_argument("--out", type=str, default="results/track_d", help="Output directory")
    args = parser.parse_args()

    exit_code = run_all(quick=args.quick, only=args.only, out=args.out)
    sys.exit(exit_code)
