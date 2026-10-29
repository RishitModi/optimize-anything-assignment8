"""Execution script for all Track B discrete optimization experiments.

Command line:
    python tracks/track_b_discrete/run_all.py [--quick] [--out results/track_b]

Experiments:
    B1: Knapsack Comparison (Greedy, DP, Branch-and-Bound, ILP, Brute Force)
    B2: Max-Cut as QUBO and Ising formulations with heuristic solvers
    B3: Feature selection as QUBO with quadratic cardinality penalty
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Ensure repository root and track root are on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _TRACK_ROOT.parents[1]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from common.io_utils import save_csv, save_json
from common.seeding import set_global_seed
from common.timing import Timer
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
    greedy_flip_local_search,
    ising_energy,
    maxcut_qubo,
    qubo_energy,
    qubo_to_ising,
    simulated_annealing_qubo,
)


def df_to_markdown(df: pd.DataFrame) -> str:
    """Render a pandas DataFrame as a GitHub-Flavored Markdown table without tabulate."""
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    rows: list[str] = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            val = row[c]
            if isinstance(val, (float, np.floating)):
                cells.append(f"{val:.4g}")
            elif val is None or pd.isna(val):
                cells.append("")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_line, sep_line] + rows)


def run_experiment_b1(
    out_dir: Path,
    quick: bool,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Experiment B1: 0/1 Knapsack benchmark across sizes and methods."""
    print("--- Running Experiment B1: Knapsack Comparison ---")
    sizes = [10, 20, 30] if quick else [10, 20, 30, 40, 50]
    n_seeds = 2 if quick else 5
    instance_seeds = list(range(n_seeds))

    runs_data: list[dict[str, Any]] = []

    for n in sizes:
        for seed in instance_seeds:
            weights, values, capacity = make_instance(n, seed)

            # 1. DP solution (exact baseline)
            dp_res = dp_knapsack(weights, values, capacity)
            dp_val = dp_res["value"]
            dp_items = dp_res["items"]

            # 2. ILP solution (scipy.optimize.milp)
            ilp_res = ilp_knapsack(weights, values, capacity)
            ilp_val = ilp_res["value"]
            ilp_items = ilp_res["items"]

            # 3. Branch-and-Bound
            bb_res = branch_and_bound(weights, values, capacity)
            bb_val = bb_res["value"]
            bb_items = bb_res["items"]
            bb_nodes = bb_res["nodes"]
            bb_proven = bb_res["proven_optimal"]

            # 4. Greedy ratio heuristic
            greedy_res = greedy_ratio(weights, values, capacity)
            greedy_val = greedy_res["value"]
            greedy_items = greedy_res["items"]

            # 5. Brute force (only for n <= 20)
            bf_res = None
            if n <= 20:
                bf_res = knapsack_brute_force(weights, values, capacity)

            # Verification assertions
            try:
                # Assert DP value == ILP value
                if dp_val != ilp_val:
                    msg = f"[B1 Fail] DP value ({dp_val}) != ILP value ({ilp_val}) for n={n}, seed={seed}"
                    verification_failures.append(msg)
                    print(msg)

                # Assert brute force == DP when run
                if bf_res is not None and bf_res["value"] != dp_val:
                    msg = f"[B1 Fail] BF value ({bf_res['value']}) != DP value ({dp_val}) for n={n}, seed={seed}"
                    verification_failures.append(msg)
                    print(msg)

                # Assert B&B equals DP when proven optimal
                if bb_proven and bb_val != dp_val:
                    msg = f"[B1 Fail] B&B value ({bb_val}) != DP value ({dp_val}) with proven_optimal=True for n={n}, seed={seed}"
                    verification_failures.append(msg)
                    print(msg)

                # Assert greedy <= DP
                if greedy_val > dp_val:
                    msg = f"[B1 Fail] Greedy value ({greedy_val}) > DP value ({dp_val}) for n={n}, seed={seed}"
                    verification_failures.append(msg)
                    print(msg)

                # Assert feasibility and value consistency for all returned sets
                all_results = [("DP", dp_res), ("ILP", ilp_res), ("B&B", bb_res), ("Greedy", greedy_res)]
                if bf_res is not None:
                    all_results.append(("BruteForce", bf_res))

                for m_name, m_dict in all_results:
                    chosen = m_dict["items"]
                    tot_w = int(np.sum(weights[chosen])) if chosen else 0
                    tot_v = int(np.sum(values[chosen])) if chosen else 0
                    if tot_w > capacity:
                        msg = f"[B1 Fail] Method {m_name} produced infeasible weight {tot_w} > {capacity} for n={n}, seed={seed}"
                        verification_failures.append(msg)
                        print(msg)
                    if tot_v != m_dict["value"]:
                        msg = f"[B1 Fail] Method {m_name} reported value {m_dict['value']} != recomputed {tot_v} for n={n}, seed={seed}"
                        verification_failures.append(msg)
                        print(msg)

            except Exception as e:
                err_msg = f"[B1 Exception] Verification check failed with exception: {e}"
                verification_failures.append(err_msg)
                print(err_msg)

            # Record runs
            methods_to_record = [
                ("DP", dp_res, None, None),
                ("ILP", ilp_res, None, None),
                ("Branch-and-Bound", bb_res, bb_nodes, bb_proven),
                ("Greedy", greedy_res, None, None),
            ]
            if bf_res is not None:
                methods_to_record.append(("BruteForce", bf_res, None, None))

            for m_name, m_res, nodes_explored, proven_opt in methods_to_record:
                m_val = m_res["value"]
                gap = ((dp_val - m_val) / dp_val * 100.0) if dp_val > 0 else 0.0
                m_items = m_res["items"]
                is_feasible = (int(np.sum(weights[m_items])) <= capacity) if m_items else True
                runs_data.append({
                    "n": n,
                    "seed": seed,
                    "method": m_name,
                    "value": m_val,
                    "optimality_gap_pct": gap,
                    "runtime_s": m_res["runtime_s"],
                    "nodes": nodes_explored if nodes_explored is not None else "",
                    "proven_optimal": proven_opt if proven_opt is not None else "",
                    "feasible": is_feasible,
                })

    runs_df = pd.DataFrame(runs_data)
    save_csv(runs_df, out_dir / "b1_knapsack_runs.csv")

    # Summary table: mean runtime and mean gap per method and size
    summary_rows = []
    for (n, method), group in runs_df.groupby(["n", "method"], sort=False):
        summary_rows.append({
            "n": n,
            "method": method,
            "mean_runtime_s": float(group["runtime_s"].mean()),
            "mean_gap_pct": float(group["optimality_gap_pct"].mean()),
        })
    summary_df = pd.DataFrame(summary_rows)
    save_csv(summary_df, out_dir / "b1_knapsack_summary.csv")

    # Plot 1: Runtime vs n (log y scale) per method
    fig, ax = plt.subplots(figsize=(8, 5))
    method_styles = {
        "DP": ("blue", "o-"),
        "ILP": ("green", "s-"),
        "Branch-and-Bound": ("purple", "^-"),
        "Greedy": ("orange", "d-"),
        "BruteForce": ("red", "x--"),
    }
    for method, group in summary_df.groupby("method"):
        color, marker = method_styles.get(method, ("black", ".-"))
        sorted_group = group.sort_values("n")
        ax.plot(
            sorted_group["n"],
            sorted_group["mean_runtime_s"],
            marker,
            color=color,
            label=method,
            linewidth=1.75,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Problem size n (number of items)", fontsize=10)
    ax.set_ylabel("Runtime (seconds)", fontsize=10)
    ax.set_title("Knapsack Solver Runtime vs Problem Size n (Log Scale)", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    runtime_plot_path = out_dir / "b1_runtime_vs_n.png"
    fig.savefig(runtime_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Greedy optimality gap (%) vs n (mean with min/max error bars)
    greedy_runs = runs_df[runs_df["method"] == "Greedy"]
    gap_summary = []
    for n, group in greedy_runs.groupby("n"):
        mean_gap = float(group["optimality_gap_pct"].mean())
        min_gap = float(group["optimality_gap_pct"].min())
        max_gap = float(group["optimality_gap_pct"].max())
        gap_summary.append({
            "n": n,
            "mean": mean_gap,
            "lower_err": mean_gap - min_gap,
            "upper_err": max_gap - mean_gap,
        })
    gap_df = pd.DataFrame(gap_summary).sort_values("n")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        gap_df["n"],
        gap_df["mean"],
        yerr=[gap_df["lower_err"], gap_df["upper_err"]],
        fmt="o-",
        color="#d95f02",
        capsize=4,
        linewidth=1.75,
        label="Greedy Ratio",
    )
    ax.set_xlabel("Problem size n (number of items)", fontsize=10)
    ax.set_ylabel("Optimality Gap (%)", fontsize=10)
    ax.set_title("Greedy Knapsack Optimality Gap vs Problem Size n (Min/Mean/Max)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    gap_plot_path = out_dir / "b1_greedy_gap_vs_n.png"
    fig.savefig(gap_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "runs_df": runs_df,
        "summary_df": summary_df,
        "figures": [runtime_plot_path.name, gap_plot_path.name],
    }


def run_experiment_b2(
    out_dir: Path,
    quick: bool,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Experiment B2: Max-Cut on 3-regular graphs as QUBO and Ising formulations."""
    print("--- Running Experiment B2: Max-Cut as QUBO/Ising ---")
    sizes = [6, 8, 10] if quick else [6, 8, 10, 12, 14, 16]
    n_seeds = 5 if quick else 10
    heuristic_seeds = list(range(n_seeds))
    sa_sweeps = 500 if quick else 2000
    restarts = 10 if quick else 20

    summary_records: list[dict[str, Any]] = []
    qubo_instances_data: list[dict[str, Any]] = []

    conventions_str = (
        "QUBO energy E(x) = x^T Q x with Q upper-triangular (diagonal = linear terms), "
        "x in {0,1}^n, to be MINIMISED. Ising energy E(s) = sum_{i<j} J[i,j] s_i s_j + sum_i h[i] s_i + offset "
        "with s in {-1,+1}^n and the substitution x_i = (1 - s_i)/2 (so s_i = +1 corresponds to x_i = 0, "
        "matching the qubit convention where Z has eigenvalue +1 on |0>)."
    )

    for n in sizes:
        graph = nx.random_regular_graph(3, n, seed=1)
        n_edges = graph.number_of_edges()

        Q = maxcut_qubo(graph)
        h, J, offset = qubo_to_ising(Q)

        # Verification of energy equivalence across all 2^n assignments for n <= 12
        max_discrepancy = 0.0
        if n <= 12:
            num_states = 1 << n
            for state_idx in range(num_states):
                x = np.array([(state_idx >> bit) & 1 for bit in range(n)], dtype=int)
                # s_i = 1 - 2 * x_i, so x_i = 0 => s_i = +1, x_i = 1 => s_i = -1
                s = 1 - 2 * x

                eq = qubo_energy(Q, x)
                es = ising_energy(h, J, offset, s)
                direct_cut = cut_value(graph, x)

                # Invariant: E_qubo(x) == E_ising(s) == -Cut(x)
                disc_ising = abs(eq - es)
                disc_cut = abs(eq - (-float(direct_cut)))
                disc = max(disc_ising, disc_cut)
                if disc > max_discrepancy:
                    max_discrepancy = disc

            if max_discrepancy > 1e-9:
                msg = f"[B2 Fail] Energy discrepancy {max_discrepancy:.2e} > 1e-9 for n={n}"
                verification_failures.append(msg)
                print(msg)

        # 1. Exact brute force baseline
        bf_res = brute_force_qubo(Q, offset=0.0)
        exact_opt_energy = bf_res["best_energy"]
        exact_opt_cut = int(round(-exact_opt_energy))
        bf_runtime = bf_res["runtime_s"]

        summary_records.append({
            "n": n,
            "n_edges": n_edges,
            "method": "BruteForce",
            "best_cut": exact_opt_cut,
            "mean_cut": float(exact_opt_cut),
            "success_rate": 1.0,
            "runtime_s": bf_runtime,
            "max_discrepancy": f"{max_discrepancy:.2e}" if n <= 12 else "N/A",
        })

        # 2. Greedy flip local search
        gf_cuts: list[int] = []
        gf_times: list[float] = []
        for s_idx in heuristic_seeds:
            res_gf = greedy_flip_local_search(Q, seed=s_idx, restarts=restarts)
            cut_val = cut_value(graph, res_gf["best_x"])
            gf_cuts.append(cut_val)
            gf_times.append(res_gf["runtime_s"])

        gf_success = float(np.mean([1.0 if c == exact_opt_cut else 0.0 for c in gf_cuts]))
        summary_records.append({
            "n": n,
            "n_edges": n_edges,
            "method": "GreedyFlip",
            "best_cut": int(np.max(gf_cuts)),
            "mean_cut": float(np.mean(gf_cuts)),
            "success_rate": gf_success,
            "runtime_s": float(np.mean(gf_times)),
            "max_discrepancy": f"{max_discrepancy:.2e}" if n <= 12 else "N/A",
        })

        # 3. Simulated annealing
        sa_cuts: list[int] = []
        sa_times: list[float] = []
        for s_idx in heuristic_seeds:
            res_sa = simulated_annealing_qubo(Q, seed=s_idx, n_sweeps=sa_sweeps)
            cut_val = cut_value(graph, res_sa["best_x"])
            sa_cuts.append(cut_val)
            sa_times.append(res_sa["runtime_s"])

        sa_success = float(np.mean([1.0 if c == exact_opt_cut else 0.0 for c in sa_cuts]))
        summary_records.append({
            "n": n,
            "n_edges": n_edges,
            "method": "SimulatedAnnealing",
            "best_cut": int(np.max(sa_cuts)),
            "mean_cut": float(np.mean(sa_cuts)),
            "success_rate": sa_success,
            "runtime_s": float(np.mean(sa_times)),
            "max_discrepancy": f"{max_discrepancy:.2e}" if n <= 12 else "N/A",
        })

    summary_df = pd.DataFrame(summary_records)
    save_csv(summary_df, out_dir / "b2_maxcut_summary.csv")

    # Generate qubo_instances.json for quantum track (always for n in [4, 6, 8, 10])
    for n_inst in [4, 6, 8, 10]:
        inst_g = nx.random_regular_graph(3, n_inst, seed=1)
        inst_Q = maxcut_qubo(inst_g)
        inst_h, inst_J, inst_offset = qubo_to_ising(inst_Q)
        inst_bf = brute_force_qubo(inst_Q, offset=0.0)

        inst_J_str = {f"{i},{j}": float(val) for (i, j), val in inst_J.items()}
        qubo_instances_data.append({
            "n": n_inst,
            "edge_list": [[int(u), int(v)] for u, v in inst_g.edges()],
            "Q": inst_Q.tolist(),
            "h": [float(val) for val in inst_h],
            "J": inst_J_str,
            "offset": float(inst_offset),
            "exact_optimal_energy": float(inst_bf["best_energy"]),
            "optimal_cut_value": int(round(-float(inst_bf["best_energy"]))),
            "optimal_bitstrings": inst_bf["best_bitstrings"],
            "convention": conventions_str,
        })

    qubo_payload = {
        "convention": conventions_str,
        "instances": qubo_instances_data,
    }
    save_json(qubo_payload, out_dir / "qubo_instances.json", indent=2)

    # Plot 1: Runtime vs n (log y)
    fig, ax = plt.subplots(figsize=(8, 5))
    method_styles = {
        "BruteForce": ("red", "x--"),
        "GreedyFlip": ("green", "o-"),
        "SimulatedAnnealing": ("blue", "s-"),
    }
    for method, group in summary_df.groupby("method"):
        color, marker = method_styles.get(method, ("black", ".-"))
        sorted_group = group.sort_values("n")
        ax.plot(
            sorted_group["n"],
            sorted_group["runtime_s"],
            marker,
            color=color,
            label=method,
            linewidth=1.75,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Graph vertices n", fontsize=10)
    ax.set_ylabel("Runtime (seconds)", fontsize=10)
    ax.set_title("Max-Cut QUBO Solvers Runtime vs Graph Size n", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    runtime_plot_path = out_dir / "b2_runtime_vs_n.png"
    fig.savefig(runtime_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Cut value found vs n
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in summary_df.groupby("method"):
        color, marker = method_styles.get(method, ("black", ".-"))
        sorted_group = group.sort_values("n")
        ax.plot(
            sorted_group["n"],
            sorted_group["mean_cut"],
            marker,
            color=color,
            label=f"{method} (mean cut)",
            linewidth=1.75,
        )
    ax.set_xlabel("Graph vertices n", fontsize=10)
    ax.set_ylabel("Max Cut Value (edge count)", fontsize=10)
    ax.set_title("Max-Cut Solution Quality vs Graph Size n", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    cut_plot_path = out_dir / "b2_cut_vs_n.png"
    fig.savefig(cut_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "summary_df": summary_df,
        "figures": [runtime_plot_path.name, cut_plot_path.name],
    }


def run_experiment_b3(
    out_dir: Path,
    quick: bool,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Experiment B3: Feature selection as QUBO with quadratic cardinality penalty."""
    print("--- Running Experiment B3: Feature Selection with QUBO Penalty ---")
    data = load_breast_cancer()
    X_full = data.data
    y = data.target

    # Standardize features
    X_scaled = StandardScaler().fit_transform(X_full)

    # Pick 12 features with np.random.default_rng(0).choice(30, 12, replace=False)
    rng_pick = np.random.default_rng(0)
    chosen_indices = sorted(rng_pick.choice(30, 12, replace=False).tolist())
    X_sub = X_scaled[:, chosen_indices]

    # Compute relevance r_i = |corr(feature_i, y)|
    relevance = np.zeros(12, dtype=np.float64)
    for i in range(12):
        relevance[i] = abs(float(np.corrcoef(X_sub[:, i], y)[0, 1]))

    # Compute redundancy R_ij = |corr(feature_i, feature_j)|
    redundancy = np.abs(np.corrcoef(X_sub, rowvar=False))

    k = 4
    alpha = 0.5

    # Helper function to compute raw unpenalized objective: -sum r_i x_i + alpha * sum_{i<j} R_ij x_i x_j
    def compute_raw_objective(x_vec: np.ndarray) -> float:
        obj = -float(np.sum(relevance * x_vec))
        quad = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                quad += redundancy[i, j] * x_vec[i] * x_vec[j]
        return obj + alpha * quad

    # Helper function for 5-fold CV accuracy with LogisticRegression
    def evaluate_cv_accuracy(feature_indices: list[int]) -> tuple[float, float]:
        if not feature_indices:
            return 0.0, 0.0
        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_sub[:, feature_indices], y, cv=cv, scoring="accuracy")
        return float(np.mean(scores)), float(np.std(scores))

    # (a) Penalty sweep
    lam_values = [0.0, 0.1, 0.5, 2.0, 5.0]
    sweep_records: list[dict[str, Any]] = []

    for lam in lam_values:
        Q, offset = feature_selection_qubo(relevance, redundancy, k=k, alpha=alpha, lam=lam)
        bf_res = brute_force_qubo(Q, offset=offset)
        x_opt = bf_res["best_x"]
        n_selected = int(np.sum(x_opt))
        card_satisfied = (n_selected == k)
        raw_obj = compute_raw_objective(x_opt)

        sweep_records.append({
            "lambda": lam,
            "selected_features_count": n_selected,
            "cardinality_satisfied": card_satisfied,
            "raw_objective": raw_obj,
        })

    sweep_df = pd.DataFrame(sweep_records)
    save_csv(sweep_df, out_dir / "b3_penalty_sweep.csv")

    # Plot for penalty sweep: selected features count vs lambda with horizontal line at k
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        sweep_df["lambda"],
        sweep_df["selected_features_count"],
        "o-",
        color="#2b5c8f",
        linewidth=2,
        label="Selected Features (QUBO exact)",
    )
    ax.axhline(k, color="red", linestyle="--", linewidth=1.5, label=f"Target Cardinality k={k}")
    ax.set_xlabel("Penalty Multiplier lambda", fontsize=10)
    ax.set_ylabel("Number of Selected Features", fontsize=10)
    ax.set_title("QUBO Feature Cardinality vs Penalty Multiplier lambda", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    penalty_plot_path = out_dir / "b3_penalty_sweep.png"
    fig.savefig(penalty_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (b) Method comparison at lambda = 2.0
    lam_fixed = 2.0
    Q_fixed, offset_fixed = feature_selection_qubo(relevance, redundancy, k=k, alpha=alpha, lam=lam_fixed)

    # 1. Exact QUBO optimum
    bf_fixed = brute_force_qubo(Q_fixed, offset=offset_fixed)
    x_exact = bf_fixed["best_x"]
    exact_indices = [i for i in range(12) if x_exact[i] == 1]
    exact_obj = compute_raw_objective(x_exact)
    exact_acc_mean, exact_acc_std = evaluate_cv_accuracy(exact_indices)

    # 2. Greedy forward selection (add feature with best marginal objective gain k times)
    greedy_selected: list[int] = []
    for _ in range(k):
        best_cand = -1
        best_cand_obj = float("inf")
        for cand in range(12):
            if cand not in greedy_selected:
                cand_set = greedy_selected + [cand]
                cand_vec = np.zeros(12, dtype=int)
                cand_vec[cand_set] = 1
                cand_obj = compute_raw_objective(cand_vec)
                if cand_obj < best_cand_obj:
                    best_cand_obj = cand_obj
                    best_cand = cand
        if best_cand != -1:
            greedy_selected.append(best_cand)

    greedy_indices = sorted(greedy_selected)
    x_greedy = np.zeros(12, dtype=int)
    x_greedy[greedy_indices] = 1
    greedy_obj = compute_raw_objective(x_greedy)
    greedy_acc_mean, greedy_acc_std = evaluate_cv_accuracy(greedy_indices)

    # 3. Simulated Annealing on QUBO
    sa_seeds = list(range(5 if quick else 10))
    sa_sweeps = 500 if quick else 2000
    sa_objs: list[float] = []
    sa_accs: list[float] = []
    sa_best_x = None
    sa_best_energy = float("inf")

    for s_idx in sa_seeds:
        res_sa = simulated_annealing_qubo(Q_fixed, seed=s_idx, n_sweeps=sa_sweeps)
        x_sa = res_sa["best_x"]
        if res_sa["best_energy"] < sa_best_energy:
            sa_best_energy = res_sa["best_energy"]
            sa_best_x = x_sa
        sa_objs.append(compute_raw_objective(x_sa))
        sa_indices = [i for i in range(12) if x_sa[i] == 1]
        mean_acc, _ = evaluate_cv_accuracy(sa_indices)
        sa_accs.append(mean_acc)

    sa_best_indices = [i for i in range(12) if sa_best_x is not None and sa_best_x[i] == 1]

    # 4. Random k-subsets
    n_random_subsets = 50 if quick else 200
    rng_subsets = np.random.default_rng(42)
    rand_objs: list[float] = []
    rand_accs: list[float] = []

    for _ in range(n_random_subsets):
        sub = sorted(rng_subsets.choice(12, k, replace=False).tolist())
        x_rand = np.zeros(12, dtype=int)
        x_rand[sub] = 1
        rand_objs.append(compute_raw_objective(x_rand))
        acc_m, _ = evaluate_cv_accuracy(sub)
        rand_accs.append(acc_m)

    comparison_records = [
        {
            "method": "QUBO Exact (BruteForce)",
            "selected_features": str(exact_indices),
            "num_features": len(exact_indices),
            "objective": exact_obj,
            "cv_accuracy_mean": exact_acc_mean,
            "cv_accuracy_std": exact_acc_std,
        },
        {
            "method": "Greedy Forward Selection",
            "selected_features": str(greedy_indices),
            "num_features": len(greedy_indices),
            "objective": greedy_obj,
            "cv_accuracy_mean": greedy_acc_mean,
            "cv_accuracy_std": greedy_acc_std,
        },
        {
            "method": f"QUBO Simulated Annealing (seeds={len(sa_seeds)})",
            "selected_features": str(sa_best_indices),
            "num_features": len(sa_best_indices),
            "objective": float(np.mean(sa_objs)),
            "cv_accuracy_mean": float(np.mean(sa_accs)),
            "cv_accuracy_std": float(np.std(sa_accs)),
        },
        {
            "method": f"Random {k}-Subsets (N={n_random_subsets})",
            "selected_features": "aggregate",
            "num_features": k,
            "objective": float(np.mean(rand_objs)),
            "cv_accuracy_mean": float(np.mean(rand_accs)),
            "cv_accuracy_std": float(np.std(rand_accs)),
        },
    ]

    comparison_df = pd.DataFrame(comparison_records)
    save_csv(comparison_df, out_dir / "b3_method_comparison.csv")

    return {
        "sweep_df": sweep_df,
        "comparison_df": comparison_df,
        "figures": [penalty_plot_path.name],
    }


def write_summary(
    out_dir: Path,
    b1_res: dict[str, Any] | None,
    b2_res: dict[str, Any] | None,
    b3_res: dict[str, Any] | None,
    verification_failures: list[str],
    quick: bool,
) -> Path:
    """Generate SUMMARY.md detailing environment, setups, markdown tables, and figures."""
    lines: list[str] = []
    lines.append("# Track B Discrete Optimization – Summary\n")

    # Environment
    lines.append("## Environment\n")
    lines.append(f"- **Python version**: {sys.version.split()[0]}")
    lines.append(f"- **NumPy version**: {np.__version__}")
    lines.append(f"- **SciPy version**: {scipy.__version__}")
    lines.append(f"- **NetworkX version**: {nx.__version__}")
    lines.append(f"- **scikit-learn version**: {sklearn.__version__}")
    lines.append(f"- **Platform**: {platform.platform()}")
    lines.append(f"- **CPU count**: {os.cpu_count()}")
    lines.append(f"- **Quick mode**: {quick}")
    lines.append("- **Base seeds**: Knapsack seeds [0..4] (quick: [0..1]); Graph seed 1; RNG seeds 0 and 42\n")

    # Experiment B1
    lines.append("## Experiment B1: Knapsack Comparison\n")
    lines.append(
        "**Setup**: Evaluated 0/1 knapsack benchmarks across item counts n in [10, 20, 30, 40, 50] "
        "(quick: [10, 20, 30]) using strongly correlated instances (weights uniform in [5, 50], "
        "values = weights + noise in [0, 10], capacity = floor(0.5 * sum(weights))). Methods tested include "
        "Dynamic Programming (DP), Integer Linear Programming (ILP via scipy.optimize.milp), Branch-and-Bound "
        "with fractional LP relaxation pruning, Greedy ratio heuristic, and Brute Force (n <= 20). "
        "Invariants tested: DP == ILP, BF == DP (n <= 20), B&B == DP (when proven optimal), Greedy <= DP, "
        "and strict weight feasibility."
    )
    lines.append("")
    if b1_res is not None and "summary_df" in b1_res:
        lines.append("### B1 Summary Table\n")
        lines.append(df_to_markdown(b1_res["summary_df"]))
        lines.append("")
        lines.append(f"**Figures**: {', '.join(b1_res['figures'])}\n")

    # Experiment B2
    lines.append("## Experiment B2: Max-Cut as QUBO and Ising\n")
    lines.append(
        "**Setup**: Formulated Max-Cut on 3-regular graphs of sizes n in [6, 8, 10, 12, 14, 16] "
        "(quick: [6, 8, 10]) into upper-triangular QUBO (minimising -Cut) and equivalent Ising spin models. "
        "For all states on n <= 12, verified that QUBO energy matches Ising energy and exact negative cut value. "
        "Compared exact brute-force enumeration baseline against greedy single-bit-flip local search and "
        "Metropolis simulated annealing across multiple random seeds."
    )
    lines.append("")
    if b2_res is not None and "summary_df" in b2_res:
        lines.append("### B2 Summary Table\n")
        lines.append(df_to_markdown(b2_res["summary_df"]))
        lines.append("")
        lines.append(f"**Figures**: {', '.join(b2_res['figures'])}\n")

    # Experiment B3
    lines.append("## Experiment B3: Feature Selection as QUBO with Cardinality Penalty\n")
    lines.append(
        "**Setup**: Formulated feature selection on 12 standardized features of the Wisconsin Breast Cancer dataset "
        "as a QUBO minimising relevance minus alpha * redundancy plus lambda * (sum x_i - k)^2 with k = 4 and alpha = 0.5. "
        "Conducted a penalty sweep across lambda in [0, 0.1, 0.5, 2.0, 5.0] to test constraint enforcement, and compared "
        "the exact QUBO optimum at lambda = 2.0 against greedy forward selection, simulated annealing, and random k-subsets "
        "using 5-fold cross-validated LogisticRegression accuracy."
    )
    lines.append("")
    if b3_res is not None:
        if "sweep_df" in b3_res:
            lines.append("### B3 Penalty Sweep Table\n")
            lines.append(df_to_markdown(b3_res["sweep_df"]))
            lines.append("")
        if "comparison_df" in b3_res:
            lines.append("### B3 Method Comparison Table\n")
            lines.append(df_to_markdown(b3_res["comparison_df"]))
            lines.append("")
        lines.append(f"**Figures**: {', '.join(b3_res['figures'])}\n")

    # Verification Failures
    lines.append("## Verification\n")
    if verification_failures:
        lines.append("**Verification Failures:**")
        for fail in verification_failures:
            lines.append(f"- {fail}")
    else:
        lines.append("No verification failures")
    lines.append("")

    summary_file = out_dir / "SUMMARY.md"
    summary_file.write_text("\n".join(lines), encoding="utf-8")
    return summary_file


def run_all(quick: bool = False, out: str = "results/track_b") -> int:
    """Run all Track B discrete experiments in sequence.

    Args:
        quick: If True, execute with smaller problem sizes and fewer sweeps.
        out: Destination directory for artifacts.

    Returns:
        Exit code: 0 on complete success, 1 if any experiment failed.
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(42)

    verification_failures: list[str] = []
    any_failed = False

    b1_res = None
    b2_res = None
    b3_res = None

    # Experiment B1
    try:
        t0 = time.perf_counter()
        b1_res = run_experiment_b1(out_dir, quick=quick, verification_failures=verification_failures)
        print(f"Experiment B1 completed in {time.perf_counter() - t0:.2f} s")
    except Exception:
        any_failed = True
        print("Experiment B1 failed:")
        traceback.print_exc()

    # Experiment B2
    try:
        t0 = time.perf_counter()
        b2_res = run_experiment_b2(out_dir, quick=quick, verification_failures=verification_failures)
        print(f"Experiment B2 completed in {time.perf_counter() - t0:.2f} s")
    except Exception:
        any_failed = True
        print("Experiment B2 failed:")
        traceback.print_exc()

    # Experiment B3
    try:
        t0 = time.perf_counter()
        b3_res = run_experiment_b3(out_dir, quick=quick, verification_failures=verification_failures)
        print(f"Experiment B3 completed in {time.perf_counter() - t0:.2f} s")
    except Exception:
        any_failed = True
        print("Experiment B3 failed:")
        traceback.print_exc()

    # Summary generation
    summary_path = write_summary(
        out_dir=out_dir,
        b1_res=b1_res,
        b2_res=b2_res,
        b3_res=b3_res,
        verification_failures=verification_failures,
        quick=quick,
    )
    print(f"Summary written to: {summary_path}")

    return 1 if any_failed else 0


def main() -> None:
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser(description="Run Track B discrete optimization experiments.")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid testing under 30s.")
    parser.add_argument("--out", default="results/track_b", help="Output directory path (default: results/track_b).")
    args = parser.parse_args()

    exit_code = run_all(quick=args.quick, out=args.out)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

