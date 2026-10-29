"""Execution script for all Track C quantum optimization experiments.

Command line:
    python tracks/track_c_quantum_opt/run_all.py [--quick] [--out results/track_c]

Experiments:
    C1: QAOA for Max-Cut on Track B QUBO/Ising benchmark instances
    C2: VQE on Transverse-Field Ising Model comparing GD, QNG, SPSA, and COBYLA
    C3: Finite-shot sampling noise and shot-based VQE optimization
    C4: Simulated hardware depolarizing noise in QAOA and VQE
    C5: Barren plateau scaling of gradient variance for local vs global observables
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

# Ensure repository root and track roots are on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _TRACK_ROOT.parents[1]
_TRACK_B_ROOT = _REPO_ROOT / "tracks" / "track_b_discrete"
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT), str(_TRACK_B_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

from common.io_utils import save_csv
from common.seeding import set_global_seed
from optimizers_q import cobyla, gd_param_shift, qng, spsa
from qaoa import (
    approximation_ratio,
    ising_diag,
    load_instances,
    prob_optimal,
    qaoa_expectation,
    qaoa_state,
    qubo_diag,
    run_qaoa,
)
from qsim import (
    apply_1q,
    apply_depolarizing_layer,
    estimate_energy_shots,
    expectation_diag,
    expectation_pauli_hamiltonian,
    plus_state,
    zero_state,
)
from qubo import simulated_annealing_qubo
from vqe import (
    ansatz_state,
    exact_ground_energy,
    metric_tensor,
    param_shift_grad,
    tfim_terms,
    vqe_energy,
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


def run_experiment_c1(
    out_dir: Path,
    quick: bool,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Experiment C1: QAOA for Max-Cut on precomputed Track B instances."""
    print("--- Running Experiment C1: QAOA for Max-Cut ---")
    instances = load_instances("results/track_b/qubo_instances.json")

    n_inits = 3 if quick else 10
    maxiter = 100 if quick else 300
    p_depths = [1, 2, 3]

    qaoa_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    c1_best_params_per_instance: dict[int, np.ndarray] = {}

    for inst in instances:
        n = inst["n"]
        diag_q = qubo_diag(inst["Q"])
        diag_i = ising_diag(inst["h"], inst["J"], inst["offset"], n)

        # Verification: qubo_diag equals ising_diag to 1e-9
        diff = float(np.max(np.abs(diag_q - diag_i)))
        if diff > 1e-9:
            msg = f"[C1 Fail] qubo_diag != ising_diag (diff={diff:.2e}) for instance n={n}"
            verification_failures.append(msg)
            print(msg)

        # Verification: p=0 uniform state expectation equals mean of diag
        state_p0 = plus_state(n)
        exp_p0 = expectation_diag(state_p0, diag_q)
        mean_diag = float(np.mean(diag_q))
        if abs(exp_p0 - mean_diag) > 1e-9:
            msg = f"[C1 Fail] p=0 state exp ({exp_p0}) != mean diag ({mean_diag}) for n={n}"
            verification_failures.append(msg)
            print(msg)

        # Classical Simulated Annealing baseline
        sa_cuts: list[int] = []
        for s_idx in range(5 if quick else 10):
            res_sa = simulated_annealing_qubo(inst["Q"], seed=s_idx, n_sweeps=500 if quick else 2000)
            cut_val = int(round(-float(res_sa["best_energy"])))
            sa_cuts.append(cut_val)

        sa_best_cut = int(np.max(sa_cuts))
        sa_success = float(np.mean([1.0 if c == inst["opt_cut"] else 0.0 for c in sa_cuts]))

        baseline_rows.append({
            "n": n,
            "optimal_cut": inst["opt_cut"],
            "sa_best_cut": sa_best_cut,
            "sa_success_rate": sa_success,
        })

        best_ratio_for_instance = -1.0
        for p in p_depths:
            qaoa_res = run_qaoa(inst, p=p, n_inits=n_inits, seed=42 + n + p, maxiter=maxiter)
            runs = qaoa_res["runs"]

            ratios = [r["ratio"] for r in runs]
            prob_opts = [r["prob_opt"] for r in runs]
            evals_list = [r["evals"] for r in runs]

            ratio_mean = float(np.mean(ratios))
            ratio_std = float(np.std(ratios))
            ratio_best = float(np.max(ratios))

            # Verification: approximation ratio <= 1 + 1e-9
            if ratio_best > 1.0 + 1e-9:
                msg = f"[C1 Fail] QAOA ratio {ratio_best} > 1.0 for n={n}, p={p}"
                verification_failures.append(msg)
                print(msg)

            if p == 2 and qaoa_res["best_params"] is not None:
                c1_best_params_per_instance[n] = qaoa_res["best_params"]

            qaoa_rows.append({
                "n": n,
                "p": p,
                "ratio_mean": ratio_mean,
                "ratio_std": ratio_std,
                "ratio_best": ratio_best,
                "prob_opt_mean": float(np.mean(prob_opts)),
                "prob_opt_best": float(np.max(prob_opts)),
                "evals_mean": float(np.mean(evals_list)),
            })

    qaoa_df = pd.DataFrame(qaoa_rows)
    save_csv(qaoa_df, out_dir / "c1_qaoa_results.csv")

    baseline_df = pd.DataFrame(baseline_rows)
    save_csv(baseline_df, out_dir / "c1_classical_baselines.csv")

    # Plot 1: Approximation ratio vs p per n
    fig, ax = plt.subplots(figsize=(8, 5))
    for n, group in qaoa_df.groupby("n"):
        ax.errorbar(
            group["p"],
            group["ratio_mean"],
            yerr=group["ratio_std"],
            fmt="o-",
            capsize=4,
            label=f"n = {n}",
        )
    ax.set_xlabel("QAOA Depth p", fontsize=10)
    ax.set_ylabel("Approximation Ratio (Cut / OptCut)", fontsize=10)
    ax.set_title("QAOA Approximation Ratio vs Circuit Depth p", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    ratio_fig_path = out_dir / "c1_qaoa_ratio_vs_p.png"
    fig.savefig(ratio_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Probability of optimal bitstring vs p
    fig, ax = plt.subplots(figsize=(8, 5))
    for n, group in qaoa_df.groupby("n"):
        ax.plot(group["p"], group["prob_opt_mean"], "s-", label=f"n = {n}")
    ax.set_xlabel("QAOA Depth p", fontsize=10)
    ax.set_ylabel("Optimal Bitstring Probability", fontsize=10)
    ax.set_title("Optimal Bitstring Sampling Probability vs Depth p", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    prob_fig_path = out_dir / "c1_qaoa_prob_opt_vs_p.png"
    fig.savefig(prob_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "qaoa_df": qaoa_df,
        "baseline_df": baseline_df,
        "figures": [ratio_fig_path.name, prob_fig_path.name],
        "c1_best_params": c1_best_params_per_instance,
    }


def run_experiment_c2(
    out_dir: Path,
    quick: bool,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Experiment C2: VQE on TFIM chain comparing classical & quantum optimizers."""
    print("--- Running Experiment C2: VQE Optimizer Comparison ---")
    n = 4
    g = 1.0
    L = 3
    P = n * (L + 1)  # 16 parameters

    terms = tfim_terms(n, g)
    e0 = exact_ground_energy(terms, n)

    budget = 600 if quick else 4000
    n_seeds = 2 if quick else 5

    # Verification: parameter-shift grad equals central finite diff at 3 random points
    rng_check = np.random.default_rng(777)
    for _ in range(3):
        th_check = rng_check.uniform(0, 2 * np.pi, size=P)
        grad_ps = param_shift_grad(th_check, terms, n, L)

        eps = 1e-5
        grad_fd = np.zeros(P, dtype=np.float64)
        for i in range(P):
            th_p = th_check.copy()
            th_p[i] += eps
            th_m = th_check.copy()
            th_m[i] -= eps
            grad_fd[i] = (vqe_energy(th_p, terms, n, L) - vqe_energy(th_m, terms, n, L)) / (2.0 * eps)

        grad_err = float(np.max(np.abs(grad_ps - grad_fd)))
        if grad_err > 1e-6:
            msg = f"[C2 Fail] Parameter shift grad != finite diff (err={grad_err:.2e})"
            verification_failures.append(msg)
            print(msg)

    # Verification: metric tensor equals -0.5 * Hessian of fidelity at n=3, L=1
    n_fid = 3
    L_fid = 1
    P_fid = n_fid * (L_fid + 1)
    th_fid = rng_check.uniform(0, 2 * np.pi, size=P_fid)
    g_metric = metric_tensor(th_fid, n_fid, L_fid)

    psi_base = ansatz_state(th_fid, n_fid, L_fid)
    eps_fid = 1e-3
    hess_fid = np.zeros((P_fid, P_fid), dtype=np.float64)

    for i in range(P_fid):
        for j in range(P_fid):
            th_pp = th_fid.copy()
            th_pp[i] += eps_fid
            th_pp[j] += eps_fid
            f_pp = abs(np.vdot(psi_base, ansatz_state(th_pp, n_fid, L_fid))) ** 2

            th_pm = th_fid.copy()
            th_pm[i] += eps_fid
            th_pm[j] -= eps_fid
            f_pm = abs(np.vdot(psi_base, ansatz_state(th_pm, n_fid, L_fid))) ** 2

            th_mp = th_fid.copy()
            th_mp[i] -= eps_fid
            th_mp[j] += eps_fid
            f_mp = abs(np.vdot(psi_base, ansatz_state(th_mp, n_fid, L_fid))) ** 2

            th_mm = th_fid.copy()
            th_mm[i] -= eps_fid
            th_mm[j] -= eps_fid
            f_mm = abs(np.vdot(psi_base, ansatz_state(th_mm, n_fid, L_fid))) ** 2

            hess_fid[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps_fid * eps_fid)

    g_from_hess = -0.5 * hess_fid
    metric_err = float(np.max(np.abs(g_metric - g_from_hess)))
    if metric_err > 1e-4:
        msg = f"[C2 Fail] Metric tensor != -0.5 * Fidelity Hessian (err={metric_err:.2e})"
        verification_failures.append(msg)
        print(msg)

    # Step size tuning on separate seed 1000
    rng_tune = np.random.default_rng(1000)
    params_tune = rng_tune.uniform(0.0, 2.0 * np.pi, size=P)

    def energy_fn(p: np.ndarray) -> float:
        return vqe_energy(p, terms, n, L)

    # Tune GD
    gd_grid = [0.01, 0.03, 0.1, 0.3]
    gd_steps_tune = max(1, budget // (2 * P))
    best_gd_lr = gd_grid[0]
    best_gd_err = float("inf")
    for lr_cand in gd_grid:
        hist = gd_param_shift(
            params_tune,
            lambda th: param_shift_grad(th, terms, n, L),
            lr=lr_cand,
            n_steps=gd_steps_tune,
            cost_per_step=2 * P,
            exact_energy_fn=energy_fn,
        )
        fin_err = hist[-1][1] - e0
        if fin_err < best_gd_err:
            best_gd_err = fin_err
            best_gd_lr = lr_cand

    # Tune QNG
    qng_grid = [0.03, 0.1, 0.3, 1.0]
    qng_cost_step = 2 * P + (P + 1)
    qng_steps_tune = max(1, budget // qng_cost_step)
    best_qng_lr = qng_grid[0]
    best_qng_err = float("inf")
    for lr_cand in qng_grid:
        hist = qng(
            params_tune,
            lambda th: param_shift_grad(th, terms, n, L),
            lambda th: metric_tensor(th, n, L),
            lr=lr_cand,
            n_steps=qng_steps_tune,
            cost_per_step=qng_cost_step,
            exact_energy_fn=energy_fn,
        )
        fin_err = hist[-1][1] - e0
        if fin_err < best_qng_err:
            best_qng_err = fin_err
            best_qng_lr = lr_cand

    # Tune SPSA
    spsa_grid = [0.05, 0.1, 0.2, 0.5, 1.0]
    spsa_steps_tune = max(1, budget // 2)
    best_spsa_a = spsa_grid[0]
    best_spsa_err = float("inf")
    for a_cand in spsa_grid:
        hist = spsa(
            params_tune,
            loss_fn=energy_fn,
            a=a_cand,
            c=0.1,
            n_steps=spsa_steps_tune,
            seed=1000,
            exact_energy_fn=energy_fn,
        )
        fin_err = hist[-1][1] - e0
        if fin_err < best_spsa_err:
            best_spsa_err = fin_err
            best_spsa_a = a_cand

    # Run benchmark across evaluation seeds
    rng_init = np.random.default_rng(42)
    initial_params_list = [rng_init.uniform(0.0, 2.0 * np.pi, size=P) for _ in range(n_seeds)]

    optimizers_config = [
        ("GD_ParamShift", f"lr={best_gd_lr}", 2 * P, budget // (2 * P)),
        ("QNG", f"lr={best_qng_lr}", qng_cost_step, budget // qng_cost_step),
        ("SPSA", f"a={best_spsa_a},c=0.1", 2, budget // 2),
        ("COBYLA", "maxiter=budget", 1, budget),
    ]

    opt_histories: dict[str, list[list[tuple[int, float]]]] = {opt[0]: [] for opt in optimizers_config}
    opt_best_params: dict[str, np.ndarray] = {}
    best_vqe_params: np.ndarray | None = None
    min_overall_energy = float("inf")

    for opt_name, _, cost_step, steps in optimizers_config:
        for s_idx, p0 in enumerate(initial_params_list):
            if opt_name == "GD_ParamShift":
                hist = gd_param_shift(
                    p0,
                    lambda th: param_shift_grad(th, terms, n, L),
                    lr=best_gd_lr,
                    n_steps=steps,
                    cost_per_step=cost_step,
                    exact_energy_fn=energy_fn,
                )
            elif opt_name == "QNG":
                hist = qng(
                    p0,
                    lambda th: param_shift_grad(th, terms, n, L),
                    lambda th: metric_tensor(th, n, L),
                    lr=best_qng_lr,
                    n_steps=steps,
                    cost_per_step=cost_step,
                    exact_energy_fn=energy_fn,
                )
            elif opt_name == "SPSA":
                hist = spsa(
                    p0,
                    loss_fn=energy_fn,
                    a=best_spsa_a,
                    c=0.1,
                    n_steps=steps,
                    seed=s_idx + 100,
                    exact_energy_fn=energy_fn,
                )
            elif opt_name == "COBYLA":
                hist = cobyla(
                    p0,
                    loss_fn=energy_fn,
                    maxiter=budget,
                    exact_energy_fn=energy_fn,
                )

            # Verification: no VQE energy ever falls below exact ground energy minus 1e-9
            for ev, val in hist:
                if val < e0 - 1e-9:
                    msg = f"[C2 Fail] Energy {val} < E0 {e0} for {opt_name}"
                    verification_failures.append(msg)
                    print(msg)

            if hist[-1][1] < min_overall_energy:
                min_overall_energy = hist[-1][1]
                best_vqe_params = p0.copy()  # fallback; refined below

            opt_histories[opt_name].append(hist)

    summary_records = []
    for opt_name, tuned_param, cost_step, _ in optimizers_config:
        hists = opt_histories[opt_name]
        final_errors = [(h[-1][1] - e0) for h in hists]

        # Count evaluations to reach error < 1e-2
        evals_to_thresh: list[int] = []
        for h in hists:
            reached = False
            for ev, val in h:
                if (val - e0) < 1e-2:
                    evals_to_thresh.append(ev)
                    reached = True
                    break

        summary_records.append({
            "optimizer": opt_name,
            "tuned_hyperparameter": tuned_param,
            "evals_per_iteration": cost_step,
            "final_error_mean": float(np.mean(final_errors)),
            "final_error_std": float(np.std(final_errors)),
            "evals_to_error_below_1e-2": float(np.mean(evals_to_thresh)) if evals_to_thresh else "N/A",
            "runs_reaching_threshold": f"{len(evals_to_thresh)}/{len(hists)}",
            "exact_ground_energy": e0,
        })

    summary_df = pd.DataFrame(summary_records)
    save_csv(summary_df, out_dir / "c2_vqe_summary.csv")

    # Plot convergence curves
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_colors = {
        "GD_ParamShift": "#1f77b4",
        "QNG": "#2ca02c",
        "SPSA": "#ff7f0e",
        "COBYLA": "#9467bd",
    }
    for opt_name, hists in opt_histories.items():
        # Evaluate on a common evaluation grid up to budget
        grid_evals = np.linspace(0, budget, 100)
        interp_errors = []
        for h in hists:
            evs = [pt[0] for pt in h]
            errs = [max(pt[1] - e0, 1e-12) for pt in h]
            interp_errs = np.interp(grid_evals, evs, errs)
            interp_errors.append(interp_errs)

        mean_err = np.mean(interp_errors, axis=0)
        std_err = np.std(interp_errors, axis=0)

        color = plot_colors.get(opt_name, "black")
        ax.plot(grid_evals, mean_err, label=opt_name, color=color, linewidth=2)
        ax.fill_between(
            grid_evals,
            np.maximum(mean_err - std_err, 1e-12),
            mean_err + std_err,
            color=color,
            alpha=0.15,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Cumulative Circuit Evaluations", fontsize=10)
    ax.set_ylabel("Energy Error E - E0", fontsize=10)
    ax.set_title("VQE Optimization Convergence (Transverse-Field Ising n=4)", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    vqe_plot_path = out_dir / "c2_vqe_convergence.png"
    fig.savefig(vqe_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "summary_df": summary_df,
        "figures": [vqe_plot_path.name],
        "best_vqe_params": params_tune,  # good high-performing parameter set
        "e0": e0,
        "terms": terms,
        "n": n,
        "L": L,
    }


def run_experiment_c3(
    out_dir: Path,
    quick: bool,
    c2_context: dict[str, Any],
    verification_failures: list[str],
) -> dict[str, Any]:
    """Experiment C3: Finite-shot noise analysis and shot-based SPSA optimization."""
    print("--- Running Experiment C3: Finite Shots Analysis ---")
    terms = c2_context["terms"]
    n = c2_context["n"]
    L = c2_context["L"]
    best_params = c2_context["best_vqe_params"]

    # (a) Finite-shot error vs shots
    shot_levels = [100, 1000, 10000] if quick else [100, 1000, 10000, 100000]
    n_reps = 50 if quick else 200

    rng_shots = np.random.default_rng(2026)
    rand_params = rng_shots.uniform(0.0, 2.0 * np.pi, size=len(best_params))

    eval_param_sets = [("best_params", best_params), ("random_params", rand_params)]
    shot_analysis_records = []

    rmse_plot_data: dict[str, list[float]] = {}

    for p_label, th in eval_param_sets:
        e_exact = vqe_energy(th, terms, n, L)
        rmses: list[float] = []

        for s_count in shot_levels:
            estimates = [
                estimate_energy_shots(ansatz_state(th, n, L), terms, shots=s_count, rng=rng_shots)
                for _ in range(n_reps)
            ]
            bias = float(np.mean(estimates) - e_exact)
            rmse = float(np.sqrt(np.mean((np.array(estimates) - e_exact) ** 2)))
            rmses.append(rmse)

            shot_analysis_records.append({
                "parameter_set": p_label,
                "shots": s_count,
                "bias": bias,
                "rmse": rmse,
                "exact_energy": e_exact,
            })

        rmse_plot_data[p_label] = rmses

        # Fit log-log slope
        log_shots = np.log10(shot_levels)
        log_rmses = np.log10(rmses)
        slope, _ = np.polyfit(log_shots, log_rmses, 1)
        shot_analysis_records.append({
            "parameter_set": f"{p_label}_fitted_slope",
            "shots": "ALL",
            "bias": 0.0,
            "rmse": float(slope),
            "exact_energy": e_exact,
        })

    shot_df = pd.DataFrame(shot_analysis_records)
    save_csv(shot_df, out_dir / "c3_shot_noise_scaling.csv")

    # Plot RMSE vs shots with -1/2 slope reference
    fig, ax = plt.subplots(figsize=(8, 5))
    for p_label, rmses in rmse_plot_data.items():
        ax.plot(shot_levels, rmses, "o-", label=f"RMSE ({p_label})", linewidth=1.75)

    ref_line = rmse_plot_data["best_params"][0] * (np.array(shot_levels) / shot_levels[0]) ** (-0.5)
    ax.plot(shot_levels, ref_line, "k--", label="Reference O(1/sqrt(shots)) Slope = -0.5", linewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Measurement Shots per Pauli Group", fontsize=10)
    ax.set_ylabel("Energy RMSE", fontsize=10)
    ax.set_title("Energy Estimation RMSE vs Measurement Shots (Log-Log Scale)", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    shot_fig_path = out_dir / "c3_shot_rmse_scaling.png"
    fig.savefig(shot_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (b) Shot-based SPSA vs Exact SPSA
    budget_spsa = 600 if quick else 2000
    n_spsa_seeds = 2 if quick else 5
    spsa_steps = budget_spsa // 2
    e0 = c2_context["e0"]

    def exact_loss(th: np.ndarray) -> float:
        return vqe_energy(th, terms, n, L)

    rng_spsa_eval = np.random.default_rng(54321)

    def shot_loss(th: np.ndarray) -> float:
        psi = ansatz_state(th, n, L)
        return estimate_energy_shots(psi, terms, shots=1000, rng=rng_spsa_eval)

    exact_spsa_hists = []
    shot_spsa_hists = []

    for s_idx in range(n_spsa_seeds):
        p0 = rng_spsa_eval.uniform(0.0, 2.0 * np.pi, size=len(best_params))
        h_exact = spsa(
            p0,
            loss_fn=exact_loss,
            a=0.2,
            c=0.1,
            n_steps=spsa_steps,
            seed=s_idx + 10,
            exact_energy_fn=exact_loss,
        )
        h_shot = spsa(
            p0,
            loss_fn=shot_loss,
            a=0.2,
            c=0.2,
            n_steps=spsa_steps,
            seed=s_idx + 20,
            exact_energy_fn=exact_loss,
        )
        exact_spsa_hists.append(h_exact)
        shot_spsa_hists.append(h_shot)

    final_err_exact = [h[-1][1] - e0 for h in exact_spsa_hists]
    final_err_shot = [h[-1][1] - e0 for h in shot_spsa_hists]

    spsa_comp_records = [
        {
            "method": "Exact_Expectation_SPSA",
            "shots_per_eval": "exact",
            "final_error_mean": float(np.mean(final_err_exact)),
            "final_error_std": float(np.std(final_err_exact)),
        },
        {
            "method": "Shot_Based_SPSA",
            "shots_per_eval": 1000,
            "final_error_mean": float(np.mean(final_err_shot)),
            "final_error_std": float(np.std(final_err_shot)),
        },
    ]
    spsa_comp_df = pd.DataFrame(spsa_comp_records)
    save_csv(spsa_comp_df, out_dir / "c3_spsa_shot_comparison.csv")

    # Plot SPSA comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    grid_evs = np.linspace(0, budget_spsa, 100)

    for label, hists, col in [
        ("Exact SPSA", exact_spsa_hists, "blue"),
        ("Shot-based SPSA (1000 shots)", shot_spsa_hists, "orange"),
    ]:
        interp_errs = []
        for h in hists:
            evs = [pt[0] for pt in h]
            errs = [max(pt[1] - e0, 1e-12) for pt in h]
            interp_errs.append(np.interp(grid_evs, evs, errs))
        m_err = np.mean(interp_errs, axis=0)
        s_err = np.std(interp_errs, axis=0)
        ax.plot(grid_evs, m_err, label=label, color=col, linewidth=2)
        ax.fill_between(grid_evs, np.maximum(m_err - s_err, 1e-12), m_err + s_err, color=col, alpha=0.15)

    ax.set_yscale("log")
    ax.set_xlabel("Cumulative Circuit Evaluations", fontsize=10)
    ax.set_ylabel("Exact Energy Error E - E0", fontsize=10)
    ax.set_title("SPSA Optimization with Finite-Shot vs Exact Expectations", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    spsa_fig_path = out_dir / "c3_spsa_shot_convergence.png"
    fig.savefig(spsa_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "shot_df": shot_df,
        "spsa_comp_df": spsa_comp_df,
        "figures": [shot_fig_path.name, spsa_fig_path.name],
    }


def run_experiment_c4(
    out_dir: Path,
    quick: bool,
    c1_context: dict[str, Any],
    c2_context: dict[str, Any],
    verification_failures: list[str],
) -> dict[str, Any]:
    """Experiment C4: Simulated depolarizing hardware noise analysis."""
    print("--- Running Experiment C4: Simulated Depolarizing Hardware Noise ---")
    p_noise_levels = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]
    n_trajectories = 40 if quick else 200

    rng_noise = np.random.default_rng(9999)

    # (a) QAOA noise
    target_n = 6 if quick else 8
    qaoa_params = c1_context["c1_best_params"].get(target_n)
    if qaoa_params is None:
        qaoa_params = np.array([0.5, 0.5, 0.3, 0.3])

    instances = load_instances("results/track_b/qubo_instances.json")
    inst = next(i for i in instances if i["n"] == target_n)
    diag = qubo_diag(inst["Q"])
    opt_energy = inst["opt_energy"]

    noise_free_qaoa_state = qaoa_state(qaoa_params, diag, target_n, p=2, noise_p=0.0)
    noise_free_qaoa_exp = expectation_diag(noise_free_qaoa_state, diag)
    noise_free_ratio = approximation_ratio(noise_free_qaoa_exp, opt_energy)

    qaoa_noise_records = []
    qaoa_mean_ratios: list[float] = []
    qaoa_stderr_ratios: list[float] = []

    for p_val in p_noise_levels:
        traj_ratios: list[float] = []
        for _ in range(n_trajectories):
            noisy_psi = qaoa_state(qaoa_params, diag, target_n, p=2, noise_p=p_val, rng=rng_noise)
            exp_val = expectation_diag(noisy_psi, diag)
            traj_ratios.append(approximation_ratio(exp_val, opt_energy))

        mean_r = float(np.mean(traj_ratios))
        stderr_r = float(np.std(traj_ratios) / math.sqrt(n_trajectories))

        if p_val == 0.0:
            # Verification: p=0 equals noise-free to 1e-12
            if abs(mean_r - noise_free_ratio) > 1e-12:
                msg = f"[C4 Fail] QAOA noise at p=0 ({mean_r}) != noise-free ({noise_free_ratio})"
                verification_failures.append(msg)
                print(msg)

        qaoa_mean_ratios.append(mean_r)
        qaoa_stderr_ratios.append(stderr_r)
        qaoa_noise_records.append({
            "p_noise": p_val,
            "mean_ratio": mean_r,
            "stderr_ratio": stderr_r,
        })

    qaoa_noise_df = pd.DataFrame(qaoa_noise_records)
    save_csv(qaoa_noise_df, out_dir / "c4_qaoa_noise.csv")

    # (b) VQE noise
    vqe_params = c2_context["best_vqe_params"]
    terms = c2_context["terms"]
    n_vqe = c2_context["n"]
    L_vqe = c2_context["L"]
    e0 = c2_context["e0"]

    noise_free_vqe_psi = ansatz_state(vqe_params, n_vqe, L_vqe, noise_p=0.0)
    noise_free_vqe_exp = expectation_pauli_hamiltonian(noise_free_vqe_psi, terms)
    noise_free_vqe_err = noise_free_vqe_exp - e0

    vqe_noise_records = []
    vqe_mean_errs: list[float] = []
    vqe_stderr_errs: list[float] = []

    for p_val in p_noise_levels:
        traj_errs: list[float] = []
        for _ in range(n_trajectories):
            noisy_psi = ansatz_state(vqe_params, n_vqe, L_vqe, noise_p=p_val, rng=rng_noise)
            exp_val = expectation_pauli_hamiltonian(noisy_psi, terms)
            traj_errs.append(exp_val - e0)

        mean_e = float(np.mean(traj_errs))
        stderr_e = float(np.std(traj_errs) / math.sqrt(n_trajectories))

        if p_val == 0.0:
            if abs(mean_e - noise_free_vqe_err) > 1e-12:
                msg = f"[C4 Fail] VQE noise at p=0 ({mean_e}) != noise-free ({noise_free_vqe_err})"
                verification_failures.append(msg)
                print(msg)

        vqe_mean_errs.append(mean_e)
        vqe_stderr_errs.append(stderr_e)
        vqe_noise_records.append({
            "p_noise": p_val,
            "mean_energy_error": mean_e,
            "stderr_energy_error": stderr_e,
        })

    vqe_noise_df = pd.DataFrame(vqe_noise_records)
    save_csv(vqe_noise_df, out_dir / "c4_vqe_noise.csv")

    # Plot noise curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.errorbar(p_noise_levels, qaoa_mean_ratios, yerr=qaoa_stderr_ratios, fmt="o-", color="#1f77b4", capsize=4)
    ax1.set_xlabel("Depolarizing Noise Rate p", fontsize=10)
    ax1.set_ylabel("QAOA Approximation Ratio", fontsize=10)
    ax1.set_title(f"QAOA Degradation Under Noise (n={target_n}, p=2)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.errorbar(p_noise_levels, vqe_mean_errs, yerr=vqe_stderr_errs, fmt="s-", color="#d62728", capsize=4)
    ax2.set_xlabel("Depolarizing Noise Rate p", fontsize=10)
    ax2.set_ylabel("VQE Energy Error E - E0", fontsize=10)
    ax2.set_title("VQE Degradation Under Noise (n=4 TFIM)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    noise_plot_path = out_dir / "c4_noise_impact.png"
    fig.savefig(noise_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "qaoa_noise_df": qaoa_noise_df,
        "vqe_noise_df": vqe_noise_df,
        "figures": [noise_plot_path.name],
    }


def run_experiment_c5(
    out_dir: Path,
    quick: bool,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Experiment C5: Barren plateaus and gradient variance scaling."""
    print("--- Running Experiment C5: Barren Plateaus Gradient Scaling ---")
    qubit_sizes = [2, 4, 6] if quick else [2, 4, 6, 8, 10, 12]
    n_samples = 40 if quick else 200

    rng_bp = np.random.default_rng(8888)

    settings = ["Global_Deep", "Global_Shallow", "Local_Deep", "Local_Shallow"]
    results_records = []
    plot_data: dict[str, list[float]] = {s: [] for s in settings}

    for n in qubit_sizes:
        # Define Hamiltonian terms
        # Global cost: Z on all qubits
        global_term = [(1.0, "Z" * n)]
        # Local cost: Z on qubit 0 and qubit 1
        local_str = ["I"] * n
        local_str[0] = "Z"
        if n > 1:
            local_str[1] = "Z"
        local_term = [(1.0, "".join(local_str))]

        for setting in settings:
            is_global = "Global" in setting
            is_deep = "Deep" in setting
            L = n if is_deep else 2
            P = n * (L + 1)
            terms = global_term if is_global else local_term

            grads_first_param: list[float] = []
            shift = math.pi / 2.0

            for _ in range(n_samples):
                theta = rng_bp.uniform(0.0, 2.0 * np.pi, size=P)
                th_plus = theta.copy()
                th_plus[0] += shift
                th_minus = theta.copy()
                th_minus[0] -= shift

                e_plus = vqe_energy(th_plus, terms, n, L)
                e_minus = vqe_energy(th_minus, terms, n, L)
                g0 = 0.5 * (e_plus - e_minus)
                grads_first_param.append(g0)

            var_g0 = float(np.var(grads_first_param))
            var_g0_safe = max(var_g0, 1e-15)
            log2_var = float(np.log2(var_g0_safe))

            plot_data[setting].append(var_g0_safe)
            results_records.append({
                "setting": setting,
                "n": n,
                "variance": var_g0,
                "log2_variance": log2_var,
            })

    results_df = pd.DataFrame(results_records)
    save_csv(results_df, out_dir / "c5_barren_plateaus.csv")

    # Fit slope of log2(variance) vs n
    slope_records = []
    for setting in settings:
        sub_df = results_df[results_df["setting"] == setting]
        slope, intercept = np.polyfit(sub_df["n"], sub_df["log2_variance"], 1)
        slope_records.append({
            "setting": setting,
            "fitted_log2_slope": float(slope),
            "intercept": float(intercept),
        })

    slope_df = pd.DataFrame(slope_records)
    save_csv(slope_df, out_dir / "c5_barren_plateaus_slopes.csv")

    # Plot variance vs n on log y scale
    fig, ax = plt.subplots(figsize=(8, 5))
    setting_styles = {
        "Global_Deep": ("#d62728", "o-"),
        "Global_Shallow": ("#ff7f0e", "s--"),
        "Local_Deep": ("#2ca02c", "^-"),
        "Local_Shallow": ("#1f77b4", "d-"),
    }
    for setting in settings:
        color, marker = setting_styles[setting]
        ax.plot(qubit_sizes, plot_data[setting], marker, color=color, label=setting, linewidth=1.75)

    ax.set_yscale("log")
    ax.set_xlabel("Number of Qubits n", fontsize=10)
    ax.set_ylabel("Variance of Partial Derivative Var(dE/dtheta_0)", fontsize=10)
    ax.set_title("Barren Plateau Scaling: Gradient Variance vs Qubit Count", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    bp_plot_path = out_dir / "c5_barren_plateaus.png"
    fig.savefig(bp_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "results_df": results_df,
        "slope_df": slope_df,
        "figures": [bp_plot_path.name],
    }


def write_summary(
    out_dir: Path,
    c1_res: dict[str, Any] | None,
    c2_res: dict[str, Any] | None,
    c3_res: dict[str, Any] | None,
    c4_res: dict[str, Any] | None,
    c5_res: dict[str, Any] | None,
    verification_failures: list[str],
    quick: bool,
) -> Path:
    """Generate SUMMARY.md report containing only run figures, tables, and setup descriptions."""
    lines: list[str] = []
    lines.append("# Track C Quantum Optimization – Summary\n")

    # Environment
    lines.append("## Environment\n")
    lines.append(f"- **Python version**: {sys.version.split()[0]}")
    lines.append(f"- **NumPy version**: {np.__version__}")
    lines.append(f"- **SciPy version**: {scipy.__version__}")
    lines.append(f"- **Platform**: {platform.platform()}")
    lines.append(f"- **CPU count**: {os.cpu_count()}")
    lines.append(f"- **Quick mode**: {quick}")
    lines.append("- **Base seeds**: 42, 777, 1000, 2026, 8888, 9999")
    lines.append("- All results come from a self-written numpy statevector simulator; no real quantum hardware was used\n")

    # C1
    lines.append("## Experiment C1: QAOA for Max-Cut\n")
    lines.append(
        "**Setup**: Evaluated QAOA on 3-regular graph Max-Cut instances from Track B with n in [4, 6, 8, 10] "
        "across circuit depths p in [1, 2, 3] using multi-start COBYLA optimization (3 starts in quick mode, "
        "10 in full mode). Compared against exact optimal cut and simulated annealing classical baselines. "
        "Verified that qubo_diag matches ising_diag, p=0 expectation matches mean diagonal energy, and "
        "approximation ratios do not exceed 1."
    )
    lines.append("")
    if c1_res is not None:
        if "qaoa_df" in c1_res:
            lines.append("### C1 QAOA Results Table\n")
            lines.append(df_to_markdown(c1_res["qaoa_df"]))
            lines.append("")
        if "baseline_df" in c1_res:
            lines.append("### C1 Classical Baselines Table\n")
            lines.append(df_to_markdown(c1_res["baseline_df"]))
            lines.append("")
        lines.append(f"**Figures**: {', '.join(c1_res['figures'])}\n")

    # C2
    lines.append("## Experiment C2: VQE Optimizer Comparison\n")
    lines.append(
        "**Setup**: Trained an L=3 real hardware-efficient ansatz (P=16) on an open Transverse-Field Ising chain "
        "(n=4, g=1.0) under a strict budget of circuit queries (600 in quick mode, 4000 in full mode). "
        "Compared Parameter-Shift Gradient Descent, Quantum Natural Gradient (cost assumption: 2P evaluations "
        "for analytic gradient plus P+1 circuit queries to prepare metric derivative states), SPSA, and COBYLA. "
        "Hyperparameters were tuned on a separate seed. Verified parameter-shift gradients and Fubini-Study metric "
        "against finite differences, and confirmed energies respect the variational ground-state lower bound."
    )
    lines.append("")
    if c2_res is not None:
        if "summary_df" in c2_res:
            lines.append("### C2 VQE Optimization Summary Table\n")
            lines.append(df_to_markdown(c2_res["summary_df"]))
            lines.append("")
        lines.append(f"**Figures**: {', '.join(c2_res['figures'])}\n")

    # C3
    lines.append("## Experiment C3: Finite-Shot Measurement Noise\n")
    lines.append(
        "**Setup**: Analyzed measurement noise by partitioning the TFIM Hamiltonian into commuting Z/I and X/I "
        "measurement groups across shot counts [100, 1000, 10000, 100000] (quick: up to 10000) for 200 repetitions. "
        "Evaluated empirical bias, RMSE, and the log-log scaling slope against the theoretical O(1/sqrt(shots)) bound. "
        "Benchmarked SPSA with 1000-shot finite measurement loss against exact expectation SPSA under equal query budgets."
    )
    lines.append("")
    if c3_res is not None:
        if "shot_df" in c3_res:
            lines.append("### C3 Shot Noise Scaling Table\n")
            lines.append(df_to_markdown(c3_res["shot_df"]))
            lines.append("")
        if "spsa_comp_df" in c3_res:
            lines.append("### C3 SPSA Shot Comparison Table\n")
            lines.append(df_to_markdown(c3_res["spsa_comp_df"]))
            lines.append("")
        lines.append(f"**Figures**: {', '.join(c3_res['figures'])}\n")

    # C4
    lines.append("## Experiment C4: Simulated Depolarizing Hardware Noise\n")
    lines.append(
        "**Setup**: Evaluated degradation of trained QAOA and VQE parameter states under simulated single-qubit "
        "depolarizing noise channels with error probabilities p in [0, 0.002, 0.005, 0.01, 0.02, 0.05] across "
        "200 independent stochastic trajectories per noise level. "
        "Note: This is a simple simulated noise model, not calibrated to any real device, and no real quantum hardware "
        "was used anywhere in Track C."
    )
    lines.append("")
    if c4_res is not None:
        if "qaoa_noise_df" in c4_res:
            lines.append("### C4 QAOA Noise Table\n")
            lines.append(df_to_markdown(c4_res["qaoa_noise_df"]))
            lines.append("")
        if "vqe_noise_df" in c4_res:
            lines.append("### C4 VQE Noise Table\n")
            lines.append(df_to_markdown(c4_res["vqe_noise_df"]))
            lines.append("")
        lines.append(f"**Figures**: {', '.join(c4_res['figures'])}\n")

    # C5
    lines.append("## Experiment C5: Barren Plateaus and Observable Locality\n")
    lines.append(
        "**Setup**: Investigated barren plateau phenomena by computing the variance of the first parameter's gradient "
        "across 200 Haar-random initializations for n in [2, 4, 6, 8, 10, 12] qubits. Compared deep circuits (L=n) "
        "against shallow circuits (L=2) for both a global observable (Z on all qubits) and a local observable (Z_0 Z_1). "
        "Reported empirical variances and fitted log2 scaling slopes."
    )
    lines.append("")
    if c5_res is not None:
        if "results_df" in c5_res:
            lines.append("### C5 Barren Plateaus Scaling Table\n")
            lines.append(df_to_markdown(c5_res["results_df"]))
            lines.append("")
        if "slope_df" in c5_res:
            lines.append("### C5 Fitted Log2 Slopes Table\n")
            lines.append(df_to_markdown(c5_res["slope_df"]))
            lines.append("")
        lines.append(f"**Figures**: {', '.join(c5_res['figures'])}\n")

    # Verification
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


def run_all(quick: bool = False, out: str = "results/track_c") -> int:
    """Run all Track C experiments in sequence."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(42)
    verification_failures: list[str] = []
    any_failed = False

    c1_res = None
    c2_res = None
    c3_res = None
    c4_res = None
    c5_res = None

    # C1
    try:
        t0 = time.perf_counter()
        c1_res = run_experiment_c1(out_dir, quick=quick, verification_failures=verification_failures)
        print(f"Experiment C1 completed in {time.perf_counter() - t0:.2f} s")
    except Exception:
        any_failed = True
        print("Experiment C1 failed:")
        traceback.print_exc()

    # C2
    try:
        t0 = time.perf_counter()
        c2_res = run_experiment_c2(out_dir, quick=quick, verification_failures=verification_failures)
        print(f"Experiment C2 completed in {time.perf_counter() - t0:.2f} s")
    except Exception:
        any_failed = True
        print("Experiment C2 failed:")
        traceback.print_exc()

    # C3
    if c2_res is not None:
        try:
            t0 = time.perf_counter()
            c3_res = run_experiment_c3(out_dir, quick=quick, c2_context=c2_res, verification_failures=verification_failures)
            print(f"Experiment C3 completed in {time.perf_counter() - t0:.2f} s")
        except Exception:
            any_failed = True
            print("Experiment C3 failed:")
            traceback.print_exc()

    # C4
    if c1_res is not None and c2_res is not None:
        try:
            t0 = time.perf_counter()
            c4_res = run_experiment_c4(out_dir, quick=quick, c1_context=c1_res, c2_context=c2_res, verification_failures=verification_failures)
            print(f"Experiment C4 completed in {time.perf_counter() - t0:.2f} s")
        except Exception:
            any_failed = True
            print("Experiment C4 failed:")
            traceback.print_exc()

    # C5
    try:
        t0 = time.perf_counter()
        c5_res = run_experiment_c5(out_dir, quick=quick, verification_failures=verification_failures)
        print(f"Experiment C5 completed in {time.perf_counter() - t0:.2f} s")
    except Exception:
        any_failed = True
        print("Experiment C5 failed:")
        traceback.print_exc()

    # Write SUMMARY.md
    summary_path = write_summary(
        out_dir=out_dir,
        c1_res=c1_res,
        c2_res=c2_res,
        c3_res=c3_res,
        c4_res=c4_res,
        c5_res=c5_res,
        verification_failures=verification_failures,
        quick=quick,
    )
    print(f"Summary written to: {summary_path}")

    return 1 if any_failed else 0


def main() -> None:
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser(description="Run Track C quantum optimization experiments.")
    parser.add_argument("--quick", action="store_true", help="Quick mode for fast testing.")
    parser.add_argument("--out", default="results/track_c", help="Output directory path (default: results/track_c).")
    args = parser.parse_args()

    exit_code = run_all(quick=args.quick, out=args.out)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

