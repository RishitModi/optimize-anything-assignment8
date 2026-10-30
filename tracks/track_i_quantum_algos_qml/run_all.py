"""Master evaluation and benchmarking script for Track I: Quantum Algorithms and QML.

Runs up to five core experiments:
  I1: Quantum algorithms (Deutsch-Jozsa, Bernstein-Vazirani, QFT, QPE, Shor's order finding).
  I2: Grover search vs classical linear search (optimal iterations, overshoot, query & runtime scaling).
  I3: Quantum annealing simulation & Chimera QUBO embedding limits.
  I4: Hardware noise modeling (depolarizing error trajectories across DJ, BV, Grover).
  I5: Quantum Machine Learning (VQC with parameter-shift, quantum kernels, and classical baselines).

Usage:
  python tracks/track_i_quantum_algos_qml/run_all.py [--quick] [--out results/track_i] [--max-minutes 8] [--only I1 I2 I3 I4 I5]
"""

from __future__ import annotations

import argparse
import math
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
import scipy
import sklearn

from qi_algos import (
    balanced_parity,
    balanced_random,
    bv_distribution,
    constant0,
    constant1,
    dj_probability_all_zero,
    grover_gate_counts,
    grover_optimal_iterations,
    grover_run,
    grover_theory,
    linear_search_queries,
    qft,
    qpe_distribution,
    shor_factors,
    shor_order_finding,
    shor_postprocess,
    true_order,
)
from qi_anneal import (
    anneal_success,
    chimera_clique_embedding,
    load_instances,
    optimal_indices,
    qubo_diag,
)
from qi_qml import (
    compute_kernel_matrix,
    evaluate_classical_baselines,
    evaluate_quantum_kernel,
    make_dataset,
    param_shift_grad,
    train_classifier,
    variational_forward,
)
from qi_sim import zero_state
from qi_util import Timer, save_fig, to_markdown, write_csv


def run_experiment_i1(
    out_dir: Path,
    quick: bool,
    deadline: float,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Experiment I1: Textbook Quantum Algorithms."""
    results: dict[str, Any] = {}
    rng = np.random.default_rng(2026)

    # -------------------------------------------------------------------------
    # (a) Deutsch-Jozsa
    # -------------------------------------------------------------------------
    dj_ns = [2, 4, 6] if quick else [2, 3, 4, 5, 6, 7, 8]
    dj_rows: list[dict[str, Any]] = []

    for n in dj_ns:
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded during I1 Deutsch-Jozsa")
        oracles = [
            ("constant0", constant0(n), True),
            ("constant1", constant1(n), True),
            ("balanced_random", balanced_random(n, seed=int(rng.integers(1000))), False),
            ("balanced_parity", balanced_parity(n), False),
        ]
        for name, table, is_const in oracles:
            p_zero = dj_probability_all_zero(n, table)
            expected = 1.0 if is_const else 0.0
            if abs(p_zero - expected) > 1e-9:
                verification_failures.append(
                    f"DJ failure n={n} {name}: P(0)={p_zero:.6f} expected {expected}"
                )
            dj_rows.append({
                "n": n,
                "oracle": name,
                "p_all_zero": p_zero,
                "quantum_queries": 1,
                "classical_queries_worst": (1 << (n - 1)) + 1,
            })

    write_csv(out_dir / "i1_deutsch_jozsa.csv", dj_rows)
    results["dj_rows"] = dj_rows

    # -------------------------------------------------------------------------
    # (b) Bernstein-Vazirani
    # -------------------------------------------------------------------------
    bv_ns = [2, 4, 6] if quick else [2, 3, 4, 5, 6, 7, 8, 9, 10]
    bv_rows: list[dict[str, Any]] = []

    for n in bv_ns:
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded during I1 Bernstein-Vazirani")
        num_secrets = 20
        probs: list[float] = []
        for _ in range(num_secrets):
            secret = rng.integers(0, 2, size=n)
            secret_idx = int(np.sum(secret * (1 << np.arange(n - 1, -1, -1))))
            dist = bv_distribution(secret)
            p_secret = float(dist[secret_idx])
            probs.append(p_secret)
            if abs(p_secret - 1.0) > 1e-9:
                verification_failures.append(
                    f"BV failure n={n} secret={secret}: P(s)={p_secret:.6f}"
                )
        bv_rows.append({
            "n": n,
            "num_secrets_tested": num_secrets,
            "mean_secret_prob": float(np.mean(probs)),
            "quantum_queries": 1,
            "classical_queries": n,
        })

    write_csv(out_dir / "i1_bernstein_vazirani.csv", bv_rows)
    results["bv_rows"] = bv_rows

    # -------------------------------------------------------------------------
    # (c) Quantum Fourier Transform
    # -------------------------------------------------------------------------
    qft_ns = list(range(1, 7)) if quick else list(range(1, 11))
    qft_rows: list[dict[str, Any]] = []

    for n in qft_ns:
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded during I1 QFT")
        N = 1 << n
        qubits = list(range(n))
        max_diff = 0.0
        max_inv_diff = 0.0

        t0 = time.perf_counter()
        for _ in range(5):
            st = rng.normal(size=N) + 1.0j * rng.normal(size=N)
            st = st / np.linalg.norm(st)

            # Forward QFT
            qft_out = qft(st, qubits, n, inverse=False)
            expected_fft = np.fft.ifft(st) * np.sqrt(N)
            diff = float(np.max(np.abs(qft_out - expected_fft)))
            if diff > max_diff:
                max_diff = diff

            # Inverse QFT
            inv_out = qft(qft_out, qubits, n, inverse=True)
            inv_diff = float(np.max(np.abs(inv_out - st)))
            if inv_diff > max_inv_diff:
                max_inv_diff = inv_diff

        sim_time = time.perf_counter() - t0

        if max_diff > 1e-9:
            verification_failures.append(f"QFT forward diff n={n}: {max_diff:.3e} > 1e-9")
        if max_inv_diff > 1e-9:
            verification_failures.append(f"QFT inverse identity diff n={n}: {max_inv_diff:.3e} > 1e-9")

        hadamards = n
        cphase = n * (n - 1) // 2
        swaps = n // 2
        qft_rows.append({
            "n": n,
            "dimension": N,
            "max_fft_diff": max_diff,
            "max_invert_diff": max_inv_diff,
            "hadamards": hadamards,
            "cphase_gates": cphase,
            "swap_gates": swaps,
            "total_gates": hadamards + cphase + swaps,
            "time_s": sim_time,
        })

    write_csv(out_dir / "i1_qft.csv", qft_rows)
    results["qft_rows"] = qft_rows

    # -------------------------------------------------------------------------
    # (d) Quantum Phase Estimation (QPE)
    # -------------------------------------------------------------------------
    # Exact check: phi = 5/16 with t = 4
    qpe_dist_exact = qpe_distribution(5.0 / 16.0, t=4)
    if abs(float(qpe_dist_exact[5]) - 1.0) > 1e-9:
        verification_failures.append(
            f"QPE exact 5/16 failure: P(5)={float(qpe_dist_exact[5]):.6f} expected 1.0"
        )

    qpe_rows: list[dict[str, Any]] = []
    phi_test = 0.3
    t_vals = list(range(3, 8)) if quick else list(range(3, 11))

    for t in t_vals:
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded during I1 QPE")
        dist = qpe_distribution(phi_test, t=t)
        most_prob_m = int(np.argmax(dist))
        prob_m = float(dist[most_prob_m])
        est_phi = most_prob_m / float(1 << t)
        error = abs(est_phi - phi_test)

        # Theoretical lower bound on peak: 4 / pi^2 ~ 0.40528
        if prob_m < (4.0 / (np.pi ** 2) - 1e-9):
            verification_failures.append(
                f"QPE peak prob below bound at t={t}: P={prob_m:.4f} < 4/pi^2"
            )

        # Mass within 1 unit 1/2^t of phi
        unit = 1.0 / float(1 << t)
        basis = np.arange(1 << t)
        in_window = np.abs(basis * unit - phi_test) <= unit
        mass_window = float(np.sum(dist[in_window]))

        qpe_rows.append({
            "t": t,
            "true_phi": phi_test,
            "best_outcome": most_prob_m,
            "best_prob": prob_m,
            "estimated_phi": est_phi,
            "phase_error": error,
            "window_mass": mass_window,
        })

    write_csv(out_dir / "i1_qpe.csv", qpe_rows)
    results["qpe_rows"] = qpe_rows

    # Plot QPE scaling
    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=150)
    ts = [r["t"] for r in qpe_rows]
    errs = [r["phase_error"] for r in qpe_rows]
    probs = [r["best_prob"] for r in qpe_rows]

    ax1.plot(ts, errs, "o-", color="tab:blue", label="Phase Error |m/2^t - phi|")
    ax1.set_xlabel("Number of Counting Qubits (t)")
    ax1.set_ylabel("Phase Estimation Error", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(ts, probs, "s--", color="tab:orange", label="Peak Probability P(m)")
    ax2.axhline(4.0 / (np.pi ** 2), color="red", linestyle=":", label="Theoretical Floor (4/pi^2)")
    ax2.set_ylabel("Probability of Peak Outcome", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0.0, 1.05)

    plt.title("QPE Convergence and Peak Probability vs Counting Qubits (phi=0.3)")
    save_fig(fig, out_dir / "i1_qpe_scaling.png")

    # -------------------------------------------------------------------------
    # (e) Shor's Order Finding & Factorization
    # -------------------------------------------------------------------------
    shor_instances: list[tuple[int, list[int], int, int]] = []
    if quick:
        shor_instances.append((15, [2, 7], 8, 4))
    else:
        shor_instances.append((15, [2, 4, 7, 8, 11, 13], 8, 4))
        shor_instances.append((21, [2, 4, 5, 8, 10, 11, 13, 16, 17, 19], 10, 5))

    shor_rows: list[dict[str, Any]] = []
    shor_dists: dict[tuple[int, int], np.ndarray] = {}

    for N, a_list, t, w in shor_instances:
        for a in a_list:
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded during I1 Shor")
            r_true = true_order(a, N)
            if r_true is None:
                continue

            t0 = time.perf_counter()
            probs, gates = shor_order_finding(N, a, t, w)
            sim_time = time.perf_counter() - t0
            shor_dists[(N, a)] = probs

            # Check if r divides 2^t for exact multiples check
            if N == 15 and (1 << t) % r_true == 0:
                step = (1 << t) // r_true
                exact_multiples = np.arange(0, 1 << t, step)
                mass_exact = float(np.sum(probs[exact_multiples]))
                if abs(mass_exact - 1.0) > 1e-9:
                    verification_failures.append(
                        f"Shor N={N} a={a}: mass on exact multiples={mass_exact:.6f} != 1.0"
                    )

            # Sum probability of finding correct order and nontrivial factors
            p_order_found = 0.0
            p_factor_found = 0.0
            for m_out in range(1 << t):
                p_m = float(probs[m_out])
                if p_m <= 1e-15:
                    continue
                cand_r = shor_postprocess(m_out, t, N, a)
                if cand_r == r_true:
                    p_order_found += p_m
                if cand_r is not None:
                    facts = shor_factors(a, cand_r, N)
                    if facts is not None and len(facts) > 0:
                        p_factor_found += p_m

            exp_reps = 1.0 / p_factor_found if p_factor_found > 1e-9 else float("inf")

            # 200 Sampled shots
            sample_rng = np.random.default_rng(42)
            sampled_counts = sample_rng.multinomial(200, probs)
            shots_success = 0
            for m_out in np.where(sampled_counts > 0)[0]:
                cand_r = shor_postprocess(int(m_out), t, N, a)
                if cand_r is not None:
                    facts = shor_factors(a, cand_r, N)
                    if facts is not None and len(facts) > 0:
                        shots_success += int(sampled_counts[m_out])

            shor_rows.append({
                "N": N,
                "a": a,
                "true_order_r": r_true,
                "p_order_exact": p_order_found,
                "p_factor_success": p_factor_found,
                "expected_reps": exp_reps,
                "shots_tested": 200,
                "shots_factored": shots_success,
                "qubits_total": t + w,
                "ctrl_mod_mul_gates": gates["controlled_mod_mul"],
                "sim_time_s": sim_time,
                "classical_trial_divs": 3,  # smallest prime factor is 3
            })

    write_csv(out_dir / "i1_shor.csv", shor_rows)
    results["shor_rows"] = shor_rows

    # Plot Shor counting distribution for N=15, a=7
    if (15, 7) in shor_dists:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        p15_7 = shor_dists[(15, 7)]
        m_indices = np.arange(len(p15_7))
        ax.bar(m_indices, p15_7, width=1.5, color="tab:purple", alpha=0.85)
        ax.set_xlabel("Counting Register Outcome m (0 .. 255)")
        ax.set_ylabel("Probability")
        ax.set_title("Shor's Counting Outcome Distribution for N=15, a=7 (r=4, multiples of 64)")
        ax.grid(True, linestyle="--", alpha=0.4)
        save_fig(fig, out_dir / "i1_shor_distribution.png")

    return results


def run_experiment_i2(
    out_dir: Path,
    quick: bool,
    deadline: float,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Experiment I2: Grover Search vs Classical Linear Search."""
    results: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # (a) Success probability vs iterations and overshoot
    # -------------------------------------------------------------------------
    test_configs = [(6, 1), (8, 1), (8, 4)]
    curve_data: list[dict[str, Any]] = []

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    for n, M in test_configs:
        N = 1 << n
        opt_k = grover_optimal_iterations(N, M)
        max_k = 2 * opt_k + 2
        marked = list(range(M))

        sim_curve: list[float] = []
        theory_curve: list[float] = []

        for k in range(max_k + 1):
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded during I2 Grover curves")
            p_sim, _, _ = grover_run(n, marked, iterations=k, record_curve=False)
            p_theo = grover_theory(N, M, k)
            sim_curve.append(p_sim)
            theory_curve.append(p_theo)

            # Verification check against theory
            if abs(p_sim - p_theo) > 1e-9:
                verification_failures.append(
                    f"Grover theory mismatch n={n} M={M} k={k}: sim={p_sim:.6f} theo={p_theo:.6f}"
                )

        curve_data.append({
            "n": n,
            "M": M,
            "N": N,
            "opt_k": opt_k,
            "iterations": list(range(max_k + 1)),
            "sim_probs": sim_curve,
            "theory_probs": theory_curve,
        })

        ax.plot(
            range(max_k + 1),
            sim_curve,
            "o-",
            label=f"Sim n={n}, M={M} (N={N})",
            markersize=4,
        )
        ax.plot(
            range(max_k + 1),
            theory_curve,
            "--",
            alpha=0.6,
            label=f"Theory n={n}, M={M}",
        )
        ax.axvline(opt_k, linestyle=":", alpha=0.5)

    ax.set_xlabel("Grover Iterations (k)")
    ax.set_ylabel("Success Probability on Marked States")
    ax.set_title("Grover Success Probability vs Iterations (Oscillation & Overshoot)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    save_fig(fig, out_dir / "i2_grover_curves.png")

    # -------------------------------------------------------------------------
    # (b) Scaling of Grover vs Classical Search
    # -------------------------------------------------------------------------
    scaling_ns = list(range(4, 10)) if quick else list(range(4, 15))
    search_rows: list[dict[str, Any]] = []

    for n in scaling_ns:
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded during I2 search scaling")
        N = 1 << n
        M = 1
        opt_k = grover_optimal_iterations(N, M)
        gates = grover_gate_counts(n, opt_k)

        # Quantum statevector simulation
        t0 = time.perf_counter()
        p_succ, q_queries, _ = grover_run(n, marked=[0], iterations=opt_k)
        q_time = time.perf_counter() - t0

        # Classical search
        n_targets = 50 if quick else 200
        c_queries_list: list[int] = []
        t0 = time.perf_counter()
        for s_idx in range(n_targets):
            targ = (s_idx * 17) % N
            c_queries_list.append(linear_search_queries(N, targ, order_seed=s_idx))
        c_time = time.perf_counter() - t0
        mean_c_queries = float(np.mean(c_queries_list))

        search_rows.append({
            "n": n,
            "search_space_N": N,
            "quantum_queries": opt_k,
            "quantum_success_prob": p_succ,
            "quantum_gates_total": sum(gates.values()),
            "quantum_sim_time_s": q_time,
            "classical_mean_queries": mean_c_queries,
            "classical_theory_queries": (N + 1) / 2.0,
            "classical_worst_queries": N,
            "classical_scan_time_s": c_time / float(n_targets),
        })

    write_csv(out_dir / "i2_search_scaling.csv", search_rows)
    results["search_rows"] = search_rows

    # Log-log plot of queries vs N with power law fits
    N_vals = np.array([r["search_space_N"] for r in search_rows], dtype=np.float64)
    q_q = np.array([r["quantum_queries"] for r in search_rows], dtype=np.float64)
    c_q = np.array([r["classical_mean_queries"] for r in search_rows], dtype=np.float64)

    # Fit log10(queries) = slope * log10(N) + intercept
    slope_q, _ = np.polyfit(np.log10(N_vals), np.log10(np.maximum(1, q_q)), 1)
    slope_c, _ = np.polyfit(np.log10(N_vals), np.log10(c_q), 1)

    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)
    ax.loglog(N_vals, q_q, "o-", color="tab:blue", label=f"Grover (slope={slope_q:.2f}, theory=0.50)")
    ax.loglog(N_vals, c_q, "s-", color="tab:orange", label=f"Classical (slope={slope_c:.2f}, theory=1.00)")
    ax.set_xlabel("Search Space Size N = 2^n")
    ax.set_ylabel("Query Count to Find Target")
    ax.set_title("Query Scaling: Grover O(sqrt(N)) vs Classical O(N)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    save_fig(fig, out_dir / "i2_queries_scaling.png")

    # Plot runtime comparison
    q_times = [r["quantum_sim_time_s"] for r in search_rows]
    c_times = [r["classical_scan_time_s"] for r in search_rows]

    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=150)
    ax.semilogy(N_vals, q_times, "o-", color="tab:red", label="Grover Statevector Simulation Time (s)")
    ax.semilogy(N_vals, c_times, "s-", color="tab:green", label="Classical Linear Scan Time (s)")
    ax.set_xlabel("Search Space Size N = 2^n")
    ax.set_ylabel("Wall-Clock Execution Time (seconds, log scale)")
    ax.set_title("Runtime Comparison: Quantum Simulation vs Classical Scan")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    save_fig(fig, out_dir / "i2_runtime_comparison.png")

    return results


def run_experiment_i3(
    out_dir: Path,
    quick: bool,
    deadline: float,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Experiment I3: Quantum Annealing Simulation and Chimera Clique Embedding."""
    results: dict[str, Any] = {}

    # Load Track B QUBO instances
    instances = load_instances("results/track_b/qubo_instances.json")

    anneal_ns = [4, 6] if quick else [4, 6, 8, 10]
    T_list = [1.0, 5.0, 20.0] if quick else [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    dt_base = 0.1 if quick else 0.05

    anneal_rows: list[dict[str, Any]] = []

    for inst in instances:
        n = inst["n"]
        if n not in anneal_ns:
            continue
        Q = inst["Q"]
        diag = qubo_diag(Q, n)
        opt_idx = optimal_indices(diag)
        e_min = float(np.min(diag))

        for T in T_list:
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded during I3 Annealing")
            p_succ, exp_e, norm_err = anneal_success(diag, n, T, dt_base, opt_idx)

            if norm_err > 1e-9:
                verification_failures.append(
                    f"Anneal norm error n={n} T={T}: {norm_err:.3e} > 1e-9"
                )

            # Energy ratio = expected_energy / optimal_energy
            e_ratio = exp_e / e_min if abs(e_min) > 1e-12 else 1.0

            anneal_rows.append({
                "n": n,
                "T": T,
                "dt": dt_base,
                "success_prob": p_succ,
                "expected_energy": exp_e,
                "optimal_energy": e_min,
                "energy_ratio": e_ratio,
                "norm_error": norm_err,
            })

    # Step-size sensitivity check: n=4, T=10, compare dt=0.05 vs dt=0.025
    inst4 = [i for i in instances if i["n"] == 4][0]
    d4 = qubo_diag(inst4["Q"], 4)
    opt4 = optimal_indices(d4)
    p_dt1, _, _ = anneal_success(d4, 4, 10.0, 0.05, opt4)
    p_dt2, _, _ = anneal_success(d4, 4, 10.0, 0.025, opt4)
    if abs(p_dt1 - p_dt2) > 1e-3:
        verification_failures.append(
            f"Anneal dt sensitivity n=4 T=10: |P(0.05)-P(0.025)|={abs(p_dt1 - p_dt2):.4e} > 1e-3"
        )

    write_csv(out_dir / "i3_annealing.csv", anneal_rows)
    results["anneal_rows"] = anneal_rows

    # Plot success prob vs T per n
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for n in anneal_ns:
        sub = [r for r in anneal_rows if r["n"] == n]
        ts = [r["T"] for r in sub]
        ps = [r["success_prob"] for r in sub]
        ax.plot(ts, ps, "o-", label=f"n = {n} qubits")

    ax.set_xlabel("Annealing Time T (arbitrary units)")
    ax.set_ylabel("Ground State Probability")
    ax.set_title("Transverse-Field Annealing Success Probability vs Time T")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    save_fig(fig, out_dir / "i3_anneal_success.png")

    # -------------------------------------------------------------------------
    # (b) Analytic Chimera Clique Embedding
    # -------------------------------------------------------------------------
    logical_ns = [4, 8, 12, 16, 20, 32, 48, 64, 65, 100]
    embed_rows: list[dict[str, Any]] = []

    for n_log in logical_ns:
        info = chimera_clique_embedding(n_log)
        embed_rows.append(info)

    write_csv(out_dir / "i3_embedding.csv", embed_rows)
    results["embed_rows"] = embed_rows

    # Plot physical qubits vs logical variables
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    logs = [r["n_logical"] for r in embed_rows]
    phys = [r["physical_qubits"] for r in embed_rows]

    ax.plot(logs, phys, "o-", color="tab:purple", label="TRIAD Physical Qubits 4*M*(M+1)")
    ax.axhline(2048, color="red", linestyle="--", label="D-Wave 2000Q Capacity (2048 Qubits)")
    ax.axvline(64, color="gray", linestyle=":", label="Max Logical Clique (n=64)")
    ax.set_xlabel("Logical QUBO Variables (Complete Graph K_n)")
    ax.set_ylabel("Physical Chimera Qubits Required")
    ax.set_title("Analytic TRIAD Embedding Requirements on 16x16 Chimera Graph")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    save_fig(fig, out_dir / "i3_embedding_scaling.png")

    return results


def run_experiment_i4(
    out_dir: Path,
    quick: bool,
    deadline: float,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Experiment I4: Simulated Depolarizing Hardware Noise Model."""
    results: dict[str, Any] = {}
    p_rates = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    n_trajectories = 40 if quick else 200
    n = 6

    noise_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(2026)

    # 1. Ideal zero-noise checks
    grover_opt_k = grover_optimal_iterations(1 << n, 1)
    p_g_ideal, _, _ = grover_run(n, marked=[0], iterations=grover_opt_k, noise_p=0.0)
    bv_ideal = float(bv_distribution(np.ones(n, dtype=int), noise_p=0.0)[(1 << n) - 1])
    dj_ideal = dj_probability_all_zero(n, balanced_parity(n), noise_p=0.0)

    if abs(p_g_ideal - grover_theory(1 << n, 1, grover_opt_k)) > 1e-12:
        verification_failures.append("I4 ideal Grover doesn't match theory to 1e-12")
    if abs(bv_ideal - 1.0) > 1e-12:
        verification_failures.append("I4 ideal BV doesn't match 1.0 to 1e-12")
    if abs(dj_ideal - 0.0) > 1e-12:
        verification_failures.append("I4 ideal DJ balanced doesn't match 0.0 to 1e-12")

    for p in p_rates:
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded during I4 Noise Model")

        g_succs: list[float] = []
        bv_succs: list[float] = []
        dj_wrongs: list[float] = []

        secret = np.ones(n, dtype=int)
        sec_idx = (1 << n) - 1
        bal_table = balanced_parity(n)

        for _ in range(n_trajectories):
            # Grover
            p_g, _, _ = grover_run(n, marked=[0], iterations=grover_opt_k, noise_p=p, rng=rng)
            g_succs.append(p_g)

            # BV
            dist_bv = bv_distribution(secret, noise_p=p, rng=rng)
            bv_succs.append(float(dist_bv[sec_idx]))

            # DJ (probability of measuring all zero on balanced = WRONG answer)
            p_dj_wrong = dj_probability_all_zero(n, bal_table, noise_p=p, rng=rng)
            dj_wrongs.append(p_dj_wrong)

        noise_rows.append({
            "noise_p": p,
            "trajectories": n_trajectories,
            "grover_succ_mean": float(np.mean(g_succs)),
            "grover_succ_stderr": float(np.std(g_succs) / np.sqrt(n_trajectories)),
            "bv_succ_mean": float(np.mean(bv_succs)),
            "bv_succ_stderr": float(np.std(bv_succs) / np.sqrt(n_trajectories)),
            "dj_wrong_mean": float(np.mean(dj_wrongs)),
            "dj_wrong_stderr": float(np.std(dj_wrongs) / np.sqrt(n_trajectories)),
        })

    write_csv(out_dir / "i4_noise_model.csv", noise_rows)
    results["noise_rows"] = noise_rows

    # Plot noise degradation curves
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    ps = [r["noise_p"] for r in noise_rows]

    g_m = [r["grover_succ_mean"] for r in noise_rows]
    g_err = [r["grover_succ_stderr"] for r in noise_rows]
    ax.errorbar(ps, g_m, yerr=g_err, fmt="o-", capsize=3, label="Grover Success P (n=6)", color="tab:blue")

    bv_m = [r["bv_succ_mean"] for r in noise_rows]
    bv_err = [r["bv_succ_stderr"] for r in noise_rows]
    ax.errorbar(ps, bv_m, yerr=bv_err, fmt="s-", capsize=3, label="Bernstein-Vazirani P(s) (n=6)", color="tab:green")

    dj_m = [r["dj_wrong_mean"] for r in noise_rows]
    dj_err = [r["dj_wrong_stderr"] for r in noise_rows]
    ax.errorbar(ps, dj_m, yerr=dj_err, fmt="^-", capsize=3, label="Deutsch-Jozsa P(wrong |0^n|) (n=6)", color="tab:red")

    ax.set_xlabel("Depolarizing Error Probability per Layer (p)")
    ax.set_ylabel("Probability")
    ax.set_title("Algorithm Performance Degradation under Depolarizing Noise (n=6)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    save_fig(fig, out_dir / "i4_noise_degradation.png")

    return results


def run_experiment_i5(
    out_dir: Path,
    quick: bool,
    deadline: float,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Experiment I5: Quantum Machine Learning vs Classical Baselines."""
    results: dict[str, Any] = {}
    datasets = ["moons", "circles"]
    n_seeds = 2 if quick else 5
    epochs = 6 if quick else 30

    # 1. Parameter-shift gradient check vs finite differences
    X_sample, _, y_sample, _ = make_dataset("moons", n=40, seed=0)
    test_params = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.1, 0.2, -0.3, 0.4, -0.2, 0.1])
    grad_ps = param_shift_grad(test_params, X_sample, y_sample)

    eps_fd = 1e-5
    grad_fd = np.empty_like(test_params)
    for j in range(len(test_params)):
        p_plus = test_params.copy()
        p_plus[j] += eps_fd
        loss_plus = float(np.mean((variational_forward(p_plus, X_sample) - y_sample) ** 2))

        p_minus = test_params.copy()
        p_minus[j] -= eps_fd
        loss_minus = float(np.mean((variational_forward(p_minus, X_sample) - y_sample) ** 2))

        grad_fd[j] = (loss_plus - loss_minus) / (2.0 * eps_fd)

    max_grad_diff = float(np.max(np.abs(grad_ps - grad_fd)))
    if max_grad_diff > 1e-6:
        verification_failures.append(
            f"I5 Parameter-shift gradient diff vs FD: {max_grad_diff:.3e} > 1e-6"
        )

    # 2. Kernel matrix symmetry and positive semidefiniteness check
    X_kcheck, _, _, _ = make_dataset("moons", n=20, seed=0)
    K_check = compute_kernel_matrix(X_kcheck, X_kcheck)
    if not np.allclose(K_check, K_check.T, atol=1e-9):
        verification_failures.append("I5 Kernel matrix is not symmetric")
    if not np.allclose(np.diag(K_check), 1.0, atol=1e-9):
        verification_failures.append("I5 Kernel matrix unit diagonal check failed")
    min_eig = float(np.min(np.linalg.eigvalsh(K_check)))
    if min_eig < -1e-9:
        verification_failures.append(f"I5 Kernel matrix min eigenvalue: {min_eig:.3e} < -1e-9")

    # 3. Model benchmarking across datasets
    all_qml_rows: list[dict[str, Any]] = []
    loss_curves: dict[str, list[list[float]]] = {"moons": [], "circles": []}

    for dname in datasets:
        if time.time() > deadline:
            raise TimeoutError(f"Deadline exceeded during I5 {dname}")

        # (a) Variational Quantum Classifier (multi-seed)
        vqc_test_accs: list[float] = []
        vqc_train_accs: list[float] = []
        vqc_times: list[float] = []
        vqc_circuits: list[int] = []

        for s in range(n_seeds):
            res_vqc = train_classifier(dname, seed=s, epochs=epochs, batch_size=20, lr=0.1)
            vqc_test_accs.append(res_vqc["test_accuracy"])
            vqc_train_accs.append(res_vqc["train_accuracy"])
            vqc_times.append(res_vqc["wall_time_s"])
            vqc_circuits.append(res_vqc["circuit_evaluations"])
            loss_curves[dname].append(res_vqc["loss_history"])

        all_qml_rows.append({
            "dataset": dname,
            "model": "variational_classifier",
            "test_accuracy_mean": float(np.mean(vqc_test_accs)),
            "test_accuracy_std": float(np.std(vqc_test_accs)),
            "train_accuracy_mean": float(np.mean(vqc_train_accs)),
            "wall_time_mean_s": float(np.mean(vqc_times)),
            "parameter_count": 12,
            "n_qubits": 2,
            "circuit_evaluations_train": int(np.mean(vqc_circuits)) - 240,
            "circuit_evaluations_pred": 240,
        })

        # (b) Quantum Kernel (Exact)
        res_k_exact = evaluate_quantum_kernel(dname, shot_based=False, seed=42)
        all_qml_rows.append({
            "dataset": dname,
            "model": "quantum_kernel_exact",
            "test_accuracy_mean": res_k_exact["test_accuracy"],
            "test_accuracy_std": 0.0,
            "train_accuracy_mean": res_k_exact["train_accuracy"],
            "wall_time_mean_s": res_k_exact["wall_time_s"],
            "parameter_count": res_k_exact["parameter_count"],
            "n_qubits": 2,
            "circuit_evaluations_train": res_k_exact["circuit_evaluations_train"],
            "circuit_evaluations_pred": res_k_exact["circuit_evaluations_pred"],
        })

        # (c) Quantum Kernel (Shot-based, 1000 shots)
        res_k_shots = evaluate_quantum_kernel(dname, shot_based=True, shots=1000, seed=42)
        all_qml_rows.append({
            "dataset": dname,
            "model": "quantum_kernel_1000shots",
            "test_accuracy_mean": res_k_shots["test_accuracy"],
            "test_accuracy_std": 0.0,
            "train_accuracy_mean": res_k_shots["train_accuracy"],
            "wall_time_mean_s": res_k_shots["wall_time_s"],
            "parameter_count": res_k_shots["parameter_count"],
            "n_qubits": 2,
            "circuit_evaluations_train": res_k_shots["circuit_evaluations_train"],
            "circuit_evaluations_pred": res_k_shots["circuit_evaluations_pred"],
        })

        # (d) Classical Baselines
        classical_res = evaluate_classical_baselines(dname, seed=42)
        for cname, cinfo in classical_res.items():
            all_qml_rows.append({
                "dataset": dname,
                "model": cname,
                "test_accuracy_mean": cinfo["test_accuracy"],
                "test_accuracy_std": 0.0,
                "train_accuracy_mean": cinfo["train_accuracy"],
                "wall_time_mean_s": cinfo["wall_time_s"],
                "parameter_count": cinfo["parameter_count"],
                "n_qubits": 0,
                "circuit_evaluations_train": 0,
                "circuit_evaluations_pred": 0,
            })

    write_csv(out_dir / "i5_qml_comparison.csv", all_qml_rows)
    results["all_qml_rows"] = all_qml_rows

    # Plot training loss curve for VQC
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    for idx, dname in enumerate(datasets):
        ax = axes[idx]
        curves = np.array(loss_curves[dname])  # shape (n_seeds, steps)
        steps = np.arange(curves.shape[1])
        mean_l = np.mean(curves, axis=0)
        std_l = np.std(curves, axis=0)

        ax.plot(steps, mean_l, color="tab:blue", label="Mean Loss")
        ax.fill_between(steps, mean_l - std_l, mean_l + std_l, color="tab:blue", alpha=0.25, label="±1 Std Dev")
        ax.set_xlabel("Adam Optimization Steps")
        ax.set_ylabel("Mean Squared Error Loss")
        ax.set_title(f"VQC Convergence: {dname.capitalize()}")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    save_fig(fig, out_dir / "i5_vqc_loss_curve.png")

    # Bar chart of test accuracy per model
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    models_order = [
        "logistic_regression",
        "mlp_hidden4",
        "svm_rbf",
        "variational_classifier",
        "quantum_kernel_exact",
        "quantum_kernel_1000shots",
    ]
    model_labels = [
        "Logistic Reg",
        "MLP (4 hidden)",
        "RBF SVM",
        "VQC (2Q, L=3)",
        "Q-Kernel (Exact)",
        "Q-Kernel (1000s)",
    ]
    x_pos = np.arange(len(models_order))
    width = 0.35

    acc_moons = [
        [r["test_accuracy_mean"] for r in all_qml_rows if r["dataset"] == "moons" and r["model"] == m][0]
        for m in models_order
    ]
    acc_circles = [
        [r["test_accuracy_mean"] for r in all_qml_rows if r["dataset"] == "circles" and r["model"] == m][0]
        for m in models_order
    ]

    ax.bar(x_pos - width / 2, acc_moons, width, label="Moons", color="tab:blue", alpha=0.85)
    ax.bar(x_pos + width / 2, acc_circles, width, label="Circles", color="tab:orange", alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_labels, rotation=20, ha="right")
    ax.set_ylabel("Test Set Accuracy")
    ax.set_title("Test Accuracy: Classical vs Quantum Models on Non-Linear 2D Datasets")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    save_fig(fig, out_dir / "i5_accuracy_comparison.png")

    return results


def main() -> None:
    """CLI orchestrator for Track I evaluation suite."""
    parser = argparse.ArgumentParser(description="Track I Quantum Algorithms and QML Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run rapid evaluation mode")
    parser.add_argument("--out", type=str, default="results/track_i", help="Output directory")
    parser.add_argument("--max-minutes", type=float, default=8.0, help="Maximum execution time before timeout")
    parser.add_argument("--only", nargs="+", default=["I1", "I2", "I3", "I4", "I5"], help="Subset of experiments")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + args.max_minutes * 60.0

    experiment_status: dict[str, str] = {}
    experiment_results: dict[str, Any] = {}
    failed_tracebacks: dict[str, str] = {}
    verification_failures: list[str] = []

    print(f"=== Starting Track I Quantum Benchmark [Quick: {args.quick}, Max: {args.max_minutes}m] ===")

    try:
        # I1
        if "I1" in args.only:
            print("\n--- Running Experiment I1: Quantum Algorithms ---")
            try:
                res_i1 = run_experiment_i1(out_dir, args.quick, deadline, verification_failures)
                experiment_results["I1"] = res_i1
                experiment_status["I1"] = "Done"
            except Exception as e:
                experiment_status["I1"] = "Failed"
                failed_tracebacks["I1"] = traceback.format_exc()
                print(f"I1 Exception: {e}")

        # I2
        if "I2" in args.only:
            print("\n--- Running Experiment I2: Grover vs Classical Search ---")
            try:
                res_i2 = run_experiment_i2(out_dir, args.quick, deadline, verification_failures)
                experiment_results["I2"] = res_i2
                experiment_status["I2"] = "Done"
            except Exception as e:
                experiment_status["I2"] = "Failed"
                failed_tracebacks["I2"] = traceback.format_exc()
                print(f"I2 Exception: {e}")

        # I3
        if "I3" in args.only:
            print("\n--- Running Experiment I3: Quantum Annealing & QUBO Embedding ---")
            try:
                res_i3 = run_experiment_i3(out_dir, args.quick, deadline, verification_failures)
                experiment_results["I3"] = res_i3
                experiment_status["I3"] = "Done"
            except Exception as e:
                experiment_status["I3"] = "Failed"
                failed_tracebacks["I3"] = traceback.format_exc()
                print(f"I3 Exception: {e}")

        # I4
        if "I4" in args.only:
            print("\n--- Running Experiment I4: Hardware Noise Model ---")
            try:
                res_i4 = run_experiment_i4(out_dir, args.quick, deadline, verification_failures)
                experiment_results["I4"] = res_i4
                experiment_status["I4"] = "Done"
            except Exception as e:
                experiment_status["I4"] = "Failed"
                failed_tracebacks["I4"] = traceback.format_exc()
                print(f"I4 Exception: {e}")

        # I5
        if "I5" in args.only:
            print("\n--- Running Experiment I5: QML vs Classical Baselines ---")
            try:
                res_i5 = run_experiment_i5(out_dir, args.quick, deadline, verification_failures)
                experiment_results["I5"] = res_i5
                experiment_status["I5"] = "Done"
            except Exception as e:
                experiment_status["I5"] = "Failed"
                failed_tracebacks["I5"] = traceback.format_exc()
                print(f"I5 Exception: {e}")

    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user! Writing SUMMARY.md before exit...")
        raise
    finally:
        # Generate SUMMARY.md
        summary_path = out_dir / "SUMMARY.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Track I: Quantum Algorithms and QML - Benchmark Summary\n\n")

            f.write("## Environment\n\n")
            f.write(f"- **Python:** {platform.python_version()}\n")
            f.write(f"- **NumPy:** {np.__version__}\n")
            f.write(f"- **SciPy:** {scipy.__version__}\n")
            f.write(f"- **Scikit-Learn:** {sklearn.__version__}\n")
            f.write(f"- **Platform:** {platform.platform()}\n")
            f.write(f"- **Quick Mode:** {args.quick}\n")
            f.write(f"- **Deadline Guard:** {args.max_minutes} minutes\n\n")

            f.write("> **System Statements:**\n")
            f.write("> 1. All quantum results are simulated with a self-written numpy statevector simulator; no real quantum hardware was used.\n")
            f.write("> 2. Noise is a simple depolarizing-trajectory model, not calibrated to a device.\n")
            f.write("> 3. Embedding numbers are an analytic literature formula, not measured.\n\n")

            # I1 Summary
            f.write("## Experiment I1: Quantum Algorithms\n\n")
            f.write("Textbook quantum algorithms (Deutsch-Jozsa, Bernstein-Vazirani, QFT, QPE, Shor's order finding) implemented gate-by-gate on top of qi_sim.\n\n")
            if "I1" in experiment_results:
                f.write("### (a) Deutsch-Jozsa Results\n\n")
                f.write(to_markdown(experiment_results["I1"]["dj_rows"]) + "\n\n")
                f.write("### (b) Bernstein-Vazirani Results\n\n")
                f.write(to_markdown(experiment_results["I1"]["bv_rows"]) + "\n\n")
                f.write("### (c) Quantum Fourier Transform (QFT)\n\n")
                f.write(to_markdown(experiment_results["I1"]["qft_rows"]) + "\n\n")
                f.write("### (d) Quantum Phase Estimation (QPE)\n\n")
                f.write(to_markdown(experiment_results["I1"]["qpe_rows"]) + "\n\n")
                f.write("### (e) Shor's Order Finding & Factorization\n\n")
                f.write(to_markdown(experiment_results["I1"]["shor_rows"]) + "\n\n")
            else:
                f.write(f"*Status: {experiment_status.get('I1', 'Skipped')}*\n\n")

            # I2 Summary
            f.write("## Experiment I2: Grover vs Classical Search\n\n")
            f.write("Query complexity and physical simulation runtime comparison between Grover unstructured search and classical linear scan.\n\n")
            f.write("> **Note on Runtime:** The statevector simulation of Grover is much slower than the classical scan, so the advantage is in the query count only.\n\n")
            if "I2" in experiment_results:
                f.write(to_markdown(experiment_results["I2"]["search_rows"]) + "\n\n")
            else:
                f.write(f"*Status: {experiment_status.get('I2', 'Skipped')}*\n\n")

            # I3 Summary
            f.write("## Experiment I3: Quantum Annealing and QUBO Embedding Limits\n\n")
            f.write("Transverse-field quantum annealing simulation via Strang splitting on Track B Max-Cut QUBO instances, and analytic TRIAD clique embedding resource requirements.\n\n")
            f.write("> **Note on QUBO Density:** A cardinality-penalty term such as lambda * (sum x - k)^2 (Track B) makes the QUBO dense, requiring the complete graph K_n, while a 3-regular Max-Cut instance has only 3n/2 couplings.\n\n")
            if "I3" in experiment_results:
                f.write("### Annealing Success Probabilities\n\n")
                f.write(to_markdown(experiment_results["I3"]["anneal_rows"]) + "\n\n")
                f.write("### Analytic Chimera Embedding Table\n\n")
                f.write(to_markdown(experiment_results["I3"]["embed_rows"]) + "\n\n")
            else:
                f.write(f"*Status: {experiment_status.get('I3', 'Skipped')}*\n\n")

            # I4 Summary
            f.write("## Experiment I4: Hardware Noise Model (Simulated)\n\n")
            f.write("Depolarizing error channel trajectory simulation across varying error rates for Grover, Bernstein-Vazirani, and Deutsch-Jozsa.\n\n")
            if "I4" in experiment_results:
                f.write(to_markdown(experiment_results["I4"]["noise_rows"]) + "\n\n")
            else:
                f.write(f"*Status: {experiment_status.get('I4', 'Skipped')}*\n\n")

            # I5 Summary
            f.write("## Experiment I5: QML vs Classical Baselines\n\n")
            f.write("Variational Quantum Classifier with data re-uploading and parameter-shift gradients, exact and shot-based Quantum Kernel SVMs, and classical baselines on moons and circles.\n\n")
            if "I5" in experiment_results:
                f.write(to_markdown(experiment_results["I5"]["all_qml_rows"]) + "\n\n")
            else:
                f.write(f"*Status: {experiment_status.get('I5', 'Skipped')}*\n\n")

            # Figures
            f.write("## Generated Figures\n\n")
            figures = [
                "i1_qpe_scaling.png",
                "i1_shor_distribution.png",
                "i2_grover_curves.png",
                "i2_queries_scaling.png",
                "i2_runtime_comparison.png",
                "i3_anneal_success.png",
                "i3_embedding_scaling.png",
                "i4_noise_degradation.png",
                "i5_vqc_loss_curve.png",
                "i5_accuracy_comparison.png",
            ]
            for fig_name in figures:
                fig_file = out_dir / fig_name
                if fig_file.is_file():
                    f.write(f"- `{fig_name}`\n")
            f.write("\n")

            # Verification Section
            f.write("## Verification\n\n")
            all_selected_done = all(experiment_status.get(exp) == "Done" for exp in args.only)
            if all_selected_done and not verification_failures and not failed_tracebacks:
                f.write("No verification failures. All theoretical invariants, exact checks, and physical constraints passed.\n")
            else:
                if failed_tracebacks:
                    f.write("### Experiment Exceptions:\n\n")
                    for exp_name, tb in failed_tracebacks.items():
                        f.write(f"- **{exp_name} Exception:**\n```text\n{tb}\n```\n\n")
                if verification_failures:
                    f.write("### Invariant Failures:\n\n")
                    for vf in verification_failures:
                        f.write(f"- {vf}\n")
                    f.write("\n")

        print(f"\n[+] SUMMARY.md written to {summary_path}")

    # Exit code
    if failed_tracebacks or verification_failures:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

