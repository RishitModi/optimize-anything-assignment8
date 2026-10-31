"""Master Benchmark and Evaluation Script for Track J (Software Engineering Optimization).

Usage:
    python tracks/track_j_software_eng/run_all.py [--quick] [--out results/track_j]

Experiments:
    J0: Codebase Metrics & Baseline Profile
    J1: Test Suite Minimization (Weighted Set Cover: Greedy vs ILP vs Random)
    J2: Test Prioritization (Greedy Additional Coverage vs Cost-Ascending vs Random)
    J3: Fault-Injection Cost & Mutation Testing Kill Matrix
"""

from __future__ import annotations

import argparse
import cProfile
from pathlib import Path
import platform
import pstats
import sys
import time
import traceback
from typing import Any

# Ensure repository root, track F, and current folder are on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACK_F = _REPO_ROOT / "tracks" / "track_f_compiler"

for _p in [str(_REPO_ROOT), str(_TRACK_ROOT), str(_TRACK_F)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from sj_select import (
    auc_normalised,
    coverage_curve,
    order_cost_ascending,
    order_greedy_additional,
    order_random,
    select_greedy,
    select_ilp,
    select_random_until_cover,
)
from sj_suite import (
    CUT_PATHS,
    collect_coverage,
    count_lines,
    executable_lines,
    get_all_mutants,
    get_test_universe,
    kill_matrix,
    measure_test_cost,
    run_test,
)
from sj_util import median, to_markdown, write_csv


# ---------------------------------------------------------------------------
# Experiment J0: Codebase & Baseline Profile
# ---------------------------------------------------------------------------


def run_experiment_j0(
    out_dir: Path,
    tests: list[tuple[str, str]],
    verification_failures: list[str],
) -> dict[str, Any]:
    """Profile baseline execution, measure codebase lines, compute coverage and top functions."""
    print("=== Running Experiment J0: Codebase & Baseline Profile ===")

    # 1. Count lines of code under test and compiler codebase
    line_data = count_lines()
    print(f"Code Under Test lines: {line_data['cut_total']}, Full Compiler: {line_data['compiler_total']}")

    if line_data["cut_total"] < 1000 and line_data["compiler_total"] < 1000:
        verification_failures.append("Codebase has fewer than 1000 lines of non-comment, non-blank code.")

    # Save line counts CSV
    lc_rows = [
        {"file": fname, "lines": count}
        for fname, count in line_data["file_counts"].items()
    ]
    lc_rows.append({"file": "TOTAL_CUT", "lines": line_data["cut_total"]})
    lc_rows.append({"file": "TOTAL_ALL_COMPILER", "lines": line_data["compiler_total"]})
    write_csv(out_dir / "j0_line_counts.csv", lc_rows, ["file", "lines"])

    # 2. Run full suite untraced 3 times to get median runtime and verify pass status
    runtimes: list[float] = []
    passing_tests = 0
    cap_runs = 5
    run_step = 0
    for r in range(3):
        run_step += 1
        if run_step > cap_runs:
            break
        t0 = time.perf_counter()
        pass_count = 0
        cap_t = len(tests) + 5
        t_step = 0
        for name, src in tests:
            t_step += 1
            if t_step > cap_t:
                break
            if run_test(src):
                pass_count += 1
        runtimes.append(time.perf_counter() - t0)
        passing_tests = pass_count

    baseline_total_time = median(runtimes)
    if passing_tests != len(tests):
        verification_failures.append(f"Baseline suite had failures: {passing_tests}/{len(tests)} passed.")

    # 3. cProfile one full-suite run
    pr = cProfile.Profile()
    pr.enable()
    cap_t = len(tests) + 5
    t_step = 0
    for _, src in tests:
        t_step += 1
        if t_step > cap_t:
            break
        run_test(src)
    pr.disable()

    ps = pstats.Stats(pr)
    sorted_stats = sorted(ps.stats.items(), key=lambda kv: kv[1][3], reverse=True)[:15]
    top15: list[dict[str, Any]] = []
    for (fname, lineno, funcname), (cc, nc, tt, ct, _) in sorted_stats:
        top15.append({
            "function": funcname,
            "file_line": f"{Path(fname).name}:{lineno}",
            "ncalls": nc,
            "tottime": tt,
            "cumtime": ct,
        })
    write_csv(out_dir / "profile_top15.csv", top15, ["function", "file_line", "ncalls", "tottime", "cumtime"])

    # 4. Measure per-test cost (median of 5 untraced runs)
    print("Measuring individual test costs...")
    cost_map: dict[str, float] = {}
    cap_t = len(tests) + 5
    t_step = 0
    for name, src in tests:
        t_step += 1
        if t_step > cap_t:
            break
        cost_map[name] = measure_test_cost(src, runs=5)

    all_costs = list(cost_map.values())
    min_cost = float(min(all_costs)) if all_costs else 0.0
    med_cost = median(all_costs)
    max_cost = float(max(all_costs)) if all_costs else 0.0
    tot_cost = float(sum(all_costs))

    sorted_costs = sorted(all_costs, reverse=True)
    top10_sum = float(sum(sorted_costs[:10]))
    top10_share = (top10_sum / tot_cost * 100.0) if tot_cost > 0 else 0.0

    cost_stats = {
        "min_cost": min_cost,
        "median_cost": med_cost,
        "max_cost": max_cost,
        "total_cost": tot_cost,
        "top10_share_pct": top10_share,
    }

    # Plot test cost distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(all_costs, bins=20, color="steelblue", edgecolor="black")
    ax1.set_xscale("log")
    ax1.set_title("Test Cost Distribution")
    ax1.set_xlabel("Runtime (s, log scale)")
    ax1.set_ylabel("Count")
    ax1.grid(True, linestyle="--", alpha=0.5)

    sorted_asc = sorted(all_costs)
    cum_share = np.cumsum(sorted_asc) / tot_cost * 100.0 if tot_cost > 0 else np.zeros(len(sorted_asc))
    x_pct = np.linspace(0, 100, len(cum_share))
    ax2.plot(x_pct, cum_share, color="darkorange", lw=2, label="Cumulative Cost")
    ax2.plot([0, 100], [0, 100], "k--", alpha=0.5, label="Uniform Share")
    ax2.set_title("Cumulative Test Cost Share")
    ax2.set_xlabel("Percentage of Tests (%)")
    ax2.set_ylabel("Cumulative Cost Share (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "j0_test_cost_distribution.png", dpi=150)
    plt.close()

    # 5. Full suite coverage
    print("Collecting full-suite line and arc coverage...")
    line_cover: dict[str, set[tuple[int, int]]] = {}
    arc_cover: dict[str, set[tuple[int, int, int]]] = {}
    full_lines: set[tuple[int, int]] = set()
    full_arcs: set[tuple[int, int, int]] = set()

    cap_t = len(tests) + 5
    t_step = 0
    for name, src in tests:
        t_step += 1
        if t_step > cap_t:
            break
        l_cov, a_cov = collect_coverage(lambda: run_test(src))
        line_cover[name] = l_cov
        arc_cover[name] = a_cov
        full_lines.update(l_cov)
        full_arcs.update(a_cov)

    # Calculate per-file and total coverage
    cov_rows: list[dict[str, Any]] = []
    total_exec = 0
    total_cov_lines = 0
    total_cov_arcs = 0

    cap_f = len(CUT_PATHS) + 5
    f_step = 0
    for idx, p in enumerate(CUT_PATHS):
        f_step += 1
        if f_step > cap_f:
            break
        ex_lines = executable_lines(p)
        cov_l = {l for fi, l in full_lines if fi == idx}
        cov_a = {(p_l, l) for fi, p_l, l in full_arcs if fi == idx}

        total_exec += len(ex_lines)
        total_cov_lines += len(cov_l)
        total_cov_arcs += len(cov_a)

        pct = (len(cov_l) / len(ex_lines) * 100.0) if ex_lines else 0.0
        cov_rows.append({
            "file": p.name,
            "executable_lines": len(ex_lines),
            "covered_lines": len(cov_l),
            "line_coverage_pct": pct,
            "covered_arcs": len(cov_a),
        })

    tot_pct = (total_cov_lines / total_exec * 100.0) if total_exec > 0 else 0.0
    cov_rows.append({
        "file": "TOTAL",
        "executable_lines": total_exec,
        "covered_lines": total_cov_lines,
        "line_coverage_pct": tot_pct,
        "covered_arcs": total_cov_arcs,
    })
    write_csv(
        out_dir / "j0_coverage_summary.csv",
        cov_rows,
        ["file", "executable_lines", "covered_lines", "line_coverage_pct", "covered_arcs"],
    )

    return {
        "line_data": line_data,
        "lc_rows": lc_rows,
        "top15": top15,
        "cost_map": cost_map,
        "cost_stats": cost_stats,
        "baseline_total_time": baseline_total_time,
        "passing_tests": passing_tests,
        "line_cover": line_cover,
        "arc_cover": arc_cover,
        "full_lines": full_lines,
        "full_arcs": full_arcs,
        "cov_rows": cov_rows,
    }


# ---------------------------------------------------------------------------
# Experiment J1: Test Suite Minimization
# ---------------------------------------------------------------------------


def run_experiment_j1(
    out_dir: Path,
    tests: list[tuple[str, str]],
    j0_data: dict[str, Any],
    n_random_seeds: int,
    verification_failures: list[str],
) -> dict[str, Any]:
    """Perform weighted set cover minimization using greedy and ILP formulations for lines and arcs."""
    print("=== Running Experiment J1: Test Suite Minimization ===")

    cost_map: dict[str, float] = j0_data["cost_map"]
    full_lines = j0_data["full_lines"]
    full_arcs = j0_data["full_arcs"]
    line_cover = j0_data["line_cover"]
    arc_cover = j0_data["arc_cover"]
    test_dict = dict(tests)
    all_mutants = get_all_mutants()

    print("Computing mutant kill matrix on full suite...")
    k_matrix = kill_matrix(tests, all_mutants)
    detectable_mutants = [
        m for m in all_mutants
        if any(k_matrix[m][t_name] for t_name, _ in tests)
    ]
    n_detectable = len(detectable_mutants)

    def eval_selection(sel_tests: list[str]) -> tuple[float, int, float]:
        """Compute actual measured runtime, fail count, and mutation score for a test subset."""
        sel_srcs = [test_dict[t] for t in sel_tests if t in test_dict]
        times: list[float] = []
        cap_r = 7
        r_step = 0
        for _ in range(5):
            r_step += 1
            if r_step > cap_r:
                break
            t0 = time.perf_counter()
            for s in sel_srcs:
                run_test(s)
            times.append(time.perf_counter() - t0)
        actual_time = median(times)

        failing = 0
        cap_s = len(sel_srcs) + 5
        s_step = 0
        for s in sel_srcs:
            s_step += 1
            if s_step > cap_s:
                break
            if not run_test(s):
                failing += 1

        # Mutation score
        killed = 0
        cap_m = len(detectable_mutants) + 5
        m_step = 0
        for m in detectable_mutants:
            m_step += 1
            if m_step > cap_m:
                break
            if any(k_matrix[m].get(t, False) for t in sel_tests):
                killed += 1

        mut_score = (killed / n_detectable * 100.0) if n_detectable > 0 else 100.0
        return actual_time, failing, mut_score

    all_test_names = [t[0] for t in tests]
    full_actual_time, full_failing, full_mut_score = eval_selection(all_test_names)
    full_cost = sum(cost_map.values())

    j1_records: list[dict[str, Any]] = []
    selections: dict[str, list[str]] = {"full": all_test_names}

    # Record full suite baseline
    j1_records.append({
        "coverage_kind": "all",
        "method": "full_suite",
        "num_tests": len(tests),
        "summed_cost": full_cost,
        "actual_runtime": full_actual_time,
        "runtime_reduction_pct": 0.0,
        "coverage_preserved_pct": 100.0,
        "failing_tests": full_failing,
        "mutation_score_pct": full_mut_score,
        "optimal": True,
        "solve_time": 0.0,
    })

    for kind, cov_dict, universe in [
        ("lines", line_cover, full_lines),
        ("arcs", arc_cover, full_arcs),
    ]:
        print(f"Minimizing for coverage kind: {kind} ({len(universe)} elements)...")

        # 1. Random until cover baseline
        rand_res = select_random_until_cover(cov_dict, cost_map, universe, seeds=n_random_seeds)
        first_rand_sel = rand_res["selections"][0] if rand_res["selections"] else []
        rand_act_time, rand_failing, rand_mut_score = eval_selection(first_rand_sel)
        rand_reduc = (
            (full_actual_time - rand_act_time) / full_actual_time * 100.0
            if full_actual_time > 0 else 0.0
        )
        j1_records.append({
            "coverage_kind": kind,
            "method": f"random_until_cover(mean_N={rand_res['mean_size']:.1f})",
            "num_tests": int(round(rand_res["mean_size"])),
            "summed_cost": rand_res["mean_cost"],
            "actual_runtime": rand_act_time,
            "runtime_reduction_pct": rand_reduc,
            "coverage_preserved_pct": 100.0,
            "failing_tests": rand_failing,
            "mutation_score_pct": rand_mut_score,
            "optimal": False,
            "solve_time": 0.0,
        })

        # 2. Greedy selection
        t0_g = time.perf_counter()
        greedy_sel = select_greedy(cov_dict, cost_map, universe)
        t_greedy = time.perf_counter() - t0_g

        selections[f"greedy_{kind}"] = greedy_sel
        g_cost = sum(cost_map.get(t, 0.0) for t in greedy_sel)
        g_act_time, g_failing, g_mut_score = eval_selection(greedy_sel)
        g_cov_elements = set().union(*(cov_dict.get(t, set()) for t in greedy_sel)) & universe
        g_cov_pct = len(g_cov_elements) / len(universe) * 100.0 if universe else 100.0
        g_reduc = (
            (full_actual_time - g_act_time) / full_actual_time * 100.0
            if full_actual_time > 0 else 0.0
        )

        j1_records.append({
            "coverage_kind": kind,
            "method": "greedy",
            "num_tests": len(greedy_sel),
            "summed_cost": g_cost,
            "actual_runtime": g_act_time,
            "runtime_reduction_pct": g_reduc,
            "coverage_preserved_pct": g_cov_pct,
            "failing_tests": g_failing,
            "mutation_score_pct": g_mut_score,
            "optimal": False,
            "solve_time": t_greedy,
        })

        # 3. ILP exact selection
        ilp_sel, ilp_opt, ilp_cost, ilp_solve_time = select_ilp(cov_dict, cost_map, universe, time_limit=60.0)
        selections[f"ilp_{kind}"] = ilp_sel
        ilp_act_time, ilp_failing, ilp_mut_score = eval_selection(ilp_sel)
        ilp_cov_elements = set().union(*(cov_dict.get(t, set()) for t in ilp_sel)) & universe
        ilp_cov_pct = len(ilp_cov_elements) / len(universe) * 100.0 if universe else 100.0
        ilp_reduc = (
            (full_actual_time - ilp_act_time) / full_actual_time * 100.0
            if full_actual_time > 0 else 0.0
        )

        j1_records.append({
            "coverage_kind": kind,
            "method": "ilp",
            "num_tests": len(ilp_sel),
            "summed_cost": ilp_cost,
            "actual_runtime": ilp_act_time,
            "runtime_reduction_pct": ilp_reduc,
            "coverage_preserved_pct": ilp_cov_pct,
            "failing_tests": ilp_failing,
            "mutation_score_pct": ilp_mut_score,
            "optimal": ilp_opt,
            "solve_time": ilp_solve_time,
        })

        # Verification checks
        if abs(g_cov_pct - 100.0) > 1e-4:
            verification_failures.append(f"Greedy {kind} failed to cover 100% of elements ({g_cov_pct:.2f}%).")
        if abs(ilp_cov_pct - 100.0) > 1e-4:
            verification_failures.append(f"ILP {kind} failed to cover 100% of elements ({ilp_cov_pct:.2f}%).")

        # Invariant checks: ILP cost <= Greedy cost <= Full cost
        if ilp_cost > g_cost + 1e-6:
            verification_failures.append(f"ILP cost ({ilp_cost:.5f}) > Greedy cost ({g_cost:.5f}) for {kind}.")
        if g_cost > full_cost + 1e-6:
            verification_failures.append(f"Greedy cost ({g_cost:.5f}) > Full cost ({full_cost:.5f}) for {kind}.")
        if ilp_cost > rand_res["mean_cost"] + 1e-6:
            verification_failures.append(f"ILP cost ({ilp_cost:.5f}) > Random cost ({rand_res['mean_cost']:.5f}) for {kind}.")

    cols = [
        "coverage_kind",
        "method",
        "num_tests",
        "summed_cost",
        "actual_runtime",
        "runtime_reduction_pct",
        "coverage_preserved_pct",
        "failing_tests",
        "mutation_score_pct",
        "optimal",
        "solve_time",
    ]
    write_csv(out_dir / "j1_minimization_summary.csv", j1_records, cols)

    # Grouped bar chart comparing tests kept and runtime reduction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_rows = [r for r in j1_records if r["method"] in ("full_suite", "greedy", "ilp")]

    labels = [f"{r['method']}\n({r['coverage_kind']})" for r in plot_rows]
    tests_kept = [r["num_tests"] for r in plot_rows]
    reduc_pcts = [r["runtime_reduction_pct"] for r in plot_rows]

    x = np.arange(len(labels))
    ax1.bar(x, tests_kept, color="cornflowerblue", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Tests Kept (count)")
    ax1.set_title("Test Suite Size After Minimization")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.bar(x, reduc_pcts, color="forestgreen", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Runtime Reduction (%)")
    ax2.set_title("Measured Execution Time Reduction")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "j1_minimization_comparison.png", dpi=150)
    plt.close()

    return {
        "j1_records": j1_records,
        "selections": selections,
        "k_matrix": k_matrix,
        "detectable_mutants": detectable_mutants,
        "all_mutants": all_mutants,
    }


# ---------------------------------------------------------------------------
# Experiment J2: Test Prioritization
# ---------------------------------------------------------------------------


def run_experiment_j2(
    out_dir: Path,
    tests: list[tuple[str, str]],
    j0_data: dict[str, Any],
    n_random_seeds: int,
) -> dict[str, Any]:
    """Evaluate test prioritization heuristics and plot coverage-versus-time curves."""
    print("=== Running Experiment J2: Test Prioritization ===")

    cost_map: dict[str, float] = j0_data["cost_map"]
    full_lines = j0_data["full_lines"]
    line_cover = j0_data["line_cover"]
    test_names = [t[0] for t in tests]

    # 1. Greedy additional coverage order
    ord_greedy = order_greedy_additional(line_cover, cost_map, full_lines)
    curve_greedy = coverage_curve(ord_greedy, line_cover, cost_map, full_lines)
    auc_greedy = auc_normalised(curve_greedy)

    # 2. Cost ascending order
    ord_cost = order_cost_ascending(cost_map)
    curve_cost = coverage_curve(ord_cost, line_cover, cost_map, full_lines)
    auc_cost = auc_normalised(curve_cost)

    # 3. Random order (average over multiple seeds)
    rand_aucs: list[float] = []
    rand_curves: list[tuple[list[float], list[float]]] = []
    cap_s = n_random_seeds + 5
    s_step = 0
    for s in range(n_random_seeds):
        s_step += 1
        if s_step > cap_s:
            break
        ord_r = order_random(test_names, seed=s)
        c_r = coverage_curve(ord_r, line_cover, cost_map, full_lines)
        rand_curves.append(c_r)
        rand_aucs.append(auc_normalised(c_r))

    mean_rand_auc = float(np.mean(rand_aucs)) if rand_aucs else 0.0

    j2_records = [
        {"strategy": "greedy_additional", "normalized_auc": auc_greedy},
        {"strategy": "cost_ascending", "normalized_auc": auc_cost},
        {
            "strategy": f"random_order(mean_over_{n_random_seeds}_seeds)",
            "normalized_auc": mean_rand_auc,
        },
    ]
    write_csv(out_dir / "j2_prioritization_summary.csv", j2_records, ["strategy", "normalized_auc"])

    # Plot prioritization curves
    plt.figure(figsize=(7, 4.5))
    plt.plot(curve_greedy[0], [f * 100 for f in curve_greedy[1]], "b-", lw=2, label=f"Greedy Additional (AUC={auc_greedy:.3f})")
    plt.plot(curve_cost[0], [f * 100 for f in curve_cost[1]], "r--", lw=1.5, label=f"Cost Ascending (AUC={auc_cost:.3f})")

    # Plot average random curve
    if rand_curves:
        max_t = curve_greedy[0][-1]
        common_t = np.linspace(0, max_t, 100)
        interp_f: list[np.ndarray] = []
        for t_arr, f_arr in rand_curves:
            interp_f.append(np.interp(common_t, t_arr, f_arr))
        mean_f = np.mean(interp_f, axis=0) * 100.0
        plt.plot(common_t, mean_f, "g-.", lw=1.5, label=f"Random Mean (AUC={mean_rand_auc:.3f})")

    plt.title("Test Prioritization: Cumulative Line Coverage vs Wall-Clock Time")
    plt.xlabel("Cumulative Execution Time (seconds)")
    plt.ylabel("Line Coverage Preserved (%)")
    plt.ylim(-2, 105)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "j2_prioritization_curves.png", dpi=150)
    plt.close()

    return {
        "j2_records": j2_records,
        "auc_greedy": auc_greedy,
        "auc_cost": auc_cost,
        "auc_random": mean_rand_auc,
    }


# ---------------------------------------------------------------------------
# Experiment J3: Fault Injection & Mutation Testing
# ---------------------------------------------------------------------------


def run_experiment_j3(
    out_dir: Path,
    j1_data: dict[str, Any],
) -> dict[str, Any]:
    """Examine fault-detection capabilities of minimized suites against injected pass mutants."""
    print("=== Running Experiment J3: Fault-Injection Cost & Mutation Testing ===")

    all_mutants = j1_data["all_mutants"]
    detectable_mutants = j1_data["detectable_mutants"]
    k_matrix = j1_data["k_matrix"]
    selections = j1_data["selections"]

    methods = ["full", "greedy_lines", "ilp_lines", "greedy_arcs", "ilp_arcs"]
    j3_records: list[dict[str, Any]] = []

    cap_m = len(methods) + 5
    m_step = 0
    for meth in methods:
        m_step += 1
        if m_step > cap_m:
            break
        sel_tests = selections.get(meth, [])
        killed_list = [
            m for m in detectable_mutants
            if any(k_matrix[m].get(t, False) for t in sel_tests)
        ]
        missed_list = [m for m in detectable_mutants if m not in killed_list]
        score = (len(killed_list) / len(detectable_mutants) * 100.0) if detectable_mutants else 100.0

        j3_records.append({
            "method": meth,
            "tests_kept": len(sel_tests),
            "detectable_killed": len(killed_list),
            "detectable_missed": len(missed_list),
            "mutation_score_pct": score,
            "missed_mutants": ";".join([f"{p}:{op}" for p, op in missed_list]) if missed_list else "None",
        })

    cols = [
        "method",
        "tests_kept",
        "detectable_killed",
        "detectable_missed",
        "mutation_score_pct",
        "missed_mutants",
    ]
    write_csv(out_dir / "j3_mutation_summary.csv", j3_records, cols)

    return {
        "j3_records": j3_records,
        "total_mutants": len(all_mutants),
        "detectable_count": len(detectable_mutants),
    }


# ---------------------------------------------------------------------------
# SUMMARY.md Generation
# ---------------------------------------------------------------------------


def write_summary_report(
    out_dir: Path,
    quick: bool,
    j0_res: dict[str, Any] | None,
    j1_res: dict[str, Any] | None,
    j2_res: dict[str, Any] | None,
    j3_res: dict[str, Any] | None,
    verification_failures: list[str],
    failed_experiments: dict[str, str],
) -> None:
    """Generate comprehensive markdown summary without interpretive speculation."""
    lines: list[str] = [
        "# Track J: Software Engineering Optimization – Experimental Summary",
        "",
        "## Environment & Platform Configuration",
        f"- **Python Version**: {platform.python_version()}",
        f"- **NumPy Version**: {np.__version__}",
        f"- **SciPy Version**: {scipy.__version__}",
        f"- **Platform / OS**: {platform.platform()}",
        f"- **Quick Execution Flag**: {quick}",
        "- **Random Seed Management**: Explicit seeds across all stochastic selections",
        "",
        "## Codebase Under Test Metrics",
    ]

    if j0_res is not None:
        lines.append(f"- **Code Under Test (CUT) Non-Comment Lines**: {j0_res['line_data']['cut_total']}")
        lines.append(f"- **Total Non-Test Compiler Lines**: {j0_res['line_data']['compiler_total']}")
        lines.append("")
        lines.append("### Codebase Line Counts by Module")
        lines.append(to_markdown(j0_res["lc_rows"], ["file", "lines"]))
        lines.append("")

        lines.append("## J0: Baseline Profiling & Test Execution")
        lines.append(f"- **Total Tests in Universe**: {len(j0_res['cost_map'])}")
        lines.append(f"- **Baseline Median Full-Suite Runtime**: {j0_res['baseline_total_time']:.4f} s")
        lines.append(f"- **Passing Tests**: {j0_res['passing_tests']}/{len(j0_res['cost_map'])}")
        lines.append(f"- **Per-Test Cost (min / median / max)**: {j0_res['cost_stats']['min_cost']:.5f} s / {j0_res['cost_stats']['median_cost']:.5f} s / {j0_res['cost_stats']['max_cost']:.5f} s")
        lines.append(f"- **Top 10 Slowest Tests Runtime Share**: {j0_res['cost_stats']['top10_share_pct']:.2f}%")
        lines.append("")
        lines.append("### Top 15 Functions by Cumulative Execution Time (cProfile)")
        lines.append(to_markdown(j0_res["top15"], ["function", "file_line", "ncalls", "tottime", "cumtime"]))
        lines.append("")
        lines.append("### Code Coverage Breakdown")
        lines.append(to_markdown(j0_res["cov_rows"], ["file", "executable_lines", "covered_lines", "line_coverage_pct", "covered_arcs"]))
        lines.append("")
        lines.append("- Output Figure: `j0_test_cost_distribution.png`")
        lines.append("")

    if j1_res is not None:
        lines.append("## J1: Test Suite Minimization (Weighted Set Cover)")
        lines.append("Evaluates full suite against random-until-cover, greedy heuristic, and exact 0/1 ILP minimization.")
        lines.append("")
        cols = [
            "coverage_kind",
            "method",
            "num_tests",
            "summed_cost",
            "actual_runtime",
            "runtime_reduction_pct",
            "coverage_preserved_pct",
            "failing_tests",
            "mutation_score_pct",
            "optimal",
            "solve_time",
        ]
        lines.append(to_markdown(j1_res["j1_records"], cols))
        lines.append("")
        lines.append("- Output Figure: `j1_minimization_comparison.png`")
        lines.append("")

    if j2_res is not None:
        lines.append("## J2: Test Prioritization")
        lines.append("Compares cumulative line coverage accumulation over time for greedy additional coverage, cost-ascending, and random orders.")
        lines.append("")
        lines.append(to_markdown(j2_res["j2_records"], ["strategy", "normalized_auc"]))
        lines.append("")
        lines.append("- Output Figure: `j2_prioritization_curves.png`")
        lines.append("")

    if j3_res is not None:
        lines.append("## J3: Fault-Injection & Mutation Testing")
        lines.append("Evaluates whether minimized test suites preserve mutant killing capability.")
        lines.append("")
        lines.append(f"- **Total Injected Mutants**: {j3_res['total_mutants']}")
        lines.append(f"- **Detectable Mutants by Full Suite**: {j3_res['detectable_count']}")
        lines.append("")
        lines.append("> **Note on Fault Injection Proxy**: Fault injection in this benchmark modifies the instruction list output of individual optimization passes. This operates as an empirical proxy for fault-detection ability; structural coverage and fault-detection capability are not theoretically identical.")
        lines.append("")
        cols = [
            "method",
            "tests_kept",
            "detectable_killed",
            "detectable_missed",
            "mutation_score_pct",
            "missed_mutants",
        ]
        lines.append(to_markdown(j3_res["j3_records"], cols))
        lines.append("")

    lines.append("## Verification Section")
    has_errors = bool(verification_failures) or bool(failed_experiments)
    if not has_errors:
        lines.append("- **No verification failures**")
    else:
        if verification_failures:
            lines.append("### Invariant Violations")
            for vf in verification_failures:
                lines.append(f"- [FAIL] {vf}")
        if failed_experiments:
            lines.append("### Failed Experiment Exceptions")
            for exp_name, tb in failed_experiments.items():
                lines.append(f"#### Experiment `{exp_name}`")
                lines.append("```python")
                lines.append(tb.strip())
                lines.append("```")

    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main Runner Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Orchestrate all Track J benchmark experiments."""
    parser = argparse.ArgumentParser(description="Track J: Software Engineering Optimization")
    parser.add_argument("--quick", action="store_true", help="Execute faster run with fewer fuzz tests")
    parser.add_argument("--out", type=str, default="results/track_j", help="Output directory path")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    quick = bool(args.quick)
    n_fuzz = 30 if quick else 150
    n_random_seeds = 20 if quick else 50

    verification_failures: list[str] = []
    failed_experiments: dict[str, str] = {}

    j0_res: dict[str, Any] | None = None
    j1_res: dict[str, Any] | None = None
    j2_res: dict[str, Any] | None = None
    j3_res: dict[str, Any] | None = None

    try:
        # Build test universe
        print(f"Building test universe (n_fuzz={n_fuzz})...")
        tests = get_test_universe(n_fuzz=n_fuzz)
        print(f"Total test universe size: {len(tests)}")

        # J0
        try:
            j0_res = run_experiment_j0(out_dir, tests, verification_failures)
        except Exception:
            failed_experiments["J0"] = traceback.format_exc()
            print(f"[ERROR] J0 failed:\n{failed_experiments['J0']}")

        # J1
        if j0_res is not None:
            try:
                j1_res = run_experiment_j1(
                    out_dir,
                    tests,
                    j0_res,
                    n_random_seeds=n_random_seeds,
                    verification_failures=verification_failures,
                )
            except Exception:
                failed_experiments["J1"] = traceback.format_exc()
                print(f"[ERROR] J1 failed:\n{failed_experiments['J1']}")

        # J2
        if j0_res is not None:
            try:
                j2_res = run_experiment_j2(
                    out_dir,
                    tests,
                    j0_res,
                    n_random_seeds=n_random_seeds,
                )
            except Exception:
                failed_experiments["J2"] = traceback.format_exc()
                print(f"[ERROR] J2 failed:\n{failed_experiments['J2']}")

        # J3
        if j1_res is not None:
            try:
                j3_res = run_experiment_j3(out_dir, j1_res)
            except Exception:
                failed_experiments["J3"] = traceback.format_exc()
                print(f"[ERROR] J3 failed:\n{failed_experiments['J3']}")

    except KeyboardInterrupt:
        failed_experiments["INTERRUPTED"] = "KeyboardInterrupt received."
        print("\nExecution interrupted by user.")
        raise
    finally:
        write_summary_report(
            out_dir=out_dir,
            quick=quick,
            j0_res=j0_res,
            j1_res=j1_res,
            j2_res=j2_res,
            j3_res=j3_res,
            verification_failures=verification_failures,
            failed_experiments=failed_experiments,
        )

    if verification_failures or failed_experiments:
        print("[FAIL] Track J completed with verification failures or errors.")
        sys.exit(1)
    else:
        print("[SUCCESS] Track J completed with all verification checks passing.")


if __name__ == "__main__":
    main()

