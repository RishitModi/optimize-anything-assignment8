"""Master Benchmark and Evaluation Script for Track F (Compiler Optimization).

Usage:
    python tracks/track_f_compiler/run_all.py [--quick] [--out results/track_f]

Experiments:
    F1: Optimization Pass Effects (None, Individual Passes, Pipeline "All")
    F2: Register Allocation as 0/1 Optimization (Linear Scan, Greedy-Weight, ILP, Brute Force)
    F3: Differential Testing & Fuzzing (Reference vs Optimized vs Register VM)
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Ensure repository root and track root are placed on sys.path
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
import scipy

from cc_fuzz import generate as fuzz_generate
from cc_ir import StepLimitExceeded, loop_depth_per_instruction, run_ir
from cc_lang import compile_to_ir, format_ir
from cc_passes import (
    PASS_REGISTRY,
    algebraic_strength_reduction,
    const_fold_prop,
    cse,
    dce,
    optimize,
    optimize_all,
)
from cc_programs import PROGRAMS
from cc_regalloc import (
    allocate_bruteforce,
    allocate_greedy_weight,
    allocate_ilp,
    allocate_linear_scan,
    assign_physical_registers,
    build_live_ranges,
)
from cc_vm import generate_code, run_machine


def render_markdown_table(df: pd.DataFrame) -> str:
    """Render a pandas DataFrame as a clean Markdown table without external dependencies."""
    headers = [str(c) for c in df.columns]
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for _, row in df.iterrows():
        row_cells: list[str] = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                row_cells.append(f"{val:.4f}")
            else:
                row_cells.append(str(val))
        lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment F1: Pass Effects
# ---------------------------------------------------------------------------


def run_experiment_f1(
    out_dir: Path,
    quick: bool,
    verification_failures: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Evaluate compile time, code size, dynamic instruction counts, and output correctness across passes."""
    print("--- Running Experiment F1: Pass Effects ---")
    configs = [
        "none",
        "const_fold_prop",
        "algebraic_strength_reduction",
        "cse",
        "dce",
        "all",
    ]
    records: list[dict[str, Any]] = []
    repeat_runs = 3 if quick else 5
    pass_impact_counts = {p: 0 for p in PASS_REGISTRY}

    for prog_name, src in PROGRAMS.items():
        base_ir = compile_to_ir(src)
        unopt_out, unopt_dyn, _ = run_ir(base_ir)
        unopt_static = len(base_ir)

        for cfg in configs:
            compile_t0 = time.perf_counter()
            if cfg == "none":
                opt_ir = list(base_ir)
                compile_time = 0.0
            elif cfg == "all":
                opt_ir, _ = optimize_all(base_ir)
                compile_time = time.perf_counter() - compile_t0
            else:
                pass_fn = PASS_REGISTRY[cfg]
                opt_ir = pass_fn(base_ir)
                compile_time = time.perf_counter() - compile_t0

            static_count = len(opt_ir)

            # Profile dynamic runtime over repeat runs
            dyn_times: list[float] = []
            final_dyn = 0
            final_out: list[int] = []
            for _ in range(repeat_runs):
                out, dyn, wall = run_ir(opt_ir)
                dyn_times.append(wall)
                final_dyn = dyn
                final_out = out

            median_time = statistics.median(dyn_times)

            # Verification: output must strictly match unoptimized program
            if final_out != unopt_out:
                msg = f"F1 Mismatch: {prog_name} with {cfg} gave {final_out}, expected {unopt_out}"
                print(f"[FAIL] {msg}")
                verification_failures.append(msg)

            reduction_pct = (
                (unopt_dyn - final_dyn) / unopt_dyn * 100.0 if unopt_dyn > 0 else 0.0
            )

            # Track if pass alone had any effect on instructions
            if cfg in pass_impact_counts and (
                final_dyn != unopt_dyn or static_count != unopt_static
            ):
                pass_impact_counts[cfg] += 1

            records.append(
                {
                    "program": prog_name,
                    "configuration": cfg,
                    "static_instructions": static_count,
                    "dynamic_instructions": final_dyn,
                    "dynamic_reduction_pct": reduction_pct,
                    "run_time_median_s": median_time,
                    "compile_time_s": compile_time,
                    "output_correct": (final_out == unopt_out),
                }
            )

    df_raw = pd.DataFrame(records)
    df_raw.to_csv(out_dir / "f1_pass_effects_raw.csv", index=False)

    # Summary table: mean reduction of dynamic instructions per config
    summary_records = []
    for cfg in configs:
        sub = df_raw[df_raw["configuration"] == cfg]
        summary_records.append(
            {
                "configuration": cfg,
                "mean_dynamic_reduction_pct": sub["dynamic_reduction_pct"].mean(),
                "mean_static_instructions": sub["static_instructions"].mean(),
                "mean_compile_time_s": sub["compile_time_s"].mean(),
            }
        )
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(out_dir / "f1_pass_effects_summary.csv", index=False)

    # Plot 1: Grouped bar chart of dynamic reduction (%)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    progs = list(PROGRAMS.keys())
    x = np.arange(len(progs))
    plot_cfgs = [c for c in configs if c != "none"]
    width = 0.8 / len(plot_cfgs)

    for i, cfg in enumerate(plot_cfgs):
        sub = df_raw[df_raw["configuration"] == cfg]
        vals = [
            sub[sub["program"] == p]["dynamic_reduction_pct"].values[0] for p in progs
        ]
        ax.bar(x + i * width, vals, width, label=cfg)

    ax.set_ylabel("Dynamic Instruction Reduction (%)")
    ax.set_title("Optimization Pass Reduction by Benchmark Program")
    ax.set_xticks(x + width * (len(plot_cfgs) - 1) / 2)
    ax.set_xticklabels(progs, rotation=35, ha="right")
    ax.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "f1_dynamic_reduction.png")
    plt.close()

    # Plot 2: Static vs Dynamic Instruction counts (None vs All)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    sub_none = df_raw[df_raw["configuration"] == "none"]
    sub_all = df_raw[df_raw["configuration"] == "all"]

    ax.scatter(
        sub_none["static_instructions"],
        sub_none["dynamic_instructions"],
        color="crimson",
        label="None (Unoptimized)",
        s=60,
        alpha=0.85,
    )
    ax.scatter(
        sub_all["static_instructions"],
        sub_all["dynamic_instructions"],
        color="royalblue",
        label="All (Optimized)",
        s=60,
        alpha=0.85,
    )

    for p in progs:
        n_row = sub_none[sub_none["program"] == p].iloc[0]
        a_row = sub_all[sub_all["program"] == p].iloc[0]
        ax.annotate(
            "",
            xy=(a_row["static_instructions"], a_row["dynamic_instructions"]),
            xytext=(n_row["static_instructions"], n_row["dynamic_instructions"]),
            arrowprops=dict(
                arrowstyle="->", color="gray", lw=1.2, linestyle="dashed"
            ),
        )

    ax.set_xlabel("Static Instruction Count")
    ax.set_ylabel("Dynamic Instruction Count")
    ax.set_yscale("log")
    ax.set_title("Static vs Dynamic Instruction Counts: Unoptimized vs Optimized")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "f1_static_vs_dynamic.png")
    plt.close()

    return df_raw, df_summary, pass_impact_counts


# ---------------------------------------------------------------------------
# Experiment F2: Register Allocation as 0/1 Optimization
# ---------------------------------------------------------------------------


def run_experiment_f2(
    out_dir: Path,
    quick: bool,
    verification_failures: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Evaluate register allocation heuristics and ILP across varying register capacities k."""
    print("--- Running Experiment F2: Register Allocation ---")
    k_values = [2, 3, 4, 6, 8]
    methods = ["linear_scan", "greedy_weight", "ilp", "bruteforce"]
    records: list[dict[str, Any]] = []
    strictly_beats_count = {"linear_scan": 0, "greedy_weight": 0}

    for prog_name, src in PROGRAMS.items():
        base_ir = compile_to_ir(src)
        opt_ir, _ = optimize_all(base_ir)
        ref_out, _, _ = run_ir(opt_ir)
        ranges = build_live_ranges(opt_ir)
        num_ranges = len(ranges)

        for k in k_values:
            alloc_results: dict[str, Any] = {}

            for m in methods:
                if m == "bruteforce" and num_ranges > 18:
                    continue

                if m == "linear_scan":
                    kept, spilled, cost, solve_t = allocate_linear_scan(ranges, k)
                elif m == "greedy_weight":
                    kept, spilled, cost, solve_t = allocate_greedy_weight(ranges, k)
                elif m == "ilp":
                    kept, spilled, cost, solve_t = allocate_ilp(ranges, k)
                elif m == "bruteforce":
                    kept, spilled, cost, solve_t = allocate_bruteforce(ranges, k)
                else:
                    raise ValueError(m)

                # Verification: Capacity constraint at every point in [start, end]
                max_point = max((r.end for r in ranges), default=0)
                kept_ranges = [r for r in ranges if r.var in kept]
                for t in range(max_point + 1):
                    overlap = sum(1 for r in kept_ranges if r.start <= t <= r.end)
                    if overlap > k:
                        msg = f"F2 Capacity Invariant Violation: {prog_name}, k={k}, {m} has {overlap} > {k} active ranges at t={t}"
                        print(f"[FAIL] {msg}")
                        verification_failures.append(msg)

                # Physical register assignment and machine execution
                reg_map, slot_map = assign_physical_registers(ranges, kept, spilled, k)
                code = generate_code(opt_ir, (reg_map, slot_map), k)
                m_out, m_dyn, m_ld, m_st, m_alu, m_wall = run_machine(code)

                # Verification: Machine output matches reference IR interpreter output
                if m_out != ref_out:
                    msg = f"F2 Machine Output Mismatch: {prog_name}, k={k}, {m} output {m_out} != ref {ref_out}"
                    print(f"[FAIL] {msg}")
                    verification_failures.append(msg)

                alloc_results[m] = {
                    "kept": kept,
                    "spilled": spilled,
                    "spill_cost": cost,
                    "solve_time": solve_t,
                    "dynamic_insts": m_dyn,
                    "loads": m_ld,
                    "stores": m_st,
                    "mem_ops": m_ld + m_st,
                }

                records.append(
                    {
                        "program": prog_name,
                        "k": k,
                        "method": m,
                        "live_ranges": num_ranges,
                        "spilled_count": len(spilled),
                        "spill_cost": cost,
                        "solve_time_s": solve_t,
                        "dynamic_instructions": m_dyn,
                        "loads": m_ld,
                        "stores": m_st,
                        "mem_ops": m_ld + m_st,
                    }
                )

            # Verification: ILP optimality bounds compared to heuristics and brute force
            if "ilp" in alloc_results:
                ilp_cost = alloc_results["ilp"]["spill_cost"]
                if "linear_scan" in alloc_results:
                    ls_cost = alloc_results["linear_scan"]["spill_cost"]
                    if ilp_cost > ls_cost + 1e-5:
                        msg = f"F2 Optimality Violation: ILP cost {ilp_cost} > LinearScan cost {ls_cost} for {prog_name}, k={k}"
                        print(f"[FAIL] {msg}")
                        verification_failures.append(msg)
                    elif ilp_cost < ls_cost - 1e-5:
                        strictly_beats_count["linear_scan"] += 1

                if "greedy_weight" in alloc_results:
                    gw_cost = alloc_results["greedy_weight"]["spill_cost"]
                    if ilp_cost > gw_cost + 1e-5:
                        msg = f"F2 Optimality Violation: ILP cost {ilp_cost} > GreedyWeight cost {gw_cost} for {prog_name}, k={k}"
                        print(f"[FAIL] {msg}")
                        verification_failures.append(msg)
                    elif ilp_cost < gw_cost - 1e-5:
                        strictly_beats_count["greedy_weight"] += 1

                if "bruteforce" in alloc_results:
                    bf_cost = alloc_results["bruteforce"]["spill_cost"]
                    if abs(ilp_cost - bf_cost) > 1e-5:
                        msg = f"F2 Exactness Violation: ILP cost {ilp_cost} != BruteForce cost {bf_cost} for {prog_name}, k={k}"
                        print(f"[FAIL] {msg}")
                        verification_failures.append(msg)

    df_raw = pd.DataFrame(records)
    df_raw.to_csv(out_dir / "f2_regalloc_raw.csv", index=False)

    # Aggregated table by (k, method)
    agg_df = (
        df_raw.groupby(["k", "method"])
        .agg(
            mean_spill_cost=("spill_cost", "mean"),
            mean_mem_ops=("mem_ops", "mean"),
            mean_loads=("loads", "mean"),
            mean_stores=("stores", "mean"),
            mean_solve_time_s=("solve_time_s", "mean"),
        )
        .reset_index()
    )
    agg_df.to_csv(out_dir / "f2_regalloc_summary.csv", index=False)

    # Plot 1: Mean dynamic memory operations vs k
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for m in ["linear_scan", "greedy_weight", "ilp"]:
        sub = agg_df[agg_df["method"] == m]
        ax.plot(
            sub["k"],
            sub["mean_mem_ops"],
            marker="o",
            label=m.replace("_", " ").title(),
            lw=2,
        )
    ax.set_xlabel("Number of Physical Registers (k)")
    ax.set_ylabel("Dynamic Memory Operations (Loads + Stores)")
    ax.set_title("Memory Traffic vs Register File Capacity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "f2_memory_ops_vs_k.png")
    plt.close()

    # Plot 2: Mean spill cost vs k
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for m in ["linear_scan", "greedy_weight", "ilp"]:
        sub = agg_df[agg_df["method"] == m]
        ax.plot(
            sub["k"],
            sub["mean_spill_cost"],
            marker="s",
            label=m.replace("_", " ").title(),
            lw=2,
        )
    ax.set_xlabel("Number of Physical Registers (k)")
    ax.set_ylabel("Weighted Spill Cost")
    ax.set_yscale("log")
    ax.set_title("Spill Cost Objective vs Register File Capacity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "f2_spill_cost_vs_k.png")
    plt.close()

    return df_raw, agg_df, strictly_beats_count


# ---------------------------------------------------------------------------
# Experiment F3: Differential Fuzzing
# ---------------------------------------------------------------------------


def run_experiment_f3(
    out_dir: Path,
    quick: bool,
    verification_failures: list[str],
) -> tuple[pd.DataFrame, float, int, int]:
    """Differential testing over randomly generated programs across all pipelines and VM."""
    print("--- Running Experiment F3: Differential Fuzzing ---")
    num_tests = 40 if quick else 300
    mismatch_count = 0
    all_reduction_pcts: list[float] = []

    for seed in range(42, 42 + num_tests):
        src = fuzz_generate(seed)
        base_ir = compile_to_ir(src)

        # Baseline execution
        base_exc: type | None = None
        base_out: list[int] = []
        base_dyn = 0
        try:
            base_out, base_dyn, _ = run_ir(base_ir)
        except Exception as e:
            base_exc = type(e)

        # 1. Test each individual pass
        variants: dict[str, Any] = {
            "const_fold_prop": const_fold_prop(base_ir),
            "algebraic_strength_reduction": algebraic_strength_reduction(base_ir),
            "cse": cse(base_ir),
            "dce": dce(base_ir),
        }

        # 2. Test "all" pass pipeline
        opt_ir, _ = optimize_all(base_ir)
        variants["all"] = opt_ir

        for var_name, var_ir in variants.items():
            try:
                var_out, var_dyn, _ = run_ir(var_ir)
                var_exc = None
            except Exception as e:
                var_out = []
                var_dyn = 0
                var_exc = type(e)

            if var_exc != base_exc:
                msg = f"F3 Exception Mismatch (Seed {seed}, Pass {var_name}): {var_exc} vs Base {base_exc}"
                print(f"[FAIL] {msg}")
                verification_failures.append(msg)
                mismatch_count += 1
            elif base_exc is None and var_out != base_out:
                msg = f"F3 Output Mismatch (Seed {seed}, Pass {var_name}): {var_out} vs Base {base_out}"
                print(f"[FAIL] {msg}")
                verification_failures.append(msg)
                mismatch_count += 1

            if var_name == "all" and base_exc is None and base_dyn > 0:
                all_reduction_pcts.append((base_dyn - var_dyn) / base_dyn * 100.0)

        # 3. Test Machine Code generation with ILP at k=3
        try:
            ranges = build_live_ranges(opt_ir)
            kept, spilled, _, _ = allocate_ilp(ranges, k=3)
            alloc = assign_physical_registers(ranges, kept, spilled, k=3)
            code = generate_code(opt_ir, alloc, k=3)
            m_out, _, _, _, _, _ = run_machine(code)
            m_exc = None
        except Exception as e:
            m_out = []
            m_exc = type(e)

        if m_exc != base_exc:
            msg = f"F3 Machine Exception Mismatch (Seed {seed}): {m_exc} vs Base {base_exc}"
            print(f"[FAIL] {msg}")
            verification_failures.append(msg)
            mismatch_count += 1
        elif base_exc is None and m_out != base_out:
            msg = f"F3 Machine Output Mismatch (Seed {seed}): {m_out} vs Base {base_out}"
            print(f"[FAIL] {msg}")
            verification_failures.append(msg)
            mismatch_count += 1

    mean_reduction = float(np.mean(all_reduction_pcts)) if all_reduction_pcts else 0.0

    df_summary = pd.DataFrame(
        [
            {
                "programs_fuzzed": num_tests,
                "mismatches": mismatch_count,
                "mean_dynamic_reduction_all_pct": mean_reduction,
            }
        ]
    )
    df_summary.to_csv(out_dir / "f3_fuzz_summary.csv", index=False)

    return df_summary, mean_reduction, num_tests, mismatch_count


# ---------------------------------------------------------------------------
# Artifact Generation: example_ir.txt & SUMMARY.md
# ---------------------------------------------------------------------------


def write_example_ir(out_dir: Path) -> None:
    """Write pretty-printed unoptimized and optimized IR for dead_code and sum_squares."""
    txt_lines: list[str] = []

    for name in ["dead_code", "sum_squares"]:
        src = PROGRAMS[name]
        unopt_ir = compile_to_ir(src)
        opt_ir, _ = optimize_all(unopt_ir)

        txt_lines.append("=" * 70)
        txt_lines.append(f"PROGRAM: {name}")
        txt_lines.append("=" * 70)
        txt_lines.append("\n[UNOPTIMIZED IR]\n")
        txt_lines.append(format_ir(unopt_ir))
        txt_lines.append("\n" + "-" * 70)
        txt_lines.append("\n[OPTIMIZED IR (Pipeline 'all')]\n")
        txt_lines.append(format_ir(opt_ir))
        txt_lines.append("\n\n")

    (out_dir / "example_ir.txt").write_text("\n".join(txt_lines), encoding="utf-8")


def write_summary_md(
    out_dir: Path,
    quick: bool,
    f1_summary: pd.DataFrame,
    f1_impacts: dict[str, int],
    f2_summary: pd.DataFrame,
    f2_strictly_beats: dict[str, int],
    f3_tests: int,
    f3_mismatches: int,
    f3_reduction: float,
    verification_failures: list[str],
    failed_experiments: dict[str, str] | None = None,
    completed_experiments: set[str] | None = None,
    interrupted: bool = False,
) -> None:
    """Render comprehensive markdown experimental report without interpretation."""
    if failed_experiments is None:
        failed_experiments = {}
    if completed_experiments is None:
        completed_experiments = set()

    md_lines: list[str] = [
        "# Track F: Compiler Optimization – Experimental Summary",
        "",
        "## Environment & Platform Configuration",
        f"- **Python Version**: {platform.python_version()}",
        f"- **NumPy Version**: {np.__version__}",
        f"- **SciPy Version**: {scipy.__version__}",
        f"- **Platform / OS**: {platform.platform()}",
        "- **Random Seeds**: Explicit seeds (range 42 to 42 + N)",
        f"- **Quick Execution Flag**: {quick}",
        "- **Note on Performance Timing**: Run times are measured on a Python-level IR interpreter and register-machine VM and serve as relative proxies; dynamic and static instruction counts are mathematically exact.",
        "",
        "## Verification Invariants & Failure Log",
    ]

    has_failures = bool(verification_failures) or bool(failed_experiments) or interrupted or (len(completed_experiments) < 3)

    if interrupted:
        md_lines.append("- :warning: **Execution interrupted by user (KeyboardInterrupt)**.")

    if failed_experiments:
        md_lines.append("### Experiments that failed:")
        for exp_name, tb in failed_experiments.items():
            md_lines.append(f"#### Experiment {exp_name}")
            md_lines.append("```")
            md_lines.append(tb.strip())
            md_lines.append("```")

    if verification_failures:
        md_lines.append(f"- :x: **{len(verification_failures)} Verification Failures Detected**:")
        for fail in verification_failures:
            md_lines.append(f"  - `{fail}`")

    if not has_failures:
        md_lines.append(
            "- :white_check_mark: **No verification failures**. All semantic, algebraic, and capacity invariants satisfied."
        )

    md_lines.extend(
        [
            "",
            "---",
            "## Experiment F1: Optimization Pass Effects",
            "",
            "### Setup Description",
            "Evaluates static IR instruction size, dynamic instruction counts, runtime, and compile-time overhead "
            "across 8 benchmark programs for six configurations: unoptimized (none), constant folding/propagation "
            "(const_fold_prop), algebraic strength reduction (algebraic_strength_reduction), common subexpression "
            "elimination (cse), dead code elimination (dce), and the complete pipeline (all).",
            "",
            "#### Table F1.1: Aggregated Optimization Pass Effects",
            render_markdown_table(f1_summary) if not f1_summary.empty else "*(Experiment F1 was not completed or failed)*",
            "",
            "#### Single-Pass Effectiveness Across Programs",
        ]
    )

    if f1_impacts:
        for p, cnt in f1_impacts.items():
            md_lines.append(f"- **{p}**: produced code changes in {cnt} of 8 benchmark programs")
    else:
        md_lines.append("*(No pass impact data available)*")

    md_lines.extend(
        [
            "",
            "#### Artifact Figures",
            "- `f1_dynamic_reduction.png`: Grouped bar chart of percentage dynamic instruction reduction per benchmark program.",
            "- `f1_static_vs_dynamic.png`: Scatter plot mapping static code size against dynamic execution steps for unoptimized vs optimized code.",
            "",
            "---",
            "## Experiment F2: Register Allocation as 0/1 Optimization",
            "",
            "### Setup Description",
            "Formulates register allocation over live range intervals [start, end] as a 0/1 Integer Linear Program (ILP) "
            "with objective maximize sum(w_i * x_i) subject to overlap capacity sum(x_i) <= k. Compares against Linear Scan "
            "and Greedy-by-Weight heuristics across physical register counts k in {2, 3, 4, 6, 8}. Emits register machine code "
            "and simulates dynamic loads, stores, and execution cycles.",
            "",
            "#### Table F2.1: Aggregated Register Allocation Metrics by Capacity (k) and Method",
            render_markdown_table(f2_summary) if not f2_summary.empty else "*(Experiment F2 was not completed or failed)*",
            "",
            "#### Exact Optimality Comparison",
            f"- ILP strictly lower spill cost than Linear Scan: {f2_strictly_beats.get('linear_scan', 0)} instances",
            f"- ILP strictly lower spill cost than Greedy Weight: {f2_strictly_beats.get('greedy_weight', 0)} instances",
            "",
            "#### Artifact Figures",
            "- `f2_memory_ops_vs_k.png`: Mean dynamic memory traffic (loads + stores) as a function of physical registers k.",
            "- `f2_spill_cost_vs_k.png`: Weighted spill cost objective versus physical register file size k.",
            "",
            "---",
            "## Experiment F3: Differential Fuzzing",
            "",
            "### Setup Description",
            f"Performs differential testing across {f3_tests} deterministically generated programs containing nested conditionals, "
            "bounded loops, arithmetic expressions, and non-zero constant divisors. Validates semantic equivalence between unoptimized IR, "
            "individual passes, the full pipeline, and machine code running on the register VM with k=3.",
            "",
            f"- **Programs Tested**: {f3_tests}",
            f"- **Semantic Mismatches**: {f3_mismatches}",
            f"- **Mean Dynamic Instruction Reduction (Pipeline 'all')**: {f3_reduction:.4f}%",
            "",
            "---",
            "## Disassembly Comparison Artifact",
            "- `example_ir.txt`: Contains textual three-address disassembly before and after optimization for `dead_code` and `sum_squares`.",
            "",
        ]
    )

    (out_dir / "SUMMARY.md").write_text("\n".join(md_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main Driver Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    """Execute all Track F experiments, export figures, CSV tables, example IR, and SUMMARY.md."""
    parser = argparse.ArgumentParser(
        description="Run Track F compiler optimization experiments."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in accelerated mode (reduced repeats and fuzz cases).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(_REPO_ROOT / "results" / "track_f"),
        help="Output directory path for artifacts.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    verification_failures: list[str] = []
    failed_experiments: dict[str, str] = {}
    completed_experiments: set[str] = set()
    interrupted = False
    keyboard_exc: BaseException | None = None

    f1_summary = pd.DataFrame()
    f1_impacts: dict[str, int] = {}
    f2_summary = pd.DataFrame()
    f2_strictly_beats: dict[str, int] = {}
    f3_tests = 0
    f3_mismatches = 0
    f3_reduction = 0.0

    try:
        # F1
        try:
            _, f1_summary, f1_impacts = run_experiment_f1(
                out_dir, args.quick, verification_failures
            )
            completed_experiments.add("F1")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[ERROR] Experiment F1 crashed:\n{tb}")
            failed_experiments["F1"] = tb
            verification_failures.append(f"Experiment F1 crashed: {e}")

        # F2
        try:
            _, f2_summary, f2_strictly_beats = run_experiment_f2(
                out_dir, args.quick, verification_failures
            )
            completed_experiments.add("F2")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[ERROR] Experiment F2 crashed:\n{tb}")
            failed_experiments["F2"] = tb
            verification_failures.append(f"Experiment F2 crashed: {e}")

        # F3
        try:
            _, f3_reduction, f3_tests, f3_mismatches = run_experiment_f3(
                out_dir, args.quick, verification_failures
            )
            completed_experiments.add("F3")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[ERROR] Experiment F3 crashed:\n{tb}")
            failed_experiments["F3"] = tb
            verification_failures.append(f"Experiment F3 crashed: {e}")

        try:
            write_example_ir(out_dir)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[ERROR] Writing example_ir.txt crashed:\n{tb}")
            failed_experiments["example_ir"] = tb
            verification_failures.append(f"Writing example_ir.txt crashed: {e}")

    except KeyboardInterrupt as ki:
        interrupted = True
        keyboard_exc = ki
        print("\nExecution interrupted by user (KeyboardInterrupt). Writing summary...")
    finally:
        try:
            write_summary_md(
                out_dir=out_dir,
                quick=args.quick,
                f1_summary=f1_summary,
                f1_impacts=f1_impacts,
                f2_summary=f2_summary,
                f2_strictly_beats=f2_strictly_beats,
                f3_tests=f3_tests,
                f3_mismatches=f3_mismatches,
                f3_reduction=f3_reduction,
                verification_failures=verification_failures,
                failed_experiments=failed_experiments,
                completed_experiments=completed_experiments,
                interrupted=interrupted,
            )
        except Exception as write_err:
            print(f"Error writing SUMMARY.md in finally block: {write_err}")
            traceback.print_exc()

    if keyboard_exc is not None:
        raise keyboard_exc

    if verification_failures or failed_experiments or len(completed_experiments) < 3:
        print(f"\n[SUMMARY] Failures encountered (verification failures: {len(verification_failures)}, failed experiments: {len(failed_experiments)}).")
        return 1
    else:
        print("\n[SUMMARY] All Track F experiments completed successfully with 0 failures.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

