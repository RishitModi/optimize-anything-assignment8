"""Master execution and benchmarking script for Track H (DBMS Index Selection Optimization).

Runs three progressive experiments:
  H0: Synthetic dataset generation, candidate index sizing and build time profiling.
  H1: Single-index empirical benefit matrix measurement and baseline query evaluation.
  H2: Index selection optimization (Greedy Static, Greedy Marginal, DP Knapsack, ILP, Brute Force)
      across three storage budgets (10%, 25%, 50%), followed by real workload evaluation and verification.

Produces:
  - dataset.csv, candidates.csv, baseline_times.csv, benefit_matrix.csv, selections.csv, per_query.csv
  - Query execution plan dumps: plans_baseline.txt, plans_ilp_25pct.txt, plans_all_indexes.txt
  - Visualizations: workload_vs_storage.png, predicted_vs_measured.png, per_query_comparison.png
  - Comprehensive report: SUMMARY.md
"""

from __future__ import annotations

import argparse
import os
import platform
import sqlite3
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

from db_data import build_database, copy_database
from db_eval import evaluate_config, measure_benefit_matrix, measure_index_sizes
from db_queries import CANDIDATES, Query, make_workload, result_hash
from db_select import (
    additive_benefit,
    predicted_benefit,
    select_bruteforce,
    select_dp_knapsack,
    select_greedy_marginal,
    select_greedy_static,
    select_ilp,
)


def _render_markdown_table(df: pd.DataFrame) -> str:
    """Format a pandas DataFrame as a standard Markdown table without external dependencies."""
    if df.empty:
        return "*Table is empty.*"
    headers = [str(col) for col in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for _, row in df.iterrows():
        row_str = [str(val) for val in row]
        data_lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join([header_line, sep_line] + data_lines)


def run_experiments(
    sf: float = 0.05,
    quick: bool = False,
    out_dir: str | Path = "results/track_h",
    max_minutes: float = 12.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute all Track H experiments with timing guards and error isolation.

    Args:
        sf: Scale factor (0.05 for full mode, 0.005 for quick mode).
        quick: Fast evaluation mode flag.
        out_dir: Directory where tables, plots, and SUMMARY.md are written.
        max_minutes: Overall timeout guard in minutes.
        seed: Random seed for data and query parameter generation.

    Returns:
        Summary dictionary of experimental results.
    """
    start_time = time.time()
    deadline = start_time + max_minutes * 60.0

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    work_dir = _REPO_ROOT / "work" / "track_h"
    work_dir.mkdir(parents=True, exist_ok=True)

    repeats = 2 if quick else 3
    verification_failures: list[str] = []
    experiment_errors: dict[str, str] = {}
    h0_data: dict[str, Any] = {}
    h1_data: dict[str, Any] = {}
    h2_data: dict[str, Any] = {}

    page_size = 4096
    db_file = work_dir / f"base_sf{sf}_seed{seed}.db"

    # -------------------------------------------------------------------------
    # Experiment H0: Setup and Candidate Sizing
    # -------------------------------------------------------------------------
    try:
        # Check deadline
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded prior to H0 setup")

        # Build or reuse base database
        dataset_csv_path = out_path / "dataset.csv"
        build_info = build_database(db_file, sf=sf, seed=seed, out_csv=dataset_csv_path)

        # Retrieve SQLite page size
        conn = sqlite3.connect(str(db_file))
        cur = conn.cursor()
        cur.execute("PRAGMA page_size")
        page_size = int(cur.fetchone()[0])
        conn.close()

        # Measure candidate index sizes and build times
        candidate_measurements = measure_index_sizes(db_file, CANDIDATES, work_dir=work_dir)

        # Save candidates.csv
        cand_rows = []
        for m in candidate_measurements:
            cand_rows.append({
                "id": m["id"],
                "table": m["table"],
                "columns": ", ".join(m["columns"]),
                "size_bytes": m["size_bytes"],
                "build_time_s": round(m["build_time_s"], 4),
            })
        df_candidates = pd.DataFrame(cand_rows)
        df_candidates.to_csv(out_path / "candidates.csv", index=False)

        h0_data = {
            "build_info": build_info,
            "page_size": page_size,
            "candidate_measurements": candidate_measurements,
            "df_candidates": df_candidates,
        }
    except Exception as e:
        experiment_errors["H0"] = traceback.format_exc()

    # -------------------------------------------------------------------------
    # Experiment H1: Benefit Matrix and Baseline Evaluation
    # -------------------------------------------------------------------------
    try:
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded prior to H1 benefit measurement")

        conn = sqlite3.connect(str(db_file))
        queries = make_workload(conn, seed=seed, sf=sf)
        conn.close()

        b_result = measure_benefit_matrix(
            base_db=db_file,
            queries=queries,
            candidates=CANDIDATES,
            repeats=repeats,
            deadline=deadline,
            work_dir=work_dir,
        )

        b_matrix = b_result["b"]
        baseline_times = b_result["baseline_times"]
        baseline_hashes = b_result["baseline_hashes"]
        if b_result["failures"]:
            verification_failures.extend(b_result["failures"])

        # Save baseline_times.csv
        base_rows = []
        for q_idx, q in enumerate(queries):
            base_rows.append({
                "query_id": q.id,
                "query_name": q.name,
                "baseline_time_s": round(float(baseline_times[q_idx]), 5),
                "result_hash": baseline_hashes[q_idx],
            })
        df_baseline = pd.DataFrame(base_rows)
        df_baseline.to_csv(out_path / "baseline_times.csv", index=False)

        # Save benefit_matrix.csv (Queries by Candidate indexes)
        cand_ids = [c[0] for c in CANDIDATES]
        q_ids = [q.id for q in queries]
        df_b = pd.DataFrame(b_matrix, index=q_ids, columns=cand_ids)
        df_b.to_csv(out_path / "benefit_matrix.csv")

        h1_data = {
            "queries": queries,
            "b_matrix": b_matrix,
            "baseline_times": baseline_times,
            "baseline_hashes": baseline_hashes,
            "df_baseline": df_baseline,
            "df_b": df_b,
            "is_partial": b_result.get("is_partial", False),
        }
    except Exception as e:
        experiment_errors["H1"] = traceback.format_exc()

    # -------------------------------------------------------------------------
    # Experiment H2: Selection Algorithms and Real Workload Evaluation
    # -------------------------------------------------------------------------
    try:
        if time.time() > deadline:
            raise TimeoutError("Deadline exceeded prior to H2 selection")

        queries = h1_data["queries"]
        b_matrix = h1_data["b_matrix"]
        cand_measurements = h0_data["candidate_measurements"]
        sizes = np.array([m["size_bytes"] for m in cand_measurements], dtype=np.int64)
        total_size = int(np.sum(sizes))

        budget_fractions = [0.10, 0.25, 0.50]
        budgets = [int(frac * total_size) for frac in budget_fractions]

        cand_id_list = [c[0] for c in CANDIDATES]

        # Cache evaluations by frozenset of selected index IDs
        eval_cache: dict[frozenset[str], dict[str, Any]] = {}

        def get_or_eval_config(idx_ids: list[str]) -> dict[str, Any]:
            key = frozenset(idx_ids)
            if key not in eval_cache:
                eval_cache[key] = evaluate_config(
                    base_db=db_file,
                    index_ids=idx_ids,
                    candidates=CANDIDATES,
                    queries=queries,
                    repeats=repeats,
                    work_dir=work_dir,
                )
            return eval_cache[key]

        # 1. Evaluate baseline (no indexes)
        base_eval = get_or_eval_config([])
        base_workload_s = base_eval["workload_time_s"]

        # 2. Evaluate all 16 indexes
        all_ids = [c[0] for c in CANDIDATES]
        all_eval = get_or_eval_config(all_ids)

        # Selections records
        selection_records: list[dict[str, Any]] = []

        methods = ["greedy_static", "greedy_marginal", "dp_knapsack", "ilp", "bruteforce"]

        for frac, budget in zip(budget_fractions, budgets):
            budget_results: dict[str, dict[str, Any]] = {}

            # Execute selection algorithms
            budget_results["greedy_static"] = select_greedy_static(b_matrix, sizes, budget)
            budget_results["greedy_marginal"] = select_greedy_marginal(b_matrix, sizes, budget)
            budget_results["dp_knapsack"] = select_dp_knapsack(b_matrix, sizes, budget)
            budget_results["ilp"] = select_ilp(b_matrix, sizes, budget)

            # Brute force run guard (instant for 16 candidates)
            bf_t0 = time.perf_counter()
            budget_results["bruteforce"] = select_bruteforce(b_matrix, sizes, budget)

            # Verification invariants for this budget:
            # 1. Budget capacity satisfaction
            for m_name, res in budget_results.items():
                if res["total_size"] > budget:
                    verification_failures.append(
                        f"Budget violation: {m_name} at fraction {frac} used {res['total_size']} > {budget}"
                    )

            # 2. ILP optimality vs heuristics under interaction-aware model
            ilp_pred = budget_results["ilp"]["predicted_benefit"]
            bf_pred = budget_results["bruteforce"]["predicted_benefit"]
            gm_pred = budget_results["greedy_marginal"]["predicted_benefit"]
            gs_pred = budget_results["greedy_static"]["predicted_benefit"]
            dp_pred = budget_results["dp_knapsack"]["predicted_benefit"]

            if abs(ilp_pred - bf_pred) > 1e-7:
                verification_failures.append(
                    f"Optimality mismatch at {frac*100}% budget: ILP ({ilp_pred:.5f}) != BruteForce ({bf_pred:.5f})"
                )
            if ilp_pred + 1e-9 < gm_pred:
                verification_failures.append(
                    f"Sub-optimality: ILP ({ilp_pred:.5f}) < Greedy Marginal ({gm_pred:.5f}) at {frac*100}% budget"
                )
            if ilp_pred + 1e-9 < gs_pred:
                verification_failures.append(
                    f"Sub-optimality: ILP ({ilp_pred:.5f}) < Greedy Static ({gs_pred:.5f}) at {frac*100}% budget"
                )
            if ilp_pred + 1e-9 < dp_pred:
                verification_failures.append(
                    f"Sub-optimality: ILP ({ilp_pred:.5f}) < DP Knapsack ({dp_pred:.5f}) at {frac*100}% budget"
                )

            # 3. DP knapsack additive optimality vs static greedy additive
            dp_add = budget_results["dp_knapsack"]["additive_benefit"]
            gs_add = budget_results["greedy_static"]["additive_benefit"]
            if dp_add + 1e-9 < gs_add:
                verification_failures.append(
                    f"Additive knapsack invariant violated at {frac*100}%: DP ({dp_add:.5f}) < Static Greedy ({gs_add:.5f})"
                )

            # Benchmark each method's configuration on the real database
            for m_name in methods:
                if time.time() > deadline:
                    raise TimeoutError("Deadline exceeded during H2 physical evaluations")

                sol = budget_results[m_name]
                sel_indices = sol["selected"]
                sel_ids = [cand_id_list[i] for i in sel_indices]

                real_eval = get_or_eval_config(sel_ids)

                # Check query hash consistency against baseline
                results_identical = True
                for q in queries:
                    if real_eval["result_hashes"].get(q.id) != base_eval["result_hashes"].get(q.id):
                        results_identical = False
                        verification_failures.append(
                            f"Hash mismatch on query {q.id} in configuration {m_name} ({frac*100}% budget)"
                        )

                meas_workload = real_eval["workload_time_s"]
                meas_benefit = max(0.0, base_workload_s - meas_workload)
                meas_speedup = (base_workload_s / meas_workload) if meas_workload > 0 else 1.0

                selection_records.append({
                    "budget_fraction": f"{int(frac*100)}%",
                    "method": m_name,
                    "selected_ids": ",".join(sel_ids) if sel_ids else "NONE",
                    "total_size_bytes": sol["total_size"],
                    "predicted_benefit_s": round(sol["predicted_benefit"], 5),
                    "additive_benefit_s": round(sol["additive_benefit"], 5),
                    "solve_time_s": round(sol["solve_time_s"], 5),
                    "measured_workload_s": round(meas_workload, 5),
                    "measured_speedup_vs_baseline": round(meas_speedup, 3),
                    "measured_benefit_s": round(meas_benefit, 5),
                    "throughput_qps": round(real_eval["throughput_qps"], 2),
                    "db_size_bytes": real_eval["db_size_bytes"],
                    "results_identical": results_identical,
                })

        df_selections = pd.DataFrame(selection_records)
        df_selections.to_csv(out_path / "selections.csv", index=False)

        # ---------------------------------------------------------------------
        # Save per_query.csv
        # ---------------------------------------------------------------------
        per_query_rows = []
        # Find 25% budget ILP configuration evaluation
        ilp_25_ids = None
        for r in selection_records:
            if r["budget_fraction"] == "25%" and r["method"] == "ilp":
                ilp_25_ids = [x for x in r["selected_ids"].split(",") if x != "NONE"]
                break
        ilp_25_eval = get_or_eval_config(ilp_25_ids if ilp_25_ids is not None else [])

        for q in queries:
            per_query_rows.append({
                "query_id": q.id,
                "baseline_s": round(base_eval["per_query_times_s"].get(q.id, 0.0), 5),
                "ilp_25pct_s": round(ilp_25_eval["per_query_times_s"].get(q.id, 0.0), 5),
                "all_indexes_s": round(all_eval["per_query_times_s"].get(q.id, 0.0), 5),
            })
        df_per_query = pd.DataFrame(per_query_rows)
        df_per_query.to_csv(out_path / "per_query.csv", index=False)

        # ---------------------------------------------------------------------
        # Write Query Plans
        # ---------------------------------------------------------------------
        def write_plans_file(filepath: Path, plans_dict: dict[str, str]) -> None:
            with open(filepath, "w", encoding="utf-8") as f:
                for q in queries:
                    f.write("=" * 80 + "\n")
                    f.write(f"Query: {q.id} ({q.name})\n")
                    f.write(f"SQL: {q.sql}\n")
                    f.write("-" * 40 + "\n")
                    f.write("EXPLAIN QUERY PLAN:\n")
                    f.write(plans_dict.get(q.id, "No plan captured") + "\n")
                    f.write("=" * 80 + "\n\n")

        write_plans_file(out_path / "plans_baseline.txt", base_eval["plans"])
        write_plans_file(out_path / "plans_ilp_25pct.txt", ilp_25_eval["plans"])
        write_plans_file(out_path / "plans_all_indexes.txt", all_eval["plans"])

        # ---------------------------------------------------------------------
        # Generate Visualizations (PNG dpi 150)
        # ---------------------------------------------------------------------
        # Plot 1: Workload time vs storage used (MB) across methods & budgets
        plt.figure(figsize=(9, 5.5), dpi=150)
        marker_map = {
            "greedy_static": "o",
            "greedy_marginal": "s",
            "dp_knapsack": "^",
            "ilp": "D",
            "bruteforce": "x",
        }
        for m_name in methods:
            sub = df_selections[df_selections["method"] == m_name]
            sizes_mb = sub["total_size_bytes"] / (1024 * 1024)
            w_times = sub["measured_workload_s"]
            plt.plot(
                sizes_mb,
                w_times,
                marker=marker_map.get(m_name, "o"),
                label=m_name,
                alpha=0.85,
            )

        # Baseline and All-Indexes horizontal reference lines
        plt.axhline(
            base_workload_s,
            color="black",
            linestyle="--",
            label=f"Baseline (no indexes): {base_workload_s:.3f}s",
        )
        all_workload_s = all_eval["workload_time_s"]
        plt.axhline(
            all_workload_s,
            color="red",
            linestyle=":",
            label=f"All 16 indexes: {all_workload_s:.3f}s",
        )

        plt.xlabel("Index Storage Allocated (MB)")
        plt.ylabel("Measured Workload Latency (seconds)")
        plt.title("Workload Latency vs. Index Storage Budget Across Selection Methods")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path / "workload_vs_storage.png", dpi=150)
        plt.close()

        # Plot 2: Predicted benefit vs measured benefit
        plt.figure(figsize=(7, 6), dpi=150)
        preds = df_selections["predicted_benefit_s"]
        meass = df_selections["measured_benefit_s"]
        plt.scatter(preds, meass, color="#1f77b4", edgecolors="k", s=60, alpha=0.85, label="Configurations")
        max_val = max(float(np.max(preds)), float(np.max(meass)), 0.01) * 1.1
        plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", label="Ideal Prediction (y = x)")
        plt.xlabel("Predicted Benefit (Interaction-Aware Model) [seconds]")
        plt.ylabel("Measured Workload Time Saved [seconds]")
        plt.title("Interaction-Aware Model: Predicted vs. Measured Benefit")
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_path / "predicted_vs_measured.png", dpi=150)
        plt.close()

        # Plot 3: Per-query latency comparison (grouped bars, log y scale)
        plt.figure(figsize=(10, 5.5), dpi=150)
        x_indices = np.arange(len(queries))
        bar_width = 0.28
        plt.bar(
            x_indices - bar_width,
            df_per_query["baseline_s"],
            width=bar_width,
            label="Baseline (unindexed)",
            color="#d62728",
            alpha=0.85,
        )
        plt.bar(
            x_indices,
            df_per_query["ilp_25pct_s"],
            width=bar_width,
            label="ILP (25% budget)",
            color="#1f77b4",
            alpha=0.85,
        )
        plt.bar(
            x_indices + bar_width,
            df_per_query["all_indexes_s"],
            width=bar_width,
            label="All 16 Indexes",
            color="#2ca02c",
            alpha=0.85,
        )
        plt.yscale("log")
        plt.xticks(x_indices, [q.id for q in queries])
        plt.xlabel("Workload Query")
        plt.ylabel("Query Execution Latency (seconds, log scale)")
        plt.title("Per-Query Latency Comparison: Baseline vs ILP (25% Budget) vs All Indexes")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path / "per_query_comparison.png", dpi=150)
        plt.close()

        h2_data = {
            "df_selections": df_selections,
            "df_per_query": df_per_query,
            "base_eval": base_eval,
            "all_eval": all_eval,
            "ilp_25_eval": ilp_25_eval,
        }
    except Exception as e:
        experiment_errors["H2"] = traceback.format_exc()

    finally:
        # ---------------------------------------------------------------------
        # Generate SUMMARY.md
        # ---------------------------------------------------------------------
        summary_md_path = out_path / "SUMMARY.md"
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write("# Track H: DBMS Query Optimization Benchmark Summary\n\n")

            f.write("## Environment\n\n")
            f.write(f"- **Python Version:** {platform.python_version()}\n")
            f.write(f"- **SQLite Version:** {sqlite3.sqlite_version}\n")
            f.write(f"- **SQLite Page Size:** {page_size} bytes\n")
            f.write(f"- **NumPy Version:** {np.__version__}\n")
            f.write(f"- **SciPy Version:** {scipy.__version__}\n")
            f.write(f"- **Platform:** {platform.platform()} ({platform.machine()})\n")
            f.write(f"- **Scale Factor (sf):** {sf}\n")
            f.write(f"- **Quick Mode:** {quick}\n")
            f.write(f"- **Random Seed:** {seed}\n")
            f.write(f"- **Measurement Repetitions:** {repeats}\n\n")

            f.write("Dataset: TPC-H-like synthetic data generated by db_data.py (TPC-H table layout and scaled row counts, not generated by dbgen)\n\n")
            f.write("Default optimiser: SQLite query planner with automatic temporary indexes enabled (the SQLite default); SQLite has no EXPLAIN ANALYZE so plans come from EXPLAIN QUERY PLAN and timings are measured directly\n\n")

            if "build_info" in h0_data:
                f.write("### Table Row Counts\n\n")
                row_counts_df = pd.DataFrame([
                    {"table_name": k, "row_count": v}
                    for k, v in h0_data["build_info"]["row_counts"].items()
                ])
                f.write(_render_markdown_table(row_counts_df) + "\n\n")

            f.write("## Experiment H0: Candidate Index Storage & Build Times\n\n")
            if "df_candidates" in h0_data:
                f.write(_render_markdown_table(h0_data["df_candidates"]) + "\n\n")
            elif "H0" in experiment_errors:
                f.write(f"```text\nH0 Failed:\n{experiment_errors['H0']}\n```\n\n")

            f.write("## Experiment H1: Baseline Query Timings & Single-Index Benefits\n\n")
            if "df_baseline" in h1_data:
                f.write("### Baseline Query Performance\n\n")
                f.write(_render_markdown_table(h1_data["df_baseline"]) + "\n\n")
            if "df_b" in h1_data:
                f.write("### Benefit Matrix B[Q, C] (seconds saved)\n\n")
                f.write(_render_markdown_table(h1_data["df_b"].round(4)) + "\n\n")
            if "H1" in experiment_errors:
                f.write(f"```text\nH1 Failed:\n{experiment_errors['H1']}\n```\n\n")

            f.write("## Experiment H2: Index Selection & Real Workload Evaluation\n\n")
            if "df_selections" in h2_data:
                f.write("### Selected Configurations & Performance Metrics\n\n")
                f.write(_render_markdown_table(h2_data["df_selections"]) + "\n\n")
            if "df_per_query" in h2_data:
                f.write("### Per-Query Execution Times across Selected Configurations\n\n")
                f.write(_render_markdown_table(h2_data["df_per_query"]) + "\n\n")
            if "H2" in experiment_errors:
                f.write(f"```text\nH2 Failed:\n{experiment_errors['H2']}\n```\n\n")

            f.write("## Generated Figures & Query Plans\n\n")
            f.write("- Figures:\n")
            f.write("  - `workload_vs_storage.png`: Measured workload latency vs index storage budget.\n")
            f.write("  - `predicted_vs_measured.png`: Interaction-aware predicted benefit vs measured empirical benefit.\n")
            f.write("  - `per_query_comparison.png`: Per-query latency comparison on log scale.\n")
            f.write("- Query Plan Dumps:\n")
            f.write("  - `plans_baseline.txt`: EXPLAIN QUERY PLAN without indexes.\n")
            f.write("  - `plans_ilp_25pct.txt`: EXPLAIN QUERY PLAN under 25% budget ILP selection.\n")
            f.write("  - `plans_all_indexes.txt`: EXPLAIN QUERY PLAN with all 16 candidate indexes.\n\n")

            f.write("## Verification\n\n")
            all_completed = len(experiment_errors) == 0
            if all_completed and len(verification_failures) == 0:
                f.write("No verification failures. All theoretical invariants, hash matches, and budget constraints passed.\n")
            else:
                if experiment_errors:
                    f.write("### Experiment Exceptions:\n\n")
                    for exp_name, err in experiment_errors.items():
                        f.write(f"- **{exp_name} Exception:**\n```text\n{err}\n```\n\n")
                if verification_failures:
                    f.write("### Invariant Failures:\n\n")
                    for vf in verification_failures:
                        f.write(f"- {vf}\n")
                    f.write("\n")

    return {
        "h0": h0_data,
        "h1": h1_data,
        "h2": h2_data,
        "verification_failures": verification_failures,
        "experiment_errors": experiment_errors,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Track H benchmarking."""
    parser = argparse.ArgumentParser(description="Track H DBMS Index Selection Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation (sf=0.005, 2 repeats)")
    parser.add_argument("--sf", type=float, default=None, help="Scale factor (defaults to 0.005 if --quick else 0.05)")
    parser.add_argument("--out", type=str, default="results/track_h", help="Output directory")
    parser.add_argument("--max-minutes", type=float, default=12.0, help="Maximum execution timeout in minutes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args(argv)

    if args.sf is None:
        sf = 0.005 if args.quick else 0.05
    else:
        sf = args.sf

    print(f"Starting Track H DBMS benchmark: sf={sf}, quick={args.quick}, out={args.out}")
    res = run_experiments(
        sf=sf,
        quick=args.quick,
        out_dir=args.out,
        max_minutes=args.max_minutes,
        seed=args.seed,
    )

    if res["experiment_errors"]:
        print(f"Benchmark completed with errors in: {list(res['experiment_errors'].keys())}")
        return 1
    if res["verification_failures"]:
        print(f"Benchmark completed with {len(res['verification_failures'])} verification failures.")
        return 1

    print("Track H DBMS benchmark completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

