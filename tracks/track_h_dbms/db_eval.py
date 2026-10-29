"""Database evaluation, index size measurement, and workload benchmarking harness for Track H."""

from __future__ import annotations

import sqlite3
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Ensure repository root and track root are placed on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from db_data import copy_database
from db_queries import CANDIDATES, Query, result_hash, run_query


def _get_work_dir(work_dir: str | Path | None = None) -> Path:
    """Return resolved working directory for heavy database copies."""
    wdir = Path(work_dir) if work_dir is not None else _REPO_ROOT / "work" / "track_h"
    wdir.mkdir(parents=True, exist_ok=True)
    return wdir


def measure_index_sizes(
    base_db: str | Path,
    candidates: list[tuple[str, str, list[str]]],
    work_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Measure the exact page-level storage size and build time of each candidate index.

    For each candidate index, creates an isolated temporary copy of the base database,
    measures used pages before and after index creation, and computes:
      size = (used_pages_after - used_pages_before) * page_size
    where used_pages = PRAGMA page_count - PRAGMA freelist_count.

    Args:
        base_db: Path to the unindexed base database.
        candidates: List of candidate index tuples (id, table, columns).
        work_dir: Workspace directory for temporary database files.

    Returns:
        List of dicts: {"id", "table", "columns", "size_bytes", "build_time_s"}.
    """
    wdir = _get_work_dir(work_dir)
    results: list[dict[str, Any]] = []

    for idx_id, table, columns in candidates:
        tmp_db = wdir / f"size_eval_{idx_id}_{uuid.uuid4().hex[:8]}.db"
        try:
            copy_database(base_db, tmp_db)
            conn = sqlite3.connect(str(tmp_db))
            cur = conn.cursor()

            cur.execute("PRAGMA page_size")
            page_size = int(cur.fetchone()[0])

            cur.execute("PRAGMA page_count")
            pc_before = int(cur.fetchone()[0])
            cur.execute("PRAGMA freelist_count")
            fc_before = int(cur.fetchone()[0])
            used_before = pc_before - fc_before

            cols_str = ", ".join(columns)
            create_sql = f"CREATE INDEX {idx_id} ON {table} ({cols_str})"

            t0 = time.perf_counter()
            cur.execute(create_sql)
            conn.commit()
            build_time = time.perf_counter() - t0

            cur.execute("PRAGMA page_count")
            pc_after = int(cur.fetchone()[0])
            cur.execute("PRAGMA freelist_count")
            fc_after = int(cur.fetchone()[0])
            used_after = pc_after - fc_after

            conn.close()

            delta_pages = max(1, used_after - used_before)
            size_bytes = delta_pages * page_size

            results.append({
                "id": idx_id,
                "table": table,
                "columns": columns,
                "size_bytes": int(size_bytes),
                "build_time_s": float(build_time),
            })
        finally:
            if tmp_db.exists():
                tmp_db.unlink()

    return results


def measure_benefit_matrix(
    base_db: str | Path,
    queries: list[Query],
    candidates: list[tuple[str, str, list[str]]],
    repeats: int = 3,
    deadline: float | None = None,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Measure the empirical benefit matrix B[Q, C] of time saved by each individual index.

    Runs ANALYZE once per database configuration to ensure SQLite optimizer statistics.
    For each candidate index alone, only executes queries whose tables overlap with the
    index's table (non-overlapping queries receive benefit 0.0 without redundant execution).
    Checks query result hashes against baseline results to ensure correctness.

    Args:
        base_db: Path to the unindexed base database.
        queries: List of 12 Query workload instances.
        candidates: List of 16 candidate index tuples.
        repeats: Repetitions per query to compute median execution time.
        deadline: Optional wall-clock timestamp deadline (time.time()).
        work_dir: Workspace directory for temporary database copies.

    Returns:
        dict containing:
          - "b": np.ndarray of shape (len(queries), len(candidates))
          - "baseline_times": np.ndarray of baseline query times (length Q)
          - "baseline_hashes": list of baseline result hashes
          - "failures": list of error messages / hash mismatches
          - "is_partial": bool indicating if execution stopped early due to deadline
    """
    wdir = _get_work_dir(work_dir)
    Q = len(queries)
    C = len(candidates)
    failures: list[str] = []
    is_partial = False

    # 1. Measure baseline (no candidate indexes)
    tmp_baseline = wdir / f"bench_baseline_{uuid.uuid4().hex[:8]}.db"
    baseline_times = np.zeros(Q, dtype=np.float64)
    baseline_hashes: list[str] = []

    try:
        copy_database(base_db, tmp_baseline)
        conn = sqlite3.connect(str(tmp_baseline))
        cur = conn.cursor()
        cur.execute("ANALYZE")
        conn.commit()

        for q_idx, q in enumerate(queries):
            query_runs: list[float] = []
            q_res_hash: str = ""
            for r in range(max(1, repeats)):
                res, el = run_query(conn, q)
                query_runs.append(el)
                if r == 0:
                    q_res_hash = result_hash(res)
            baseline_times[q_idx] = float(statistics.median(query_runs))
            baseline_hashes.append(q_res_hash)

        conn.close()
    finally:
        if tmp_baseline.exists():
            tmp_baseline.unlink()

    # 2. Measure each candidate index alone
    b_matrix = np.zeros((Q, C), dtype=np.float64)

    for c_idx, (idx_id, table, columns) in enumerate(candidates):
        if deadline is not None and time.time() > deadline:
            is_partial = True
            break

        # Check if any query accesses this table
        relevant_queries = [q_idx for q_idx, q in enumerate(queries) if table in q.tables]
        if not relevant_queries:
            continue

        tmp_idx_db = wdir / f"bench_idx_{idx_id}_{uuid.uuid4().hex[:8]}.db"
        try:
            copy_database(base_db, tmp_idx_db)
            conn = sqlite3.connect(str(tmp_idx_db))
            cur = conn.cursor()

            cols_str = ", ".join(columns)
            cur.execute(f"CREATE INDEX {idx_id} ON {table} ({cols_str})")
            cur.execute("ANALYZE")
            conn.commit()

            for q_idx in relevant_queries:
                q = queries[q_idx]
                query_runs = []
                for r in range(max(1, repeats)):
                    res, el = run_query(conn, q)
                    query_runs.append(el)
                    if r == 0:
                        h = result_hash(res)
                        if h != baseline_hashes[q_idx]:
                            failures.append(
                                f"Hash mismatch on query {q.id} with candidate {idx_id}: "
                                f"got {h[:8]} vs baseline {baseline_hashes[q_idx][:8]}"
                            )
                time_with_i = float(statistics.median(query_runs))
                saved = baseline_times[q_idx] - time_with_i
                b_matrix[q_idx, c_idx] = max(0.0, saved)

            conn.close()
        finally:
            if tmp_idx_db.exists():
                tmp_idx_db.unlink()

    return {
        "b": b_matrix,
        "baseline_times": baseline_times,
        "baseline_hashes": baseline_hashes,
        "failures": failures,
        "is_partial": is_partial,
    }


def evaluate_config(
    base_db: str | Path,
    index_ids: list[str],
    candidates: list[tuple[str, str, list[str]]],
    queries: list[Query],
    repeats: int = 3,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate an exact configuration of indexes on the full query workload.

    Creates the selected indexes, runs ANALYZE, records EXPLAIN QUERY PLAN outputs,
    and runs the full workload `repeats` times to measure throughput and latency.

    Args:
        base_db: Path to unindexed base database.
        index_ids: List of index ids to create (e.g. ['I1', 'I6']).
        candidates: Full list of 16 candidate index tuples.
        queries: List of 12 Query workload instances.
        repeats: Repetitions to compute median times.
        work_dir: Workspace directory for temporary database copies.

    Returns:
        dict containing workload_time_s, per_query_times_s, result_hashes, plans,
        db_size_bytes, num_executions, throughput_qps.
    """
    wdir = _get_work_dir(work_dir)
    cand_map = {c[0]: (c[1], c[2]) for c in candidates}
    tmp_db = wdir / f"eval_cfg_{uuid.uuid4().hex[:8]}.db"

    try:
        copy_database(base_db, tmp_db)
        conn = sqlite3.connect(str(tmp_db))
        cur = conn.cursor()

        # Create selected indexes
        for i_id in index_ids:
            if i_id in cand_map:
                tbl, cols = cand_map[i_id]
                cols_str = ", ".join(cols)
                cur.execute(f"CREATE INDEX {i_id} ON {tbl} ({cols_str})")

        cur.execute("ANALYZE")
        conn.commit()

        # Capture EXPLAIN QUERY PLAN for each query (using first parameter set)
        plans: dict[str, str] = {}
        for q in queries:
            first_param = q.params[0] if q.params else ()
            cur.execute(f"EXPLAIN QUERY PLAN {q.sql}", first_param)
            plan_rows = cur.fetchall()
            # row[3] is the detail explanation string in SQLite
            plans[q.id] = "\n".join(str(r[3]) for r in plan_rows)

        # Benchmark workload across `repeats` iterations
        workload_totals: list[float] = []
        per_query_runs: dict[str, list[float]] = {q.id: [] for q in queries}
        recorded_hashes: dict[str, str] = {}

        for r in range(max(1, repeats)):
            w_total = 0.0
            for q in queries:
                res, el = run_query(conn, q)
                w_total += el
                per_query_runs[q.id].append(el)
                if r == 0:
                    recorded_hashes[q.id] = result_hash(res)
            workload_totals.append(w_total)

        conn.close()

        median_workload = float(statistics.median(workload_totals))
        per_query_medians = {q.id: float(statistics.median(per_query_runs[q.id])) for q in queries}
        db_size = tmp_db.stat().st_size
        num_executions = sum(len(q.params) for q in queries)
        throughput = float(num_executions / median_workload) if median_workload > 0 else 0.0

        return {
            "workload_time_s": median_workload,
            "per_query_times_s": per_query_medians,
            "result_hashes": recorded_hashes,
            "plans": plans,
            "db_size_bytes": db_size,
            "num_executions": num_executions,
            "throughput_qps": throughput,
        }
    finally:
        if tmp_db.exists():
            tmp_db.unlink()

