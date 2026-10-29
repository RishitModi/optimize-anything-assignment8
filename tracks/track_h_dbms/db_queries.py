"""Query workload definitions, candidate index definitions, and execution harness for Track H."""

from __future__ import annotations

import dataclasses
import hashlib
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root and track root are placed on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# Exactly the 16 candidate secondary indexes specified for Track H
# Format: list of tuple (id, table, list_of_columns)
CANDIDATES: list[tuple[str, str, list[str]]] = [
    ("I1", "lineitem", ["l_orderkey"]),
    ("I2", "lineitem", ["l_shipdate"]),
    ("I3", "lineitem", ["l_partkey"]),
    ("I4", "lineitem", ["l_suppkey"]),
    ("I5", "lineitem", ["l_partkey", "l_suppkey"]),
    ("I6", "orders", ["o_orderkey"]),
    ("I7", "orders", ["o_custkey"]),
    ("I8", "orders", ["o_orderdate"]),
    ("I9", "orders", ["o_totalprice"]),
    ("I10", "orders", ["o_orderstatus", "o_orderdate"]),
    ("I11", "customer", ["c_custkey"]),
    ("I12", "customer", ["c_mktsegment", "c_nationkey"]),
    ("I13", "part", ["p_partkey"]),
    ("I14", "part", ["p_brand", "p_size"]),
    ("I15", "supplier", ["s_suppkey"]),
    ("I16", "supplier", ["s_nationkey"]),
]


@dataclasses.dataclass(frozen=True)
class Query:
    """Dataclass encapsulating an analytical query template and its parameters.

    Attributes:
        id: Unique identifier (e.g., 'Q1' .. 'Q12').
        name: Short descriptive name of query logic.
        sql: Parameterized SQL string with ? placeholders.
        params: List of tuples, one per parameter set execution.
        tables: List of relational tables accessed by this query.
    """

    id: str
    name: str
    sql: str
    params: list[tuple[Any, ...]]
    tables: list[str]


def result_hash(results: Any) -> str:
    """Compute deterministic SHA-256 hex digest of query result rows.

    Args:
        results: List of result row lists or tuples returned by run_query.

    Returns:
        Hexadecimal SHA-256 string.
    """
    return hashlib.sha256(repr(results).encode("utf-8")).hexdigest()


def run_query(conn: sqlite3.Connection, q: Query) -> tuple[list[list[Any]], float]:
    """Execute all parameter sets of query q on the given connection.

    Args:
        conn: SQLite connection.
        q: Query object with SQL and list of parameter tuples.

    Returns:
        tuple (list of result row lists for all parameter sets, elapsed wall-clock seconds).
    """
    results: list[list[Any]] = []
    t0 = time.perf_counter()
    cursor = conn.cursor()
    for p in q.params:
        cursor.execute(q.sql, p)
        results.append(cursor.fetchall())
    cursor.close()
    elapsed = time.perf_counter() - t0
    return results, elapsed


def make_workload(conn: sqlite3.Connection, seed: int = 42, sf: float = 0.05) -> list[Query]:
    """Generate the standard 12-query workload with deterministic parameter sets.

    Parameters are sampled deterministically using np.random.default_rng(seed).
    Q3 fetches 15 valid (l_partkey, l_suppkey) pairs directly from existing lineitem rows.

    Args:
        conn: Open SQLite connection to the base database.
        seed: Random seed for deterministic parameter generation.
        sf: Scale factor used when generating the database.

    Returns:
        List of exactly 12 Query instances (Q1 through Q12).
    """
    rng = np.random.default_rng(seed)
    cur = conn.cursor()

    # Query table cardinalities for valid parameter ranges
    cur.execute("SELECT COUNT(*) FROM customer")
    n_cust = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM orders")
    n_orders = int(cur.fetchone()[0])
    cur.execute("SELECT MAX(rowid) FROM lineitem")
    row_max = cur.fetchone()[0]
    n_lineitems = int(row_max) if row_max is not None else 1000

    # Q1 [5] lineitem shipdate window (14-day window)
    q1_starts = rng.integers(0, 2400 - 14, size=5)
    q1_params = [(int(s), int(s + 14)) for s in q1_starts]
    q1 = Query(
        id="Q1",
        name="Lineitem Shipdate 14-day Window",
        sql="SELECT COUNT(*), SUM(l_extendedprice) FROM lineitem WHERE l_shipdate BETWEEN ? AND ?",
        params=q1_params,
        tables=["lineitem"],
    )

    # Q2 [15] lineitem orderkey point lookup
    q2_orders = rng.integers(1, n_orders + 1, size=15)
    q2_params = [(int(o),) for o in q2_orders]
    q2 = Query(
        id="Q2",
        name="Lineitem Quantity by Orderkey",
        sql="SELECT SUM(l_quantity) FROM lineitem WHERE l_orderkey = ?",
        params=q2_params,
        tables=["lineitem"],
    )

    # Q3 [15] lineitem composite point lookup from existing rows
    q3_rowids = rng.integers(1, n_lineitems + 1, size=15)
    q3_params: list[tuple[Any, ...]] = []
    for rid in q3_rowids:
        cur.execute("SELECT l_partkey, l_suppkey FROM lineitem WHERE rowid = ?", (int(rid),))
        row = cur.fetchone()
        if row:
            q3_params.append((int(row[0]), int(row[1])))
        else:
            q3_params.append((1, 1))
    q3 = Query(
        id="Q3",
        name="Lineitem Count by Partkey and Suppkey",
        sql="SELECT COUNT(*) FROM lineitem WHERE l_partkey = ? AND l_suppkey = ?",
        params=q3_params,
        tables=["lineitem"],
    )

    # Q4 [2] orders JOIN lineitem date range (6-day window)
    q4_starts = rng.integers(0, 2400 - 6, size=2)
    q4_params = [(int(s), int(s + 6)) for s in q4_starts]
    q4 = Query(
        id="Q4",
        name="Orders Lineitem Join 6-day Window",
        sql="SELECT COUNT(*), SUM(o.o_totalprice) FROM orders o JOIN lineitem l ON l.l_orderkey = o.o_orderkey WHERE o.o_orderdate BETWEEN ? AND ?",
        params=q4_params,
        tables=["orders", "lineitem"],
    )

    # Q5 [2] customer JOIN orders segment and nation filter
    segments = ["AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"]
    q5_params: list[tuple[Any, ...]] = []
    for _ in range(2):
        seg = str(rng.choice(segments))
        nat = int(rng.integers(0, 25))
        q5_params.append((seg, nat))
    q5 = Query(
        id="Q5",
        name="Customer Orders Join by Segment and Nation",
        sql="SELECT COUNT(*) FROM customer c JOIN orders o ON o.o_custkey = c.c_custkey WHERE c.c_mktsegment = ? AND c.c_nationkey = ?",
        params=q5_params,
        tables=["customer", "orders"],
    )

    # Q6 [3] orders top-10 totalprice (no params, repeated 3 times)
    q6_params = [(), (), ()]
    q6 = Query(
        id="Q6",
        name="Orders Top-10 Totalprice",
        sql="SELECT o_orderkey, o_totalprice FROM orders ORDER BY o_totalprice DESC, o_orderkey LIMIT 10",
        params=q6_params,
        tables=["orders"],
    )

    # Q7 [3] orders status and date window (30-day window)
    q7_statuses = ["F", "O", "F"]
    q7_starts = rng.integers(0, 2400 - 30, size=3)
    q7_params = [(q7_statuses[i], int(q7_starts[i]), int(q7_starts[i] + 30)) for i in range(3)]
    q7 = Query(
        id="Q7",
        name="Orders Status and 30-day Window",
        sql="SELECT COUNT(*) FROM orders WHERE o_orderstatus = ? AND o_orderdate BETWEEN ? AND ?",
        params=q7_params,
        tables=["orders"],
    )

    # Q8 [2] part JOIN lineitem brand and size filter
    brands = [f"Brand#{i}{j}" for i in range(1, 6) for j in range(1, 6)]
    q8_params: list[tuple[Any, ...]] = []
    for _ in range(2):
        b = str(rng.choice(brands))
        s = int(rng.integers(1, 51))
        q8_params.append((b, s))
    q8 = Query(
        id="Q8",
        name="Part Lineitem Join by Brand and Size",
        sql="SELECT SUM(l.l_extendedprice) FROM part p JOIN lineitem l ON l.l_partkey = p.p_partkey WHERE p.p_brand = ? AND p.p_size = ?",
        params=q8_params,
        tables=["part", "lineitem"],
    )

    # Q9 [2] supplier JOIN lineitem nation filter
    q9_nats = rng.integers(0, 25, size=2)
    q9_params = [(int(n),) for n in q9_nats]
    q9 = Query(
        id="Q9",
        name="Supplier Lineitem Join by Nation",
        sql="SELECT COUNT(*) FROM supplier s JOIN lineitem l ON l.l_suppkey = s.s_suppkey WHERE s.s_nationkey = ?",
        params=q9_params,
        tables=["supplier", "lineitem"],
    )

    # Q10 [15] orders by custkey point lookup
    q10_custs = rng.integers(1, n_cust + 1, size=15)
    q10_params = [(int(c),) for c in q10_custs]
    q10 = Query(
        id="Q10",
        name="Orders Aggregates by Custkey",
        sql="SELECT COUNT(*), SUM(o_totalprice) FROM orders WHERE o_custkey = ?",
        params=q10_params,
        tables=["orders"],
    )

    # Q11 [1] lineitem returnflag group by with late shipdate (~10% qualifying)
    # Late threshold at 2220 out of max ~2521
    q11_params = [(2220,)]
    q11 = Query(
        id="Q11",
        name="Lineitem Aggregation with Late Shipdate",
        sql="SELECT l_returnflag, COUNT(*), SUM(l_quantity) FROM lineitem WHERE l_shipdate >= ? GROUP BY l_returnflag ORDER BY l_returnflag",
        params=q11_params,
        tables=["lineitem"],
    )

    # Q12 [2] customer JOIN orders JOIN nation 3-day window
    q12_starts = rng.integers(0, 2400 - 3, size=2)
    q12_params = [(int(s), int(s + 3)) for s in q12_starts]
    q12 = Query(
        id="Q12",
        name="Three-Way Join Aggregated by Nation",
        sql="SELECT n.n_name, COUNT(*) FROM customer c JOIN orders o ON o.o_custkey = c.c_custkey JOIN nation n ON n.n_nationkey = c.c_nationkey WHERE o.o_orderdate BETWEEN ? AND ? GROUP BY n.n_name ORDER BY n.n_name",
        params=q12_params,
        tables=["customer", "orders", "nation"],
    )

    cur.close()
    return [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12]

