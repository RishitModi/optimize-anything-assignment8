"""TPC-H-like synthetic data generation and SQLite database builder for Track H.

Generates relational tables without declared primary/foreign keys or secondary indexes.
The default SQLite planner has only full table scans, enabling controlled evaluation of
index selection algorithms under storage budget constraints.
"""

from __future__ import annotations

import csv
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Ensure repository root and track root are placed on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

# Fixed 25 TPC-H nation entries (n_nationkey, n_name, n_regionkey)
NATIONS = [
    (0, "ALGERIA", 0),
    (1, "ARGENTINA", 1),
    (2, "BRAZIL", 1),
    (3, "CANADA", 1),
    (4, "EGYPT", 4),
    (5, "ETHIOPIA", 0),
    (6, "FRANCE", 3),
    (7, "GERMANY", 3),
    (8, "INDIA", 2),
    (9, "INDONESIA", 2),
    (10, "IRAN", 4),
    (11, "IRAQ", 4),
    (12, "JAPAN", 2),
    (13, "JORDAN", 4),
    (14, "KENYA", 0),
    (15, "MOROCCO", 0),
    (16, "MOZAMBIQUE", 0),
    (17, "PERU", 1),
    (18, "CHINA", 2),
    (19, "ROMANIA", 3),
    (20, "SAUDI ARABIA", 4),
    (21, "VIETNAM", 2),
    (22, "RUSSIA", 3),
    (23, "UNITED KINGDOM", 3),
    (24, "UNITED STATES", 1),
]

MKTSEGMENTS = ["AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"]


def copy_database(src: str | Path, dst: str | Path) -> None:
    """Copy an SQLite database file using the sqlite3 online backup API.

    Args:
        src: Path to source SQLite database.
        dst: Path to destination SQLite database.
    """
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()

    src_conn = sqlite3.connect(str(src_path))
    dst_conn = sqlite3.connect(str(dst_path))
    try:
        with dst_conn:
            src_conn.backup(dst_conn)
    finally:
        src_conn.close()
        dst_conn.close()


def _check_existing_db(db_path: Path, sf: float, seed: int) -> dict[str, int] | None:
    """Check if an existing database matches the required sf and seed.

    Returns row counts dict if valid, else None.
    """
    if not db_path.is_file():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='_meta'")
        if not cur.fetchone():
            conn.close()
            return None
        cur.execute("SELECT value FROM _meta WHERE key='sf'")
        row_sf = cur.fetchone()
        cur.execute("SELECT value FROM _meta WHERE key='seed'")
        row_seed = cur.fetchone()
        if not row_sf or not row_seed:
            conn.close()
            return None
        if abs(float(row_sf[0]) - sf) > 1e-9 or int(row_seed[0]) != seed:
            conn.close()
            return None

        # Verify tables and get row counts
        tables = ["customer", "orders", "lineitem", "part", "supplier", "nation"]
        row_counts: dict[str, int] = {}
        for tbl in tables:
            cur.execute(f"SELECT COUNT(*) FROM {tbl}")
            row_counts[tbl] = int(cur.fetchone()[0])
        conn.close()
        return row_counts
    except Exception:
        return None


def build_database(
    path: str | Path,
    sf: float,
    seed: int,
    out_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Build a TPC-H-like synthetic SQLite database scaled by scale factor sf.

    Tables are created without primary keys or indexes so that the default SQLite planner
    must use full table scans. Chunked insertion is performed with PRAGMA journal_mode=OFF
    and synchronous=OFF for fast ingestion.

    Args:
        path: Path where the SQLite database will be written.
        sf: Scale factor (e.g., 0.05 for standard benchmark, 0.005 for quick mode).
        seed: Random seed for deterministic generation.
        out_csv: Optional path for dataset.csv (defaults to results/track_h/dataset.csv).

    Returns:
        dict containing row counts, database size, SQLite version, sf, and seed.
    """
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if database file already exists with same sf and seed
    existing_counts = _check_existing_db(db_path, sf, seed)
    if existing_counts is not None:
        db_size = db_path.stat().st_size
        result_info = {
            "row_counts": existing_counts,
            "db_size_bytes": db_size,
            "sqlite_version": sqlite3.sqlite_version,
            "sf": sf,
            "seed": seed,
        }
        _save_dataset_csv(out_csv, result_info)
        return result_info

    # Remove existing file if mismatched or corrupted
    if db_path.exists():
        db_path.unlink()

    rng = np.random.default_rng(seed)

    # Row counts scaled by sf
    n_customer = max(100, int(150000 * sf))
    n_orders = max(1000, int(1500000 * sf))
    n_part = max(100, int(200000 * sf))
    n_supplier = max(10, int(10000 * sf))
    n_nation = 25

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Fast insertion pragma settings
    cur.execute("PRAGMA journal_mode = OFF")
    cur.execute("PRAGMA synchronous = OFF")

    # DDL: Tables without keys and without secondary indexes
    # Note: Explicitly unindexed so the baseline planner uses full scans.
    cur.execute("""
        CREATE TABLE customer (
            c_custkey INTEGER,
            c_nationkey INTEGER,
            c_mktsegment TEXT,
            c_acctbal INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE orders (
            o_orderkey INTEGER,
            o_custkey INTEGER,
            o_orderstatus TEXT,
            o_totalprice INTEGER,
            o_orderdate INTEGER,
            o_orderpriority INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE lineitem (
            l_orderkey INTEGER,
            l_linenumber INTEGER,
            l_partkey INTEGER,
            l_suppkey INTEGER,
            l_quantity INTEGER,
            l_extendedprice INTEGER,
            l_discount INTEGER,
            l_shipdate INTEGER,
            l_returnflag TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE part (
            p_partkey INTEGER,
            p_brand TEXT,
            p_type INTEGER,
            p_size INTEGER,
            p_retailprice INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE supplier (
            s_suppkey INTEGER,
            s_nationkey INTEGER,
            s_acctbal INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE nation (
            n_nationkey INTEGER,
            n_name TEXT,
            n_regionkey INTEGER
        )
    """)
    cur.execute("CREATE TABLE _meta (key TEXT PRIMARY KEY, value TEXT)")

    # Insert nation table (fixed 25 rows)
    cur.executemany("INSERT INTO nation VALUES (?, ?, ?)", NATIONS)

    # 1. Insert customer table
    c_custkey = np.arange(1, n_customer + 1, dtype=np.int64)
    c_nationkey = rng.integers(0, 25, size=n_customer, dtype=np.int64)
    seg_idx = rng.integers(0, len(MKTSEGMENTS), size=n_customer)
    c_mktsegment = np.array(MKTSEGMENTS)[seg_idx]
    c_acctbal = rng.integers(-99999, 1000000, size=n_customer, dtype=np.int64)

    chunk_size = 100000
    for start in range(0, n_customer, chunk_size):
        end = min(start + chunk_size, n_customer)
        rows = list(zip(
            c_custkey[start:end].tolist(),
            c_nationkey[start:end].tolist(),
            c_mktsegment[start:end].tolist(),
            c_acctbal[start:end].tolist(),
        ))
        cur.executemany("INSERT INTO customer VALUES (?, ?, ?, ?)", rows)

    # 2. Insert orders table
    o_orderkey = np.arange(1, n_orders + 1, dtype=np.int64)
    o_custkey = rng.integers(1, n_customer + 1, size=n_orders, dtype=np.int64)
    # Status F/O/P with probabilities 0.49 / 0.49 / 0.02
    status_choices = np.array(["F", "O", "P"])
    o_orderstatus = rng.choice(status_choices, size=n_orders, p=[0.49, 0.49, 0.02])
    o_totalprice = rng.integers(1000, 50000001, size=n_orders, dtype=np.int64)
    o_orderdate = rng.integers(0, 2401, size=n_orders, dtype=np.int64)
    o_orderpriority = rng.integers(1, 6, size=n_orders, dtype=np.int64)

    for start in range(0, n_orders, chunk_size):
        end = min(start + chunk_size, n_orders)
        rows = list(zip(
            o_orderkey[start:end].tolist(),
            o_custkey[start:end].tolist(),
            o_orderstatus[start:end].tolist(),
            o_totalprice[start:end].tolist(),
            o_orderdate[start:end].tolist(),
            o_orderpriority[start:end].tolist(),
        ))
        cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", rows)

    # 3. Generate and insert lineitem table
    # Each order has 1..7 lineitems uniformly distributed
    items_per_order = rng.integers(1, 8, size=n_orders, dtype=np.int64)
    n_lineitems = int(np.sum(items_per_order))

    l_orderkey = np.repeat(o_orderkey, items_per_order)
    # Generate 1-based linenumbers per order efficiently using cumsum step diffs
    ends = np.cumsum(items_per_order)
    diff_arr = np.ones(n_lineitems, dtype=np.int64)
    diff_arr[ends[:-1]] -= items_per_order[:-1]
    l_linenumber = np.cumsum(diff_arr)

    l_partkey = rng.integers(1, n_part + 1, size=n_lineitems, dtype=np.int64)
    l_suppkey = rng.integers(1, n_supplier + 1, size=n_lineitems, dtype=np.int64)
    l_quantity = rng.integers(1, 51, size=n_lineitems, dtype=np.int64)
    l_unitprice = rng.integers(100, 10001, size=n_lineitems, dtype=np.int64)
    l_extendedprice = l_quantity * l_unitprice
    l_discount = rng.integers(0, 11, size=n_lineitems, dtype=np.int64)

    l_orderdate = np.repeat(o_orderdate, items_per_order)
    ship_delay = rng.integers(1, 122, size=n_lineitems, dtype=np.int64)
    l_shipdate = l_orderdate + ship_delay

    ret_choices = np.array(["R", "A", "N"])
    l_returnflag = rng.choice(ret_choices, size=n_lineitems)

    for start in range(0, n_lineitems, chunk_size):
        end = min(start + chunk_size, n_lineitems)
        rows = list(zip(
            l_orderkey[start:end].tolist(),
            l_linenumber[start:end].tolist(),
            l_partkey[start:end].tolist(),
            l_suppkey[start:end].tolist(),
            l_quantity[start:end].tolist(),
            l_extendedprice[start:end].tolist(),
            l_discount[start:end].tolist(),
            l_shipdate[start:end].tolist(),
            l_returnflag[start:end].tolist(),
        ))
        cur.executemany("INSERT INTO lineitem VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    # 4. Insert part table
    p_partkey = np.arange(1, n_part + 1, dtype=np.int64)
    brands = [f"Brand#{i}{j}" for i in range(1, 6) for j in range(1, 6)]
    brand_idx = rng.integers(0, len(brands), size=n_part)
    p_brand = np.array(brands)[brand_idx]
    p_type = rng.integers(0, 150, size=n_part, dtype=np.int64)
    p_size = rng.integers(1, 51, size=n_part, dtype=np.int64)
    p_retailprice = rng.integers(100, 10001, size=n_part, dtype=np.int64)

    for start in range(0, n_part, chunk_size):
        end = min(start + chunk_size, n_part)
        rows = list(zip(
            p_partkey[start:end].tolist(),
            p_brand[start:end].tolist(),
            p_type[start:end].tolist(),
            p_size[start:end].tolist(),
            p_retailprice[start:end].tolist(),
        ))
        cur.executemany("INSERT INTO part VALUES (?, ?, ?, ?, ?)", rows)

    # 5. Insert supplier table
    s_suppkey = np.arange(1, n_supplier + 1, dtype=np.int64)
    s_nationkey = rng.integers(0, 25, size=n_supplier, dtype=np.int64)
    s_acctbal = rng.integers(-99999, 1000000, size=n_supplier, dtype=np.int64)

    for start in range(0, n_supplier, chunk_size):
        end = min(start + chunk_size, n_supplier)
        rows = list(zip(
            s_suppkey[start:end].tolist(),
            s_nationkey[start:end].tolist(),
            s_acctbal[start:end].tolist(),
        ))
        cur.executemany("INSERT INTO supplier VALUES (?, ?, ?)", rows)

    # Store metadata for caching and verification
    cur.execute("INSERT INTO _meta VALUES ('sf', ?)", (str(sf),))
    cur.execute("INSERT INTO _meta VALUES ('seed', ?)", (str(seed),))
    conn.commit()
    conn.close()

    row_counts = {
        "customer": n_customer,
        "orders": n_orders,
        "lineitem": n_lineitems,
        "part": n_part,
        "supplier": n_supplier,
        "nation": n_nation,
    }
    db_size = db_path.stat().st_size
    result_info = {
        "row_counts": row_counts,
        "db_size_bytes": db_size,
        "sqlite_version": sqlite3.sqlite_version,
        "sf": sf,
        "seed": seed,
    }
    _save_dataset_csv(out_csv, result_info)
    return result_info


def _save_dataset_csv(out_csv: str | Path | None, result_info: dict[str, Any]) -> None:
    """Save dataset metadata and row counts to a standardized CSV file."""
    csv_path = Path(out_csv) if out_csv is not None else _REPO_ROOT / "results" / "track_h" / "dataset.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for tbl, count in result_info["row_counts"].items():
        rows.append({
            "table_name": tbl,
            "row_count": count,
            "database_size_bytes": result_info["db_size_bytes"],
            "sqlite_version": result_info["sqlite_version"],
            "scale_factor": result_info["sf"],
            "seed": result_info["seed"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

