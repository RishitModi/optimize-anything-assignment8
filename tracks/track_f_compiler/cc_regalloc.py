"""Register Allocation Formulated as a 0/1 Integer Linear Program and Heuristics.

Mathematical Formulation:
-------------------------
Given a program with n live ranges (each corresponding to a variable v_i active over
interval [start_i, end_i] in linear instruction space) and k available registers:

Decision Variables:
    x_i in {0, 1} for each live range i in {1, ..., n}
    where x_i = 1 if range i is assigned to a physical register,
          x_i = 0 if range i is spilled to memory.

Objective:
    Maximise total weight of variables kept in registers:
        max sum_{i=1}^n w_i * x_i
    Equivalently, minimise weighted spill cost:
        min sum_{i=1}^n w_i * (1 - x_i)

Constraints:
    At every instruction index t in {0, ..., T}:
        sum_{i: start_i <= t <= end_i} x_i <= k

Theoretical Property:
    Because each live range is a contiguous interval along the linear instruction sequence,
    the column for range i in the constraint matrix contains a single contiguous block of 1s
    (the consecutive-ones property for interval graphs). Consequently, the constraint matrix
    is totally unimodular, guaranteeing that the basic feasible solutions of the continuous
    Linear Programming (LP) relaxation are naturally integral.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure repository root and track root are placed on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import scipy.optimize

from cc_ir import (
    get_def_use,
    live_at_each_instruction,
    loop_depth_per_instruction,
)
from cc_lang import Instruction


@dataclass(frozen=True)
class LiveRange:
    """Represents the lifetime interval and execution weight of a variable."""

    var: str
    start: int
    end: int
    weight: int


def build_live_ranges(ir: list[Instruction]) -> list[LiveRange]:
    """Construct live range intervals [start, end] and loop-depth-based spill weights.

    start is the first instruction index where the variable is defined or live.
    end is the last instruction index where the variable is used or live.
    weight w_i = sum over every definition and use of 10 ** loop_depth(instruction).
    """
    if not ir:
        return []

    live_before = live_at_each_instruction(ir)
    depths = loop_depth_per_instruction(ir)

    all_vars: set[str] = set()
    defs_at: list[set[str]] = []
    uses_at: list[set[str]] = []

    for inst in ir:
        d, u = get_def_use(inst)
        defs_at.append(d)
        uses_at.append(u)
        all_vars.update(d)
        all_vars.update(u)

    for s in live_before:
        all_vars.update(s)

    ranges: list[LiveRange] = []
    for v in sorted(all_vars):
        def_or_live = [
            i for i in range(len(ir)) if (v in live_before[i] or v in defs_at[i])
        ]
        use_or_live = [
            i for i in range(len(ir)) if (v in live_before[i] or v in uses_at[i])
        ]

        if not def_or_live and not use_or_live:
            continue

        start_idx = min(def_or_live) if def_or_live else min(use_or_live)
        end_idx = max(use_or_live) if use_or_live else max(def_or_live)
        end_idx = max(start_idx, end_idx)

        # Spill weight: 10 ** loop_depth for each def and each use
        total_weight = 0
        for i, inst in enumerate(ir):
            op, dst, a, b = inst
            inst_weight = 10 ** depths[i]
            if dst == v:
                total_weight += inst_weight
            if a == v:
                total_weight += inst_weight
            if b == v:
                total_weight += inst_weight

        if total_weight == 0:
            total_weight = 1

        ranges.append(
            LiveRange(var=v, start=start_idx, end=end_idx, weight=total_weight)
        )

    # Sort ranges by start instruction, breaking ties deterministically
    ranges.sort(key=lambda r: (r.start, r.end, r.var))
    return ranges


# ---------------------------------------------------------------------------
# Heuristic 1: Linear Scan (Poletto & Sarkar)
# ---------------------------------------------------------------------------


def allocate_linear_scan(
    ranges: list[LiveRange], k: int
) -> tuple[set[str], set[str], float, float]:
    """Allocate registers using Poletto & Sarkar's Linear Scan algorithm.

    Iterates through live ranges ordered by start point. When active ranges exceed k,
    spills the active range with the furthest end point (which may be the current range).
    """
    t0 = time.perf_counter()
    if k <= 0 or not ranges:
        spilled = {r.var for r in ranges}
        cost = float(sum(r.weight for r in ranges))
        return set(), spilled, cost, time.perf_counter() - t0

    sorted_ranges = sorted(ranges, key=lambda r: (r.start, r.end, r.var))
    active: list[LiveRange] = []
    kept: set[str] = set()
    spilled: set[str] = set()

    for r in sorted_ranges:
        # Expire old ranges that ended strictly before current start
        active = [act for act in active if act.end >= r.start]

        if len(active) < k:
            active.append(r)
            kept.add(r.var)
        else:
            # Select range among active + current with furthest end point
            furthest = max(active, key=lambda a: (a.end, -a.weight))
            if furthest.end > r.end:
                active.remove(furthest)
                kept.remove(furthest.var)
                spilled.add(furthest.var)
                active.append(r)
                kept.add(r.var)
            else:
                spilled.add(r.var)

    spill_cost = float(sum(r.weight for r in ranges if r.var in spilled))
    elapsed = time.perf_counter() - t0
    return kept, spilled, spill_cost, elapsed


# ---------------------------------------------------------------------------
# Heuristic 2: Greedy by Weight
# ---------------------------------------------------------------------------


def allocate_greedy_weight(
    ranges: list[LiveRange], k: int
) -> tuple[set[str], set[str], float, float]:
    """Greedy register allocation ordering ranges by weight descending.

    Admits a range if and only if adding it keeps register occupancy <= k at all points.
    """
    t0 = time.perf_counter()
    if k <= 0 or not ranges:
        spilled = {r.var for r in ranges}
        cost = float(sum(r.weight for r in ranges))
        return set(), spilled, cost, time.perf_counter() - t0

    sorted_ranges = sorted(ranges, key=lambda r: (-r.weight, r.start, r.var))
    max_pt = max(r.end for r in ranges)
    occupancy = [0] * (max_pt + 1)

    kept: set[str] = set()
    spilled: set[str] = set()

    for r in sorted_ranges:
        can_fit = all(occupancy[t] < k for t in range(r.start, r.end + 1))
        if can_fit:
            for t in range(r.start, r.end + 1):
                occupancy[t] += 1
            kept.add(r.var)
        else:
            spilled.add(r.var)

    spill_cost = float(sum(r.weight for r in ranges if r.var in spilled))
    elapsed = time.perf_counter() - t0
    return kept, spilled, spill_cost, elapsed


# ---------------------------------------------------------------------------
# Exact 1: Integer Linear Programming (ILP) via scipy.optimize.milp
# ---------------------------------------------------------------------------


def allocate_ilp(
    ranges: list[LiveRange], k: int
) -> tuple[set[str], set[str], float, float]:
    """Solve optimal register allocation via 0/1 Integer Linear Programming.

    Formulates constraints sum_{r containing t} x_r <= k and maximizes sum w_i x_i.
    Deduplicates identical constraint rows across time points.
    """
    t0 = time.perf_counter()
    n = len(ranges)
    if n == 0 or k <= 0:
        spilled = {r.var for r in ranges}
        cost = float(sum(r.weight for r in ranges))
        return set(), spilled, cost, time.perf_counter() - t0

    # Maximize sum(w_i * x_i) <=> Minimize sum(-w_i * x_i)
    c = np.array([-float(r.weight) for r in ranges], dtype=float)
    integrality = np.ones(n, dtype=int)
    bounds = scipy.optimize.Bounds(lb=0, ub=1)

    min_pt = min(r.start for r in ranges)
    max_pt = max(r.end for r in ranges)

    # Collect distinct overlap sets of range indices
    seen_rows: set[tuple[int, ...]] = set()
    row_list: list[tuple[int, ...]] = []

    for t in range(min_pt, max_pt + 1):
        indices = tuple(i for i, r in enumerate(ranges) if r.start <= t <= r.end)
        if len(indices) > k and indices not in seen_rows:
            seen_rows.add(indices)
            row_list.append(indices)

    if not row_list:
        # All ranges can fit simultaneously without any conflict
        kept = {r.var for r in ranges}
        elapsed = time.perf_counter() - t0
        return kept, set(), 0.0, elapsed

    # Construct constraint matrix A x <= k
    A = np.zeros((len(row_list), n), dtype=float)
    for row_idx, indices in enumerate(row_list):
        for col_idx in indices:
            A[row_idx, col_idx] = 1.0

    constraints = scipy.optimize.LinearConstraint(A, lb=-np.inf, ub=k)
    res = scipy.optimize.milp(
        c=c, integrality=integrality, bounds=bounds, constraints=constraints
    )

    if not res.success:
        # Fallback to greedy if solver reports issue
        return allocate_greedy_weight(ranges, k)

    x = np.round(res.x).astype(int)
    kept = {ranges[i].var for i in range(n) if x[i] == 1}
    spilled = {ranges[i].var for i in range(n) if x[i] == 0}
    spill_cost = float(sum(r.weight for r in ranges if r.var in spilled))
    elapsed = time.perf_counter() - t0

    return kept, spilled, spill_cost, elapsed


# ---------------------------------------------------------------------------
# Exact 2: Brute Force for Small Problem Instances
# ---------------------------------------------------------------------------


def allocate_bruteforce(
    ranges: list[LiveRange], k: int
) -> tuple[set[str], set[str], float, float]:
    """Exhaustive search over all 2^n assignment subsets (applicable for n <= 18)."""
    t0 = time.perf_counter()
    n = len(ranges)
    if n > 18:
        raise ValueError(f"Exhaustive brute force only supported for n <= 18 (got {n})")
    if n == 0 or k <= 0:
        spilled = {r.var for r in ranges}
        cost = float(sum(r.weight for r in ranges))
        return set(), spilled, cost, time.perf_counter() - t0

    min_pt = min(r.start for r in ranges)
    max_pt = max(r.end for r in ranges)

    overlap_masks: list[int] = []
    for t in range(min_pt, max_pt + 1):
        mask = 0
        count = 0
        for i, r in enumerate(ranges):
            if r.start <= t <= r.end:
                mask |= 1 << i
                count += 1
        if count > k:
            overlap_masks.append(mask)

    best_mask = 0
    best_kept_weight = -1

    for subset_mask in range(1 << n):
        # Verify capacity constraint at all overlap points
        feasible = True
        for o_mask in overlap_masks:
            intersect = subset_mask & o_mask
            if intersect.bit_count() > k:
                feasible = False
                break

        if feasible:
            current_weight = sum(
                ranges[i].weight for i in range(n) if (subset_mask & (1 << i))
            )
            if current_weight > best_kept_weight:
                best_kept_weight = current_weight
                best_mask = subset_mask

    kept = {ranges[i].var for i in range(n) if (best_mask & (1 << i))}
    spilled = {ranges[i].var for i in range(n) if not (best_mask & (1 << i))}
    spill_cost = float(sum(r.weight for r in ranges if r.var in spilled))
    elapsed = time.perf_counter() - t0

    return kept, spilled, spill_cost, elapsed


# ---------------------------------------------------------------------------
# Interval Coloring and Spill Slot Assignment
# ---------------------------------------------------------------------------


@dataclass
class AllocationResult:
    """Holds physical register mapping, memory slot assignments, and metrics."""

    kept: set[str]
    spilled: set[str]
    spill_cost: float
    solve_time: float
    reg_map: dict[str, int]
    slot_map: dict[str, int]


def assign_physical_registers(
    ranges: list[LiveRange],
    kept: set[str],
    spilled: set[str],
    k: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """Assign physical registers 0..k-1 to kept ranges and memory slots 0, 1... to spilled variables."""
    kept_ranges = [r for r in ranges if r.var in kept]
    kept_ranges.sort(key=lambda r: (r.start, r.end, r.var))

    # Interval coloring: sweep by start point, reuse registers
    active: list[tuple[int, int, str]] = []  # (end_time, reg, var)
    reg_map: dict[str, int] = {}

    for r in kept_ranges:
        # Expire ranges ending strictly before r.start
        active = [item for item in active if item[0] >= r.start]
        used_regs = {item[1] for item in active}

        chosen_reg = 0
        for reg in range(k):
            if reg not in used_regs:
                chosen_reg = reg
                break

        reg_map[r.var] = chosen_reg
        active.append((r.end, chosen_reg, r.var))

    slot_map: dict[str, int] = {
        var: slot_idx for slot_idx, var in enumerate(sorted(spilled))
    }
    return reg_map, slot_map

