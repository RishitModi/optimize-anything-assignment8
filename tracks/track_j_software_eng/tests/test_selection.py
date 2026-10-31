"""Unit tests for Track J: selection heuristics, ILP optimality, coverage tracer, and smoke test."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import sys
from typing import Any

import pytest

_TRACK_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]

for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sj_select import (
    select_greedy,
    select_ilp,
    select_random_until_cover,
)
from sj_suite import (
    apply_mutation_op,
    collect_coverage,
    executable_lines,
)


def brute_force_set_cover(
    cover: dict[str, set[Any]],
    cost: dict[str, float],
    universe: set[Any],
) -> tuple[list[str], float]:
    """Exhaustive search over all 2^N subsets to find the exact global minimum cost cover."""
    test_keys = list(cover.keys())
    n = len(test_keys)
    best_cost = float("inf")
    best_subset: list[str] = []

    cap = (1 << n) + 5
    steps = 0
    for mask in range(1 << n):
        steps += 1
        if steps > cap:
            break
        subset = [test_keys[i] for i in range(n) if (mask & (1 << i))]
        covered: set[Any] = set()
        for t in subset:
            covered.update(cover[t])

        if universe.issubset(covered):
            c = sum(cost[t] for t in subset)
            if c < best_cost:
                best_cost = c
                best_subset = subset

    return best_subset, best_cost


def test_handmade_set_cover_instance() -> None:
    """Verify that greedy yields a valid cover and ILP yields the global optimum on a hand-crafted instance."""
    universe = {1, 2, 3, 4, 5}
    cover = {
        "t1": {1, 2},
        "t2": {2, 3, 4},
        "t3": {4, 5},
        "t4": {1, 3, 5},
    }
    cost = {
        "t1": 2.0,
        "t2": 3.0,
        "t3": 2.0,
        "t4": 4.0,
    }

    # 1. Greedy selection
    greedy_sel = select_greedy(cover, cost, universe)
    covered_greedy = set().union(*(cover[t] for t in greedy_sel))
    assert universe.issubset(covered_greedy), "Greedy selection must cover the entire universe"

    # 2. ILP selection compared against brute force
    ilp_sel, optimal, ilp_cost, _ = select_ilp(cover, cost, universe)
    assert optimal, "ILP solver should report optimal status"
    covered_ilp = set().union(*(cover[t] for t in ilp_sel))
    assert universe.issubset(covered_ilp), "ILP selection must cover the entire universe"

    _, bf_cost = brute_force_set_cover(cover, cost, universe)
    assert abs(ilp_cost - bf_cost) < 1e-5, f"ILP cost ({ilp_cost}) must equal brute-force optimum ({bf_cost})"


def test_random_instances_optimality() -> None:
    """Verify ILP cost equals brute force and greedy cost >= ILP across 10 random small instances."""
    rng = random.Random(42)

    for i in range(10):
        n_tests = rng.randint(6, 12)
        universe = set(range(rng.randint(5, 8)))

        cover: dict[str, set[Any]] = {}
        cost: dict[str, float] = {}

        # Ensure every universe element is covered by at least one test
        for u in universe:
            t_name = f"t_{rng.randint(0, n_tests - 1)}"
            cover.setdefault(t_name, set()).add(u)

        for t_idx in range(n_tests):
            t_name = f"t_{t_idx}"
            if t_name not in cover:
                cover[t_name] = set()
            # Randomly add elements
            for u in universe:
                if rng.random() < 0.4:
                    cover[t_name].add(u)
            cost[t_name] = round(rng.uniform(1.0, 10.0), 2)

        greedy_sel = select_greedy(cover, cost, universe)
        greedy_cost = sum(cost[t] for t in greedy_sel)

        ilp_sel, optimal, ilp_cost, _ = select_ilp(cover, cost, universe)
        assert optimal, "ILP solver must find an optimal solution"

        _, bf_cost = brute_force_set_cover(cover, cost, universe)

        assert abs(ilp_cost - bf_cost) < 1e-4, f"Instance {i}: ILP ({ilp_cost}) != BruteForce ({bf_cost})"
        assert greedy_cost >= ilp_cost - 1e-4, f"Instance {i}: Greedy ({greedy_cost}) < ILP ({ilp_cost})"


def test_random_until_cover_always_covers() -> None:
    """Verify that select_random_until_cover covers 100% of the universe for all seeds."""
    universe = {10, 20, 30, 40}
    cover = {
        "a": {10, 20},
        "b": {20, 30},
        "c": {30, 40},
        "d": {10, 40},
    }
    cost = {"a": 1.0, "b": 1.5, "c": 2.0, "d": 2.5}

    res = select_random_until_cover(cover, cost, universe, seeds=15)
    for sel in res["selections"]:
        covered = set().union(*(cover[t] for t in sel))
        assert universe.issubset(covered), "Every random-until-cover trial must cover the universe"


def test_coverage_tracer_on_temporary_module(tmp_path: Path) -> None:
    """Verify sys.settrace coverage tracer accurately records executed line numbers."""
    mod_path = tmp_path / "temp_target.py"
    mod_code = (
        "def compute(x):\n"
        "    a = x + 1\n"
        "    if a > 2:\n"
        "        return a * 2\n"
        "    return a\n"
    )
    mod_path.write_text(mod_code, encoding="utf-8")

    # Load module dynamically
    spec = importlib.util.spec_from_file_location("temp_target", str(mod_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Collect executable lines
    ex_lines = executable_lines(mod_path)
    assert len(ex_lines) >= 3, "Module must contain executable lines"

    # Trace execution of compute(5)
    covered_lines, covered_arcs = collect_coverage(
        lambda: mod.compute(5),
        cut_paths=[mod_path],
    )

    line_nums = {lineno for f_idx, lineno in covered_lines if f_idx == 0}
    assert 2 in line_nums, "Line 2 (a = x + 1) should be executed"
    assert 3 in line_nums, "Line 3 (if a > 2) should be executed"
    assert 4 in line_nums, "Line 4 (return a * 2) should be executed"
    assert 5 not in line_nums, "Line 5 (return a) should not be executed for x=5"


def test_mutant_operator_no_site_leaves_unchanged() -> None:
    """Verify that a mutation operator finding no applicable site returns the IR unchanged."""
    ir = [
        ("mov", "a", 42, None),
        ("print", None, "a", None),
        ("halt", None, None, None),
    ]

    # M1 modifies add / sub. IR has neither.
    mutated = apply_mutation_op(ir, "M1")
    assert mutated == ir, "Operator M1 must leave instruction list unchanged when no add/sub present"

    # M5 modifies lt / le. IR has neither.
    mutated_m5 = apply_mutation_op(ir, "M5")
    assert mutated_m5 == ir, "Operator M5 must leave instruction list unchanged when no lt/le present"


def test_smoke_run_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke test running quick mode of run_all into tmp_path without importing run_all directly."""
    run_all_path = _TRACK_ROOT / "run_all.py"
    spec = importlib.util.spec_from_file_location("track_j_run_all", str(run_all_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["track_j_run_all"] = mod
    spec.loader.exec_module(mod)

    out_dir = tmp_path / "smoke_out"
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--quick", "--out", str(out_dir)])

    # Execute main()
    try:
        mod.main()
    except SystemExit as exc:
        assert exc.code == 0, f"run_all exited with non-zero code {exc.code}"

    # Assert outputs generated
    assert (out_dir / "SUMMARY.md").exists(), "SUMMARY.md must be generated"
    assert any(out_dir.glob("*.csv")), "At least one CSV output file must exist"
    assert any(out_dir.glob("*.png")), "At least one PNG figure file must exist"

