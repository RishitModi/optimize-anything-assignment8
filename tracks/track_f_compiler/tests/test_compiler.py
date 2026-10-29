"""Unit and integration test suite for Track F (Compiler Optimization).

Tests cover:
- Lexing, parsing, and execution of language constructs (loops, conditionals, floor division, modulo).
- Textual assignment validation (CompileError on use before definition).
- Individual optimization passes (const_fold_prop, algebraic_strength_reduction, cse, dce).
- CFG construction and backward dataflow liveness analysis.
- End-to-end benchmark semantic preservation across all optimization levels.
- 0/1 Register allocation equivalence between ILP and Brute Force, and optimality bounds vs heuristics.
- Target register-machine code generation and execution across varying register capacities.
- Smoke test of run_all.py execution under quick mode via importlib.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

# Ensure repository root and track root are placed on sys.path
_TRACK_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest

from cc_ir import live_at_each_instruction, run_ir
from cc_lang import CompileError, compile_to_ir, eval_op
from cc_passes import (
    PASS_REGISTRY,
    algebraic_strength_reduction,
    const_fold_prop,
    cse,
    dce,
    optimize_all,
)
from cc_programs import PROGRAMS
from cc_regalloc import (
    LiveRange,
    allocate_bruteforce,
    allocate_greedy_weight,
    allocate_ilp,
    allocate_linear_scan,
    assign_physical_registers,
    build_live_ranges,
)
from cc_vm import generate_code, run_machine


def test_small_programs_execution() -> None:
    """Verify execution semantics of factorial loop, nested if/else, and negative floor div/mod."""
    # 1. Factorial loop
    fact_src = """
    n = 5;
    res = 1;
    i = 1;
    while (i <= n) {
        res = res * i;
        i = i + 1;
    }
    print res;
    """
    ir = compile_to_ir(fact_src)
    out, _, _ = run_ir(ir)
    assert out == [120]

    # 2. Nested if / else
    nested_if_src = """
    x = 10;
    y = 20;
    if (x < y) {
        if (x == 10) {
            ans = 1;
        } else {
            ans = 2;
        }
    } else {
        ans = 3;
    }
    print ans;
    """
    ir = compile_to_ir(nested_if_src)
    out, _, _ = run_ir(ir)
    assert out == [1]

    # 3. Floor division of negative numbers
    neg_div_src = """
    a = -9;
    b = 4;
    res = a / b;
    print res;
    """
    ir = compile_to_ir(neg_div_src)
    out, _, _ = run_ir(ir)
    assert out == [-3]  # Python floor division: -9 // 4 == -3

    # 4. Modulo of negative numbers
    neg_mod_src = """
    a = -9;
    b = 4;
    res = a % b;
    print res;
    """
    ir = compile_to_ir(neg_mod_src)
    out, _, _ = run_ir(ir)
    assert out == [3]  # Python modulo: -9 % 4 == 3


def test_use_before_assignment_raises_compile_error() -> None:
    """Check that reading a variable not textually assigned earlier raises CompileError with line number."""
    bad_src = """
    x = 1;
    z = y + 1;
    print z;
    """
    with pytest.raises(CompileError) as exc_info:
        compile_to_ir(bad_src)
    assert "Variable 'y' used before textual assignment" in str(exc_info.value)
    assert exc_info.value.line == 3


def test_const_fold_prop() -> None:
    """Verify constant folding for arithmetic, copy propagation, and preservation of division by zero."""
    # 1. Folds 2 + 3 * 4 to 14
    src = "x = 2 + 3 * 4; print x;"
    ir = compile_to_ir(src)
    opt = const_fold_prop(ir)
    assert any(inst[0] == "mov" and inst[2] == 14 for inst in opt)
    out, _, _ = run_ir(opt)
    assert out == [14]

    # 2. Does NOT fold division by constant zero
    div_zero_src = "x = 5 / 0; print x;"
    ir_zero = compile_to_ir(div_zero_src)
    opt_zero = const_fold_prop(ir_zero)
    # The div instruction with 0 divisor must remain in the IR
    assert any(inst[0] == "div" and inst[3] == 0 for inst in opt_zero)
    with pytest.raises(ZeroDivisionError):
        run_ir(opt_zero)


def test_algebraic_strength_reduction() -> None:
    """Verify algebraic simplifications: multiplication by 8 becomes shl, and values match across domain."""
    src = "x = 10; y = x * 8; print y;"
    ir = compile_to_ir(src)
    opt = algebraic_strength_reduction(ir)
    assert any(inst[0] == "shl" and inst[3] == 3 for inst in opt)

    # Equivalence of x*8, x/4, and x%16 for x in [-9, -1, 0, 1, 7, 100]
    for val in [-9, -1, 0, 1, 7, 100]:
        prog = f"""
        x = {val};
        m = x * 8;
        d = x / 4;
        r = x % 16;
        print m;
        print d;
        print r;
        """
        base_ir = compile_to_ir(prog)
        reduced_ir = algebraic_strength_reduction(base_ir)
        out_base, _, _ = run_ir(base_ir)
        out_opt, _, _ = run_ir(reduced_ir)
        assert out_base == out_opt
        assert out_base == [val * 8, val // 4, val % 16]


def test_dce() -> None:
    """Verify DCE removes pure unused assignments but retains unremovable operations with variable divisors."""
    # Pure unused assignment eliminated
    pure_src = "unused = 100 + 200; res = 42; print res;"
    ir_pure = compile_to_ir(pure_src)
    opt_pure = dce(ir_pure)
    assert not any(inst[1] == "unused" for inst in opt_pure)

    # Unused division with variable divisor must be kept (can trap at runtime)
    impure_src = "a = 10; b = 2; unused = a / b; res = 42; print res;"
    ir_impure = compile_to_ir(impure_src)
    opt_impure = dce(ir_impure)
    assert any(inst[0] == "div" for inst in opt_impure)


def test_cse() -> None:
    """Verify CSE replaces repeated subexpressions, but not when an operand is redefined."""
    # Case 1: Identical expressions without redefinition
    cse_src1 = """
    a = 10;
    b = 20;
    c = a + b;
    d = a + b;
    print c;
    print d;
    """
    ir1 = compile_to_ir(cse_src1)
    opt1 = cse(ir1)
    add_ops1 = [inst for inst in opt1 if inst[0] == "add"]
    assert len(add_ops1) == 1

    # Case 2: Redefinition of 'a' invalidates previous computation
    cse_src2 = """
    a = 10;
    b = 20;
    c = a + b;
    a = 30;
    d = a + b;
    print c;
    print d;
    """
    ir2 = compile_to_ir(cse_src2)
    opt2 = cse(ir2)
    add_ops2 = [inst for inst in opt2 if inst[0] == "add"]
    assert len(add_ops2) == 2


def test_liveness_handwritten_loop() -> None:
    """Verify backward dataflow liveness analysis on a small loop."""
    ir = [
        ("mov", "x", 0, None),
        ("label", None, "L_head", None),
        ("lt", "t0", "x", 10),
        ("jz", None, "t0", "L_exit"),
        ("add", "x", "x", 1),
        ("jmp", None, "L_head", None),
        ("label", None, "L_exit", None),
        ("print", None, "x", None),
        ("halt", None, None, None),
    ]
    live_before = live_at_each_instruction(ir)
    # x must be live entering the loop body and before print
    assert "x" in live_before[2]  # lt t0, x, 10
    assert "x" in live_before[4]  # add x, x, 1
    assert "x" in live_before[5]  # jmp L_head
    assert "x" in live_before[7]  # print x
    # t0 is defined at 2 and used at 3; live entering 3
    assert "t0" in live_before[3]


def test_benchmark_programs_all_passes_correctness() -> None:
    """Verify semantic equivalence of unoptimized IR against individual passes and the full pipeline."""
    for name, src in PROGRAMS.items():
        base_ir = compile_to_ir(src)
        ref_out, _, _ = run_ir(base_ir)

        for p_name, p_fn in PASS_REGISTRY.items():
            opt_ir = p_fn(base_ir)
            opt_out, _, _ = run_ir(opt_ir)
            assert opt_out == ref_out, f"Program {name} failed with pass {p_name}"

        all_ir, _ = optimize_all(base_ir)
        all_out, _, _ = run_ir(all_ir)
        assert all_out == ref_out, f"Program {name} failed with pipeline 'all'"


def test_regalloc_ilp_vs_bruteforce_and_heuristics() -> None:
    """Verify ILP against Brute Force on 20 random small problem instances, and check heuristic bounds."""
    rng = random.Random(12345)

    for trial in range(20):
        n = rng.randint(5, 10)
        ranges: list[LiveRange] = []
        for i in range(n):
            start = rng.randint(0, 15)
            end = start + rng.randint(0, 8)
            weight = rng.choice([1, 10, 100, 1000])
            ranges.append(
                LiveRange(var=f"v{i}", start=start, end=end, weight=weight)
            )

        k = rng.randint(2, 4)

        kept_ilp, spilled_ilp, cost_ilp, _ = allocate_ilp(ranges, k)
        kept_bf, spilled_bf, cost_bf, _ = allocate_bruteforce(ranges, k)
        _, _, cost_ls, _ = allocate_linear_scan(ranges, k)
        _, _, cost_gw, _ = allocate_greedy_weight(ranges, k)

        # ILP must exactly match brute force
        assert (
            abs(cost_ilp - cost_bf) < 1e-5
        ), f"Trial {trial}: ILP {cost_ilp} != BF {cost_bf}"

        # ILP cost must be <= Linear Scan and Greedy-Weight
        assert (
            cost_ilp <= cost_ls + 1e-5
        ), f"Trial {trial}: ILP {cost_ilp} > LS {cost_ls}"
        assert (
            cost_ilp <= cost_gw + 1e-5
        ), f"Trial {trial}: ILP {cost_ilp} > GW {cost_gw}"


def test_machine_code_execution_benchmarks() -> None:
    """Verify machine code execution against reference IR interpreter for k in [2, 3, 8] on two benchmarks."""
    for prog_name in ["fib_iter", "sum_squares"]:
        src = PROGRAMS[prog_name]
        base_ir = compile_to_ir(src)
        opt_ir, _ = optimize_all(base_ir)
        ref_out, _, _ = run_ir(opt_ir)
        ranges = build_live_ranges(opt_ir)

        for k in [2, 3, 8]:
            kept, spilled, _, _ = allocate_ilp(ranges, k)
            reg_map, slot_map = assign_physical_registers(
                ranges, kept, spilled, k
            )
            code = generate_code(opt_ir, (reg_map, slot_map), k)
            m_out, m_dyn, m_ld, m_st, m_alu, _ = run_machine(code)
            assert (
                m_out == ref_out
            ), f"{prog_name} with k={k} machine output {m_out} != ref {ref_out}"


def test_smoke_run_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke test: execute run_all.py under quick mode into tmp_path and verify artifact creation."""
    run_all_path = _TRACK_ROOT / "run_all.py"
    spec = importlib.util.spec_from_file_location("track_f_run_all", run_all_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(
        sys, "argv", ["run_all.py", "--quick", "--out", str(tmp_path)]
    )
    ret = mod.main()
    assert ret == 0

    assert (tmp_path / "SUMMARY.md").exists()
    assert (tmp_path / "f1_pass_effects_raw.csv").exists()
    assert (tmp_path / "f1_pass_effects_summary.csv").exists()
    assert (tmp_path / "f2_regalloc_raw.csv").exists()
    assert (tmp_path / "f2_regalloc_summary.csv").exists()
    assert (tmp_path / "f3_fuzz_summary.csv").exists()
    assert (tmp_path / "example_ir.txt").exists()
    assert (tmp_path / "f1_dynamic_reduction.png").exists()
    assert (tmp_path / "f1_static_vs_dynamic.png").exists()
    assert (tmp_path / "f2_memory_ops_vs_k.png").exists()
    assert (tmp_path / "f2_spill_cost_vs_k.png").exists()

