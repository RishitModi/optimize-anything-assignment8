"""Intermediate Representation (IR) Optimization Passes.

Implements the four core optimization passes:
- const_fold_prop: Local constant and copy propagation, constant folding, and branch folding.
- algebraic_strength_reduction: Identity elimination, constant multiplication/division/modulo
  reduction to shifts and bitwise ANDs, and zeroing identities.
- cse: Local Common Subexpression Elimination via local value numbering.
- dce: Dead Code Elimination (pure unused defs via CFG liveness, unreachable code, redundant jumps/labels).

Also provides pass registry and fixed-point iterative optimization harness with timing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

# Ensure repository root and track root are placed on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cc_ir import (
    build_cfg,
    cfg_to_ir,
    live_after_each_instruction,
    pure,
)
from cc_lang import Instruction, eval_op

PassFunction = Callable[[list[Instruction]], list[Instruction]]


# ---------------------------------------------------------------------------
# Helper: Power of Two Check
# ---------------------------------------------------------------------------


def is_power_of_two(n: Any) -> int | None:
    """If n is a positive integer power of two (n = 2^k), return k; else None."""
    if isinstance(n, int) and n > 0 and (n & (n - 1)) == 0:
        return n.bit_length() - 1
    return None


# ---------------------------------------------------------------------------
# Pass (a): Constant Folding & Propagation
# ---------------------------------------------------------------------------


def const_fold_prop(ir: list[Instruction]) -> list[Instruction]:
    """Propagate constants and copies within each basic block, fold constant ops, and fold branches."""
    blocks = build_cfg(ir)

    for block in blocks:
        env: dict[str, int | str] = {}
        new_insts: list[Instruction] = []

        def resolve(val: Any) -> Any:
            # Single lookup per invariant, protected by bounded loop guard
            if not (isinstance(val, str) and val in env):
                return val
            curr = val
            limit = len(env) + 1
            steps = 0
            while isinstance(curr, str) and curr in env:
                steps += 1
                if steps > limit:
                    raise RuntimeError("copy chain cycle")
                curr = env[curr]
            return curr

        def invalidate(v: str) -> None:
            env.pop(v, None)
            to_remove = [k for k, val in env.items() if val == v]
            for k in to_remove:
                del env[k]

        for inst in block.instructions:
            op, dst, a, b = inst

            if op == "mov":
                if isinstance(a, str):
                    src2 = env.get(a, a)
                    src2 = resolve(src2)
                else:
                    src2 = a

                invalidate(dst)
                if src2 == dst:
                    # No-op self copy: record nothing in env; preserve instruction
                    new_insts.append(("mov", dst, src2, None))
                else:
                    env[dst] = src2
                    new_insts.append(("mov", dst, src2, None))

            elif op == "neg":
                resolved_a = resolve(a)
                invalidate(dst)
                if isinstance(resolved_a, int):
                    folded = eval_op("neg", resolved_a, None)
                    env[dst] = folded
                    new_insts.append(("mov", dst, folded, None))
                else:
                    new_insts.append(("neg", dst, resolved_a, None))

            elif op in (
                "add",
                "sub",
                "mul",
                "div",
                "mod",
                "shl",
                "shr",
                "and",
                "lt",
                "le",
                "gt",
                "ge",
                "eq",
                "ne",
            ):
                resolved_a = resolve(a)
                resolved_b = resolve(b)
                invalidate(dst)

                if isinstance(resolved_a, int) and isinstance(resolved_b, int):
                    # Never fold div or mod when divisor is zero
                    if op in ("div", "mod") and resolved_b == 0:
                        new_insts.append((op, dst, resolved_a, resolved_b))
                    else:
                        folded = eval_op(op, resolved_a, resolved_b)
                        env[dst] = folded
                        new_insts.append(("mov", dst, folded, None))
                else:
                    new_insts.append((op, dst, resolved_a, resolved_b))

            elif op == "jz":
                resolved_a = resolve(a)
                target = b
                if isinstance(resolved_a, int):
                    if resolved_a == 0:
                        # Condition is false (0), branch taken: convert to unconditional jmp
                        new_insts.append(("jmp", None, target, None))
                    else:
                        # Condition is true (!= 0), branch never taken: delete branch
                        pass
                else:
                    new_insts.append(("jz", None, resolved_a, target))

            elif op == "print":
                resolved_a = resolve(a)
                new_insts.append(("print", None, resolved_a, None))

            elif op in ("label", "jmp", "halt"):
                new_insts.append(inst)

            else:
                if dst is not None and isinstance(dst, str):
                    invalidate(dst)
                new_insts.append(inst)

        block.instructions = new_insts

    return cfg_to_ir(blocks)


# ---------------------------------------------------------------------------
# Pass (b): Algebraic Strength Reduction
# ---------------------------------------------------------------------------


def algebraic_strength_reduction(ir: list[Instruction]) -> list[Instruction]:
    """Replace costly arithmetic operations with cheaper equivalents, identities, and shifts."""
    new_ir: list[Instruction] = []

    for inst in ir:
        op, dst, a, b = inst

        if op == "add":
            # x + 0 -> mov x, 0 + x -> mov x
            if b == 0:
                new_ir.append(("mov", dst, a, None))
            elif a == 0:
                new_ir.append(("mov", dst, b, None))
            else:
                new_ir.append(inst)

        elif op == "sub":
            # x - 0 -> mov x, x - x -> 0
            if b == 0:
                new_ir.append(("mov", dst, a, None))
            elif a == b and not isinstance(a, int):
                new_ir.append(("mov", dst, 0, None))
            else:
                new_ir.append(inst)

        elif op == "mul":
            # x * 0 -> 0, 0 * x -> 0
            if b == 0 or a == 0:
                new_ir.append(("mov", dst, 0, None))
            # x * 1 -> x, 1 * x -> x
            elif b == 1:
                new_ir.append(("mov", dst, a, None))
            elif a == 1:
                new_ir.append(("mov", dst, b, None))
            # x * 2^k -> shl x, k
            elif is_power_of_two(b) is not None and isinstance(b, int) and b > 1:
                k = is_power_of_two(b)
                new_ir.append(("shl", dst, a, k))
            elif is_power_of_two(a) is not None and isinstance(a, int) and a > 1:
                k = is_power_of_two(a)
                new_ir.append(("shl", dst, b, k))
            else:
                new_ir.append(inst)

        elif op == "div":
            # x / 1 -> mov x
            if b == 1:
                new_ir.append(("mov", dst, a, None))
            # x / 2^k -> shr x, k (valid for Python floor division)
            elif is_power_of_two(b) is not None and isinstance(b, int) and b > 1:
                k = is_power_of_two(b)
                new_ir.append(("shr", dst, a, k))
            else:
                new_ir.append(inst)

        elif op == "mod":
            # x % 2^k -> and x, 2^k - 1 (valid for Python modulo)
            if is_power_of_two(b) is not None and isinstance(b, int) and b > 1:
                mask = b - 1
                new_ir.append(("and", dst, a, mask))
            else:
                new_ir.append(inst)

        else:
            new_ir.append(inst)

    return new_ir


# ---------------------------------------------------------------------------
# Pass (c): Common Subexpression Elimination (CSE via Local Value Numbering)
# ---------------------------------------------------------------------------


def cse(ir: list[Instruction]) -> list[Instruction]:
    """Eliminate redundant computations in each basic block using local value numbering."""
    blocks = build_cfg(ir)

    for block in blocks:
        next_vn = 1
        var_to_vn: dict[str, int] = {}
        const_to_vn: dict[int, int] = {}
        expr_table: dict[tuple[str, Any, Any], tuple[str, int]] = {}
        new_insts: list[Instruction] = []

        def get_vn(val: Any) -> int:
            nonlocal next_vn
            if isinstance(val, int):
                if val not in const_to_vn:
                    const_to_vn[val] = next_vn
                    next_vn += 1
                return const_to_vn[val]
            elif isinstance(val, str):
                if val not in var_to_vn:
                    var_to_vn[val] = next_vn
                    next_vn += 1
                return var_to_vn[val]
            raise ValueError(f"Unexpected operand type: {type(val)}")

        for inst in block.instructions:
            op, dst, a, b = inst

            if op in (
                "add",
                "sub",
                "mul",
                "div",
                "mod",
                "shl",
                "shr",
                "and",
                "lt",
                "le",
                "gt",
                "ge",
                "eq",
                "ne",
                "neg",
            ):
                vn_a = get_vn(a)
                vn_b = get_vn(b) if b is not None else None

                # Commutative operators have canonical operand order
                if op in ("add", "mul", "and", "eq", "ne"):
                    assert vn_b is not None
                    key = (op, min(vn_a, vn_b), max(vn_a, vn_b))
                elif op == "neg":
                    key = (op, vn_a, None)
                else:
                    key = (op, vn_a, vn_b)

                # Check if available
                if key in expr_table:
                    holder_var, holder_vn = expr_table[key]
                    if var_to_vn.get(holder_var) == holder_vn:
                        # Reuse computed value
                        # Invalidate any entries whose holder was dst
                        for k in list(expr_table.keys()):
                            if expr_table[k][0] == dst:
                                del expr_table[k]
                        var_to_vn[dst] = holder_vn
                        new_insts.append(("mov", dst, holder_var, None))
                        continue

                # Compute new expression
                for k in list(expr_table.keys()):
                    if expr_table[k][0] == dst:
                        del expr_table[k]
                new_vn = next_vn
                next_vn += 1
                var_to_vn[dst] = new_vn

                # Record in table if operation cannot trap
                if op not in ("div", "mod") or (isinstance(b, int) and b != 0):
                    expr_table[key] = (dst, new_vn)

                new_insts.append(inst)

            elif op == "mov":
                for k in list(expr_table.keys()):
                    if expr_table[k][0] == dst:
                        del expr_table[k]
                var_to_vn[dst] = get_vn(a)
                new_insts.append(inst)

            else:
                if dst is not None and isinstance(dst, str):
                    for k in list(expr_table.keys()):
                        if expr_table[k][0] == dst:
                            del expr_table[k]
                    var_to_vn[dst] = next_vn
                    next_vn += 1
                new_insts.append(inst)

        block.instructions = new_insts

    return cfg_to_ir(blocks)


# ---------------------------------------------------------------------------
# Pass (d): Dead Code Elimination (DCE)
# ---------------------------------------------------------------------------


def dce(ir: list[Instruction]) -> list[Instruction]:
    """Eliminate dead instructions, unreachable code, redundant jumps, and unused labels."""
    current_ir = list(ir)
    changed = True
    max_iters = 10 * (len(ir) + 10)
    iters = 0

    while changed:
        iters += 1
        if iters > max_iters:
            raise RuntimeError(
                f"DCE fixed-point iteration cap exceeded ({max_iters} iterations)"
            )
        changed = False

        # 1. Identify referenced labels
        referenced_labels = set()
        for inst in current_ir:
            if inst[0] == "jmp":
                referenced_labels.add(inst[2])
            elif inst[0] == "jz":
                referenced_labels.add(inst[3])

        # 2. Eliminate unreachable code after unconditional jmp or halt
        filtered_ir: list[Instruction] = []
        unreachable = False
        for inst in current_ir:
            op = inst[0]
            if unreachable:
                if op == "label" and inst[2] in referenced_labels:
                    unreachable = False
                    filtered_ir.append(inst)
                else:
                    changed = True
                    continue
            else:
                filtered_ir.append(inst)
                if op in ("jmp", "halt"):
                    unreachable = True

        # 3. Eliminate jumps to immediately following label
        compact_ir: list[Instruction] = []
        for i, inst in enumerate(filtered_ir):
            if inst[0] == "jmp":
                target = inst[2]
                if (
                    i + 1 < len(filtered_ir)
                    and filtered_ir[i + 1][0] == "label"
                    and filtered_ir[i + 1][2] == target
                ):
                    changed = True
                    continue
            compact_ir.append(inst)

        # 4. Eliminate unreferenced labels
        ref_labels2 = set()
        for inst in compact_ir:
            if inst[0] == "jmp":
                ref_labels2.add(inst[2])
            elif inst[0] == "jz":
                ref_labels2.add(inst[3])

        retained_ir: list[Instruction] = []
        for inst in compact_ir:
            if inst[0] == "label" and inst[2] not in ref_labels2:
                changed = True
                continue
            retained_ir.append(inst)

        # 5. Pure dead instruction elimination via CFG liveness
        live_after_map = live_after_each_instruction(retained_ir)
        live_pruned_ir: list[Instruction] = []
        for idx, inst in enumerate(retained_ir):
            op, dst, a, b = inst
            if pure(inst) and dst is not None and isinstance(dst, str):
                if dst not in live_after_map[idx]:
                    # Destination is not live afterwards; safe to remove
                    changed = True
                    continue
            live_pruned_ir.append(inst)

        if len(live_pruned_ir) != len(current_ir) or live_pruned_ir != current_ir:
            changed = True
            current_ir = live_pruned_ir

    return current_ir


# ---------------------------------------------------------------------------
# Pass Registry and Fixed-Point Driver
# ---------------------------------------------------------------------------

PASS_REGISTRY: dict[str, PassFunction] = {
    "const_fold_prop": const_fold_prop,
    "algebraic_strength_reduction": algebraic_strength_reduction,
    "cse": cse,
    "dce": dce,
}

ALL_PASSES: list[PassFunction] = [
    const_fold_prop,
    algebraic_strength_reduction,
    cse,
    const_fold_prop,
    dce,
]


def optimize(
    ir: list[Instruction],
    passes: list[PassFunction],
    max_iters: int = 10,
) -> tuple[list[Instruction], dict[str, float]]:
    """Iteratively apply an ordered list of passes to IR until a fixed point or max_iters.

    Returns:
        (optimized IR instruction list, dictionary of cumulative wall time per pass name).
    """
    current_ir = list(ir)
    pass_times: dict[str, float] = {p.__name__: 0.0 for p in passes}

    for _ in range(max_iters):
        changed = False
        for p in passes:
            t0 = time.perf_counter()
            new_ir = p(current_ir)
            elapsed = time.perf_counter() - t0
            pass_times[p.__name__] += elapsed

            if new_ir != current_ir:
                changed = True
                current_ir = new_ir

        if not changed:
            break

    return current_ir, pass_times


def optimize_all(
    ir: list[Instruction],
    max_iters: int = 10,
) -> tuple[list[Instruction], dict[str, float]]:
    """Run standard full optimization pipeline ("all") to fixed point."""
    return optimize(ir, ALL_PASSES, max_iters=max_iters)

