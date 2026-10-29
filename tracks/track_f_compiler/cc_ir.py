"""Control Flow Graph (CFG), Basic Blocks, Liveness Analysis, and IR Interpreter.

Provides:
- BasicBlock and CFG construction from linear three-address IR and reverse conversion.
- Iterative backward dataflow liveness analysis (live_in, live_out, and instruction-level live sets).
- Loop depth calculation for weight estimation in register allocation.
- run_ir: reference IR interpreter returning (output, dynamic_inst_count, wall_time).
- pure(op): purity analysis respecting exception semantics of div/mod.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure repository root and track root are placed on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cc_lang import Instruction, eval_op


def pure(op_or_inst: str | Instruction, b: Any = None) -> bool:
    """Determine if an operation or instruction is pure (side-effect free and cannot trap).

    Instructions other than print, jmp, jz, label, and halt are pure, EXCEPT div and mod,
    which are only treated as pure/removable when the divisor operand is a non-zero integer constant.
    """
    if isinstance(op_or_inst, tuple):
        op = op_or_inst[0]
        divisor = op_or_inst[3]
    else:
        op = op_or_inst
        divisor = b

    if op in ("print", "jmp", "jz", "label", "halt"):
        return False
    if op in ("div", "mod"):
        return isinstance(divisor, int) and divisor != 0
    return True


def get_def_use(inst: Instruction) -> tuple[set[str], set[str]]:
    """Return the set of variables defined and used by a single IR instruction."""
    op, dst, a, b = inst
    defs: set[str] = set()
    uses: set[str] = set()

    if dst is not None and isinstance(dst, str):
        defs.add(dst)

    if op == "print":
        if isinstance(a, str):
            uses.add(a)
    elif op == "jz":
        if isinstance(a, str):
            uses.add(a)
        # b is a jump label, not a variable
    elif op in ("jmp", "label", "halt"):
        pass
    elif op in ("mov", "neg"):
        if isinstance(a, str):
            uses.add(a)
    else:
        # Binary ALU / comparison / bitwise ops
        if isinstance(a, str):
            uses.add(a)
        if isinstance(b, str):
            uses.add(b)

    return defs, uses


@dataclass
class BasicBlock:
    """A maximal sequence of consecutive instructions with single-entry and single-exit."""

    block_id: int
    instructions: list[Instruction] = field(default_factory=list)
    predecessors: list[BasicBlock] = field(default_factory=list)
    successors: list[BasicBlock] = field(default_factory=list)
    live_in: set[str] = field(default_factory=set)
    live_out: set[str] = field(default_factory=set)
    def_set: set[str] = field(default_factory=set)
    use_set: set[str] = field(default_factory=set)

    def compute_local_def_use(self) -> None:
        """Compute the variables used before definition and defined within this block."""
        self.def_set.clear()
        self.use_set.clear()
        for inst in self.instructions:
            defs, uses = get_def_use(inst)
            for u in uses:
                if u not in self.def_set:
                    self.use_set.add(u)
            for d in defs:
                self.def_set.add(d)


def build_cfg(ir: list[Instruction]) -> list[BasicBlock]:
    """Partition linear IR into basic blocks and establish control-flow edges."""
    if not ir:
        return []

    # Identify leader instruction indices
    leaders = {0}
    for idx, inst in enumerate(ir):
        op = inst[0]
        if op == "label":
            leaders.add(idx)
        elif op in ("jmp", "jz", "halt"):
            if idx + 1 < len(ir):
                leaders.add(idx + 1)

    sorted_leaders = sorted(leaders)
    blocks: list[BasicBlock] = []
    label_to_block: dict[str, BasicBlock] = {}

    for i, start_idx in enumerate(sorted_leaders):
        end_idx = sorted_leaders[i + 1] if i + 1 < len(sorted_leaders) else len(ir)
        block_insts = ir[start_idx:end_idx]
        b = BasicBlock(block_id=i, instructions=block_insts)
        b.compute_local_def_use()
        blocks.append(b)
        # If the block begins with a label, register it for jump target resolution
        if block_insts and block_insts[0][0] == "label":
            label_name = block_insts[0][2]
            label_to_block[label_name] = b

    # Connect CFG edges
    for i, b in enumerate(blocks):
        if not b.instructions:
            continue
        last = b.instructions[-1]
        op = last[0]
        if op == "jmp":
            target = last[2]
            if target in label_to_block:
                succ = label_to_block[target]
                b.successors.append(succ)
                succ.predecessors.append(b)
        elif op == "jz":
            target = last[3]
            if target in label_to_block:
                succ = label_to_block[target]
                b.successors.append(succ)
                succ.predecessors.append(b)
            # Fall-through to next linear block
            if i + 1 < len(blocks):
                fallthrough = blocks[i + 1]
                b.successors.append(fallthrough)
                fallthrough.predecessors.append(b)
        elif op == "halt":
            # No successors
            pass
        else:
            # Fall-through to next linear block
            if i + 1 < len(blocks):
                fallthrough = blocks[i + 1]
                b.successors.append(fallthrough)
                fallthrough.predecessors.append(b)

    return blocks


def cfg_to_ir(blocks: list[BasicBlock]) -> list[Instruction]:
    """Convert a CFG basic block list back into a linear sequence of instructions."""
    ir: list[Instruction] = []
    for b in blocks:
        ir.extend(b.instructions)
    return ir


def compute_liveness(blocks: list[BasicBlock]) -> None:
    """Perform iterative backward dataflow analysis to compute live_in and live_out sets."""
    for b in blocks:
        b.compute_local_def_use()
        b.live_in.clear()
        b.live_out.clear()

    changed = True
    while changed:
        changed = False
        # Iterate in reverse topological/linear order for efficient backward convergence
        for b in reversed(blocks):
            new_out: set[str] = set()
            for succ in b.successors:
                new_out.update(succ.live_in)

            new_in = b.use_set | (new_out - b.def_set)
            if new_in != b.live_in or new_out != b.live_out:
                b.live_in = new_in
                b.live_out = new_out
                changed = True


def live_at_each_instruction(ir: list[Instruction]) -> list[set[str]]:
    """Return the set of variables live immediately before each instruction in the IR."""
    if not ir:
        return []

    blocks = build_cfg(ir)
    compute_liveness(blocks)

    live_before_map: list[set[str]] = [set() for _ in range(len(ir))]
    inst_idx = 0

    for b in blocks:
        curr_live = set(b.live_out)
        block_live_before: list[set[str]] = []
        for inst in reversed(b.instructions):
            defs, uses = get_def_use(inst)
            live_before = (curr_live - defs) | uses
            block_live_before.append(set(live_before))
            curr_live = live_before

        block_live_before.reverse()
        for s in block_live_before:
            live_before_map[inst_idx] = s
            inst_idx += 1

    return live_before_map


def live_after_each_instruction(ir: list[Instruction]) -> list[set[str]]:
    """Return the set of variables live immediately after each instruction in the IR."""
    if not ir:
        return []

    blocks = build_cfg(ir)
    compute_liveness(blocks)

    live_after_map: list[set[str]] = [set() for _ in range(len(ir))]
    inst_idx = 0

    for b in blocks:
        curr_live = set(b.live_out)
        block_live_after: list[set[str]] = []
        for inst in reversed(b.instructions):
            block_live_after.append(set(curr_live))
            defs, uses = get_def_use(inst)
            curr_live = (curr_live - defs) | uses

        block_live_after.reverse()
        for s in block_live_after:
            live_after_map[inst_idx] = s
            inst_idx += 1

    return live_after_map


def loop_depth_per_instruction(ir: list[Instruction]) -> list[int]:
    """Calculate loop nest depth for every instruction based on backward jumps.

    For every backward jump (jmp to a label that appears earlier in the linear order),
    all instructions between that label and the jump instruction are inside one loop.
    The loop depth of an instruction is the number of such intervals containing it.
    """
    label_positions: dict[str, int] = {}
    for idx, inst in enumerate(ir):
        if inst[0] == "label":
            label_positions[inst[2]] = idx

    loop_intervals: list[tuple[int, int]] = []
    for idx, inst in enumerate(ir):
        if inst[0] == "jmp":
            target = inst[2]
            if target in label_positions and label_positions[target] < idx:
                loop_intervals.append((label_positions[target], idx))

    depths: list[int] = []
    for idx in range(len(ir)):
        count = sum(1 for start, end in loop_intervals if start <= idx <= end)
        depths.append(count)
    return depths


def run_ir(ir: list[Instruction], step_limit: int = 2_000_000) -> tuple[list[int], int, float]:
    """Execute IR program on an abstract register/variable environment.

    Returns:
        (output list, dynamic instruction count excluding labels, execution wall time in seconds).
    Raises:
        RuntimeError: when dynamic instructions exceed step_limit.
        ZeroDivisionError: upon division or modulo by zero.
    """
    label_map: dict[str, int] = {}
    for idx, inst in enumerate(ir):
        if inst[0] == "label":
            label_map[inst[2]] = idx

    env: dict[str, int] = {}
    output: list[int] = []
    dynamic_count = 0
    pc = 0

    start_time = time.perf_counter()

    while pc < len(ir):
        inst = ir[pc]
        op = inst[0]

        if op == "label":
            pc += 1
            continue

        dynamic_count += 1
        if dynamic_count > step_limit:
            raise RuntimeError(f"Step limit {step_limit} exceeded during IR execution")

        if op == "halt":
            break
        elif op == "jmp":
            pc = label_map[inst[2]]
            continue
        elif op == "jz":
            cond_val = env[inst[2]] if isinstance(inst[2], str) else inst[2]
            if cond_val == 0:
                pc = label_map[inst[3]]
                continue
            else:
                pc += 1
                continue
        elif op == "print":
            val = env[inst[2]] if isinstance(inst[2], str) else inst[2]
            output.append(val)
            pc += 1
        elif op == "mov":
            dst = inst[1]
            val = env[inst[2]] if isinstance(inst[2], str) else inst[2]
            env[dst] = val
            pc += 1
        elif op == "neg":
            dst = inst[1]
            val = env[inst[2]] if isinstance(inst[2], str) else inst[2]
            env[dst] = eval_op("neg", val, None)
            pc += 1
        else:
            dst = inst[1]
            val_a = env[inst[2]] if isinstance(inst[2], str) else inst[2]
            val_b = env[inst[3]] if isinstance(inst[3], str) else inst[3]
            env[dst] = eval_op(op, val_a, val_b)
            pc += 1

    wall_time = time.perf_counter() - start_time
    return output, dynamic_count, wall_time

