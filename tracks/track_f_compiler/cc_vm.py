"""Register-Machine Code Generator and Virtual Machine (VM).

Translates 3-address IR to machine code for a target architecture with:
- k allocatable general-purpose registers: r0, ..., r(k-1)
- 2 dedicated scratch registers: s0, s1 (used for load/store operand buffering)
- Stack/memory slots for spilled variables
- Resolved direct branch targets (instruction indices)

Provides:
- generate_code(ir, allocation, k): emits machine instructions
- run_machine(code, step_limit): simulates the machine, tracking instruction and memory operation counts.
"""

from __future__ import annotations

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

from cc_ir import StepLimitExceeded
from cc_lang import Instruction, eval_op

MachineOperand = tuple[str, Any]
MachineInstruction = tuple[str, Any, Any, Any]

ALU_OPS = {
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
}


def generate_code(
    ir: list[Instruction],
    allocation: Any,
    k: int,
) -> list[MachineInstruction]:
    """Translate linear IR instructions into machine instructions using the assigned register allocation.

    Spilled source operands are loaded into scratch registers s0 / s1 (ld s, slot).
    Spilled destination operands are computed into scratch register s0 and stored (st slot, s0).
    Label names are resolved to direct instruction indices.
    """
    if isinstance(allocation, tuple):
        reg_map, slot_map = allocation[0], allocation[1]
    elif hasattr(allocation, "reg_map"):
        reg_map = allocation.reg_map
        slot_map = allocation.slot_map
    else:
        reg_map, slot_map = {}, {}

    machine_code: list[MachineInstruction] = []
    label_pc: dict[str, int] = {}

    for inst in ir:
        op, dst, a, b = inst

        if op == "label":
            label_name = a
            label_pc[label_name] = len(machine_code)
            continue

        if op == "jmp":
            machine_code.append(("jmp", None, a, None))
            continue

        if op == "halt":
            machine_code.append(("halt", None, None, None))
            continue

        # Resolve first source operand 'a'
        opnd_a: MachineOperand | None = None
        if a is not None:
            if isinstance(a, int):
                opnd_a = ("imm", a)
            elif a in reg_map:
                opnd_a = ("r", reg_map[a])
            elif a in slot_map:
                machine_code.append(("ld", ("s", 0), slot_map[a], None))
                opnd_a = ("s", 0)
            else:
                opnd_a = ("imm", 0)

        # Resolve second source operand 'b' (for binary ops)
        opnd_b: MachineOperand | None = None
        if b is not None and op != "jz":
            if isinstance(b, int):
                opnd_b = ("imm", b)
            elif b in reg_map:
                opnd_b = ("r", reg_map[b])
            elif b in slot_map:
                machine_code.append(("ld", ("s", 1), slot_map[b], None))
                opnd_b = ("s", 1)
            else:
                opnd_b = ("imm", 0)

        if op == "print":
            machine_code.append(("print", None, opnd_a, None))
            continue

        if op == "jz":
            target_label = b
            machine_code.append(("jz", None, opnd_a, target_label))
            continue

        # Destination handling for assignment / ALU ops
        if dst in reg_map:
            dst_loc: MachineOperand = ("r", reg_map[dst])
            spilled_dst = False
        else:
            dst_loc = ("s", 0)
            spilled_dst = True

        machine_code.append((op, dst_loc, opnd_a, opnd_b))

        if spilled_dst and dst in slot_map:
            machine_code.append(("st", slot_map[dst], ("s", 0), None))

    # Resolve labels to machine code indices
    resolved_code: list[MachineInstruction] = []
    for inst in machine_code:
        op, dst, a, b = inst
        if op == "jmp":
            target_lbl = a
            target_idx = (
                label_pc[target_lbl] if target_lbl in label_pc else len(machine_code)
            )
            resolved_code.append(("jmp", None, target_idx, None))
        elif op == "jz":
            target_lbl = b
            target_idx = (
                label_pc[target_lbl] if target_lbl in label_pc else len(machine_code)
            )
            resolved_code.append(("jz", None, a, target_idx))
        else:
            resolved_code.append(inst)

    return resolved_code


def run_machine(
    code: list[MachineInstruction],
    step_limit: int = 5_000_000,
) -> tuple[list[int], int, int, int, int, float]:
    """Execute machine code on a register/memory simulator.

    Returns:
        (output list, total dynamic instructions, dynamic loads, dynamic stores, dynamic ALU ops, wall time).
    """
    reg_file: dict[int, int] = {}
    scratch: dict[int, int] = {0: 0, 1: 0}
    memory: dict[int, int] = {}
    output: list[int] = []

    dynamic_total = 0
    dynamic_loads = 0
    dynamic_stores = 0
    dynamic_alu = 0
    pc = 0

    def read_operand(opnd: MachineOperand) -> int:
        kind, val = opnd
        if kind == "imm":
            return int(val)
        elif kind == "r":
            return reg_file.get(int(val), 0)
        elif kind == "s":
            return scratch.get(int(val), 0)
        raise ValueError(f"Unknown operand kind: {kind}")

    def write_dest(dst: MachineOperand, val: int) -> None:
        kind, idx = dst
        if kind == "r":
            reg_file[int(idx)] = val
        elif kind == "s":
            scratch[int(idx)] = val
        else:
            raise ValueError(f"Cannot write to non-register destination: {kind}")

    start_time = time.perf_counter()

    while pc < len(code):
        dynamic_total += 1
        if dynamic_total > step_limit:
            raise StepLimitExceeded(
                f"Machine step limit {step_limit} exceeded during execution"
            )

        inst = code[pc]
        op = inst[0]

        if op == "halt":
            break
        elif op == "ld":
            dynamic_loads += 1
            dst_reg = inst[1]
            slot = inst[2]
            write_dest(dst_reg, memory.get(slot, 0))
            pc += 1
        elif op == "st":
            dynamic_stores += 1
            slot = inst[1]
            src_reg = inst[2]
            memory[slot] = read_operand(src_reg)
            pc += 1
        elif op == "mov":
            dst = inst[1]
            val = read_operand(inst[2])
            write_dest(dst, val)
            pc += 1
        elif op == "neg":
            dynamic_alu += 1
            dst = inst[1]
            val = read_operand(inst[2])
            write_dest(dst, eval_op("neg", val, None))
            pc += 1
        elif op in ALU_OPS:
            dynamic_alu += 1
            dst = inst[1]
            val_a = read_operand(inst[2])
            val_b = read_operand(inst[3])
            write_dest(dst, eval_op(op, val_a, val_b))
            pc += 1
        elif op == "print":
            val = read_operand(inst[2])
            output.append(val)
            pc += 1
        elif op == "jmp":
            pc = inst[2]
        elif op == "jz":
            cond_val = read_operand(inst[2])
            if cond_val == 0:
                pc = inst[3]
            else:
                pc += 1
        else:
            raise ValueError(f"Unknown machine opcode: {op}")

    wall_time = time.perf_counter() - start_time
    return (
        output,
        dynamic_total,
        dynamic_loads,
        dynamic_stores,
        dynamic_alu,
        wall_time,
    )

