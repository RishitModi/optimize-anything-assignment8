"""Random Program Generator for Differential Testing and Fuzzing.

Generates syntactically and semantically valid programs in the toy integer language:
- Variables a..e assigned small positive integer constants at program start.
- Arithmetic expressions built from +, -, * with variables/constants, and /, % strictly
  with non-zero constant divisors (preventing runtime ZeroDivisionError).
- Conditional statements (if / else) nested up to depth 2.
- While loops driven by dedicated counters with small constant bounds (guaranteeing termination).
- Print statements to expose internal variable state for differential validation.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Ensure repository root and track root are placed on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

VARS = ["a", "b", "c", "d", "e"]


class Fuzzer:
    """Configurable syntax-directed program generator."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.loop_counter = 0

    def gen_primary(self) -> str:
        if self.rng.random() < 0.6:
            return self.rng.choice(VARS)
        return str(self.rng.randint(0, 10))

    def gen_expr(self, max_depth: int = 1) -> str:
        if max_depth <= 0 or self.rng.random() < 0.4:
            return self.gen_primary()

        op = self.rng.choice(["+", "-", "*", "/", "%"])
        if op in ("/", "%"):
            # Division and modulo strictly require non-zero constant divisor
            left = self.gen_expr(max_depth - 1)
            divisor = self.rng.choice([1, 2, 3, 4, 8, 16])
            return f"({left} {op} {divisor})"
        else:
            left = self.gen_expr(max_depth - 1)
            right = self.gen_expr(max_depth - 1)
            return f"({left} {op} {right})"

    def gen_cond(self) -> str:
        op = self.rng.choice(["==", "!=", "<", "<=", ">", ">="])
        left = self.gen_primary()
        right = self.gen_primary()
        return f"{left} {op} {right}"

    def gen_stmts(self, count: int, depth: int) -> list[str]:
        lines: list[str] = []
        for _ in range(count):
            choice = self.rng.random()
            if choice < 0.5 or depth >= 2:
                # Assignment or print
                if self.rng.random() < 0.8:
                    var = self.rng.choice(VARS)
                    expr = self.gen_expr(max_depth=1)
                    lines.append(f"{var} = {expr};")
                else:
                    lines.append(f"print {self.rng.choice(VARS)};")
            elif choice < 0.8:
                # If / Else block
                cond = self.gen_cond()
                then_stmts = self.gen_stmts(count=self.rng.randint(1, 2), depth=depth + 1)
                has_else = self.rng.random() < 0.5
                lines.append(f"if ({cond}) {{")
                for s in then_stmts:
                    lines.append(f"    {s}")
                if has_else:
                    else_stmts = self.gen_stmts(
                        count=self.rng.randint(1, 2), depth=depth + 1
                    )
                    lines.append("} else {")
                    for s in else_stmts:
                        lines.append(f"    {s}")
                lines.append("}")
            else:
                # Bounded While loop
                lp_name = f"lp{self.loop_counter}"
                self.loop_counter += 1
                bound = self.rng.randint(2, 4)
                # Ensure textual assignment before loop
                lines.append(f"{lp_name} = 0;")
                body_stmts = self.gen_stmts(count=self.rng.randint(1, 2), depth=depth + 1)
                lines.append(f"while ({lp_name} < {bound}) {{")
                for s in body_stmts:
                    lines.append(f"    {s}")
                lines.append(f"    {lp_name} = {lp_name} + 1;")
                lines.append("}")
        return lines

    def generate(self) -> str:
        # Pre-assign all five primary variables with small constants
        lines: list[str] = [
            f"a = {self.rng.randint(1, 8)};",
            f"b = {self.rng.randint(1, 8)};",
            f"c = {self.rng.randint(1, 8)};",
            f"d = {self.rng.randint(1, 8)};",
            f"e = {self.rng.randint(1, 8)};",
        ]

        # Generate main statement body
        body = self.gen_stmts(count=self.rng.randint(6, 10), depth=0)
        lines.extend(body)

        # Append final print statements for all variables
        for v in VARS:
            lines.append(f"print {v};")

        return "\n".join(lines) + "\n"


def generate(seed: int) -> str:
    """Generate deterministic random program source text given an integer seed."""
    fuzzer = Fuzzer(seed)
    return fuzzer.generate()

