"""Lexer, AST, recursive-descent parser, and IR generator for a toy integer language.

Language Semantics:
- Arbitrary-precision Python integers.
- '/' and '%' are Python floor division (//) and modulo (%).
- Division or modulo by zero is a runtime error (raises ZeroDivisionError).
- A variable must be assigned textually before it is read, otherwise a
  CompileError with the offending line number is raised.
- A condition is treated as truthy when non-zero, and falsy when zero.
- The 'print' statement appends the evaluated integer expression to an output list.

Three-Address Code (IR):
- Instructions are represented as 4-tuples: (op, dst, a, b), where unused fields are None.
- Operands are integers or variable names (user variables and compiler temporaries t0, t1, ...).
- Supported ops:
    mov(dst, a), add, sub, mul, div, mod, shl, shr, and, lt, le, gt, ge, eq, ne (dst, a, b),
    neg(dst, a), print(a), label(name), jmp(label), jz(a, label), halt.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure repository root and track root are placed on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class CompileError(Exception):
    """Raised when compilation fails (e.g. variable used before textual assignment)."""

    def __init__(self, message: str, line: int = 1) -> None:
        super().__init__(f"Line {line}: {message}")
        self.message = message
        self.line = line


# ---------------------------------------------------------------------------
# Shared Semantic Evaluation
# ---------------------------------------------------------------------------


def eval_op(op: str, a: int, b: int | None = None) -> int:
    """Single shared implementation of arithmetic, bitwise, and comparison semantics.

    Used uniformly across the IR interpreter, constant folding pass, and the register VM.
    """
    if op == "neg":
        return -a
    if op == "mov":
        return a
    if b is None:
        raise ValueError(f"Binary operator {op} called without second operand")
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        if b == 0:
            raise ZeroDivisionError("division by zero")
        return a // b
    if op == "mod":
        if b == 0:
            raise ZeroDivisionError("modulo by zero")
        return a % b
    if op == "shl":
        if b < 0:
            raise ValueError("negative shift count")
        return a << b
    if op == "shr":
        if b < 0:
            raise ValueError("negative shift count")
        return a >> b
    if op == "and":
        return a & b
    if op == "lt":
        return 1 if a < b else 0
    if op == "le":
        return 1 if a <= b else 0
    if op == "gt":
        return 1 if a > b else 0
    if op == "ge":
        return 1 if a >= b else 0
    if op == "eq":
        return 1 if a == b else 0
    if op == "ne":
        return 1 if a != b else 0
    raise ValueError(f"Unknown operator: {op}")


# ---------------------------------------------------------------------------
# AST Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ASTNode:
    """Base class for AST nodes with source line tracking."""

    line: int


@dataclass
class Expr(ASTNode):
    """Base expression node."""

    pass


@dataclass
class Const(Expr):
    """Integer literal."""

    value: int


@dataclass
class Var(Expr):
    """Variable identifier reference."""

    name: str


@dataclass
class UnaryOp(Expr):
    """Unary operator expression (e.g. -x)."""

    op: str
    operand: Expr


@dataclass
class BinaryOp(Expr):
    """Binary operator expression."""

    op: str
    left: Expr
    right: Expr


@dataclass
class Stmt(ASTNode):
    """Base statement node."""

    pass


@dataclass
class Assign(Stmt):
    """Assignment statement: IDENT = expr;"""

    name: str
    expr: Expr


@dataclass
class Print(Stmt):
    """Print statement: print expr;"""

    expr: Expr


@dataclass
class If(Stmt):
    """Conditional statement: if (cond) then_block [else else_block]"""

    cond: Expr
    then_block: list[Stmt]
    else_block: list[Stmt] | None


@dataclass
class While(Stmt):
    """Loop statement: while (cond) block"""

    cond: Expr
    body: list[Stmt]


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------


@dataclass
class Token:
    """Lexical token containing type, value, and source line number."""

    type: str
    value: Any
    line: int


def tokenize(source: str) -> list[Token]:
    """Tokenize source string into a list of Tokens.

    Recognizes identifiers, integer literals, language keywords (print, if, else, while),
    multi-character and single-character operators, and strips comments starting with '#'.
    """
    token_specification = [
        ("COMMENT", r"#[^\n]*"),
        ("NUMBER", r"\d+"),
        ("EQ", r"=="),
        ("NE", r"!="),
        ("LE", r"<="),
        ("GE", r">="),
        ("LT", r"<"),
        ("GT", r">"),
        ("ASSIGN", r"="),
        ("PLUS", r"\+"),
        ("MINUS", r"-"),
        ("MUL", r"\*"),
        ("DIV", r"/"),
        ("MOD", r"%"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("LBRACE", r"\{"),
        ("RBRACE", r"\}"),
        ("SEMI", r";"),
        ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"),
        ("NEWLINE", r"\n"),
        ("SKIP", r"[ \t\r]+"),
    ]
    tok_regex = "|".join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_specification)
    line_num = 1
    tokens: list[Token] = []

    for mo in re.finditer(tok_regex, source):
        kind = mo.lastgroup
        value = mo.group()
        if kind == "NEWLINE":
            line_num += 1
        elif kind == "SKIP" or kind == "COMMENT":
            continue
        elif kind == "NUMBER":
            tokens.append(Token("INT", int(value), line_num))
        elif kind == "IDENT":
            if value in ("print", "if", "else", "while"):
                tokens.append(Token(value.upper(), value, line_num))
            else:
                tokens.append(Token("IDENT", value, line_num))
        else:
            tokens.append(Token(kind, value, line_num))

    tokens.append(Token("EOF", None, line_num))
    return tokens


# ---------------------------------------------------------------------------
# Parser with Textual Assignment Checking
# ---------------------------------------------------------------------------


class Parser:
    """Recursive-descent parser enforcing textual assignment before variable use."""

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.defined_vars: set[str] = set()

    def current_token(self) -> Token:
        return self.tokens[self.pos]

    def match(self, expected_type: str) -> Token:
        tok = self.current_token()
        if tok.type != expected_type:
            raise CompileError(
                f"Expected token {expected_type}, got {tok.type} ('{tok.value}')",
                line=tok.line,
            )
        self.pos += 1
        return tok

    def parse_program(self) -> list[Stmt]:
        stmts: list[Stmt] = []
        while self.current_token().type != "EOF":
            stmts.append(self.parse_stmt())
        return stmts

    def parse_stmt(self) -> Stmt:
        tok = self.current_token()
        if tok.type == "IDENT":
            ident_name = tok.value
            line = tok.line
            self.match("IDENT")
            self.match("ASSIGN")
            expr = self.parse_expr()
            self.match("SEMI")
            self.defined_vars.add(ident_name)
            return Assign(line=line, name=ident_name, expr=expr)
        elif tok.type == "PRINT":
            line = tok.line
            self.match("PRINT")
            expr = self.parse_expr()
            self.match("SEMI")
            return Print(line=line, expr=expr)
        elif tok.type == "IF":
            line = tok.line
            self.match("IF")
            self.match("LPAREN")
            cond = self.parse_expr()
            self.match("RPAREN")
            then_block = self.parse_block()
            else_block = None
            if self.current_token().type == "ELSE":
                self.match("ELSE")
                else_block = self.parse_block()
            return If(line=line, cond=cond, then_block=then_block, else_block=else_block)
        elif tok.type == "WHILE":
            line = tok.line
            self.match("WHILE")
            self.match("LPAREN")
            cond = self.parse_expr()
            self.match("RPAREN")
            body = self.parse_block()
            return While(line=line, cond=cond, body=body)
        else:
            raise CompileError(
                f"Unexpected token in statement: {tok.type} ('{tok.value}')",
                line=tok.line,
            )

    def parse_block(self) -> list[Stmt]:
        self.match("LBRACE")
        stmts: list[Stmt] = []
        while self.current_token().type != "RBRACE":
            if self.current_token().type == "EOF":
                raise CompileError("Unclosed block, reached EOF", line=self.current_token().line)
            stmts.append(self.parse_stmt())
        self.match("RBRACE")
        return stmts

    def parse_expr(self) -> Expr:
        return self.parse_comparison()

    def parse_comparison(self) -> Expr:
        left = self.parse_additive()
        comp_map = {
            "EQ": "==",
            "NE": "!=",
            "LT": "<",
            "LE": "<=",
            "GT": ">",
            "GE": ">=",
        }
        while self.current_token().type in comp_map:
            tok = self.current_token()
            self.pos += 1
            right = self.parse_additive()
            left = BinaryOp(line=tok.line, op=comp_map[tok.type], left=left, right=right)
        return left

    def parse_additive(self) -> Expr:
        left = self.parse_multiplicative()
        add_map = {"PLUS": "+", "MINUS": "-"}
        while self.current_token().type in add_map:
            tok = self.current_token()
            self.pos += 1
            right = self.parse_multiplicative()
            left = BinaryOp(line=tok.line, op=add_map[tok.type], left=left, right=right)
        return left

    def parse_multiplicative(self) -> Expr:
        left = self.parse_unary()
        mul_map = {"MUL": "*", "DIV": "/", "MOD": "%"}
        while self.current_token().type in mul_map:
            tok = self.current_token()
            self.pos += 1
            right = self.parse_unary()
            left = BinaryOp(line=tok.line, op=mul_map[tok.type], left=left, right=right)
        return left

    def parse_unary(self) -> Expr:
        tok = self.current_token()
        if tok.type == "MINUS":
            self.match("MINUS")
            operand = self.parse_unary()
            return UnaryOp(line=tok.line, op="-", operand=operand)
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        tok = self.current_token()
        if tok.type == "INT":
            self.match("INT")
            return Const(line=tok.line, value=tok.value)
        elif tok.type == "IDENT":
            ident = tok.value
            if ident not in self.defined_vars:
                raise CompileError(
                    f"Variable '{ident}' used before textual assignment",
                    line=tok.line,
                )
            self.match("IDENT")
            return Var(line=tok.line, name=ident)
        elif tok.type == "LPAREN":
            self.match("LPAREN")
            expr = self.parse_expr()
            self.match("RPAREN")
            return expr
        else:
            raise CompileError(
                f"Expected primary expression, got {tok.type} ('{tok.value}')",
                line=tok.line,
            )


def parse(source: str) -> list[Stmt]:
    """Parse source string into AST statements, verifying variable definitions."""
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse_program()


# ---------------------------------------------------------------------------
# IR Code Generator
# ---------------------------------------------------------------------------

Instruction = tuple[str, Any, Any, Any]


class IRGenerator:
    """Compiles an AST statement list into 3-address linear intermediate representation."""

    def __init__(self) -> None:
        self.instructions: list[Instruction] = []
        self.temp_count = 0
        self.label_count = 0

    def new_temp(self) -> str:
        t = f"t{self.temp_count}"
        self.temp_count += 1
        return t

    def new_label(self) -> str:
        lbl = f"L{self.label_count}"
        self.label_count += 1
        return lbl

    def emit(self, op: str, dst: Any = None, a: Any = None, b: Any = None) -> None:
        self.instructions.append((op, dst, a, b))

    def compile_program(self, stmts: list[Stmt]) -> list[Instruction]:
        for stmt in stmts:
            self.compile_stmt(stmt)
        self.emit("halt", None, None, None)
        return self.instructions

    def compile_stmt(self, stmt: Stmt) -> None:
        if isinstance(stmt, Assign):
            # Evaluate expression and assign to user variable
            res = self.compile_expr(stmt.expr)
            self.emit("mov", stmt.name, res, None)
        elif isinstance(stmt, Print):
            res = self.compile_expr(stmt.expr)
            self.emit("print", None, res, None)
        elif isinstance(stmt, If):
            cond_res = self.compile_expr(stmt.cond)
            if stmt.else_block:
                l_else = self.new_label()
                l_exit = self.new_label()
                self.emit("jz", None, cond_res, l_else)
                for s in stmt.then_block:
                    self.compile_stmt(s)
                self.emit("jmp", None, l_exit, None)
                self.emit("label", None, l_else, None)
                for s in stmt.else_block:
                    self.compile_stmt(s)
                self.emit("label", None, l_exit, None)
            else:
                l_exit = self.new_label()
                self.emit("jz", None, cond_res, l_exit)
                for s in stmt.then_block:
                    self.compile_stmt(s)
                self.emit("label", None, l_exit, None)
        elif isinstance(stmt, While):
            # Loop structure: head label, condition, jz exit, body, jmp head, exit label
            l_head = self.new_label()
            l_exit = self.new_label()
            self.emit("label", None, l_head, None)
            cond_res = self.compile_expr(stmt.cond)
            self.emit("jz", None, cond_res, l_exit)
            for s in stmt.body:
                self.compile_stmt(s)
            self.emit("jmp", None, l_head, None)
            self.emit("label", None, l_exit, None)

    def compile_expr(self, expr: Expr) -> Any:
        if isinstance(expr, Const):
            return expr.value
        elif isinstance(expr, Var):
            return expr.name
        elif isinstance(expr, UnaryOp):
            val = self.compile_expr(expr.operand)
            t = self.new_temp()
            self.emit("neg", t, val, None)
            return t
        elif isinstance(expr, BinaryOp):
            l = self.compile_expr(expr.left)
            r = self.compile_expr(expr.right)
            op_map = {
                "+": "add",
                "-": "sub",
                "*": "mul",
                "/": "div",
                "%": "mod",
                "==": "eq",
                "!=": "ne",
                "<": "lt",
                "<=": "le",
                ">": "gt",
                ">=": "ge",
            }
            ir_op = op_map[expr.op]
            t = self.new_temp()
            self.emit(ir_op, t, l, r)
            return t
        raise ValueError(f"Unknown expression node: {type(expr)}")


def compile_to_ir(source: str) -> list[Instruction]:
    """Compile program source text to three-address IR instructions."""
    stmts = parse(source)
    gen = IRGenerator()
    return gen.compile_program(stmts)


def format_ir(ir: list[Instruction]) -> str:
    """Pretty-print IR instructions into a readable textual disassembly."""
    lines: list[str] = []
    for inst in ir:
        op, dst, a, b = inst
        if op == "label":
            lines.append(f"{a}:")
        elif op == "mov":
            lines.append(f"    {dst} = {a}")
        elif op == "neg":
            lines.append(f"    {dst} = -{a}")
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
            lines.append(f"    {dst} = {a} {op} {b}")
        elif op == "print":
            lines.append(f"    print {a}")
        elif op == "jmp":
            lines.append(f"    jmp {a}")
        elif op == "jz":
            lines.append(f"    jz {a}, {b}")
        elif op == "halt":
            lines.append("    halt")
        else:
            lines.append(f"    {op} {dst}, {a}, {b}")
    return "\n".join(lines)

