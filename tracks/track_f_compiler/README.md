# Track F: Compiler Optimization

## Objective
The objective of Track F is to formulate, implement, and benchmark principled compiler optimization passes and optimal register allocation for an imperative integer language. We target:
1. Significant static and dynamic instruction count reduction through classical dataflow and algebraic transformations.
2. Minimization of memory traffic (dynamic spill loads and stores) by casting register allocation as an exact 0/1 Integer Linear Program (ILP) and comparing it against standard industrial heuristics (Linear Scan and Greedy-by-Weight).
3. Preservation of strict semantics across all optimization levels and machine code generation, verified by differential testing and fuzzing.

### Language Grammar
```ebnf
program = stmt* ;
stmt    = IDENT "=" expr ";"
        | "print" expr ";"
        | "if" "(" expr ")" block ["else" block]
        | "while" "(" expr ")" block ;
block   = "{" stmt* "}" ;
expr    = comparison ;
comparison     = additive (("==" | "!=" | "<" | "<=" | ">" | ">=") additive)* ;
additive       = multiplicative (("+" | "-") multiplicative)* ;
multiplicative = unary (("*" | "/" | "%") unary)* ;
unary          = "-" unary | primary ;
primary        = INT | IDENT | "(" expr ")" ;
```
Comments begin with `#` and run to the end of the line.

### Language Semantics
- **Arithmetic**: Arbitrary-precision Python integers. `/` and `%` represent Python floor division (`//`) and modulo (`%`).
- **Exceptions**: Division or modulo by zero is a runtime exception (`ZeroDivisionError`).
- **Scope & Assignment**: A variable must be assigned textually before it is read in any expression; violation raises a `CompileError` reporting the offending source line number.
- **Conditionals**: Any non-zero integer evaluates to true; zero evaluates to false.
- **I/O**: The `print` statement evaluates an expression and appends the integer value to an output list.

### Intermediate Representation (Three-Address Code)
The IR is represented as a list of 4-tuples `(op, dst, a, b)` with unused fields set to `None`:
- Data movement: `mov(dst, a)`
- Arithmetic & Bitwise ALU: `add`, `sub`, `mul`, `div`, `mod`, `shl`, `shr`, `and`, `neg(dst, a)`
- Comparisons: `eq`, `ne`, `lt`, `le`, `gt`, `ge` (evaluating to 1 or 0)
- Control Flow: `label(name)`, `jmp(label)`, `jz(a, label)` (branch if operand `a == 0`), `halt`
- Output: `print(a)`
- Evaluation helper: `eval_op(op, a, b)` serves as the single source of truth for runtime semantics across the IR interpreter, constant folding, and the target virtual machine.
- Loop compilation: Emits `head label`, condition evaluation, `jz exit`, loop body, `jmp head`, and `exit label`.

## Baseline
The baseline consists of unoptimized three-address linear IR generated directly by the AST compiler without code elimination, constant folding, or subexpression reuse. In register allocation, baseline performance is established by the Linear Scan heuristic (Poletto & Sarkar, 1999) under constrained physical register budgets ($k \in \{2, 3, 4, 6, 8\}$).

## Formulation
Register allocation is modeled as a 0/1 combinatorial optimization problem over program live ranges.

### Decision Variables
Let $n$ be the number of program live ranges, where live range $i$ corresponds to variable $v_i$ active over the contiguous instruction interval $[start_i, end_i]$ (the convex hull of definition, use, and liveness points).
$$x_i \in \{0, 1\} \quad \forall i \in \{1, \dots, n\}$$
where $x_i = 1$ if live range $i$ is kept in a physical register, and $x_i = 0$ if it is spilled to a stack memory slot.

### Objective Function
Each live range $i$ is assigned a loop-nesting weight:
$$w_i = \sum_{\text{inst } \in \text{ defs}(v_i) \cup \text{ uses}(v_i)} 10^{\text{loop\_depth}(\text{inst})}$$
We maximize total weight kept in registers (equivalently minimizing weighted spill cost):
$$\max \sum_{i=1}^n w_i x_i \iff \min \sum_{i=1}^n w_i (1 - x_i)$$

### Capacity Constraints
At each discrete instruction index $t \in \{0, \dots, T\}$ along the linear program order, the number of active ranges assigned to physical registers cannot exceed capacity $k$:
$$\sum_{i: start_i \le t \le end_i} x_i \le k \quad \forall t \in \{0, \dots, T\}$$

### Integrality & Consecutive-Ones Property
Because each live range is a contiguous interval along the linear instruction sequence, the column corresponding to variable $i$ in the constraint matrix contains a single contiguous block of 1s in rows $start_i, \dots, end_i$. The constraint matrix therefore satisfies the **consecutive-ones property**, making it totally unimodular. Consequently, the extreme points of the continuous Linear Programming (LP) relaxation are guaranteed to be integral, allowing exact solution via polynomial-time interior point or simplex methods.

## Methods

### Optimization Passes
1. **Constant Folding & Copy Propagation (`const_fold_prop`)**:
   Tracks constant values and variable copies within each basic block, propagating them into subsequent operand positions until redefinition. Computes constant expressions at compile time using `eval_op` (preserving runtime `ZeroDivisionError` when the divisor is zero). Folds conditional branches (`jz` on constant 0 becomes `jmp`; `jz` on non-zero constant is deleted).
2. **Algebraic Strength Reduction (`algebraic_strength_reduction`)**:
   Eliminates identity operations ($x+0, x-0, x*1, x/1 \to \text{mov } x$; $x*0 \to \text{mov } 0$; $x-x \to \text{mov } 0$). Lowers power-of-two multiplication to arithmetic left shifts ($x * 2^k \to \text{shl } x, k$), power-of-two floor division to arithmetic right shifts ($x / 2^k \to \text{shr } x, k$), and power-of-two modulo to bitwise masking ($x \% 2^k \to \text{and } x, 2^k - 1$).
3. **Common Subexpression Elimination (`cse`)**:
   Implements Local Value Numbering (LVN) within each basic block. Assigns unique value numbers to variables and expressions, sorting operands for commutative operators (`add`, `mul`, `and`, `eq`, `ne`). If an expression has already been computed and its holder variable remains unmodified, redundant recomputations are replaced by `mov dst, holder`. Entries are invalidated upon holder redefinition.
4. **Dead Code Elimination (`dce`)**:
   Constructs the Control Flow Graph (CFG) and computes live variable sets via iterative backward dataflow analysis. Removes pure operations whose destination variable is not live immediately following the instruction. Respects trapping semantics: `div` and `mod` are treated as pure only when the divisor is a known non-zero integer constant. Also prunes unreachable instructions following unconditional jumps/halts, eliminates jumps to adjacent labels, and deletes unreferenced labels.
5. **Full Pipeline (`all`)**:
   Executes `[const_fold_prop, algebraic_strength_reduction, cse, const_fold_prop, dce]` repeatedly to a fixed point (up to 10 iterations).

### Register Allocation Algorithms
- **Linear Scan**: Poletto & Sarkar (1999) interval sweep ordering ranges by start point; when active ranges exceed $k$, spills the active range with furthest endpoint.
- **Greedy-by-Weight**: Orders ranges by spill weight descending, admitting a range into physical registers if and only if capacity $k$ is respected across all points in its interval.
- **Exact 0/1 ILP**: Solves the integer linear program using `scipy.optimize.milp` with deduplicated constraint rows.
- **Brute Force**: Exhaustive subset enumeration ($2^n$) for verification on instances with $n \le 18$.
- **Interval Coloring & Machine VM**: Assigns physical registers $r_0 \dots r_{k-1}$ by interval sweeping, assigns memory slots to spilled variables, buffers operands using scratch registers $s_0, s_1$, and simulates execution on a register-machine virtual machine.

## How to run
```bash
# Execute master experiment suite and generate figures/tables/SUMMARY.md:
python tracks/track_f_compiler/run_all.py

# Accelerated quick mode (under 30s):
python tracks/track_f_compiler/run_all.py --quick

# Run automated test suite:
python -m pytest tracks/track_f_compiler/tests -q
```

## Results
<!-- Experimental outcomes, comparison tables, and plots saved in results/ -->

## Correctness check
<!-- Verification steps, assertions, unit tests, and theoretical invariants checked -->

## Limitations
<!-- Computational bottlenecks, approximations, assumptions, and failure modes -->
