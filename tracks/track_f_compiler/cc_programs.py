"""Suite of Benchmark Programs for Compiler Evaluation.

Contains at least 8 benchmark programs as source text, designed to exercise
specific optimization passes and register allocation algorithms:
- sum_squares: loop with constant and redundant arithmetic subexpressions
- fib_iter: iterative Fibonacci computation testing register pressure in loop body
- primes_trial_division: nested loops and modulo operations testing prime counting
- collatz_steps: Collatz conjecture trajectory steps across starts 1..100
- horner_poly: polynomial evaluation exercising constant folding & strength reduction
- cse_heavy: repeated compound expressions inside loop body exercising CSE
- gcd_sum: double nested Euclidean GCD computation over integer grid
- dead_code: dead assignments, folded conditionals, and unreachable basic blocks
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root and track root are placed on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


PROGRAMS: dict[str, str] = {
    "sum_squares": """# Sum of squares with redundant and foldable subexpressions
i = 1;
total = 0;
while (i <= 100) {
    term = i * i + (3 + 4) * i;
    redundant = i * i + (3 + 4) * i;
    total = total + term + redundant;
    i = i + 1;
}
print total;
""",
    "fib_iter": """# Iterative Fibonacci computation
n = 35;
a = 0;
b = 1;
i = 0;
while (i < n) {
    temp = a + b;
    a = b;
    b = temp;
    i = i + 1;
}
print a;
""",
    "primes_trial_division": """# Trial division counting primes strictly below 300
count = 0;
n = 2;
while (n < 300) {
    is_prime = 1;
    d = 2;
    while (d * d <= n) {
        if (n % d == 0) {
            is_prime = 0;
        }
        d = d + 1;
    }
    if (is_prime == 1) {
        count = count + 1;
    }
    n = n + 1;
}
print count;
""",
    "collatz_steps": """# Total Collatz sequence stopping steps for initial values 1 to 100
total_steps = 0;
start = 1;
while (start <= 100) {
    n = start;
    steps = 0;
    while (n != 1) {
        if (n % 2 == 0) {
            n = n / 2;
        } else {
            n = 3 * n + 1;
        }
        steps = steps + 1;
    }
    total_steps = total_steps + steps;
    start = start + 1;
}
print total_steps;
""",
    "horner_poly": """# Polynomial evaluation with constant expressions and powers of two
total = 0;
x = 0;
while (x < 150) {
    c3 = 2 + 3;
    c2 = 4 * 2;
    c1 = 10 - 3;
    c0 = 12 / 3;
    p = ((c3 * x + c2) * x + c1) * x + c0;
    s1 = x * 8;
    s2 = x / 4;
    s3 = x % 16;
    total = total + p + s1 + s2 + s3;
    x = x + 1;
}
print total;
""",
    "cse_heavy": """# Repeated subexpression evaluation within a loop
sum_val = 0;
i = 0;
while (i < 200) {
    a = i + 1;
    b = i + 2;
    c = i + 3;
    e1 = a * b + c;
    e2 = a * b + c;
    e3 = a * b + c;
    e4 = a * b + c;
    sum_val = sum_val + e1 + e2 + e3 + e4;
    i = i + 1;
}
print sum_val;
""",
    "gcd_sum": """# Double-loop GCD sum using Euclidean algorithm
total_gcd = 0;
i = 1;
while (i <= 50) {
    j = 1;
    while (j <= 50) {
        a = i;
        b = j;
        while (b != 0) {
            r = a % b;
            a = b;
            b = r;
        }
        total_gcd = total_gcd + a;
        j = j + 1;
    }
    i = i + 1;
}
print total_gcd;
""",
    "dead_code": """# Dead assignments, folded conditions, and unreachable branches
unused1 = 100 + 200;
unused2 = unused1 * 3;
res = 0;
if (1 < 2) {
    res = res + 42;
    dead_in_then = 99 * 88;
} else {
    dead_branch = 12345;
    res = 0;
}
if (10 == 20) {
    unreachable_then = 555;
    res = 999;
} else {
    res = res + 58;
}
i = 0;
dead_accum = 0;
while (i < 50) {
    dead_temp = i * 17;
    dead_accum = dead_accum + dead_temp;
    res = res + 1;
    i = i + 1;
}
print res;
""",
}

