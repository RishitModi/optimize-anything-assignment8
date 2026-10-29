# Track F: Compiler Optimization – Experimental Summary

## Environment & Platform Configuration
- **Python Version**: 3.15.1
- **NumPy Version**: 2.5.2
- **SciPy Version**: 1.18.1
- **Platform / OS**: Windows-11-10.0.26200-SP0
- **Random Seeds**: Explicit seeds (range 42 to 42 + N)
- **Quick Execution Flag**: False
- **Note on Performance Timing**: Run times are measured on a Python-level IR interpreter and register-machine VM and serve as relative proxies; dynamic and static instruction counts are mathematically exact.

## Verification Invariants & Failure Log
- :white_check_mark: **No verification failures**. All semantic, algebraic, and capacity invariants satisfied.

---
## Experiment F1: Optimization Pass Effects

### Setup Description
Evaluates static IR instruction size, dynamic instruction counts, runtime, and compile-time overhead across 8 benchmark programs for six configurations: unoptimized (none), constant folding/propagation (const_fold_prop), algebraic strength reduction (algebraic_strength_reduction), common subexpression elimination (cse), dead code elimination (dce), and the complete pipeline (all).

#### Table F1.1: Aggregated Optimization Pass Effects
| configuration | mean_dynamic_reduction_pct | mean_static_instructions | mean_compile_time_s |
| --- | --- | --- | --- |
| none | 0.0000 | 31.2500 | 0.0000 |
| const_fold_prop | 0.0219 | 31.1250 | 0.0001 |
| algebraic_strength_reduction | 0.0000 | 31.2500 | 0.0000 |
| cse | 0.0000 | 31.2500 | 0.0001 |
| dce | 0.1311 | 30.2500 | 0.0002 |
| all | 19.9157 | 23.7500 | 0.0005 |

#### Single-Pass Effectiveness Across Programs
- **const_fold_prop**: produced code changes in 1 of 8 benchmark programs
- **algebraic_strength_reduction**: produced code changes in 0 of 8 benchmark programs
- **cse**: produced code changes in 0 of 8 benchmark programs
- **dce**: produced code changes in 1 of 8 benchmark programs

#### Artifact Figures
- `f1_dynamic_reduction.png`: Grouped bar chart of percentage dynamic instruction reduction per benchmark program.
- `f1_static_vs_dynamic.png`: Scatter plot mapping static code size against dynamic execution steps for unoptimized vs optimized code.

---
## Experiment F2: Register Allocation as 0/1 Optimization

### Setup Description
Formulates register allocation over live range intervals [start, end] as a 0/1 Integer Linear Program (ILP) with objective maximize sum(w_i * x_i) subject to overlap capacity sum(x_i) <= k. Compares against Linear Scan and Greedy-by-Weight heuristics across physical register counts k in {2, 3, 4, 6, 8}. Emits register machine code and simulates dynamic loads, stores, and execution cycles.

#### Table F2.1: Aggregated Register Allocation Metrics by Capacity (k) and Method
| k | method | mean_spill_cost | mean_mem_ops | mean_loads | mean_stores | mean_solve_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | bruteforce | 619.2500 | 10462.6250 | 6219.6250 | 4243.0000 | 0.0071 |
| 2 | greedy_weight | 1017.0000 | 16733.0000 | 8829.0000 | 7904.0000 | 0.0000 |
| 2 | ilp | 619.2500 | 10462.5000 | 6219.5000 | 4243.0000 | 0.0069 |
| 2 | linear_scan | 665.7500 | 11499.1250 | 7783.5000 | 3715.6250 | 0.0000 |
| 3 | bruteforce | 213.7500 | 4674.0000 | 3257.5000 | 1416.5000 | 0.0136 |
| 3 | greedy_weight | 287.6250 | 5921.3750 | 3410.7500 | 2510.6250 | 0.0000 |
| 3 | ilp | 213.7500 | 4674.0000 | 3257.5000 | 1416.5000 | 0.0062 |
| 3 | linear_scan | 226.7500 | 4843.5000 | 3426.6250 | 1416.8750 | 0.0000 |
| 4 | bruteforce | 87.8750 | 1475.6250 | 887.8750 | 587.7500 | 0.0172 |
| 4 | greedy_weight | 98.6250 | 1675.1250 | 975.1250 | 700.0000 | 0.0000 |
| 4 | ilp | 87.8750 | 1475.6250 | 887.8750 | 587.7500 | 0.0053 |
| 4 | linear_scan | 115.1250 | 2418.0000 | 1898.2500 | 519.7500 | 0.0000 |
| 6 | bruteforce | 2.5000 | 37.5000 | 18.7500 | 18.7500 | 0.0186 |
| 6 | greedy_weight | 2.5000 | 37.5000 | 18.7500 | 18.7500 | 0.0000 |
| 6 | ilp | 2.5000 | 37.5000 | 18.7500 | 18.7500 | 0.0003 |
| 6 | linear_scan | 2.7500 | 37.7500 | 18.8750 | 18.8750 | 0.0000 |
| 8 | bruteforce | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0187 |
| 8 | greedy_weight | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 8 | ilp | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| 8 | linear_scan | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

#### Exact Optimality Comparison
- ILP strictly lower spill cost than Linear Scan: 14 instances
- ILP strictly lower spill cost than Greedy Weight: 17 instances

#### Artifact Figures
- `f2_memory_ops_vs_k.png`: Mean dynamic memory traffic (loads + stores) as a function of physical registers k.
- `f2_spill_cost_vs_k.png`: Weighted spill cost objective versus physical register file size k.

---
## Experiment F3: Differential Fuzzing

### Setup Description
Performs differential testing across 300 deterministically generated programs containing nested conditionals, bounded loops, arithmetic expressions, and non-zero constant divisors. Validates semantic equivalence between unoptimized IR, individual passes, the full pipeline, and machine code running on the register VM with k=3.

- **Programs Tested**: 300
- **Semantic Mismatches**: 0
- **Mean Dynamic Instruction Reduction (Pipeline 'all')**: 16.3747%

---
## Disassembly Comparison Artifact
- `example_ir.txt`: Contains textual three-address disassembly before and after optimization for `dead_code` and `sum_squares`.
