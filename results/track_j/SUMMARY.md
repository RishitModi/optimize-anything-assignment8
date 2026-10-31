# Track J: Software Engineering Optimization – Experimental Summary

## Environment & Platform Configuration
- **Python Version**: 3.15.1
- **NumPy Version**: 2.5.2
- **SciPy Version**: 1.18.1
- **Platform / OS**: Windows-11-10.0.26200-SP0
- **Quick Execution Flag**: True
- **Random Seed Management**: Explicit seeds across all stochastic selections

## Codebase Under Test Metrics
- **Code Under Test (CUT) Non-Comment Lines**: 1644
- **Total Non-Test Compiler Lines**: 2582

### Codebase Line Counts by Module
| file | lines |
| --- | --- |
| cc_lang.py | 465 |
| cc_ir.py | 291 |
| cc_passes.py | 376 |
| cc_regalloc.py | 286 |
| cc_vm.py | 226 |
| TOTAL_CUT | 1644 |
| TOTAL_ALL_COMPILER | 2582 |

## J0: Baseline Profiling & Test Execution
- **Total Tests in Universe**: 38
- **Baseline Median Full-Suite Runtime**: 1.2545 s
- **Passing Tests**: 38/38
- **Per-Test Cost (min / median / max)**: 0.00102 s / 0.01754 s / 0.22472 s
- **Top 10 Slowest Tests Runtime Share**: 57.20%

### Top 15 Functions by Cumulative Execution Time (cProfile)
| function | file_line | ncalls | tottime | cumtime |
| --- | --- | --- | --- | --- |
| run_test | sj_suite.py:116 | 38 | 0.002985 | 2.541 |
| run_ir | cc_ir.py:296 | 228 | 0.787 | 1.211 |
| allocate_ilp | cc_regalloc.py:226 | 38 | 0.007269 | 0.3935 |
| run_machine | cc_vm.py:161 | 38 | 0.1808 | 0.3732 |
| milp | _milp.py:154 | 37 | 0.00464 | 0.3713 |
| _highs_wrapper | _highs_wrapper.py:9 | 37 | 0.02213 | 0.3505 |
| <built-in method scipy.optimize._highspy._core.run> | ~:0 | 37 | 0.3194 | 0.3194 |
| optimize_all | cc_passes.py:500 | 38 | 8.17e-05 | 0.2815 |
| optimize | cc_passes.py:469 | 38 | 0.001958 | 0.2814 |
| dce | cc_passes.py:361 | 125 | 0.0272 | 0.2536 |
| live_after_each_instruction | cc_ir.py:243 | 235 | 0.03274 | 0.206 |
| <built-in method builtins.isinstance> | ~:0 | 1488385 | 0.2051 | 0.2057 |
| build_cfg | cc_ir.py:111 | 610 | 0.04515 | 0.1942 |
| <built-in method builtins.len> | ~:0 | 1445751 | 0.1728 | 0.1728 |
| compute_local_def_use | cc_ir.py:98 | 15813 | 0.0545 | 0.1586 |

### Code Coverage Breakdown
| file | executable_lines | covered_lines | line_coverage_pct | covered_arcs |
| --- | --- | --- | --- | --- |
| cc_lang.py | 392 | 236 | 60.2 | 254 |
| cc_ir.py | 257 | 206 | 80.16 | 260 |
| cc_passes.py | 300 | 233 | 77.67 | 299 |
| cc_regalloc.py | 234 | 89 | 38.03 | 125 |
| cc_vm.py | 182 | 139 | 76.37 | 163 |
| TOTAL | 1365 | 903 | 66.15 | 1101 |

- Output Figure: `j0_test_cost_distribution.png`

## J1: Test Suite Minimization (Weighted Set Cover)
Evaluates full suite against random-until-cover, greedy heuristic, and exact 0/1 ILP minimization.

| coverage_kind | method | num_tests | summed_cost | actual_runtime | runtime_reduction_pct | coverage_preserved_pct | failing_tests | mutation_score_pct | optimal | solve_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | full_suite | 38 | 1.053 | 0.7738 | 0 | 100 | 0 | 100 | True | 0 |
| lines | random_until_cover(mean_N=30.6) | 31 | 0.8451 | 0.6682 | 13.64 | 100 | 0 | 100 | False | 0 |
| lines | greedy | 8 | 0.1142 | 0.09298 | 87.98 | 100 | 0 | 100 | False | 0.002173 |
| lines | ilp | 7 | 0.1021 | 0.07741 | 90 | 100 | 0 | 100 | True | 0.03599 |
| arcs | random_until_cover(mean_N=30.6) | 31 | 0.8451 | 0.7345 | 5.078 | 100 | 0 | 100 | False | 0 |
| arcs | greedy | 9 | 0.2082 | 0.1843 | 76.19 | 100 | 0 | 100 | False | 0.003224 |
| arcs | ilp | 8 | 0.196 | 0.1735 | 77.58 | 100 | 0 | 100 | True | 0.03962 |

- Output Figure: `j1_minimization_comparison.png`

## J2: Test Prioritization
Compares cumulative line coverage accumulation over time for greedy additional coverage, cost-ascending, and random orders.

| strategy | normalized_auc |
| --- | --- |
| greedy_additional | 0.9973 |
| cost_ascending | 0.9937 |
| random_order(mean_over_20_seeds) | 0.9811 |

- Output Figure: `j2_prioritization_curves.png`

## J3: Fault-Injection & Mutation Testing
Evaluates whether minimized test suites preserve mutant killing capability.

- **Total Injected Mutants**: 20
- **Detectable Mutants by Full Suite**: 20

> **Note on Fault Injection Proxy**: Fault injection in this benchmark modifies the instruction list output of individual optimization passes. This operates as an empirical proxy for fault-detection ability; structural coverage and fault-detection capability are not theoretically identical.

| method | tests_kept | detectable_killed | detectable_missed | mutation_score_pct | missed_mutants |
| --- | --- | --- | --- | --- | --- |
| full | 38 | 20 | 0 | 100 | None |
| greedy_lines | 8 | 20 | 0 | 100 | None |
| ilp_lines | 7 | 20 | 0 | 100 | None |
| greedy_arcs | 9 | 20 | 0 | 100 | None |
| ilp_arcs | 8 | 20 | 0 | 100 | None |

## Verification Section
- **No verification failures**