# Track J: Software Engineering Optimization

## Objective
Measure whether mathematical optimization (weighted set cover via 0/1 Integer Linear Programming and greedy heuristics) can significantly reduce regression test suite execution time and size while preserving 100% of structural code coverage (lines and branch arcs) and maintaining fault-detection effectiveness (mutation score). The code under test is the core compiler pipeline of Track F (>1000 lines).

## Baseline
<!-- Describe the baseline method or initial unoptimized benchmark -->

## Formulation
- **Decision variables:**
  $$x_t \in \{0, 1\} \quad \text{for each candidate test } t \in T$$
  where $x_t = 1$ if test $t$ is selected and retained in the minimized suite, and $x_t = 0$ if discarded.
- **Objective function:**
  Minimise total test execution cost (measured untraced wall-clock runtime):
  $$\min \sum_{t \in T} c_t \cdot x_t$$
  where $c_t > 0$ represents the empirical execution cost of test $t$.
- **Constraints (Weighted Set Cover):**
  Every structural code element $e \in E$ (lines or branch arcs exercised by the full suite) must be covered by at least one selected test:
  $$\sum_{t: e \in \text{cover}(t)} x_t \ge 1 \quad \forall e \in E$$

## Methods
- **Codebase Under Test:** The five core compiler modules of Track F (`cc_lang.py`, `cc_ir.py`, `cc_passes.py`, `cc_regalloc.py`, `cc_vm.py`), comprising an optimizing compiler codebase of over 1000 non-blank, non-comment lines.
- **Profile First, Then Optimise:** Baseline test execution is profiled using `cProfile` and traced with `sys.settrace` to measure individual test durations, identify runtime bottlenecks, and extract exact statement and branch coverage sets before optimization.
- **Minimization Algorithms:**
  - *Exact 0/1 ILP:* Solves the weighted set cover problem to provable mathematical optimality using `scipy.optimize.milp` over a sparse constraint matrix.
  - *Greedy Heuristic:* Iteratively selects tests maximizing cost-effectiveness (ratio of newly covered elements to test cost: $\frac{|\text{cover}(t) \cap U|}{c_t}$).
  - *Random Baseline:* Repeatedly samples tests in random order until full universe coverage is achieved (`select_random_until_cover`).
- **Test Prioritization:** Evaluates greedy additional coverage, cost-ascending, and random permutations via cumulative coverage progression curves and normalized Area Under the Curve (AUC).
- **Fault-Injection Validation:** Injects 20 pass output mutants (operators M1–M5 across 4 compiler passes) to evaluate empirical mutation scores of minimized suites.

## How to run
```bash
python tracks/track_j_software_eng/run_all.py [--quick]
python -m pytest tracks/track_j_software_eng/tests -q
```

## Results
<!-- Experimental outcomes, comparison tables, and plots saved in results/ -->

## Correctness check
<!-- Verification steps, assertions, unit tests, and theoretical invariants checked -->

## Limitations
<!-- Computational bottlenecks, approximations, assumptions, and failure modes -->
