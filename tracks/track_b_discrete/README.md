# Track B: Discrete Optimization

## Objective
Benchmark, formulate, and analyze classical discrete and combinatorial optimization algorithms across three primary paradigms:
1. Selection under capacity constraints (0/1 Knapsack) comparing heuristic, dynamic programming, branch-and-bound, and integer linear programming methods.
2. Unconstrained binary quadratic optimization (Max-Cut) formulated as QUBO and Ising spin models, comparing exact exhaustive search against greedy and simulated annealing heuristics.
3. Constrained combinatorial feature selection mapped to QUBO via quadratic penalty terms enforcing target cardinality constraints.

## Baseline
<!-- Describe the baseline method or initial unoptimized benchmark -->

## Formulation
### 1. 0/1 Knapsack
Given item weights $w_i > 0$, values $v_i > 0$, and knapsack capacity $C > 0$:
$$\max \sum_{i=0}^{n-1} v_i x_i \quad \text{subject to} \quad \sum_{i=0}^{n-1} w_i x_i \le C, \quad x_i \in \{0, 1\}$$

### 2. Max-Cut as QUBO and Ising
- **QUBO minimization**: Let $x \in \{0, 1\}^n$. Cut size across edge set $E$ is:
  $$\text{Cut}(x) = \sum_{(i,j) \in E, i<j} (x_i + x_j - 2 x_i x_j)$$
  Formulated as minimizing $E(x) = -\text{Cut}(x) = x^T Q x$ where $Q$ is upper-triangular:
  $$Q_{ii} = -\text{degree}(i), \quad Q_{ij} = +2 \quad (\forall (i, j) \in E, i < j)$$
- **Ising conversion**: Substituting $x_i = \frac{1 - s_i}{2}$ with spins $s_i \in \{-1, +1\}$ (where $s_i = +1 \iff x_i = 0$):
  $$E(s) = \sum_{i<j} J_{ij} s_i s_j + \sum_i h_i s_i + \text{offset}$$
  with exact algebraic parameters:
  $$J_{ij} = \frac{Q_{ij}}{4}, \quad h_i = -\frac{Q_{ii}}{2} - \frac{1}{4} \sum_{j \neq i} Q_{\min(i,j), \max(i,j)}, \quad \text{offset} = \sum_i \frac{Q_{ii}}{2} + \sum_{i<j} \frac{Q_{ij}}{4}$$

### 3. Feature Selection with Quadratic Penalty
Given feature relevance scores $r_i = |\text{corr}(x_i, y)|$, pairwise redundancies $R_{ij} = |\text{corr}(x_i, x_j)|$, redundancy weight $\alpha$, target cardinality $k$, and penalty multiplier $\lambda$:
$$\min_{x \in \{0, 1\}^n} -\sum_i r_i x_i + \alpha \sum_{i<j} R_{ij} x_i x_j + \lambda \left(\sum_i x_i - k\right)^2$$
Expanding $\lambda (\sum_i x_i - k)^2 = \lambda \sum_i (1 - 2k) x_i + 2\lambda \sum_{i<j} x_i x_j + \lambda k^2$ yields QUBO:
$$Q_{ii} = -r_i + \lambda (1 - 2k), \quad Q_{ij} = \alpha R_{ij} + 2\lambda \quad (i < j), \quad \text{offset} = \lambda k^2$$

## Methods
- **0/1 Knapsack**:
  - `brute_force`: Vectorized exhaustive enumeration over all $2^n$ configurations ($n \le 20$).
  - `greedy_ratio`: Packing in descending order of value-to-weight efficiency ratio $v_i / w_i$.
  - `dp_knapsack`: Exact 2D dynamic programming table over capacity with backtracking.
  - `branch_and_bound`: Depth-first search with continuous fractional knapsack LP upper bounding and pruning.
  - `ilp_knapsack`: Exact binary integer programming via `scipy.optimize.milp`.
- **QUBO / Ising**:
  - `brute_force_qubo`: Vectorized chunked enumeration over all $2^n$ bitstrings.
  - `greedy_flip_local_search`: Multi-start descent flipping the best single bit until a 1-flip local minimum is found.
  - `simulated_annealing_qubo`: Single-bit-flip Metropolis algorithm with geometric temperature cooling.
  - `greedy_forward_selection`: Stepwise addition of the best marginal objective gain for feature selection.

## How to run
```bash
python tracks/track_b_discrete/run_all.py [--quick]
python -m pytest tracks/track_b_discrete/tests -q
```

## Results
All tables and figures are produced by python tracks/track_b_discrete/run_all.py and listed in results/track_b/SUMMARY.md. Summary: DP, ILP, branch-and-bound and brute force returned identical optimal values on every knapsack instance; greedy was within 0.3 to 2.4 percent of optimal. For Max-Cut on 3-regular graphs (n = 6 to 16) brute force, greedy local search and simulated annealing all reached the exact optimum on every instance. For feature selection the exact QUBO optimum, greedy forward selection and simulated annealing reached 95.0 to 95.4 percent cross-validated accuracy versus 92.0 percent for random subsets.

## Correctness check
- 5 pytest tests pass (all exact knapsack methods agree; QUBO energy equals Ising energy equals minus cut for all assignments; feature-selection QUBO with penalty equals the direct formula for all 256 assignments).
- In-run assertions (DP = ILP, brute force = DP, branch-and-bound = DP when proven optimal, greedy <= optimum, feasibility) reported no verification failures.

## Limitations
- Small instance sizes (knapsack n <= 50, Max-Cut n <= 16, 12 features).
- 5 instance seeds per size so the greedy gap trend is noisy.
- Runtimes are single-process wall-clock times including Python overhead.
- DP is only applicable to integer weights.
- In the first feature-selection sweep (k = 4) the unconstrained optimum already had 4 features so the penalty was never active, which the k = 7 sweep (b3_penalty_k.py) addresses.
