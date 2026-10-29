# Track B Discrete Optimization – Summary

## Environment

- **Python version**: 3.12.10
- **NumPy version**: 2.5.4
- **SciPy version**: 1.18.1
- **NetworkX version**: 3.7
- **scikit-learn version**: 1.9.1
- **Platform**: Windows-11-10.0.26200-SP0
- **CPU count**: 18
- **Quick mode**: False
- **Base seeds**: Knapsack seeds [0..4] (quick: [0..1]); Graph seed 1; RNG seeds 0 and 42

## Experiment B1: Knapsack Comparison

**Setup**: Evaluated 0/1 knapsack benchmarks across item counts n in [10, 20, 30, 40, 50] (quick: [10, 20, 30]) using strongly correlated instances (weights uniform in [5, 50], values = weights + noise in [0, 10], capacity = floor(0.5 * sum(weights))). Methods tested include Dynamic Programming (DP), Integer Linear Programming (ILP via scipy.optimize.milp), Branch-and-Bound with fractional LP relaxation pruning, Greedy ratio heuristic, and Brute Force (n <= 20). Invariants tested: DP == ILP, BF == DP (n <= 20), B&B == DP (when proven optimal), Greedy <= DP, and strict weight feasibility.

### B1 Summary Table

| n | method | mean_runtime_s | mean_gap_pct |
| --- | --- | --- | --- |
| 10 | DP | 4.842e-05 | 0 |
| 10 | ILP | 0.01709 | 0 |
| 10 | Branch-and-Bound | 0.000169 | 0 |
| 10 | Greedy | 1.614e-05 | 2.351 |
| 10 | BruteForce | 0.0001223 | 0 |
| 20 | DP | 7.296e-05 | 0 |
| 20 | ILP | 0.01014 | 0 |
| 20 | Branch-and-Bound | 0.0001365 | 0 |
| 20 | Greedy | 1.216e-05 | 1.37 |
| 20 | BruteForce | 0.1719 | 0 |
| 30 | DP | 0.000134 | 0 |
| 30 | ILP | 0.01969 | 0 |
| 30 | Branch-and-Bound | 0.0004407 | 0 |
| 30 | Greedy | 1.83e-05 | 1.412 |
| 40 | DP | 0.0001805 | 0 |
| 40 | ILP | 0.02036 | 0 |
| 40 | Branch-and-Bound | 0.0007394 | 0 |
| 40 | Greedy | 1.772e-05 | 0.3312 |
| 50 | DP | 0.0002144 | 0 |
| 50 | ILP | 0.01511 | 0 |
| 50 | Branch-and-Bound | 0.0007671 | 0 |
| 50 | Greedy | 2.162e-05 | 0.7042 |

**Figures**: b1_runtime_vs_n.png, b1_greedy_gap_vs_n.png

## Experiment B2: Max-Cut as QUBO and Ising

**Setup**: Formulated Max-Cut on 3-regular graphs of sizes n in [6, 8, 10, 12, 14, 16] (quick: [6, 8, 10]) into upper-triangular QUBO (minimising -Cut) and equivalent Ising spin models. For all states on n <= 12, verified that QUBO energy matches Ising energy and exact negative cut value. Compared exact brute-force enumeration baseline against greedy single-bit-flip local search and Metropolis simulated annealing across multiple random seeds.

### B2 Summary Table

| n | n_edges | method | best_cut | mean_cut | success_rate | runtime_s | max_discrepancy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 9 | BruteForce | 9 | 9 | 1 | 7.71e-05 | 0.00e+00 |
| 6 | 9 | GreedyFlip | 9 | 9 | 1 | 0.002882 | 0.00e+00 |
| 6 | 9 | SimulatedAnnealing | 9 | 9 | 1 | 0.08437 | 0.00e+00 |
| 8 | 12 | BruteForce | 10 | 10 | 1 | 0.0001145 | 0.00e+00 |
| 8 | 12 | GreedyFlip | 10 | 10 | 1 | 0.002951 | 0.00e+00 |
| 8 | 12 | SimulatedAnnealing | 10 | 10 | 1 | 0.1183 | 0.00e+00 |
| 10 | 15 | BruteForce | 12 | 12 | 1 | 0.0001333 | 0.00e+00 |
| 10 | 15 | GreedyFlip | 12 | 12 | 1 | 0.003845 | 0.00e+00 |
| 10 | 15 | SimulatedAnnealing | 12 | 12 | 1 | 0.1326 | 0.00e+00 |
| 12 | 18 | BruteForce | 16 | 16 | 1 | 0.001172 | 0.00e+00 |
| 12 | 18 | GreedyFlip | 16 | 16 | 1 | 0.009281 | 0.00e+00 |
| 12 | 18 | SimulatedAnnealing | 16 | 16 | 1 | 0.1576 | 0.00e+00 |
| 14 | 21 | BruteForce | 19 | 19 | 1 | 0.005346 | N/A |
| 14 | 21 | GreedyFlip | 19 | 19 | 1 | 0.008385 | N/A |
| 14 | 21 | SimulatedAnnealing | 19 | 19 | 1 | 0.1828 | N/A |
| 16 | 24 | BruteForce | 22 | 22 | 1 | 0.01464 | N/A |
| 16 | 24 | GreedyFlip | 22 | 22 | 1 | 0.01031 | N/A |
| 16 | 24 | SimulatedAnnealing | 22 | 22 | 1 | 0.2141 | N/A |

**Figures**: b2_runtime_vs_n.png, b2_cut_vs_n.png

## Experiment B3: Feature Selection as QUBO with Cardinality Penalty

**Setup**: Formulated feature selection on 12 standardized features of the Wisconsin Breast Cancer dataset as a QUBO minimising relevance minus alpha * redundancy plus lambda * (sum x_i - k)^2 with k = 4 and alpha = 0.5. Conducted a penalty sweep across lambda in [0, 0.1, 0.5, 2.0, 5.0] to test constraint enforcement, and compared the exact QUBO optimum at lambda = 2.0 against greedy forward selection, simulated annealing, and random k-subsets using 5-fold cross-validated LogisticRegression accuracy.

### B3 Penalty Sweep Table

| lambda | selected_features_count | cardinality_satisfied | raw_objective |
| --- | --- | --- | --- |
| 0 | 4 | True | -1.267 |
| 0.1 | 4 | True | -1.267 |
| 0.5 | 4 | True | -1.267 |
| 2 | 4 | True | -1.267 |
| 5 | 4 | True | -1.267 |

### B3 Method Comparison Table

| method | selected_features | num_features | objective | cv_accuracy_mean | cv_accuracy_std |
| --- | --- | --- | --- | --- | --- |
| QUBO Exact (BruteForce) | [0, 1, 2, 11] | 4 | -1.267 | 0.9508 | 0.008849 |
| Greedy Forward Selection | [1, 2, 9, 11] | 4 | -1.233 | 0.9543 | 0.01022 |
| QUBO Simulated Annealing (seeds=10) | [0, 1, 2, 11] | 4 | -1.266 | 0.9498 | 0.003163 |
| Random 4-Subsets (N=200) | aggregate | 4 | -0.7876 | 0.9204 | 0.02821 |

**Figures**: b3_penalty_sweep.png

## Verification

No verification failures
