# Track D: Submodular, Parallel, and Distributed Optimization – Experimental Summary

## Environment & Platform Configuration
- **Python Version**: 3.12.10
- **NumPy Version**: 2.5.4
- **SciPy Version**: 1.18.1
- **Scikit-Learn Version**: 1.9.1
- **Platform / OS**: Windows-11-10.0.26200-SP0
- **Logical CPU Cores**: 18
- **Calibrated Burn Iterations / ms**: 12795
- **Execution Mode**: Full Mode (~5m)

## Verification Invariants & Failure Log
- :white_check_mark: **No verification failures**. All theoretical invariants satisfied.

---
## Experiment D1: Submodular Optimization

### Setup Description
Evaluates maximum coverage over 40 random subsets (universe size 300) under cardinality constraints k from 1 to 10. Compares standard greedy, lazy greedy (CELF with max-heap), and exact integer linear programming (scipy.optimize.milp). Examines empirical submodularity testing and feature selection via OLS R^2 on a synthetic regression instance containing suppressor interactions.

- Sub-experiment D1(a) Coverage: **simulated / algorithmic evaluation**
- Sub-experiment D1(b) R^2 Feature Selection: **simulated / algorithmic evaluation**

*(1 - 1/e) theoretical approximation guarantee is proven and verified for Part (a) coverage; it is not claimed for Part (b) because the suppressor dataset generates positive complementarity violating submodularity.*

#### Table D1.1: Maximum Coverage Benchmark
| k | greedy_value | lazy_value | exact_value | ratio_greedy_exact | bound_1_minus_1_over_e | greedy_evals | lazy_evals |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 40.0000 | 40.0000 | 40.0000 | 1.0000 | 0.6321 | 40.0000 | 40.0000 |
| 2.0000 | 75.0000 | 75.0000 | 75.0000 | 1.0000 | 0.6321 | 79.0000 | 45.0000 |
| 3.0000 | 105.0000 | 105.0000 | 105.0000 | 1.0000 | 0.6321 | 117.0000 | 54.0000 |
| 4.0000 | 131.0000 | 131.0000 | 131.0000 | 1.0000 | 0.6321 | 154.0000 | 60.0000 |
| 5.0000 | 153.0000 | 153.0000 | 153.0000 | 1.0000 | 0.6321 | 190.0000 | 72.0000 |
| 6.0000 | 171.0000 | 171.0000 | 171.0000 | 1.0000 | 0.6321 | 225.0000 | 85.0000 |
| 7.0000 | 187.0000 | 187.0000 | 187.0000 | 1.0000 | 0.6321 | 259.0000 | 94.0000 |
| 8.0000 | 200.0000 | 200.0000 | 200.0000 | 1.0000 | 0.6321 | 292.0000 | 108.0000 |
| 9.0000 | 211.0000 | 211.0000 | 212.0000 | 0.9953 | 0.6321 | 324.0000 | 116.0000 |
| 10.0000 | 221.0000 | 221.0000 | 223.0000 | 0.9910 | 0.6321 | 355.0000 | 120.0000 |

#### Submodularity Test Results
- Coverage Function: 0 violations across test trials (Worst violation: 0.0000)
- R^2 Feature Objective: 691 violations across test trials (Worst violation: 0.0554)

#### Table D1.2: R^2 Feature Selection Comparison
| k | greedy_r2 | exact_r2 | ratio |
| --- | --- | --- | --- |
| 1.0000 | 0.1996 | 0.1996 | 1.0000 |
| 2.0000 | 0.2780 | 0.2780 | 1.0000 |
| 3.0000 | 0.3247 | 0.3247 | 1.0000 |
| 4.0000 | 0.3467 | 0.3467 | 1.0000 |
| 5.0000 | 0.3677 | 0.3731 | 0.9856 |

#### Artifact Figures
- `d1_coverage_scaling.png`: Greedy/exact ratio and evaluation query counts vs cardinality k.

---
## Experiment D3: Federated Learning (FedAvg vs Centralized)

### Setup Description
Simulates a federation of K=10 clients optimizing a regularized binary logistic regression model on 6000 samples. Evaluates FedAvg under IID and non-IID Dirichlet label skew (alpha=0.3) across varying local computation steps (E in {1, 5, 20}), benchmarking against centralized mini-batch SGD with matched sample throughput.

- Sub-experiment D3: **measured on simulated clients (wall-clock algorithmic simulation)**

#### Table D3.1: Federated Learning Convergence & Communication
| partition | local_steps | method | final_acc_mean | final_acc_std | rounds_to_85 | communicated_floats_to_85 |
| --- | --- | --- | --- | --- | --- | --- |
| IID | 1 | FedAvg | 0.7378 | 0.0010 | not reached | not reached |
| IID | 5 | FedAvg | 0.7394 | 0.0016 | not reached | not reached |
| IID | 20 | FedAvg | 0.7431 | 0.0021 | not reached | not reached |
| Non-IID (Label Skew alpha=0.3) | 1 | FedAvg | 0.7381 | 0.0016 | not reached | not reached |
| Non-IID (Label Skew alpha=0.3) | 5 | FedAvg | 0.7408 | 0.0049 | not reached | not reached |
| Non-IID (Label Skew alpha=0.3) | 20 | FedAvg | 0.7408 | 0.0014 | not reached | not reached |
| Centralized | 5 | Centralized SGD | 0.7394 | 0.0017 | 5 | 0 (local) |

#### Artifact Figures
- `d3_federated_convergence.png`: Test accuracy convergence curves across rounds and partitions.

---
## Experiment D4: Multi-Task Learning & Gradient Balancing

### Setup Description
Investigates optimization dynamics on a two-task regression neural network with a shared representation layer (32 tanh units) and disparate task loss scales (scale ratio 10:1). Evaluates three gradient balancing mechanisms: Equal Weights (unweighted sum), MGDA (Pareto-stationary minimum-norm descent), and GradNorm (dynamic loss weights balancing relative gradient norms).

- Sub-experiment D4: **measured / algorithmic training simulation**

#### Table D4.1: Multi-Task Performance Metrics
| scheme | r2_task1_mean | r2_task1_std | r2_task2_mean | r2_task2_std | mean_r2_mean | worst_task_r2_mean | worst_task_r2_std | scheme_diagnostic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EQUAL | 0.9991 | 0.0004 | 0.9999 | 1.413e-05 | 0.9995 | 0.9991 | 0.0004 | w = [1.0, 1.0] |
| MGDA | 0.9996 | 8.656e-05 | 0.9232 | 0.0121 | 0.9614 | 0.9232 | 0.0121 | mean alpha = 1.000 |
| GRADNORM | 0.9994 | 0.0001 | 0.9989 | 0.0004 | 0.9992 | 0.9989 | 0.0004 | final weights = [1.71, 0.29] |

#### Artifact Figures
- `d4_multitask_convergence.png`: Per-task MSE loss trajectories and dynamic weight balancing parameters.