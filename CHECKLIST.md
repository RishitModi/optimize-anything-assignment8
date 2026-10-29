# Assignment Checklist

Progress tracker across Chapter 6 optimization areas and techniques.

| Chapter 6 area | Method/technique | Experiment | Code location | Result | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| Foundations | ML as optimization (parameters, loss, regularization) | exp6 | tracks/track_a_continuous/exp6_backprop_mlp.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Foundations | Continuous vs discrete | exp1 | tracks/track_a_continuous/exp1_optimizer_comparison.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Foundations | Convex vs non-convex | exp2 | tracks/track_a_continuous/exp2_landscapes.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Foundations | Local minima / global minima / saddle points | exp2 | tracks/track_a_continuous/exp2_landscapes.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Foundations | Basic quantum computing concepts | | | | TODO |
| Foundations | Role of optimization in QML | | | | TODO |
| Continuous | GD | exp1, exp3 | tracks/track_a_continuous/exp1_optimizer_comparison.py, tracks/track_a_continuous/exp3_stepsize_sensitivity.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Continuous | SGD | exp4 | tracks/track_a_continuous/exp4_stochastic_and_noise.py | GD, SGD-256, SGD-32, SGD-1 valid (exp4a); Momentum and Adam stochastic rows invalid | Partial |
| Continuous | Mini-batch SGD | exp4 | tracks/track_a_continuous/exp4_stochastic_and_noise.py | GD, SGD-256, SGD-32, SGD-1 valid (exp4a); Momentum and Adam stochastic rows invalid | Partial |
| Continuous | Momentum | exp1, exp3 | tracks/track_a_continuous/exp1_optimizer_comparison.py, tracks/track_a_continuous/exp3_stepsize_sensitivity.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Continuous | NAG | exp1, exp3 | tracks/track_a_continuous/exp1_optimizer_comparison.py, tracks/track_a_continuous/exp3_stepsize_sensitivity.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Continuous | AdaGrad | exp1, exp3 | tracks/track_a_continuous/exp1_optimizer_comparison.py, tracks/track_a_continuous/exp3_stepsize_sensitivity.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Continuous | RMSProp | exp1, exp3 | tracks/track_a_continuous/exp1_optimizer_comparison.py, tracks/track_a_continuous/exp3_stepsize_sensitivity.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Continuous | Adam | exp1, exp3 | tracks/track_a_continuous/exp1_optimizer_comparison.py, tracks/track_a_continuous/exp3_stepsize_sensitivity.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Continuous | Newton | exp5 | tracks/track_a_continuous/exp5_second_order.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Continuous | Gauss-Newton | exp5 | tracks/track_a_continuous/exp5_second_order.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Continuous | Backpropagation | exp6 | tracks/track_a_continuous/exp6_backprop_mlp.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Continuous | Fisher / diagonal / low-rank curvature approximations | exp5 | tracks/track_a_continuous/exp5_second_order.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Continuous | Hessian-vector products | exp5 | tracks/track_a_continuous/exp5_second_order.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Continuous | Convergence for convex / strongly convex / non-convex | exp2 | tracks/track_a_continuous/exp2_landscapes.py | qualitative final gaps and plots only; rate fit invalid, see Failure Log case 8 | Partial |
| Continuous | Step-size and optimizer tuning | exp3 | tracks/track_a_continuous/exp3_stepsize_sensitivity.py | see results/track_a/SUMMARY.md, exp1 to exp6 | Done |
| Continuous | Noisy objectives | exp4 | tracks/track_a_continuous/exp4_stochastic_and_noise.py | exp4b invalid, see Failure Log case 7 | Known bug - not covered |
| Discrete | Greedy | B1 | tracks/track_b_discrete/knapsack.py | | Code ready - results pending |
| Discrete | ILP | B1 | tracks/track_b_discrete/knapsack.py | | Code ready - results pending |
| Discrete | Branch-and-Bound | B1 | tracks/track_b_discrete/knapsack.py | | Code ready - results pending |
| Discrete | Dynamic programming | B1 | tracks/track_b_discrete/knapsack.py | | Code ready - results pending |
| Discrete | Feature selection | B3 | tracks/track_b_discrete/qubo.py | | Code ready - results pending |
| Discrete | QUBO and Ising (binary-to-spin conversion, penalty terms) | B2, B3 | tracks/track_b_discrete/qubo.py | | Code ready - results pending |
| Discrete | Max-Cut | B2 | tracks/track_b_discrete/qubo.py | | Code ready - results pending |
| Quantum optimization | QAOA | | | | TODO |
| Quantum optimization | VQE | | | | TODO |
| Quantum optimization | PQCs / QNNs | | | | TODO |
| Quantum optimization | Parameter-shift | | | | TODO |
| Quantum optimization | SPSA | | | | TODO |
| Quantum optimization | Quantum natural gradient | | | | TODO |
| Quantum optimization | Deutsch-Jozsa | | | | TODO |
| Quantum optimization | Bernstein-Vazirani | | | | TODO |
| Quantum optimization | Grover | | | | TODO |
| Quantum optimization | QFT | | | | TODO |
| Quantum optimization | QPE | | | | TODO |
| Quantum optimization | Shor | | | | TODO |
| Quantum optimization | Quantum annealing | | | | TODO |
| Quantum optimization | Variational classifiers | | | | TODO |
| Quantum optimization | Quantum kernels | | | | TODO |
| Quantum optimization | Barren plateaus | | | | TODO |
| Quantum optimization | Finite-shot cost | | | | TODO |
| Quantum optimization | Hardware noise | | | | TODO |
| Quantum optimization | QUBO embedding limits | | | | TODO |
| Submodular | Diminishing returns | | | | TODO |
| Submodular | Assumption testing | | | | TODO |
| Submodular | Greedy selection | | | | TODO |
| Submodular | Approximation guarantee conditions | | | | TODO |
| Parallel | Load balancing | | | | TODO |
| Parallel | Makespan | | | | TODO |
| Parallel | Work stealing | | | | TODO |
| Parallel | Locality | | | | TODO |
| Parallel | Synchronization | | | | TODO |
| Parallel | Communication overhead | | | | TODO |
| Parallel | Data / model / pipeline parallelism | | | | TODO |
| Parallel | Speedup | | | | TODO |
| Parallel | Parallel efficiency | | | | TODO |
| Parallel | Amdahl | | | | TODO |
| Parallel | Gustafson | | | | TODO |
| Strategy selection | Strategy selection | | | | TODO |
| Distributed / multi-task | FedAvg | | | | TODO |
| Distributed / multi-task | MGDA | | | | TODO |
| Distributed / multi-task | GradNorm | | | | TODO |
