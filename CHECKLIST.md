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
| Discrete | Greedy | B1 | tracks/track_b_discrete/knapsack.py | see results/track_b/SUMMARY.md | Done |
| Discrete | ILP | B1 | tracks/track_b_discrete/knapsack.py | see results/track_b/SUMMARY.md | Done |
| Discrete | Branch-and-Bound | B1 | tracks/track_b_discrete/knapsack.py | see results/track_b/SUMMARY.md | Done |
| Discrete | Dynamic programming | B1 | tracks/track_b_discrete/knapsack.py | see results/track_b/SUMMARY.md | Done |
| Discrete | Feature selection | B3 | tracks/track_b_discrete/qubo.py | see results/track_b/SUMMARY.md | Done |
| Discrete | QUBO and Ising (binary-to-spin conversion, penalty terms) | B2, B3 | tracks/track_b_discrete/qubo.py | see results/track_b/SUMMARY.md | Done |
| Discrete | Max-Cut | B2 | tracks/track_b_discrete/qubo.py | see results/track_b/SUMMARY.md | Done |
| Quantum optimization | QAOA | C1 | tracks/track_c_quantum_opt/qaoa.py | see results/track_c/SUMMARY.md | Done |
| Quantum optimization | VQE | C2 | tracks/track_c_quantum_opt/vqe.py | see results/track_c/SUMMARY.md | Done |
| Quantum optimization | PQCs | C2, C5 | tracks/track_c_quantum_opt/vqe.py | see results/track_c/SUMMARY.md | Done |
| Quantum optimization | QNNs | Track I (variational classifier) | | | TODO |
| Quantum optimization | Parameter-shift | C2 | tracks/track_c_quantum_opt/vqe.py | see results/track_c/SUMMARY.md | Done |
| Quantum optimization | SPSA | C2, C3 | tracks/track_c_quantum_opt/optimizers_q.py | see results/track_c/SUMMARY.md | Done |
| Quantum optimization | Quantum natural gradient | C2 | tracks/track_c_quantum_opt/optimizers_q.py | see results/track_c/SUMMARY.md | Done |
| Quantum optimization | Deutsch-Jozsa | | | | TODO |
| Quantum optimization | Bernstein-Vazirani | | | | TODO |
| Quantum optimization | Grover | | | | TODO |
| Quantum optimization | QFT | | | | TODO |
| Quantum optimization | QPE | | | | TODO |
| Quantum optimization | Shor | | | | TODO |
| Quantum optimization | Quantum annealing | | | | TODO |
| Quantum optimization | Variational classifiers | | | | TODO |
| Quantum optimization | Quantum kernels | | | | TODO |
| Quantum optimization | Barren plateaus | C5 | tracks/track_c_quantum_opt/run_all.py | see results/track_c/SUMMARY.md | Done |
| Quantum optimization | Finite-shot cost | C3 | tracks/track_c_quantum_opt/run_all.py | see results/track_c/SUMMARY.md | Done |
| Quantum optimization | Hardware noise | C4 (simulated noise model) | tracks/track_c_quantum_opt/run_all.py | see results/track_c/SUMMARY.md | Done (simulated, QAOA only) |
| Quantum optimization | QUBO embedding limits (Track B QUBO solved with QAOA) | C1 | tracks/track_c_quantum_opt/qaoa.py | see results/track_c/SUMMARY.md | Done |
| Submodular | Diminishing returns | D1 | tracks/track_d_submodular_parallel_distributed/submodular.py | | Code ready - results pending |
| Submodular | Assumption testing | D1 | tracks/track_d_submodular_parallel_distributed/submodular.py | | Code ready - results pending |
| Submodular | Greedy selection | D1 | tracks/track_d_submodular_parallel_distributed/submodular.py | | Code ready - results pending |
| Submodular | Approximation guarantee conditions | D1 | tracks/track_d_submodular_parallel_distributed/submodular.py | | Code ready - results pending |
| Parallel | Load balancing | D2(c)(d) | tracks/track_d_submodular_parallel_distributed/parallel_tools.py | | Code ready - results pending |
| Parallel | Makespan | D2(c)(d) | tracks/track_d_submodular_parallel_distributed/parallel_tools.py | | Code ready - results pending |
| Parallel | Work stealing | D2(c)(d) | tracks/track_d_submodular_parallel_distributed/parallel_tools.py | | Code ready - results pending |
| Parallel | Locality | D2(e)(f) | tracks/track_d_submodular_parallel_distributed/parallel_tools.py | | Code ready - results pending |
| Parallel | Synchronization | D2(e)(f) | tracks/track_d_submodular_parallel_distributed/parallel_tools.py | | Code ready - results pending |
| Parallel | Communication overhead | D2(e)(f) | tracks/track_d_submodular_parallel_distributed/parallel_tools.py | | Code ready - results pending |
| Parallel | Data / model / pipeline parallelism | D2(f), D2(g) (simulated) | tracks/track_d_submodular_parallel_distributed/parallel_tools.py | | Code ready - results pending |
| Parallel | Speedup | D2(a)(b) | tracks/track_d_submodular_parallel_distributed/parallel_tools.py | | Code ready - results pending |
| Parallel | Parallel efficiency | D2(a)(b) | tracks/track_d_submodular_parallel_distributed/parallel_tools.py | | Code ready - results pending |
| Parallel | Amdahl | D2(a)(b) | tracks/track_d_submodular_parallel_distributed/parallel_tools.py | | Code ready - results pending |
| Parallel | Gustafson | D2(a)(b) | tracks/track_d_submodular_parallel_distributed/parallel_tools.py | | Code ready - results pending |
| Strategy selection | Strategy selection | | | | TODO |
| Distributed / multi-task | FedAvg | D3 | tracks/track_d_submodular_parallel_distributed/federated.py | | Code ready - results pending |
| Distributed / multi-task | MGDA | D4 | tracks/track_d_submodular_parallel_distributed/multitask.py | | Code ready - results pending |
| Distributed / multi-task | GradNorm | D4 | tracks/track_d_submodular_parallel_distributed/multitask.py | | Code ready - results pending |
