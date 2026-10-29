# Track D: Submodular, Parallel, and Distributed Optimization

## Objective
Investigate non-convex combinatorial submodular optimization, parallel scheduling and scalability bounds, federated distributed learning under data heterogeneity, and gradient-balanced multi-task optimization. Quantify empirical efficiency versus theoretical guarantees across discrete-event scheduling models and multi-process workloads.

## Baseline
<!-- Describe the baseline method or initial unoptimized benchmark -->

## Formulation
### 1. Submodular Maximization under Cardinality Constraint
Given a finite ground set $V$ and a monotone submodular set function $f: 2^V \to \mathbb{R}_{\ge 0}$ satisfying diminishing marginal returns:
$$f(A \cup \{e\}) - f(A) \ge f(B \cup \{e\}) - f(B), \quad \forall A \subseteq B \subset V, \; e \in V \setminus B$$
The problem seeks:
$$\max_{S \subseteq V, |S| \le k} f(S)$$
The greedy algorithm guarantees an approximation ratio of:
$$f(S_{\text{greedy}}) \ge \left(1 - \frac{1}{e}\right) f(S^*) \approx 0.6321 \cdot f(S^*)$$
assuming $f$ is non-negative, monotone ($f(A) \le f(B)$ for $A \subseteq B$), and submodular.

### 2. Parallel Computing, Scheduling, and Scalability
For $n$ tasks with execution durations $\{d_1, \dots, d_n\}$ scheduled across $p$ identical parallel processors:
- **Makespan Minimization**:
  $$M = \max_{i \in \{1, \dots, p\}} \sum_{j \in \text{Assigned}(i)} d_j, \quad M \ge \max\left(\frac{1}{p} \sum_{j=1}^n d_j, \max_j d_j\right)$$
- **Speedup and Efficiency**:
  $$\text{Speedup } S(p) = \frac{T_1}{T_p}, \quad \text{Efficiency } E(p) = \frac{S(p)}{p}$$
- **Amdahl's Law (Strong Scaling)**: For fixed workload with serial fraction $s \in [0, 1]$:
  $$S(p) = \frac{1}{s + \frac{1 - s}{p}}$$
- **Gustafson's Law (Weak Scaling)**: For scaled problem size where parallel work scales with $p$:
  $$S(p) = p - s(p - 1)$$

### 3. Federated Learning (FedAvg)
Federation of $K$ clients with private local datasets $\mathcal{D}_k$ of size $n_k$ ($n = \sum_{k=1}^K n_k$):
$$\min_{w \in \mathbb{R}^d} F(w) = \sum_{k=1}^K \frac{n_k}{n} F_k(w), \quad \text{where } F_k(w) = \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} \ell(w; x_i, y_i) + \frac{\lambda}{2} \|w\|^2$$
Each communication round, all clients initialize at $w^{(t)}$, execute $E$ local mini-batch SGD updates, and the server aggregates:
$$w^{(t+1)} = \sum_{k=1}^K \frac{n_k}{n} w_k^{(t, E)}$$

### 4. Multi-Task Learning & Gradient Balancing
Two-task regression with shared representation parameters $\theta_{\text{sh}} = (W_1, b_1)$ and task heads $\theta_t = (v_t, c_t)$:
$$\min_{\theta} L(\theta) = \sum_{t=1}^2 w_t L_t(\theta_{\text{sh}}, \theta_t)$$
- **Multiple Gradient Descent Algorithm (MGDA)**: Solves for Pareto-stationary minimum-norm shared gradient:
  $$\alpha^* = \text{clip}\left( \frac{(g_2 - g_1)^T g_2}{\|g_1 - g_2\|^2}, 0, 1 \right), \quad g_{\text{shared}} = \alpha^* g_1 + (1 - \alpha^*) g_2$$
- **GradNorm**: Dynamically updates task loss weights $w_t$ to balance relative gradient norms:
  $$G_t = w_t \|\nabla_{W_1} L_t\|_2, \quad r_t = \frac{L_t / L_t(0)}{\frac{1}{2}\sum_j L_j / L_j(0)}, \quad \text{target}_t = \bar{G} \cdot r_t^{1.5}$$
  updating $w_t \leftarrow w_t - \eta \cdot \text{sign}(G_t - \text{target}_t) \frac{G_t}{w_t}$, clipped to $\ge 0.01$ and normalized such that $\sum_t w_t = 2$.

## Methods
- `submodular.py`: Monotone coverage generator, greedy and lazy greedy (CELF), exact MILP coverage (`scipy.optimize.milp`), submodularity verification, and OLS $R^2$ feature selection.
- `parallel_tools.py`: CPU-bound calibration, discrete-event scheduling simulators (static block, round-robin, list scheduling, LPT, work stealing), GPipe pipeline recurrence simulator, and worker-resident vs shipped data-parallel gradient evaluation.
- `federated.py`: Standardized logistic regression, IID and Dirichlet label skew partitioners, Federated Averaging (`fedavg`), and centralized mini-batch SGD baseline.
- `multitask.py`: Multi-task neural network with shared tanh hidden layer, manual analytical backpropagation, numerical gradient checking, MGDA solver, and GradNorm dynamic balancer.

## How to run
```bash
python tracks/track_d_submodular_parallel_distributed/run_all.py [--quick] [--only D1 D2 D3 D4]
python -m pytest tracks/track_d_submodular_parallel_distributed/tests -q
```

## Results
Tables and figures are produced by python tracks/track_d_submodular_parallel_distributed/run_all.py (D1, D3, D4) and listed in results/track_d/SUMMARY.md. D1: greedy coverage reached 99.1 to 100 percent of the exact ILP optimum (guarantee 63.2 percent); lazy greedy needed 120 instead of 355 evaluations at k = 10; the R^2 feature objective violated submodularity (691 violating triples, worst 0.055) and greedy was suboptimal at k = 5. D3: all FedAvg settings reached 73.8 to 74.3 percent test accuracy, equal to centralized SGD (73.9 percent) within about half a point; label skew caused no measurable loss. D4: equal weights reached R^2 above 0.999 on both tasks; MGDA put all weight on the smaller-gradient task (task 2 R^2 0.923); GradNorm matched equal weighting. D2 (parallel scaling) was not completed.

## Correctness check
- 4 unit tests passed before the suite stalled in the work-stealing simulator (Failure Log case 16).
- In-run checks of D1, D3 and D4 reported no failures (coverage had 0 submodularity violations, lazy greedy equalled greedy, greedy met the 1 - 1/e bound, manual multi-task gradients matched numerical gradients).

## Limitations
- D2 not completed so no measured speedup, efficiency or overhead is reported (Failure Log cases 15 and 16).
- The D3 rounds-to-target column was wrong and is not used (case 17).
- Synthetic data and a convex model in D3, one toy problem in D4; the D3 and D4 results are single configurations and not generalised.
