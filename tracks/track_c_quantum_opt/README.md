# Track C: Quantum Optimization

## Objective
Investigate and benchmark variational quantum algorithms (QAOA and VQE) on classical quantum circuit simulators. Evaluate convergence under parameter-shift gradient descent, quantum natural gradients (QNG), SPSA, and derivative-free COBYLA optimizers under strict circuit-evaluation query budgets. Analyze the impact of finite-shot projective measurement noise, phenomenological depolarizing gate noise, and barren plateau variance scaling across local and global observables.

## Baseline
<!-- Describe the baseline method or initial unoptimized benchmark -->

## Formulation
### 1. QAOA State and Cost Hamiltonian
For an $n$-qubit system, QAOA prepares the state:
$$|\psi(\gamma, \beta)\rangle = \prod_{l=1}^p \left( e^{-i \beta_l H_M} e^{-i \gamma_l H_C} \right) |+\rangle^{\otimes n}$$
where $H_M = \sum_{j=0}^{n-1} X_j$ is the transverse mixer Hamiltonian. The problem cost Hamiltonian $H_C$ is derived directly from the Track B Ising formulation of Max-Cut:
$$H_C = \sum_{i<j} J_{ij} Z_i Z_j + \sum_i h_i Z_i + \text{offset} \cdot I$$
with eigenvalues matching $x^T Q x$ for $s_i = 1 - 2x_i$.

### 2. VQE Hamiltonian and Ansatz
The target model is the 1D open Transverse-Field Ising Model (TFIM):
$$H = -\sum_{i=0}^{n-2} Z_i Z_{i+1} - g \sum_{i=0}^{n-1} X_i$$
We deploy a real hardware-efficient parameterized quantum circuit (PQC) with $L$ layers ($P = n(L+1)$ parameters):
$$|\psi(\theta)\rangle = \left[ \prod_{l=1}^L \left( \prod_{c=0}^{n-2} \text{CNOT}_{c, c+1} \prod_{q=0}^{n-1} R_Y(\theta_{l, q}) \right) \right] \prod_{q=0}^{n-1} R_Y(\theta_{0, q}) |0\rangle^{\otimes n}$$

### 3. Parameter-Shift Rule
For each generator $G = \frac{1}{2} Y$, the exact analytic gradient is computed without numerical differencing error as:
$$\frac{\partial \langle H \rangle}{\partial \theta_k} = \frac{1}{2} \left[ \langle H \rangle_{\theta + \frac{\pi}{2} e_k} - \langle H \rangle_{\theta - \frac{\pi}{2} e_k} \right]$$
requiring $2P$ circuit evaluations per gradient step.

### 4. Quantum Natural Gradient (QNG) and Fubini-Study Metric
QNG updates parameters along the steepest descent direction on the Fubini-Study Riemannian manifold:
$$\theta^{(t+1)} = \theta^{(t)} - \eta (g(\theta^{(t)}) + \lambda I)^{-1} \nabla_\theta \langle H \rangle$$
where the real Fubini-Study metric tensor $g_{ij}$ is given by:
$$g_{ij} = \frac{1}{4} \text{Re} \left[ \langle \phi_i | \phi_j \rangle - \langle \phi_i | \psi \rangle \langle \psi | \phi_j \rangle \right]$$
with $|\phi_i\rangle$ denoting the state with Pauli $Y$ inserted immediately after gate $i$.

### 5. SPSA (Simultaneous Perturbation Stochastic Approximation)
SPSA approximates the gradient in 2 circuit evaluations independent of dimension $P$:
$$\hat{g}_k(\theta) = \frac{\langle H \rangle_{\theta + c_k \Delta} - \langle H \rangle_{\theta - c_k \Delta}}{2 c_k} \Delta^{-1}$$
where $\Delta \in \{-1, +1\}^P$ is a Rademacher random vector, and $a_k, c_k$ follow standard asymptotic gain sequences.

### 6. Cost Measure
All optimizer comparisons enforce an identical budget measured in total cumulative circuit queries (statevector executions or shot allocations).

### Simulation versus hardware
All experiments are executed on a custom self-written NumPy statevector simulator. Phenomenological gate errors are simulated via an independent single-qubit depolarizing-trajectory model. No real quantum hardware or external cloud quantum backends were used anywhere in Track C.

## Methods
- `qsim`: Custom statevector simulator supporting arbitrary single-qubit rotations, CNOT networks, Pauli expectation evaluations, finite-shot multinomial sampling, and depolarizing channel trajectories.
- `qaoa`: Multi-start QAOA parameter optimization via COBYLA.
- `vqe`: Exact Transverse-Field Ising ground state preparation.
- `optimizers_q`: Parameter-shift gradient descent, Quantum Natural Gradient (Fubini-Study metric), SPSA, and COBYLA.

## How to run
```bash
python tracks/track_c_quantum_opt/run_all.py [--quick]
python -m pytest tracks/track_c_quantum_opt/tests -q
```

## Results
All tables and figures are produced by python tracks/track_c_quantum_opt/run_all.py and listed in results/track_c/SUMMARY.md. Everything is simulated with a self-written numpy statevector simulator; no real quantum hardware was used. QAOA ratio rises with depth (p = 3: 1.00, 0.92, 0.89, 0.89 for n = 4, 6, 8, 10) but does not beat the classical baselines; QNG was the most evaluation-efficient VQE optimizer; shot-noise RMSE scales as shots^-0.5; simulated depolarizing noise lowers the QAOA ratio from 0.878 to 0.771 at p = 0.05; gradient variance decays exponentially for a global cost with deep circuits (slope -0.98 per qubit) but not for a local cost with shallow circuits (-0.09).

## Correctness check
9 of 10 pytest tests pass (simulator against dense matrices, parameter shift against finite differences, metric against fidelity Hessian, TFIM limits, shot estimator unbiased, noise p = 0 identity); the failing test is a module-name collision (Failure Log case 11).

## Limitations
small sizes (n <= 10 QAOA, 4-qubit VQE); QAOA runs at n = 10, p = 3 hit the 300-evaluation cap; learning-rate grids for GD and SPSA tuned at the upper edge; C4 VQE part and C3(b) are invalid or inconclusive (cases 12 and 13); the noise model is a simple depolarizing-trajectory model.
