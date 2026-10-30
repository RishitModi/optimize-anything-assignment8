# Track I: Quantum Algorithms and QML

## Objective
The objective of Track I is exploring foundational quantum algorithms, oracle-based query complexity advantages, transverse-field quantum annealing with physical graph embeddings, and Quantum Machine Learning (QML) paradigms compared directly against classical optimization and learning baselines.

## Baseline
- **Oracle & Query Problems:** Classical deterministic query complexity ($2^{n-1} + 1$ for Deutsch-Jozsa, $n$ for Bernstein-Vazirani), classical unstructured random linear scan ($O(N)$ query expectation), and classical trial division for prime factorization.
- **Machine Learning Baselines:** Standard scikit-learn classical models evaluated on identical stratified train/test splits: Logistic Regression (linear baseline), Support Vector Classifier with RBF kernel, and Multi-Layer Perceptron (4 hidden units).

## Formulation

### Algorithmic Complexity and Quantum Speedups
- **Deutsch-Jozsa:** Distinguishes constant vs balanced boolean functions in **1 quantum query** versus $2^{n-1} + 1$ classically deterministic queries in the worst case.
- **Bernstein-Vazirani:** Reconstructs a hidden bitstring $s \in \{0, 1\}^n$ in **1 quantum query** versus $n$ classical queries.
- **Grover's Search:** Amplifies marked state amplitudes in an unstructured database of size $N = 2^n$ in $O(\sqrt{N/M})$ queries, achieving a quadratic speedup over the classical expectation $O(N)$.
- **Quantum Fourier Transform (QFT):** Operates on $n$ qubits using $O(n^2)$ elementary quantum gates ($n$ Hadamards, $n(n-1)/2$ controlled-phase rotations, $\lfloor n/2 \rfloor$ swaps), compared to classical Fast Fourier Transform (FFT) requiring $O(n 2^n)$ operations on $2^n$ amplitudes.
- **Shor's Algorithm:** Reduces integer factorization to order finding via phase estimation, achieving polynomial gate scaling $O(n^3)$ versus the best-known classical sub-exponential algorithms (General Number Field Sieve).

### Variational Quantum Classifier (VQC)
- **Data Re-Uploading:** 2-qubit circuit with $L = 3$ layers combining input feature rotations $R_X(x_j)$ with trainable rotations $R_Y(\theta) R_Z(\theta)$ and entangling CNOT gates ($P = 12$ parameters).
- **Observable & Loss:** Predicts label via expectation $f(x) = \langle Z_0 \rangle \in [-1, 1]$. Mean squared error objective:
  $$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^B (f(x_i; \theta) - y_i)^2, \quad y_i \in \{-1, +1\}$$
- **Parameter-Shift Rule:** Evaluates exact analytical gradients on quantum circuits:
  $$\frac{\partial f(x)}{\partial \theta_j} = \frac{1}{2} \left[ f\left(x; \theta + \frac{\pi}{2} e_j\right) - f\left(x; \theta - \frac{\pi}{2} e_j\right) \right]$$

### Quantum Kernel Method
- Evaluates transition amplitudes under a repeated non-linear feature map:
  $$K(x_i, x_j) = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$$
  generating a symmetric positive-semidefinite Gram matrix utilized by a dual-form Support Vector Classifier (evaluated in exact and 1000-shot finite-sampling regimes).

### Quantum Annealing Hamiltonian
- Continuous transverse-field annealing schedule:
  $$H(s) = -(1 - s) \sum_{i=0}^{n-1} X_i + s H_P, \quad s = \frac{t}{T} \in [0, 1]$$
  where $H_P = \text{diag}(x^T Q x)$ represents the QUBO problem Hamiltonian.

### Simulation vs. Hardware Disclaimer
All quantum algorithms, circuits, noise trajectories, and annealing dynamics are simulated numerically using a self-written NumPy statevector simulator; no real quantum hardware was used. Noise is simulated via a depolarizing-trajectory error model.

## Methods
- `qi_sim.py`: Pure NumPy statevector simulator implementing tensor contractions via `np.einsum`, batched circuit execution, single- and two-qubit unitary operations, and stochastic error layers.
- `qi_algos.py`: Gate-level implementations of Deutsch-Jozsa, Bernstein-Vazirani, QFT, QPE, Shor's order finding (with continued fractions post-processing), and Grover search.
- `qi_anneal.py`: Symmetric Trotter (Strang) splitting simulator for continuous annealing, QUBO energy spectrum analysis, and analytic TRIAD Chimera clique embedding models.
- `qi_qml.py`: Variational Quantum Classifier trained with mini-batch Adam, analytical parameter-shift gradient computation, exact and shot-based quantum kernel SVMs, and classical ML baselines.

## How to run
```bash
python tracks/track_i_quantum_algos_qml/run_all.py [--quick]
python -m pytest tracks/track_i_quantum_algos_qml/tests -q
```

## Results
<!-- Experimental outcomes, comparison tables, and plots saved in results/ -->

## Correctness check
<!-- Verification steps, assertions, unit tests, and theoretical invariants checked -->

## Limitations
<!-- Computational bottlenecks, approximations, assumptions, and failure modes -->
