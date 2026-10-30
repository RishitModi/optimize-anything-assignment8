# Track I: Quantum Algorithms and QML - Benchmark Summary

## Environment

- **Python:** 3.12.10
- **NumPy:** 2.5.4
- **SciPy:** 1.18.1
- **Scikit-Learn:** 1.9.1
- **Platform:** Windows-11-10.0.26200-SP0
- **Quick Mode:** False
- **Deadline Guard:** 8.0 minutes

> **System Statements:**
> 1. All quantum results are simulated with a self-written numpy statevector simulator; no real quantum hardware was used.
> 2. Noise is a simple depolarizing-trajectory model, not calibrated to a device.
> 3. Embedding numbers are an analytic literature formula, not measured.

## Experiment I1: Quantum Algorithms

Textbook quantum algorithms (Deutsch-Jozsa, Bernstein-Vazirani, QFT, QPE, Shor's order finding) implemented gate-by-gate on top of qi_sim.

### (a) Deutsch-Jozsa Results

| n | oracle | p_all_zero | quantum_queries | classical_queries_worst |
| --- | --- | --- | --- | --- |
| 2 | constant0 | 1 | 1 | 3 |
| 2 | constant1 | 1 | 1 | 3 |
| 2 | balanced_random | 0 | 1 | 3 |
| 2 | balanced_parity | 0 | 1 | 3 |
| 3 | constant0 | 1 | 1 | 5 |
| 3 | constant1 | 1 | 1 | 5 |
| 3 | balanced_random | 0 | 1 | 5 |
| 3 | balanced_parity | 0 | 1 | 5 |
| 4 | constant0 | 1 | 1 | 9 |
| 4 | constant1 | 1 | 1 | 9 |
| 4 | balanced_random | 0 | 1 | 9 |
| 4 | balanced_parity | 0 | 1 | 9 |
| 5 | constant0 | 1 | 1 | 17 |
| 5 | constant1 | 1 | 1 | 17 |
| 5 | balanced_random | 0 | 1 | 17 |
| 5 | balanced_parity | 0 | 1 | 17 |
| 6 | constant0 | 1 | 1 | 33 |
| 6 | constant1 | 1 | 1 | 33 |
| 6 | balanced_random | 1.926e-34 | 1 | 33 |
| 6 | balanced_parity | 0 | 1 | 33 |
| 7 | constant0 | 1 | 1 | 65 |
| 7 | constant1 | 1 | 1 | 65 |
| 7 | balanced_random | 2.438e-34 | 1 | 65 |
| 7 | balanced_parity | 0 | 1 | 65 |
| 8 | constant0 | 1 | 1 | 129 |
| 8 | constant1 | 1 | 1 | 129 |
| 8 | balanced_random | 1.204e-35 | 1 | 129 |
| 8 | balanced_parity | 0 | 1 | 129 |

### (b) Bernstein-Vazirani Results

| n | num_secrets_tested | mean_secret_prob | quantum_queries | classical_queries |
| --- | --- | --- | --- | --- |
| 2 | 20 | 1 | 1 | 2 |
| 3 | 20 | 1 | 1 | 3 |
| 4 | 20 | 1 | 1 | 4 |
| 5 | 20 | 1 | 1 | 5 |
| 6 | 20 | 1 | 1 | 6 |
| 7 | 20 | 1 | 1 | 7 |
| 8 | 20 | 1 | 1 | 8 |
| 9 | 20 | 1 | 1 | 9 |
| 10 | 20 | 1 | 1 | 10 |

### (c) Quantum Fourier Transform (QFT)

| n | dimension | max_fft_diff | max_invert_diff | hadamards | cphase_gates | swap_gates | total_gates | time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 2.238e-16 | 1.57e-16 | 1 | 0 | 0 | 1 | 0.0003801 |
| 2 | 4 | 2.483e-16 | 4.003e-16 | 2 | 1 | 1 | 4 | 0.0004804 |
| 3 | 8 | 2.776e-16 | 3.554e-16 | 3 | 3 | 1 | 7 | 0.0006396 |
| 4 | 16 | 2.483e-16 | 4.775e-16 | 4 | 6 | 2 | 12 | 0.0009994 |
| 5 | 32 | 2.001e-16 | 3.237e-16 | 5 | 10 | 2 | 17 | 0.001438 |
| 6 | 64 | 2.289e-16 | 3.389e-16 | 6 | 15 | 3 | 24 | 0.004681 |
| 7 | 128 | 1.706e-16 | 2.31e-16 | 7 | 21 | 3 | 31 | 0.002796 |
| 8 | 256 | 1.309e-16 | 2.455e-16 | 8 | 28 | 4 | 40 | 0.00425 |
| 9 | 512 | 1.241e-16 | 1.707e-16 | 9 | 36 | 4 | 49 | 0.007755 |
| 10 | 1024 | 8.777e-17 | 1.522e-16 | 10 | 45 | 5 | 60 | 0.009289 |

### (d) Quantum Phase Estimation (QPE)

| t | true_phi | best_outcome | best_prob | estimated_phi | phase_error | window_mass |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.3 | 2 | 0.5775 | 0.25 | 0.05 | 0.8369 |
| 4 | 0.3 | 5 | 0.8756 | 0.3125 | 0.0125 | 0.9307 |
| 5 | 0.3 | 10 | 0.5731 | 0.3125 | 0.0125 | 0.8279 |
| 6 | 0.3 | 19 | 0.8752 | 0.2969 | 0.003125 | 0.9299 |
| 7 | 0.3 | 38 | 0.5728 | 0.2969 | 0.003125 | 0.8274 |
| 8 | 0.3 | 77 | 0.8751 | 0.3008 | 0.0007813 | 0.9298 |
| 9 | 0.3 | 154 | 0.5728 | 0.3008 | 0.0007813 | 0.8274 |
| 10 | 0.3 | 307 | 0.8751 | 0.2998 | 0.0001953 | 0.9298 |

### (e) Shor's Order Finding & Factorization

| N | a | true_order_r | p_order_exact | p_factor_success | expected_reps | shots_tested | shots_factored | qubits_total | ctrl_mod_mul_gates | sim_time_s | classical_trial_divs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15 | 2 | 4 | 0.75 | 0.75 | 1.333 | 200 | 143 | 12 | 8 | 0.00403 | 3 |
| 15 | 4 | 2 | 0.5 | 0.5 | 2 | 200 | 97 | 12 | 8 | 0.001754 | 3 |
| 15 | 7 | 4 | 0.75 | 0.75 | 1.333 | 200 | 143 | 12 | 8 | 0.003458 | 3 |
| 15 | 8 | 4 | 0.75 | 0.75 | 1.333 | 200 | 143 | 12 | 8 | 0.001707 | 3 |
| 15 | 11 | 2 | 0.5 | 0.5 | 2 | 200 | 97 | 12 | 8 | 0.001758 | 3 |
| 15 | 13 | 4 | 0.75 | 0.75 | 1.333 | 200 | 143 | 12 | 8 | 0.001802 | 3 |
| 21 | 2 | 6 | 0.8162 | 0.8165 | 1.225 | 200 | 156 | 15 | 10 | 0.01721 | 3 |
| 21 | 4 | 3 | 0.6548 | 0 | inf | 200 | 0 | 15 | 10 | 0.01813 | 3 |
| 21 | 5 | 6 | 0.8162 | 0 | inf | 200 | 0 | 15 | 10 | 0.01825 | 3 |
| 21 | 8 | 2 | 0.5 | 0.5 | 2 | 200 | 97 | 15 | 10 | 0.01792 | 3 |
| 21 | 10 | 6 | 0.8162 | 0.8165 | 1.225 | 200 | 156 | 15 | 10 | 0.01801 | 3 |
| 21 | 11 | 6 | 0.8162 | 0.8165 | 1.225 | 200 | 156 | 15 | 10 | 0.01789 | 3 |
| 21 | 13 | 2 | 0.5 | 0.5 | 2 | 200 | 97 | 15 | 10 | 0.01784 | 3 |
| 21 | 16 | 3 | 0.6548 | 0 | inf | 200 | 0 | 15 | 10 | 0.01819 | 3 |
| 21 | 17 | 6 | 0.8162 | 0 | inf | 200 | 0 | 15 | 10 | 0.01721 | 3 |
| 21 | 19 | 6 | 0.8162 | 0.8165 | 1.225 | 200 | 156 | 15 | 10 | 0.01909 | 3 |

## Experiment I2: Grover vs Classical Search

Query complexity and physical simulation runtime comparison between Grover unstructured search and classical linear scan.

> **Note on Runtime:** The statevector simulation of Grover is much slower than the classical scan, so the advantage is in the query count only.

| n | search_space_N | quantum_queries | quantum_success_prob | quantum_gates_total | quantum_sim_time_s | classical_mean_queries | classical_theory_queries | classical_worst_queries | classical_scan_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 16 | 3 | 0.9613 | 61 | 0.0001368 | 8.32 | 8.5 | 16 | 1.614e-05 |
| 5 | 32 | 4 | 0.9992 | 97 | 7.86e-05 | 16.32 | 16.5 | 32 | 1.959e-05 |
| 6 | 64 | 6 | 0.9966 | 168 | 8.53e-05 | 32.41 | 32.5 | 64 | 2.122e-05 |
| 7 | 128 | 8 | 0.9956 | 255 | 0.0001371 | 62.77 | 64.5 | 128 | 1.704e-05 |
| 8 | 256 | 12 | 0.9999 | 428 | 0.0001408 | 129 | 128.5 | 256 | 1.921e-05 |
| 9 | 512 | 17 | 0.9994 | 672 | 0.0002022 | 266.5 | 256.5 | 512 | 2.386e-05 |
| 10 | 1024 | 25 | 0.9995 | 1085 | 0.0003323 | 508.9 | 512.5 | 1024 | 3.719e-05 |
| 11 | 2048 | 35 | 1 | 1656 | 0.000524 | 994.4 | 1024 | 2048 | 4.694e-05 |
| 12 | 4096 | 50 | 0.9999 | 2562 | 0.001793 | 2130 | 2048 | 4096 | 8.726e-05 |
| 13 | 8192 | 71 | 0.9999 | 3918 | 0.001942 | 4075 | 4096 | 8192 | 0.0001583 |
| 14 | 16384 | 100 | 1 | 5914 | 0.004703 | 7982 | 8192 | 16384 | 0.0003228 |

## Experiment I3: Quantum Annealing and QUBO Embedding Limits

Transverse-field quantum annealing simulation via Strang splitting on Track B Max-Cut QUBO instances, and analytic TRIAD clique embedding resource requirements.

> **Note on QUBO Density:** A cardinality-penalty term such as lambda * (sum x - k)^2 (Track B) makes the QUBO dense, requiring the complete graph K_n, while a 3-regular Max-Cut instance has only 3n/2 couplings.

### Annealing Success Probabilities

| n | T | dt | success_prob | expected_energy | optimal_energy | energy_ratio | norm_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 1 | 0.05 | 0.4917 | -3.365 | -4 | 0.8412 | 4.441e-16 |
| 4 | 2 | 0.05 | 0.6628 | -3.658 | -4 | 0.9146 | 8.882e-16 |
| 4 | 5 | 0.05 | 0.7702 | -3.769 | -4 | 0.9423 | 1.776e-15 |
| 4 | 10 | 0.05 | 0.8711 | -3.871 | -4 | 0.9677 | 7.772e-16 |
| 4 | 20 | 0.05 | 0.953 | -3.953 | -4 | 0.9883 | 1.332e-14 |
| 4 | 50 | 0.05 | 0.9943 | -3.994 | -4 | 0.9986 | 3.331e-15 |
| 4 | 100 | 0.05 | 0.9987 | -3.999 | -4 | 0.9997 | 1.998e-15 |
| 6 | 1 | 0.05 | 0.06538 | -4.926 | -9 | 0.5473 | 1.665e-15 |
| 6 | 2 | 0.05 | 0.1672 | -5.638 | -9 | 0.6265 | 6.661e-16 |
| 6 | 5 | 0.05 | 0.4935 | -7.065 | -9 | 0.785 | 1.11e-15 |
| 6 | 10 | 0.05 | 0.8512 | -8.443 | -9 | 0.9381 | 5.551e-15 |
| 6 | 20 | 0.05 | 0.9707 | -8.903 | -9 | 0.9892 | 1.998e-15 |
| 6 | 50 | 0.05 | 0.9998 | -8.999 | -9 | 0.9999 | 1.399e-14 |
| 6 | 100 | 0.05 | 1 | -9 | -9 | 1 | 8.882e-16 |
| 8 | 1 | 0.05 | 0.03437 | -6.656 | -10 | 0.6656 | 1.776e-15 |
| 8 | 2 | 0.05 | 0.09018 | -7.566 | -10 | 0.7566 | 4.441e-16 |
| 8 | 5 | 0.05 | 0.2542 | -8.536 | -10 | 0.8536 | 8.882e-16 |
| 8 | 10 | 0.05 | 0.5646 | -9.352 | -10 | 0.9352 | 4.441e-15 |
| 8 | 20 | 0.05 | 0.9373 | -9.923 | -10 | 0.9923 | 1.732e-14 |
| 8 | 50 | 0.05 | 0.9947 | -9.995 | -10 | 0.9995 | 8.882e-15 |
| 8 | 100 | 0.05 | 0.9985 | -9.998 | -10 | 0.9998 | 5.995e-15 |
| 10 | 1 | 0.05 | 0.009864 | -8.338 | -12 | 0.6948 | 2.887e-15 |
| 10 | 2 | 0.05 | 0.03092 | -9.404 | -12 | 0.7837 | 7.772e-16 |
| 10 | 5 | 0.05 | 0.11 | -10.35 | -12 | 0.8625 | 1.332e-15 |
| 10 | 10 | 0.05 | 0.3058 | -11.06 | -12 | 0.9221 | 5.44e-15 |
| 10 | 20 | 0.05 | 0.6659 | -11.63 | -12 | 0.969 | 2.531e-14 |
| 10 | 50 | 0.05 | 0.9471 | -11.95 | -12 | 0.9955 | 1.044e-14 |
| 10 | 100 | 0.05 | 0.996 | -12 | -12 | 0.9997 | 4.219e-15 |

### Analytic Chimera Embedding Table

| n_logical | M | chain_length | physical_qubits | fits_in_2048 | chip_fraction | note |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 1 | 2 | 8 | True | 0.003906 | analytic (literature formula), not measured |
| 8 | 2 | 3 | 24 | True | 0.01172 | analytic (literature formula), not measured |
| 12 | 3 | 4 | 48 | True | 0.02344 | analytic (literature formula), not measured |
| 16 | 4 | 5 | 80 | True | 0.03906 | analytic (literature formula), not measured |
| 20 | 5 | 6 | 120 | True | 0.05859 | analytic (literature formula), not measured |
| 32 | 8 | 9 | 288 | True | 0.1406 | analytic (literature formula), not measured |
| 48 | 12 | 13 | 624 | True | 0.3047 | analytic (literature formula), not measured |
| 64 | 16 | 17 | 1088 | True | 0.5312 | analytic (literature formula), not measured |
| 65 | 17 | 18 | 1224 | False | 0.5977 | analytic (literature formula), not measured |
| 100 | 25 | 26 | 2600 | False | 1.27 | analytic (literature formula), not measured |

## Experiment I4: Hardware Noise Model (Simulated)

Depolarizing error channel trajectory simulation across varying error rates for Grover, Bernstein-Vazirani, and Deutsch-Jozsa.

| noise_p | trajectories | grover_succ_mean | grover_succ_stderr | bv_succ_mean | bv_succ_stderr | dj_wrong_mean | dj_wrong_stderr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 200 | 0.9966 | 7.85e-18 | 1 | 0 | 0 | 0 |
| 0.001 | 200 | 0.9679 | 0.01145 | 0.995 | 0.004987 | 0 | 0 |
| 0.002 | 200 | 0.9607 | 0.01143 | 0.98 | 0.009899 | 0 | 0 |
| 0.005 | 200 | 0.8612 | 0.02264 | 0.95 | 0.01541 | 0 | 0 |
| 0.01 | 200 | 0.7775 | 0.02694 | 0.89 | 0.02212 | 0 | 0 |
| 0.02 | 200 | 0.5912 | 0.03214 | 0.73 | 0.03139 | 0 | 0 |
| 0.05 | 200 | 0.3074 | 0.02817 | 0.535 | 0.03527 | 0 | 0 |

## Experiment I5: QML vs Classical Baselines

Variational Quantum Classifier with data re-uploading and parameter-shift gradients, exact and shot-based Quantum Kernel SVMs, and classical baselines on moons and circles.

| dataset | model | test_accuracy_mean | test_accuracy_std | train_accuracy_mean | wall_time_mean_s | parameter_count | n_qubits | circuit_evaluations_train | circuit_evaluations_pred |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| moons | variational_classifier | 0.925 | 0.00527 | 0.9233 | 0.878 | 12 | 2 | 90000 | 240 |
| moons | quantum_kernel_exact | 0.8083 | 0 | 0.7917 | 0.002325 | 87 | 2 | 7140 | 14400 |
| moons | quantum_kernel_1000shots | 0.8167 | 0 | 0.7833 | 0.005323 | 87 | 2 | 7140 | 14400 |
| moons | logistic_regression | 0.8833 | 0 | 0.8917 | 0.002558 | 3 | 0 | 0 | 0 |
| moons | svm_rbf | 0.9417 | 0 | 0.95 | 0.0007783 | 38 | 0 | 0 | 0 |
| moons | mlp_hidden4 | 0.8833 | 0 | 0.9083 | 0.2076 | 17 | 0 | 0 | 0 |
| circles | variational_classifier | 0.9683 | 0.02656 | 0.985 | 0.9261 | 12 | 2 | 90000 | 240 |
| circles | quantum_kernel_exact | 0.9583 | 0 | 0.9667 | 0.002809 | 81 | 2 | 7140 | 14400 |
| circles | quantum_kernel_1000shots | 0.9583 | 0 | 0.9667 | 0.006311 | 78 | 2 | 7140 | 14400 |
| circles | logistic_regression | 0.475 | 0 | 0.5417 | 0.001922 | 3 | 0 | 0 | 0 |
| circles | svm_rbf | 0.9917 | 0 | 1 | 0.001419 | 40 | 0 | 0 | 0 |
| circles | mlp_hidden4 | 0.9833 | 0 | 0.9583 | 0.404 | 17 | 0 | 0 | 0 |

## Generated Figures

- `i1_qpe_scaling.png`
- `i1_shor_distribution.png`
- `i2_grover_curves.png`
- `i2_queries_scaling.png`
- `i2_runtime_comparison.png`
- `i3_anneal_success.png`
- `i3_embedding_scaling.png`
- `i4_noise_degradation.png`
- `i5_vqc_loss_curve.png`
- `i5_accuracy_comparison.png`

## Verification

No verification failures. All theoretical invariants, exact checks, and physical constraints passed.
