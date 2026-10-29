# Track C Quantum Optimization – Summary

## Environment

- **Python version**: 3.12.10
- **NumPy version**: 2.5.4
- **SciPy version**: 1.18.1
- **Platform**: Windows-11-10.0.26200-SP0
- **CPU count**: 18
- **Quick mode**: False
- **Base seeds**: 42, 777, 1000, 2026, 8888, 9999
- All results come from a self-written numpy statevector simulator; no real quantum hardware was used

## Experiment C1: QAOA for Max-Cut

**Setup**: Evaluated QAOA on 3-regular graph Max-Cut instances from Track B with n in [4, 6, 8, 10] across circuit depths p in [1, 2, 3] using multi-start COBYLA optimization (3 starts in quick mode, 10 in full mode). Compared against exact optimal cut and simulated annealing classical baselines. Verified that qubo_diag matches ising_diag, p=0 expectation matches mean diagonal energy, and approximation ratios do not exceed 1.

### C1 QAOA Results Table

| n | p | ratio_mean | ratio_std | ratio_best | prob_opt_mean | prob_opt_best | evals_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 1 | 0.9244 | 1.117e-08 | 0.9244 | 0.7391 | 0.7392 | 42.4 |
| 4 | 2 | 0.9964 | 0.01074 | 1 | 0.9906 | 1 | 260.2 |
| 4 | 3 | 1 | 3.238e-05 | 1 | 1 | 1 | 219.6 |
| 6 | 1 | 0.6925 | 6.464e-09 | 0.6925 | 0.3342 | 0.3342 | 39 |
| 6 | 2 | 0.8611 | 0.03713 | 0.8911 | 0.7277 | 0.7614 | 142.2 |
| 6 | 3 | 0.9238 | 0.0559 | 0.9842 | 0.8818 | 0.9714 | 264 |
| 8 | 1 | 0.8007 | 8.876e-09 | 0.8007 | 0.1427 | 0.1427 | 40.8 |
| 8 | 2 | 0.8695 | 0.009137 | 0.8784 | 0.3133 | 0.3494 | 171 |
| 8 | 3 | 0.8924 | 0.02491 | 0.9361 | 0.4165 | 0.6167 | 287.9 |
| 10 | 1 | 0.818 | 2.373e-09 | 0.818 | 0.0489 | 0.0489 | 40.1 |
| 10 | 2 | 0.8813 | 0.01604 | 0.8926 | 0.1745 | 0.2242 | 248.4 |
| 10 | 3 | 0.8899 | 0.06497 | 0.9314 | 0.2502 | 0.401 | 300 |

### C1 Classical Baselines Table

| n | optimal_cut | sa_best_cut | sa_success_rate |
| --- | --- | --- | --- |
| 4 | 4 | 4 | 1 |
| 6 | 9 | 9 | 1 |
| 8 | 10 | 10 | 1 |
| 10 | 12 | 12 | 1 |

**Figures**: c1_qaoa_ratio_vs_p.png, c1_qaoa_prob_opt_vs_p.png

## Experiment C2: VQE Optimizer Comparison

**Setup**: Trained an L=3 real hardware-efficient ansatz (P=16) on an open Transverse-Field Ising chain (n=4, g=1.0) under a strict budget of circuit queries (600 in quick mode, 4000 in full mode). Compared Parameter-Shift Gradient Descent, Quantum Natural Gradient (cost assumption: 2P evaluations for analytic gradient plus P+1 circuit queries to prepare metric derivative states), SPSA, and COBYLA. Hyperparameters were tuned on a separate seed. Verified parameter-shift gradients and Fubini-Study metric against finite differences, and confirmed energies respect the variational ground-state lower bound.

### C2 VQE Optimization Summary Table

| optimizer | tuned_hyperparameter | evals_per_iteration | final_error_mean | final_error_std | evals_to_error_below_1e-2 | runs_reaching_threshold | exact_ground_energy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GD_ParamShift | lr=0.3 | 32 | 0.02577 | 0.02193 | 3072 | 1/5 | -4.759 |
| QNG | lr=0.1 | 49 | 0.002573 | 0.00513 | 943.2 | 4/5 | -4.759 |
| SPSA | a=1.0,c=0.1 | 2 | 0.02049 | 0.0168 | 1692 | 1/5 | -4.759 |
| COBYLA | maxiter=budget | 1 | 0.01034 | 0.003353 | 1990 | 2/5 | -4.759 |

**Figures**: c2_vqe_convergence.png

## Experiment C3: Finite-Shot Measurement Noise

**Setup**: Analyzed measurement noise by partitioning the TFIM Hamiltonian into commuting Z/I and X/I measurement groups across shot counts [100, 1000, 10000, 100000] (quick: up to 10000) for 200 repetitions. Evaluated empirical bias, RMSE, and the log-log scaling slope against the theoretical O(1/sqrt(shots)) bound. Benchmarked SPSA with 1000-shot finite measurement loss against exact expectation SPSA under equal query budgets.

### C3 Shot Noise Scaling Table

| parameter_set | shots | bias | rmse | exact_energy |
| --- | --- | --- | --- | --- |
| best_params | 100 | -0.02267 | 0.2402 | 0.4234 |
| best_params | 1000 | -0.001023 | 0.08606 | 0.4234 |
| best_params | 10000 | 0.002678 | 0.02644 | 0.4234 |
| best_params | 100000 | -0.000467 | 0.008128 | 0.4234 |
| best_params_fitted_slope | ALL | 0 | -0.4925 | 0.4234 |
| random_params | 100 | -0.004112 | 0.233 | -0.4301 |
| random_params | 1000 | 0.003578 | 0.07276 | -0.4301 |
| random_params | 10000 | 0.0008301 | 0.02586 | -0.4301 |
| random_params | 100000 | 0.0001425 | 0.007135 | -0.4301 |
| random_params_fitted_slope | ALL | 0 | -0.4991 | -0.4301 |

### C3 SPSA Shot Comparison Table

| method | shots_per_eval | final_error_mean | final_error_std |
| --- | --- | --- | --- |
| Exact_Expectation_SPSA | exact | 0.155 | 0.08426 |
| Shot_Based_SPSA | 1000 | 0.1254 | 0.08493 |

**Figures**: c3_shot_rmse_scaling.png, c3_spsa_shot_convergence.png

## Experiment C4: Simulated Depolarizing Hardware Noise

**Setup**: Evaluated degradation of trained QAOA and VQE parameter states under simulated single-qubit depolarizing noise channels with error probabilities p in [0, 0.002, 0.005, 0.01, 0.02, 0.05] across 200 independent stochastic trajectories per noise level. Note: This is a simple simulated noise model, not calibrated to any real device, and no real quantum hardware was used anywhere in Track C.

### C4 QAOA Noise Table

| p_noise | mean_ratio | stderr_ratio |
| --- | --- | --- |
| 0 | 0.8784 | 1.57e-17 |
| 0.002 | 0.8744 | 0.001364 |
| 0.005 | 0.8664 | 0.002554 |
| 0.01 | 0.8553 | 0.003239 |
| 0.02 | 0.8224 | 0.004813 |
| 0.05 | 0.7713 | 0.006033 |

### C4 VQE Noise Table

| p_noise | mean_energy_error | stderr_energy_error |
| --- | --- | --- |
| 0 | 5.182 | 0 |
| 0.002 | 5.139 | 0.02799 |
| 0.005 | 5.189 | 0.02709 |
| 0.01 | 5.111 | 0.0372 |
| 0.02 | 5.116 | 0.05069 |
| 0.05 | 4.971 | 0.06727 |

**Figures**: c4_noise_impact.png

## Experiment C5: Barren Plateaus and Observable Locality

**Setup**: Investigated barren plateau phenomena by computing the variance of the first parameter's gradient across 200 Haar-random initializations for n in [2, 4, 6, 8, 10, 12] qubits. Compared deep circuits (L=n) against shallow circuits (L=2) for both a global observable (Z on all qubits) and a local observable (Z_0 Z_1). Reported empirical variances and fitted log2 scaling slopes.

### C5 Barren Plateaus Scaling Table

| setting | n | variance | log2_variance |
| --- | --- | --- | --- |
| Global_Deep | 2 | 0.216 | -2.211 |
| Global_Shallow | 2 | 0.2186 | -2.194 |
| Local_Deep | 2 | 0.2163 | -2.209 |
| Local_Shallow | 2 | 0.242 | -2.047 |
| Global_Deep | 4 | 0.06281 | -3.993 |
| Global_Shallow | 4 | 0.07445 | -3.748 |
| Local_Deep | 4 | 0.07014 | -3.834 |
| Local_Shallow | 4 | 0.1143 | -3.129 |
| Global_Deep | 6 | 0.01729 | -5.854 |
| Global_Shallow | 6 | 0.02507 | -5.318 |
| Local_Deep | 6 | 0.02295 | -5.446 |
| Local_Shallow | 6 | 0.08566 | -3.545 |
| Global_Deep | 8 | 0.003792 | -8.043 |
| Global_Shallow | 8 | 0.01127 | -6.471 |
| Local_Deep | 8 | 0.01047 | -6.577 |
| Local_Shallow | 8 | 0.09714 | -3.364 |
| Global_Deep | 10 | 0.0009815 | -9.993 |
| Global_Shallow | 10 | 0.003156 | -8.307 |
| Local_Deep | 10 | 0.004225 | -7.887 |
| Local_Shallow | 10 | 0.09676 | -3.369 |
| Global_Deep | 12 | 0.0002623 | -11.9 |
| Global_Shallow | 12 | 0.001598 | -9.289 |
| Local_Deep | 12 | 0.002348 | -8.734 |
| Local_Shallow | 12 | 0.1055 | -3.244 |

### C5 Fitted Log2 Slopes Table

| setting | fitted_log2_slope | intercept |
| --- | --- | --- |
| Global_Deep | -0.9802 | -0.1365 |
| Global_Shallow | -0.7187 | -0.8568 |
| Local_Deep | -0.656 | -1.189 |
| Local_Shallow | -0.09324 | -2.464 |

**Figures**: c5_barren_plateaus.png

## Verification

No verification failures
