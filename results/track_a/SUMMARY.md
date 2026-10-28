# Track A Continuous Optimization – Summary

## Environment

- **Python version**: 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
- **NumPy version**: 2.5.4
- **SciPy version**: 1.18.1
- **scikit-learn version**: 1.9.1
- **Platform**: Windows-11-10.0.26200-SP0
- **CPU count**: 18
- **Quick mode**: False
- **Fixed base seed**: 42
- **Stochastic experiments seeds**: 5 (2 in quick mode)

---

## EXP1

**Status**: SUCCESS
**Runtime**: 115.65 s

**Setup**: **Optimizer Comparison.** Problems: quadratic(cond=100, dim=10) and rosenbrock(dim=2, x0=(-1.2,1)). Methods: GradientDescent, Momentum, NAG, AdaGrad, RMSProp, Adam (all first-order) and Newton (second-order, line_search=True for Rosenbrock). Budget: 2000 iters (quadratic), 20000 iters (Rosenbrock) in full mode. Learning rate grid: np.logspace(-4, 0, 9) per method per problem; best lr selected by lowest final f among non-diverged runs. Momentum beta=0.9, Adam betas=(0.9,0.999), RMSProp rho=0.9. No seeds (deterministic problems). Outputs: convergence plots (log y) and results table.


### tuned_lr_table

| problem | method | tuned_lr |
| --- | --- | --- |
| quadratic | GradientDescent | 0.01 |
| quadratic | Momentum | 0.0316228 |
| quadratic | NAG | 0.00316228 |
| quadratic | AdaGrad | 1 |
| quadratic | RMSProp | 0.000316228 |
| quadratic | Adam | 0.1 |
| quadratic | Newton | N/A |
| rosenbrock | GradientDescent | 0.001 |
| rosenbrock | Momentum | 0.00316228 |
| rosenbrock | NAG | 0.001 |
| rosenbrock | AdaGrad | 0.316228 |
| rosenbrock | RMSProp | 0.000316228 |
| rosenbrock | Adam | 0.000316228 |
| rosenbrock | Newton | N/A |


### results_table

| problem | method | tuned_lr | final_f | final_gradnorm | iters_to_tol | runtime_s |
| --- | --- | --- | --- | --- | --- | --- |
| quadratic | GradientDescent | 0.01 | -0.396443 | 9.96535e-07 | 609 | 0.012362 |
| quadratic | Momentum | 0.0316228 | -0.396443 | 9.39779e-07 | 115 | 0.0028042 |
| quadratic | NAG | 0.00316228 | -0.396443 | 9.42807e-07 | 118 | 0.0062816 |
| quadratic | AdaGrad | 1 | -0.396443 | 9.78101e-07 | 248 | 0.007513 |
| quadratic | RMSProp | 0.000316228 | -0.396434 | 0.0391914 | not reached | 0.0304848 |
| quadratic | Adam | 0.1 | -0.396443 | 8.39205e-07 | 137 | 0.0050007 |
| quadratic | Newton | N/A | -0.396443 | 7.16465e-15 | 1 | 0.0427683 |
| rosenbrock | GradientDescent | 0.001 | 1.93849e-08 | 0.000124445 | 15068 | 0.65524 |
| rosenbrock | Momentum | 0.00316228 | 1.24427e-12 | 9.9691e-07 | 394 | 0.0275646 |
| rosenbrock | NAG | 0.001 | 1.24433e-12 | 9.96935e-07 | 1426 | 0.588069 |
| rosenbrock | AdaGrad | 0.316228 | 1.11435e-12 | 9.99293e-07 | 8663 | 4.56765 |
| rosenbrock | RMSProp | 0.000316228 | 2.25295e-05 | 0.21241 | not reached | 5.46287 |
| rosenbrock | Adam | 0.000316228 | 9.33131e-13 | 9.95265e-07 | 17931 | 4.95472 |
| rosenbrock | Newton | N/A | 3.74398e-21 | 4.47333e-10 | 19 | 0.0061517 |


**Figures**: exp1_convergence_quadratic.png, exp1_convergence_rosenbrock.png

---

## EXP2

**Status**: SUCCESS
**Runtime**: 4.38 s

**Setup**: **Landscape Analysis.** (a) GradientDescent on three convexity classes: strongly convex quadratic(cond=10), convex-not-SC sum_quartic(dim=5), non-convex Rosenbrock. Log-scale convergence plot; slope of log(gap) fitted on second half. (b) Saddle function: GD, Momentum, Adam, Newton (undamped and damped) from a small perturbation (1e-3, 1e-3); 2-D contour plot with trajectories; stationary-point classification via Hessian eigenvalues. (c) Double-well: GD and Adam from three starting points; contour plot with trajectories; table of final points and whether they reach the global minimum (checked via grid search).


### part_a/table

| objective | convexity | expected_rate | log_gap_slope_2nd_half | final_gap |
| --- | --- | --- | --- | --- |
| quadratic(cond=10) | strongly convex | linear | 0 | 1e-16 |
| sum_quartic(dim=5) | convex (not SC) | sublinear | -0.000544678 | 1.24583e-06 |
| rosenbrock(dim=2) | non-convex | none guaranteed | nan | 0.00376092 |


### part_b/table

| method | final_x0 | final_x1 | final_f | classification |
| --- | --- | --- | --- | --- |
| GradientDescent | 3.42661e-12 | 1.41421 | -1 | minimum |
| Momentum | -1.03294e-13 | 1.41421 | -1 | minimum |
| Adam | -3.7614e-12 | 1.41421 | -1 | minimum |
| Newton (undamped) | 0 | 0 | 0 | saddle |
| Newton (damped) | 2.78813e-11 | 1.41421 | -1 | minimum |


### part_c/table

| start | method | final_x0 | final_x1 | final_f | is_global_minimum |
| --- | --- | --- | --- | --- | --- |
| negative_x | GradientDescent | -1.30084 | 0 | -3.51391 | True |
| negative_x | Adam | -1.30084 | 0 | -3.51391 | True |
| near_zero | GradientDescent | -1.30084 | 0 | -3.51391 | True |
| near_zero | Adam | -1.30084 | 0 | -3.51391 | True |
| positive_x | GradientDescent | 1.1309 | 0 | -1.07023 | False |
| positive_x | Adam | 1.1309 | 0 | -1.07023 | False |


**Figures**: exp2a_convexity.png, exp2b_saddle_trajectories.png, exp2c_double_well_trajectories.png

---

## EXP3

**Status**: SUCCESS
**Runtime**: 4.60 s

**Setup**: **Step-Size Sensitivity.** Problem: quadratic(cond=50, dim=10). Methods: GradientDescent, Momentum, NAG, AdaGrad, RMSProp, Adam. Budget: 500 iterations (fixed). LR sweep: np.logspace(-4, 0.5, 20) in full mode. Diverged runs recorded as NaN. Robustness metric: width of lr range (in decades) where final gap < 1e-3.


### results_table

| optimizer | lr | final_gap | diverged |
| --- | --- | --- | --- |
| GradientDescent | 0.0001 | 0.323657 | False |
| GradientDescent | 0.000172521 | 0.256886 | False |
| GradientDescent | 0.000297635 | 0.190088 | False |
| GradientDescent | 0.000513483 | 0.132682 | False |
| GradientDescent | 0.000885867 | 0.0859623 | False |
| GradientDescent | 0.00152831 | 0.0447658 | False |
| GradientDescent | 0.00263665 | 0.0147391 | False |
| GradientDescent | 0.00454878 | 0.00216297 | False |
| GradientDescent | 0.0078476 | 7.82446e-05 | False |
| GradientDescent | 0.0135388 | 2.48366e-07 | False |
| GradientDescent | 0.0233572 | 1.12413e-11 | False |
| GradientDescent | 0.0402961 | 22417.3 | False |
| GradientDescent | 0.0695193 | nan | True |
| GradientDescent | 0.119935 | nan | True |
| GradientDescent | 0.206914 | nan | True |
| GradientDescent | 0.35697 | nan | True |
| GradientDescent | 0.615848 | nan | True |
| GradientDescent | 1.06247 | nan | True |
| GradientDescent | 1.83298 | nan | True |
| GradientDescent | 3.16228 | nan | True |
| Momentum | 0.0001 | 0.0769525 | False |
| Momentum | 0.000172521 | 0.0369162 | False |
| Momentum | 0.000297635 | 0.0102047 | False |
| Momentum | 0.000513483 | 0.00102377 | False |
| Momentum | 0.000885867 | 1.45852e-05 | False |
| Momentum | 0.00152831 | 2.87783e-09 | False |
| Momentum | 0.00263665 | -1.11022e-16 | False |
| Momentum | 0.00454878 | 1.11022e-16 | False |
| Momentum | 0.0078476 | 1.11022e-16 | False |
| Momentum | 0.0135388 | 0 | False |
| Momentum | 0.0233572 | 0 | False |
| Momentum | 0.0402961 | 3.33067e-16 | False |
| Momentum | 0.0695193 | 2.22045e-16 | False |
| Momentum | 0.119935 | nan | True |
| Momentum | 0.206914 | nan | True |
| Momentum | 0.35697 | nan | True |
| Momentum | 0.615848 | nan | True |
| Momentum | 1.06247 | nan | True |
| Momentum | 1.83298 | nan | True |
| Momentum | 3.16228 | nan | True |
| NAG | 0.0001 | 0.0770338 | False |
| NAG | 0.000172521 | 0.0370163 | False |
| NAG | 0.000297635 | 0.0102903 | False |
| NAG | 0.000513483 | 0.00105141 | False |
| NAG | 0.000885867 | 1.59682e-05 | False |
| NAG | 0.00152831 | 4.14461e-09 | False |
| NAG | 0.00263665 | 1.11022e-16 | False |
| NAG | 0.00454878 | 4.44089e-16 | False |
| NAG | 0.0078476 | 1.11022e-16 | False |
| NAG | 0.0135388 | 3.33067e-16 | False |
| NAG | 0.0233572 | 2.22045e-16 | False |
| NAG | 0.0402961 | nan | True |
| NAG | 0.0695193 | nan | True |
| NAG | 0.119935 | nan | True |
| NAG | 0.206914 | nan | True |
| NAG | 0.35697 | nan | True |
| NAG | 0.615848 | nan | True |
| NAG | 1.06247 | nan | True |
| NAG | 1.83298 | nan | True |
| NAG | 3.16228 | nan | True |
| AdaGrad | 0.0001 | 0.53183 | False |
| AdaGrad | 0.000172521 | 0.508808 | False |
| AdaGrad | 0.000297635 | 0.472632 | False |
| AdaGrad | 0.000513483 | 0.418629 | False |
| AdaGrad | 0.000885867 | 0.344908 | False |
| AdaGrad | 0.00152831 | 0.25835 | False |
| AdaGrad | 0.00263665 | 0.176476 | False |
| AdaGrad | 0.00454878 | 0.110661 | False |
| AdaGrad | 0.0078476 | 0.0473641 | False |
| AdaGrad | 0.0135388 | 0.00671988 | False |
| AdaGrad | 0.0233572 | 4.2768e-05 | False |
| AdaGrad | 0.0402961 | 2.12411e-10 | False |
| AdaGrad | 0.0695193 | 4.44089e-16 | False |
| AdaGrad | 0.119935 | 3.33067e-16 | False |
| AdaGrad | 0.206914 | 4.44089e-16 | False |
| AdaGrad | 0.35697 | 2.22045e-16 | False |
| AdaGrad | 0.615848 | 3.33067e-16 | False |
| AdaGrad | 1.06247 | 1.11022e-16 | False |
| AdaGrad | 1.83298 | 1.11022e-16 | False |
| AdaGrad | 3.16228 | 2.22045e-16 | False |
| RMSProp | 0.0001 | 0.287433 | False |
| RMSProp | 0.000172521 | 0.192212 | False |
| RMSProp | 0.000297635 | 0.126386 | False |
| RMSProp | 0.000513483 | 0.0608355 | False |
| RMSProp | 0.000885867 | 0.00462123 | False |
| RMSProp | 0.00152831 | 0.000111564 | False |
| RMSProp | 0.00263665 | 0.000363609 | False |
| RMSProp | 0.00454878 | 0.00126019 | False |
| RMSProp | 0.0078476 | 0.00367256 | False |
| RMSProp | 0.0135388 | 0.00807444 | False |
| RMSProp | 0.0233572 | 0.0330772 | False |
| RMSProp | 0.0402961 | 0.106794 | False |
| RMSProp | 0.0695193 | 0.125708 | False |
| RMSProp | 0.119935 | 1.09894 | False |
| RMSProp | 0.206914 | 2.22549 | False |
| RMSProp | 0.35697 | 6.98984 | False |
| RMSProp | 0.615848 | 17.0401 | False |
| RMSProp | 1.06247 | 102.31 | False |
| RMSProp | 1.83298 | 214.824 | False |
| RMSProp | 3.16228 | 754.513 | False |
| Adam | 0.0001 | 0.306883 | False |
| Adam | 0.000172521 | 0.221085 | False |
| Adam | 0.000297635 | 0.149446 | False |
| Adam | 0.000513483 | 0.0894951 | False |
| Adam | 0.000885867 | 0.0322129 | False |
| Adam | 0.00152831 | 0.0031519 | False |
| Adam | 0.00263665 | 9.25775e-06 | False |
| Adam | 0.00454878 | 1.11022e-16 | False |
| Adam | 0.0078476 | 3.33067e-16 | False |
| Adam | 0.0135388 | 1.11022e-16 | False |
| Adam | 0.0233572 | 1.11022e-16 | False |
| Adam | 0.0402961 | 1.02734e-11 | False |
| Adam | 0.0695193 | 6.66134e-16 | False |
| Adam | 0.119935 | 1.96509e-14 | False |
| Adam | 0.206914 | 1.80608e-06 | False |
| Adam | 0.35697 | 9.72112e-07 | False |
| Adam | 0.615848 | 0.0130102 | False |
| Adam | 1.06247 | 2.22045e-16 | False |
| Adam | 1.83298 | 2.77097e-06 | False |
| Adam | 3.16228 | 9.19894e-10 | False |


### robustness_table

| optimizer | best_lr | best_gap | range_width_decades |
| --- | --- | --- | --- |
| GradientDescent | 0.0233572 | 1.12413e-11 | 0.473684 |
| Momentum | 0.00263665 | -1.11022e-16 | 1.89474 |
| NAG | 0.00263665 | 1.11022e-16 | 1.42105 |
| AdaGrad | 1.06247 | 1.11022e-16 | 2.13158 |
| RMSProp | 0.00152831 | 0.000111564 | 0.236842 |
| Adam | 0.00454878 | 1.11022e-16 | 3.07895 |


**Figures**: exp3_stepsize_sensitivity.png

---

## EXP4

**Status**: SUCCESS
**Runtime**: 84.46 s

**Setup**: **Stochastic and Noisy Optimization.** (a) Mini-batch study: logistic regression (n=2000, d=20, l2=1e-2). Configurations: GD (full batch), SGD-256, SGD-32, SGD-1, Momentum-32, Adam-32. 30 epochs, lr tuned per configuration from np.logspace(-3,0,7), 5 seeds. Plots: full-batch loss vs epoch and vs wall-clock time (mean ± std band). (b) Noisy objective: quadratic(cond=20, dim=10) with gradient noise sigma in [0, 0.1, 0.5, 1.0]; methods: GD (lr=1/L), Momentum, Adam, GD-Decay (lr_t = lr0/(1+0.01*t)); 10 seeds, 1000 iters. Plot: final gap vs sigma (log y) with std bars.


### part_a/table

| config | batch_size | tuned_lr | final_loss_mean | final_loss_std | time_per_epoch_s | grad_evals_per_epoch |
| --- | --- | --- | --- | --- | --- | --- |
| GD (n) | 2000 | 1 | 0.609653 | 8.59975e-17 | 0.000989599 | 1 |
| SGD-256 | 256 | 0.1 | 0.609669 | 2.94566e-06 | 0.00146539 | 8 |
| SGD-32 | 32 | 0.0316228 | 0.609688 | 1.24031e-05 | 0.00674473 | 63 |
| SGD-1 | 1 | 0.001 | 0.609669 | 6.31182e-06 | 0.0553606 | 2000 |
| Momentum-32 | 32 | 0.0316228 | 0.609688 | 1.24031e-05 | 0.00106067 | 63 |
| Adam-32 | 32 | 0.0316228 | 0.609688 | 1.24031e-05 | 0.00315721 | 63 |


### part_b/table

| method | sigma | mean_gap | std_gap |
| --- | --- | --- | --- |
| GD (lr=1/L) | 0 | 0 | 0 |
| GD (lr=1/L) | 0.1 | 0 | 0 |
| GD (lr=1/L) | 0.5 | 0 | 0 |
| GD (lr=1/L) | 1 | 0 | 0 |
| Momentum | 0 | 0 | 0 |
| Momentum | 0.1 | 0 | 0 |
| Momentum | 0.5 | 0 | 0 |
| Momentum | 1 | 0 | 0 |
| Adam | 0 | 1.11022e-16 | 0 |
| Adam | 0.1 | 1.11022e-16 | 0 |
| Adam | 0.5 | 1.11022e-16 | 0 |
| Adam | 1 | 1.11022e-16 | 0 |
| GD-Decay | 0 | 6.02918e-12 | 0 |
| GD-Decay | 0.1 | 6.02918e-12 | 0 |
| GD-Decay | 0.5 | 6.02918e-12 | 0 |
| GD-Decay | 1 | 6.02918e-12 | 0 |


**Figures**: exp4a_loss_vs_epoch.png, exp4a_loss_vs_time.png, exp4b_noise_sensitivity.png

---

## EXP5

**Status**: SUCCESS
**Runtime**: 3.48 s

**Setup**: **Second-Order Methods and Hessian Analysis.** (a) Newton (line search) vs GD on logistic regression (n=2000, d=20, l2=1e-2); f_star from a long Newton run; convergence plot (log y). (b) Gauss-Newton, Levenberg-Marquardt, GD, Adam on make_exp_decay_fit from a poor start (a=1, b=2, c=0); convergence plot and parameter recovery table. (c) Curvature approximations at the Newton solution: HVP finite-difference vs exact (5 vectors, max relative error); top-5 eigenvalues power iteration vs numpy.linalg.eigh; low-rank Frobenius error for k in {1,2,5,10,20}; empirical Fisher diagonal vs Hessian diagonal (bar chart + correlation); diagonal-preconditioned GD vs plain GD (same budget, convergence plot).


### part_a/table

| method | iters | converged | final_f | final_gradnorm | runtime_s |
| --- | --- | --- | --- | --- | --- |
| GradientDescent | 30 | True | 0.609652 | 5.83385e-09 | 0.0045134 |
| Newton (line search) | 5 | True | 0.609652 | 1.33843e-11 | 0.0058559 |


### part_b/params_table

| method | a_recovered | b_recovered | c_recovered | a_true | b_true | c_true | final_f | iters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gauss-Newton | 1.40811 | -2.7658 | 0.624171 | 2 | 0.5 | 0.3 | 7.49829e+14 | 2 |
| LM | 1.96914 | 0.511866 | 0.328026 | 2 | 0.5 | 0.3 | 0.0899793 | 13 |
| GD | 1.95167 | 0.544292 | 0.366168 | 2 | 0.5 | 0.3 | 0.0950577 | 1000 |
| Adam | 1.96745 | 0.514291 | 0.331209 | 2 | 0.5 | 0.3 | 0.0900113 | 1000 |


### part_c/hvp_table

| vector_idx | max_relative_error |
| --- | --- |
| 0 | 6.30833e-10 |
| 1 | 1.59429e-10 |
| 2 | 7.43317e-11 |
| 3 | 1.57978e-10 |
| 4 | 1.02995e-09 |


### part_c/eigvals_table

| rank | power_iter_eigval | exact_eigval | abs_error |
| --- | --- | --- | --- |
| 1 | 0.263644 | 0.263644 | 2.71864e-07 |
| 2 | 0.258996 | 0.258996 | 1.35856e-07 |
| 3 | 0.251044 | 0.251044 | 6.98379e-08 |
| 4 | 0.245773 | 0.245773 | 2.80823e-08 |
| 5 | 0.240262 | 0.240262 | 3.79997e-08 |


### part_c/lowrank_table

| k | rel_frob_error |
| --- | --- |
| 1 | 0.964189 |
| 2 | 0.928319 |
| 5 | 0.823682 |
| 10 | 0.641775 |
| 20 | 9.78487e-06 |


### part_c/fisher_table

| coord | fisher_diag | hess_diag |
| --- | --- | --- |
| 0 | 0.222969 | 0.231607 |
| 1 | 0.206801 | 0.220534 |
| 2 | 0.207352 | 0.217622 |
| 3 | 0.19059 | 0.208891 |
| 4 | 0.210502 | 0.216954 |
| 5 | 0.221823 | 0.22171 |
| 6 | 0.198938 | 0.216117 |
| 7 | 0.215574 | 0.229209 |
| 8 | 0.192639 | 0.212119 |
| 9 | 0.198291 | 0.221701 |
| 10 | 0.202953 | 0.219906 |
| 11 | 0.212571 | 0.232353 |
| 12 | 0.206478 | 0.220565 |
| 13 | 0.203612 | 0.219637 |
| 14 | 0.217131 | 0.228162 |
| 15 | 0.19937 | 0.213729 |
| 16 | 0.217995 | 0.228428 |
| 17 | 0.200382 | 0.215793 |
| 18 | 0.213261 | 0.225958 |
| 19 | 0.206365 | 0.217972 |
| correlation | 0.82943 | 0.82943 |


**Figures**: exp5a_convergence.png, exp5b_convergence.png, exp5c_lowrank_error.png, exp5c_fisher_diag.png, exp5c_diag_precond.png

---

## EXP6

**Status**: SUCCESS
**Runtime**: 39.34 s

**Setup**: **MLP Backpropagation.** Dataset: make_moons(n=1000, noise=0.25, seed=0), 70/30 split, standardised, TwoLayerMLP with 16 hidden units. (a) Numerical gradient check – max relative error CSV. (b) GD (full batch), SGD-32, Momentum-32, Adam-32; 100 epochs; lr tuned over a small grid; 5 seeds; plots: train loss vs epoch and test accuracy vs epoch. (c) Adam with l2 in {0, 1e-3, 1e-2, 1e-1, 1.0}; table and 4-panel plot of train loss, test loss, test accuracy, param norm vs l2. (d) Decision boundary of Adam model (l2=1e-2) overlaid on data.


### part_a/table

| max_relative_error |
| --- |
| 1.14924e-09 |


### part_b/table

| config | tuned_lr | batch_size | final_train_loss_mean | final_train_loss_std | final_test_acc_mean | final_test_acc_std | runtime_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GD (full) | 0.3 | 700 | 0.310898 | 0.00706788 | 0.853333 | 0.0101105 | 0.585904 |
| SGD-32 | 0.3 | 32 | 0.158879 | 0.0137245 | 0.920667 | 0.00904311 | 0.824247 |
| Momentum-32 | 0.1 | 32 | 0.14846 | 0.0107953 | 0.928 | 0.0124007 | 0.793459 |
| Adam-32 | 0.1 | 32 | 0.135542 | 0.0137723 | 0.933333 | 0.0128236 | 0.990658 |


### part_c/table

| l2 | train_loss | test_loss | test_accuracy | param_norm |
| --- | --- | --- | --- | --- |
| 0 | 0.151937 | 0.174943 | 0.936667 | 9.79385 |
| 0.001 | 0.202886 | 0.215977 | 0.933333 | 7.10372 |
| 0.01 | 0.337455 | 0.330216 | 0.86 | 2.20035 |
| 0.1 | 0.500071 | 0.49353 | 0.85 | 1.64075 |
| 1 | 0.693149 | 0.693149 | 0.5 | 0.00150971 |


**Figures**: exp6b_train_loss.png, exp6b_test_acc.png, exp6c_regularisation.png, exp6d_decision_boundary.png
