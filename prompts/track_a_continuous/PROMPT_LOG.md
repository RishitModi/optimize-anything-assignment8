## Entry 1
- Date: 28/10
- Tool and model version: Antigravity
- Original prompt: 
# Prompt Log: Track A: Continuous Optimization
STRICT RULES FOR THIS TASK: Do NOT run any command, script, test, python, pip, git or shell command. Do NOT create scratch or test scripts anywhere. Do NOT install anything. Do NOT modify any file in common/. Do NOT use torch anywhere. Use only numpy, scipy, scikit-learn, matplotlib and pytest. Create only the files listed below. Before writing code, read the files in common/ so you use their exact function signatures.

Create these files inside tracks/track_a_continuous/. Every module must start with a sys.path insert of the repository root (Path(__file__).resolve().parents[2]) so "from common..." imports work. Use type hints, docstrings, a deterministic seed argument (np.random.default_rng(seed)) wherever randomness is used, and short comments explaining the optimization logic.

1. landscapes.py
   - @dataclass Objective: name, dim, f(x), grad(x), hess(x) (optional), x_star (optional), f_star (optional), description.
   - quadratic(cond=50.0, dim=10, seed=0): f = 0.5 x^T A x - b^T x, A = Q diag(linspace(1, cond, dim)) Q^T with a random orthogonal Q; so mu=1, L=cond; give exact x_star and f_star.
   - rosenbrock(dim=2); sum_quartic(dim) = sum x_i^4 (convex, not strongly convex); saddle_function(): f = x^2 - y^2 + 0.25*y^4 (saddle at origin, minima at y = +-sqrt(2)); double_well(): f = x^4 - 3x^2 + x + y^2 (one local and one global minimum in x). All with analytic grad and hess.
   - classify_stationary_point(obj, x, tol=1e-6): uses the eigenvalues of hess(x); returns "minimum", "maximum", "saddle" or "degenerate".
   - make_logistic_regression(n=2000, d=20, l2=1e-2, seed=0): synthetic binary data; returns an Objective for the L2-regularised logistic loss (full-batch f, grad, hess) plus a data object exposing X, y, per_sample_grads(w), and batch_grad(w, idx).
   - make_exp_decay_fit(seed=0): nonlinear least squares y = a*exp(-b*t) + c with noisy data; returns residual(x), jacobian(x), and f = 0.5*||r||^2 with its grad.

2. first_order.py
   - Stateful optimizer classes, each with step(x, grad_fn) -> x_new, where grad_fn(x) returns a gradient: GradientDescent(lr), Momentum(lr, beta), NAG(lr, beta) (evaluates the gradient at the look-ahead point), AdaGrad(lr, eps), RMSProp(lr, rho, eps), Adam(lr, beta1, beta2, eps) with bias correction. Add a reset() method.
   - @dataclass OptResult: x, f_history, gradnorm_history, n_iters, runtime_s, converged, diverged.
   - minimize(obj, x0, optimizer, max_iter, tol=1e-6, grad_noise_sigma=0.0, seed=0): deterministic or noisy-gradient loop (adds Gaussian noise to the gradient when sigma>0); stops at tol, max_iter, or divergence (non-finite or f > 1e12). Time it with common.timing.
   - minimize_stochastic(data, w0, optimizer, batch_size, epochs, seed=0): epoch-based loop using data.batch_grad with shuffling; batch_size=1 is SGD, 1<b<n is mini-batch SGD, b=n is full-batch GD. Record the full-batch loss after every epoch.

3. second_order.py
   - newton(obj, x0, max_iter, tol, damping=0.0, line_search=False): solves (H + damping*I) p = -g; optional Armijo backtracking.
   - gauss_newton(residual, jacobian, x0, max_iter, tol) and levenberg_marquardt(..., lam0=1e-2) with adaptive lam.
   - hvp_finite_difference(grad, x, v, eps=1e-5) using central differences; hvp_exact(obj, x, v).
   - top_eigenpairs_from_hvp(hvp, dim, k, seed=0): power iteration with deflation, or Lanczos, using only HVPs.
   - empirical_fisher_diag(per_sample_grads): mean of squared per-sample gradients.
   - low_rank_hessian_approx(hvp, dim, k) returns (eigvals, eigvecs, H_approx).
   - diag_preconditioned_gd(obj, x0, diag, lr, max_iter) where the step is lr * g / (diag + damping).

4. mlp_backprop.py
   - class TwoLayerMLP(d_in, d_hidden, d_out=1, l2=0.0, seed=0) with tanh hidden activation and sigmoid output with binary cross-entropy loss. Methods: flat parameter get/set; forward; loss (with L2 term); backward (manual backpropagation, no autograd, with comments for each chain-rule step); grad_flat; predict; accuracy.
   - numerical_gradient_check(model, X, y, eps=1e-5) returning the maximum relative error against the analytic gradient.
   - train_mlp(model, X, y, optimizer, batch_size, epochs, seed) reusing the optimizer classes from first_order.py on the flattened parameters; returns the per-epoch loss history.

5. tests/conftest.py inserting the repository root and tracks/track_a_continuous into sys.path. Then pytest files (write them, do not run them):
   - tests/test_landscapes.py: analytic grad matches finite differences for every Objective; hess matches finite differences of grad; classify_stationary_point returns "saddle" at the origin of saddle_function and "minimum" at quadratic's x_star.
   - tests/test_first_order.py: every optimizer drives quadratic(cond=10, dim=5) to f - f_star < 1e-6 with a suitable lr in the stated max_iter; with cond=50 plain GD with lr=1/L matches the theoretical linear rate within a tolerance; minimize_stochastic with batch_size=n equals full-batch GD.
   - tests/test_second_order.py: newton solves a quadratic in exactly one iteration; hvp_finite_difference agrees with hvp_exact (1e-4 relative); top_eigenpairs_from_hvp matches numpy.linalg.eigh on a small quadratic; Gauss-Newton fits make_exp_decay_fit to a small residual.
   - tests/test_mlp_backprop.py: numerical_gradient_check max relative error below 1e-5 with and without L2; training with Adam reduces the loss on make_moons data.

Do not edit README.md files of the tracks, CHECKLIST.md, or any prompt log in this task.

Finally, update the "AI Tools Register" table in README.md: add or extend the single row for Antigravity IDE / Claude Sonnet 5.5 so that Tracks includes "A", Dates includes 2026-10-28, and Kind of work includes "code generation, unit-test writing". Do not create duplicate rows and do not change anything else in README.md.

- AI response / relevant excerpt: 
Let me first read all the `common/` files to understand the exact signatures I need to use.

Viewed timing.py:1-113
Viewed plotting.py:1-180
Viewed stats.py:1-82
Viewed io_utils.py:1-139
Listed directory tracks

Good, I have all the common/ signatures. Now let me write all the files. I'll create them one by one.

Created landscapes.py
Created first_order.py
Created second_order.py
Created mlp_backprop.py

Now the test files:

Created conftest.py
Created test_landscapes.py
Created test_first_order.py
Created test_second_order.py
Created test_mlp_backprop.py

Now update the README:

Edited README.md

All files created. Here's a summary of everything produced:

---

### Files created in `tracks/track_a_continuous/`

| File | What it does |
|---|---|
| [`landscapes.py`](file:///c:/personal/projects/optimize-anything-assignment8/tracks/track_a_continuous/landscapes.py) | `Objective` dataclass; quadratic, Rosenbrock, sum-quartic, saddle, double-well; logistic regression; NLS exp-decay; `classify_stationary_point` |
| [`first_order.py`](file:///c:/personal/projects/optimize-anything-assignment8/tracks/track_a_continuous/first_order.py) | `GradientDescent`, `Momentum`, `NAG`, `AdaGrad`, `RMSProp`, `Adam` (all with `reset()`); `OptResult`; `minimize`; `minimize_stochastic` |
| [`second_order.py`](file:///c:/personal/projects/optimize-anything-assignment8/tracks/track_a_continuous/second_order.py) | `newton` (damping + Armijo); `gauss_newton`; `levenberg_marquardt`; `hvp_finite_difference` / `hvp_exact`; `top_eigenpairs_from_hvp`; `empirical_fisher_diag`; `low_rank_hessian_approx`; `diag_preconditioned_gd` |
| [`mlp_backprop.py`](file:///c:/personal/projects/optimize-anything-assignment8/tracks/track_a_continuous/mlp_backprop.py) | `TwoLayerMLP` (tanh/sigmoid, manual backprop with chain-rule comments, flat param interface); `numerical_gradient_check`; `train_mlp` |
| [`tests/conftest.py`](file:///c:/personal/projects/optimize-anything-assignment8/tracks/track_a_continuous/tests/conftest.py) | Inserts repo root + track root into `sys.path` |
| [`tests/test_landscapes.py`](file:///c:/personal/projects/optimize-anything-assignment8/tracks/track_a_continuous/tests/test_landscapes.py) | Grad/Hessian FD checks for all objectives; stationary-point classification |
| [`tests/test_first_order.py`](file:///c:/personal/projects/optimize-anything-assignment8/tracks/track_a_continuous/tests/test_first_order.py) | All 6 optimizers converge on quadratic; GD linear rate; stochastic ≡ full-batch |
| [`tests/test_second_order.py`](file:///c:/personal/projects/optimize-anything-assignment8/tracks/track_a_continuous/tests/test_second_order.py) | Newton 1-step; HVP FD vs exact; eigenpairs vs eigh; GN + LM on exp-decay; Fisher diag; preconditioned GD |
| [`tests/test_mlp_backprop.py`](file:///c:/personal/projects/optimize-anything-assignment8/tracks/track_a_continuous/tests/test_mlp_backprop.py) | Gradient check ±L2; Adam training reduces loss on make_moons (>70% acc) |

**Key design decisions:**
- `minimize_stochastic` uses `optimizer.lr` directly for the inner loop (avoids a double grad evaluation), which makes the full-batch equivalence test exact
- `newton` imports `scipy.linalg.solve` with `assume_a="sym"` for numerical stability
- `top_eigenpairs_from_hvp` uses power iteration with deflation — purely matrix-free
- All randomness uses `np.random.default_rng(seed)`, never the legacy `np.random` global state

- Revised prompt (if any): 
STRICT RULES FOR THIS TASK: Do NOT run any command, script, test, python, pip, git or shell command. Do NOT create scratch or test scripts anywhere. Do NOT install anything. Do NOT use torch. Edit only the files named below with file edit tools, and change only the functions named. Do not touch common/.

A. tracks/track_a_continuous/first_order.py, function minimize():
Bug: f_history and gradnorm_history currently have max_iter entries instead of n_iters + 1. Required behaviour: f_history[0] and gradnorm_history[0] are the values at x0, and one more entry is appended after every executed step, including the final step, whichever way the loop ends (tolerance reached, max_iter reached or divergence). So len(f_history) == n_iters + 1 always. Find the cause (likely the final step's value is not recorded, or the history is appended before the update) and fix it. Check that minimize_stochastic is unaffected.

B. tracks/track_a_continuous/landscapes.py, make_exp_decay_fit():
Use r_i = a*exp(-b*t_i) + c - y_i, f = 0.5*sum(r_i^2). The Jacobian rows are [exp(-b t_i), -a t_i exp(-b t_i), 1]. The gradient is J^T r. The exact Hessian is H = J^T J + sum_i r_i * Hess(r_i), where the only non-zero second derivatives of r_i are d2r/dadb = -t_i exp(-b t_i) and d2r/db2 = a t_i^2 exp(-b t_i). Rewrite the hess function exactly per these formulas (symmetric 3x3) and make sure grad, residual and jacobian use the same sign convention.

C. tracks/track_a_continuous/landscapes.py, make_logistic_regression():
Enforce one single convention in f, grad, hess, per_sample_grads and batch_grad, with labels y in {0,1}, z = X @ w, p = sigmoid(z):
  f(w) = mean_i[ logaddexp(0, z_i) - y_i * z_i ] + (l2/2) * ||w||^2   (use numerically stable logaddexp)
  grad(w) = X^T (p - y) / n + l2 * w
  hess(w) = X^T diag(p*(1-p)) X / n + l2 * I
  batch_grad(w, idx) = X[idx]^T (p[idx] - y[idx]) / len(idx) + l2 * w
  per_sample_grads(w): row i = (p_i - y_i) * X[i] (data-term only, no l2); add a docstring line stating this.
Check that the generated labels are in {0,1}, not {-1,+1}, and that nothing else in the function uses a different scaling (sum vs mean) or a different regularisation factor. Do not modify any test for this item.

D. tracks/track_a_continuous/tests/test_first_order.py:
1. test_optimizer_converges_quadratic: change the parametrization to (optimizer, max_iter, gap_tol). Use gap_tol = 1e-6 for GD, Momentum, NAG, AdaGrad and Adam, and gap_tol = 5e-2 for RMSProp. Add a comment explaining that RMSProp with a constant learning rate has no decaying step, so it hovers in a neighbourhood of the optimum whose size is O(lr) instead of converging exactly. Also assert for RMSProp that the final gap is below 1% of the initial gap.
2. test_gd_linear_convergence_rate: rewrite it. For lr = 1/L, every eigen-direction contracts by |1 - lambda/L| <= 1 - mu/L, so the correct rate is rho = 1 - 1/cond (mu = 1, L = cond), and f - f_star contracts by at most rho^2 per step. Use cond=50, dim=10, x0 = ones, max_iter=500, tol=0 so it runs all iterations. Assert (i) f_gap_T <= rho**(2*T) * f_gap_0 * (1 + 1e-6) + 1e-14, using T = n_iters; (ii) the empirical per-step contraction of the gap, computed from f_history between iterations 100 and 300 as (gap[300]/gap[100])**(1/200), is within 2% of rho**2. Fix the docstring to state the correct theory and mention that (kappa-1)/(kappa+1) applies only to lr = 2/(L+mu).

E. FAILURE_LOG.md (repository root):
Replace the three empty template rows of the table with the five rows below and keep the existing column headers (Case # | Track | What the AI got wrong | How it was detected | Fix | Lesson). Keep the text as given. Do not change any other section of the file.
1 | A | Test for gradient descent with lr = 1/L used the contraction rate (kappa-1)/(kappa+1), which is only correct for the optimal step 2/(L+mu). | pytest failure: observed gap 8.4e-10 versus predicted 5.9e-16. | Used rho = 1 - mu/L and also checked the empirical per-step contraction. | The AI recalls textbook formulas without checking their step-size assumptions; verify theory against the exact setting.
2 | A | Test expected RMSProp with a constant learning rate to reach f - f* < 1e-6 on a quadratic. | pytest failure: gap stayed at 1.45e-2, matching the O(lr) oscillation floor for lr = 0.05. | Per-optimizer tolerance plus a comment explaining the noise floor. | A test must encode the method's real behaviour, not an assumption that every optimizer converges exactly.
3 | A | minimize() returned max_iter history entries instead of n_iters + 1, dropping the final iterate. | pytest failure: stochastic full-batch run had 20 entries versus 19 for the GD run (off-by-one). | History now records the starting point and every executed step. | Off-by-one errors in logging code silently corrupt convergence plots; test sequence lengths explicitly.
4 | A | Analytic Hessian of the nonlinear least-squares exp-decay objective disagreed with the finite-difference Hessian in the entries involving parameter b. | pytest failure: Hessian test, e.g. entry [1,1] was 48.2 analytic versus 18.8 numeric. | Rewrote the Hessian as J^T J + sum r_i * Hess(r_i) with the exact second derivatives. | AI-derived second derivatives are error-prone; always check them against finite differences.
5 | A | Analytic logistic-regression gradient disagreed with the finite-difference gradient of its own loss in every component. | pytest failure: all 5 gradient components mismatched. | Enforced one convention (labels in {0,1}, mean loss, l2/2 regularisation) across f, grad, hess and batch gradients. | The AI can leave a loss and its gradient written under different conventions; gradient checks catch this immediately.

Do not edit any other README, CHECKLIST.md or prompt log.

F. README.md (repository root), "AI Tools Register" table only:
Add or extend the single row for Antigravity IDE / Claude Sonnet 5.5 so that Tracks includes "A", Dates includes 2026-10-28, and Kind of work includes "debugging, test correction". Do not create duplicate rows and do not change anything else in README.md.

STRICT RULES FOR THIS TASK: Do NOT run any command, script, test, python, pip, git or shell command. Do NOT create scratch or test scripts anywhere. Do NOT install anything. Do NOT use torch. Use only numpy, scipy, scikit-learn, matplotlib, pandas and pytest. Do NOT modify any existing file except the ones explicitly named in sections G, H and I. Before writing code, read all files in common/ and tracks/track_a_continuous/ (landscapes.py, first_order.py, second_order.py, mlp_backprop.py and the tests) so you use their exact names, signatures and return types. If a helper you need does not exist, write it inside the new experiment file; do not edit the existing modules.

Create the files below inside tracks/track_a_continuous/. Each experiment file starts with a sys.path insert of the repository root (Path(__file__).resolve().parents[2]) and exposes a function run(quick: bool = False, out_dir: str = "results/track_a") -> dict. quick=True shrinks sizes (fewer iterations, seeds and grid points) so the whole suite finishes in well under 30 seconds. quick=False must finish all experiments in under about 5 minutes total on a laptop. Every figure is saved as PNG (dpi 150) in out_dir with labelled axes, units, legend and title; every table is saved as CSV and also returned in the dict. Use common.seeding.set_global_seed, common.timing for runtimes, common.stats for repeated runs, common.io_utils for saving. Use fixed seeds; stochastic experiments use 5 seeds (2 in quick mode) and report mean and std; plots show a mean line with a shaded std band. Add short comments explaining the optimization logic, and docstrings.

1. exp1_optimizer_comparison.py
   Problems: quadratic(cond=100, dim=10), rosenbrock(dim=2, start x0=(-1.2, 1.0)).
   Methods: GradientDescent, Momentum, NAG, AdaGrad, RMSProp, Adam, plus newton from second_order.py (with line_search=True for Rosenbrock).
   Fairness: every first-order method gets the same budget (max_iter=2000 for quadratic, 20000 for Rosenbrock; 200 and 2000 in quick mode), same x0, same tolerance on the gradient norm (1e-6). Tune lr per method per problem over np.logspace(-4, 0, 9) (5 points in quick mode), choosing the lr with the lowest final f among non-diverged runs; momentum beta fixed at 0.9, Adam betas (0.9, 0.999), RMSProp rho 0.9. Save the chosen lr table.
   Outputs: convergence plot of f - f_star (log y-axis; for Rosenbrock f_star = 0) versus iteration for each problem; table (CSV) with columns problem, method, tuned_lr, final_f, final_gradnorm, iters_to_tol (or "not reached"), runtime_s.

2. exp2_landscapes.py
   a) Convexity classes: show convergence of GradientDescent on three objectives in one log-scale plot of f - f_star versus iteration: strongly convex (quadratic cond=10), convex but not strongly convex (sum_quartic dim=5, start ones), non-convex (rosenbrock). Annotate the expected rate (linear, sublinear, and none guaranteed) in the legend text. Fit the slope of log(gap) on the second half of the run for the first two and print it in the table.
   b) Saddle point: saddle_function started at (1e-3, 1e-3)-style small perturbations. Run GradientDescent, Momentum, Adam and Newton (undamped, then Newton with damping chosen so that H + damping*I is positive definite, e.g. damping = 1 - min_eigenvalue + 0.1). Record the final point and call classify_stationary_point on it. Plot the 2D trajectories on a contour plot of the function. Expected behaviour to be reported from the data, not asserted: undamped Newton is attracted to the saddle.
   c) Local versus global minimum: double_well from three starting points (negative x, near zero, positive x); run GradientDescent and Adam; table of start, method, final point, final f, whether it is the global minimum (compare against a fine grid search of f); contour plot with trajectories.
   Outputs: the plots above and tables (CSV).

3. exp3_stepsize_sensitivity.py
   On quadratic(cond=50, dim=10) with a fixed budget of 500 iterations (100 in quick mode), sweep lr over np.logspace(-4, 0.5, 20) (8 points in quick mode) for GradientDescent, Momentum, NAG, AdaGrad, RMSProp, Adam. Record final f - f_star, marking diverged runs as NaN. Plot final gap versus lr (log-log), one line per optimizer, with divergence shown by breaks in the line. Table: for each optimizer the best lr, and the width of the lr range (in decades) within which final gap < 1e-3 (a robustness measure).

4. exp4_stochastic_and_noise.py
   a) Mini-batch study on make_logistic_regression(n=2000, d=20, l2=1e-2, seed=0) (attach data.f = obj.f as the existing tests do). Configurations: GD (batch_size=n), mini-batch SGD with batch sizes 256, 32, and SGD with batch_size=1, plus Momentum and Adam with batch_size=32. 30 epochs (5 in quick mode), lr chosen from a small grid (np.logspace(-3, 0, 7)) per configuration by lowest final loss. 5 seeds. Plots: full-batch loss versus epoch, and full-batch loss versus wall-clock seconds (mean with std band). Table: config, tuned_lr, final_loss mean and std, time_per_epoch_s, number of gradient evaluations per epoch.
   b) Noisy objective: on quadratic(cond=20, dim=10), use minimize(..., grad_noise_sigma=sigma) for sigma in [0, 0.1, 0.5, 1.0] with GradientDescent (lr=1/L), Momentum, and Adam, 10 seeds (3 in quick mode), 1000 iterations (200 in quick mode), and also a GradientDescent variant with a decaying step size lr_t = lr0 / (1 + 0.01 t) implemented inside this file as a small wrapper optimizer class with the same step(x, grad_fn) interface. Plot final f - f_star versus sigma (log y) per method with std bars. Table: method, sigma, mean gap, std gap.

5. exp5_second_order.py
   a) Newton versus GradientDescent on logistic regression (n=2000, d=20, l2=1e-2): iterations to gradient norm 1e-8, runtime, and f history; plot f - f_star (f_star from a long Newton run) versus iteration, log y-axis.
   b) Gauss-Newton and Levenberg-Marquardt versus GradientDescent and Adam on make_exp_decay_fit from a poor start (e.g. a=1, b=2, c=0): plot f versus iteration, table of iterations and final f and the recovered parameters versus the true parameters.
   c) Curvature approximations on the logistic regression objective at the solution found by Newton: (i) check hvp_finite_difference against hvp_exact for 5 random vectors and report the max relative error; (ii) top-5 eigenvalues via top_eigenpairs_from_hvp versus numpy.linalg.eigh of the exact Hessian (table); (iii) low-rank approximation: relative Frobenius error of the rank-k Hessian approximation for k in 1, 2, 5, 10, 20 (plot error versus k); (iv) empirical Fisher diagonal versus true Hessian diagonal (bar chart for the 20 coordinates plus the correlation coefficient); (v) diag_preconditioned_gd using the Fisher diagonal versus plain GradientDescent, same budget, convergence plot.
   Outputs: plots and CSV tables for each part.

6. exp6_backprop_mlp.py
   Use sklearn make_moons(n_samples=1000, noise=0.25, random_state=0) with a 70/30 train/test split (fixed seed), standardised features, TwoLayerMLP with 16 hidden units.
   a) Run numerical_gradient_check and save the maximum relative error in a one-row CSV.
   b) Train with GradientDescent (full batch), SGD with batch 32, Momentum and Adam (batch 32), 100 epochs (15 in quick mode), lr tuned over a small grid per optimizer, 5 seeds (2 in quick). Plot training loss versus epoch and test accuracy versus epoch; table of final train loss, test accuracy mean and std, runtime.
   c) Regularisation as part of the objective: train with Adam for l2 in [0, 1e-3, 1e-2, 1e-1, 1.0]; table and plot of train loss, test loss, test accuracy and parameter norm versus l2.
   d) Plot the decision boundary of the final Adam model (l2=1e-2) over the data.

7. run_all.py
   Command line: python tracks/track_a_continuous/run_all.py [--quick] [--only exp1 exp2 ...] [--out results/track_a]. Imports each experiment module, runs them in order with timing, prints a one-line status per experiment, and wraps each in try/except so one failure does not stop the others (print the traceback and continue; exit code 1 at the end if any failed). At the end write results/track_a/SUMMARY.md containing: environment info (python version, numpy/scipy/sklearn versions, platform, CPU count, seeds), then for every experiment a heading, a one-paragraph description of the setup (problem, methods, budget, seeds, tuning grid) and the experiment's tables rendered as markdown tables (use pandas to_markdown only if tabulate is guaranteed; otherwise write your own small markdown renderer), and the filenames of its figures. SUMMARY.md must contain only numbers actually produced by the run; do not write any hand-made conclusions or interpretations.

8. tests/test_experiments_smoke.py
   One pytest per experiment module calling run(quick=True, out_dir=<pytest tmp_path>) and asserting that the returned dict is non-empty, that expected CSV and PNG files exist, and that no table contains NaN in the final_f / final_loss style columns except where divergence is intentionally recorded (exp3). Write it but do not run it.

G. Track README: edit tracks/track_a_continuous/README.md. Fill only the Objective, Formulation, Methods and How to run sections (How to run: python tracks/track_a_continuous/run_all.py, with the --quick and --only options, and the pytest command). Leave Results, Correctness check and Limitations unchanged for now.

H. CHECKLIST.md (repository root): for every row belonging to the Foundations and Continuous areas, fill the Experiment column (exp1 to exp6 as appropriate: e.g. GD/Momentum/NAG/AdaGrad/RMSProp/Adam -> exp1 and exp3; SGD/mini-batch/noisy objectives -> exp4; Newton/Gauss-Newton/HVP/Fisher/low-rank -> exp5; backpropagation/regularisation/loss as parameters -> exp6; convex vs non-convex/saddle/local minima/convergence rates -> exp2) and the Code location column (file path inside tracks/track_a_continuous/), and set Status to "Code ready - results pending". Do not touch rows of other areas and do not fill the Result column.

I. README.md (repository root), "AI Tools Register" table only: add or extend the single row for Antigravity IDE / Claude Sonnet 5.5 so that Tracks includes "A", Dates includes 2026-10-28, and Kind of work includes "experiment scripting". Do not create duplicate rows and do not change anything else in README.md.

- Resulting commit link: https://github.com/RishitModi/optimize-anything-assignment8/commit/a2b2eb6708f75705dd5d82a8eb68abdfa722475e
- Verification note (done by the student): 
- Chat export / screenshot filename: 
