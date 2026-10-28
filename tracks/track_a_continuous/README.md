# Track A: Continuous Optimization

## Objective
Benchmark and analyse continuous optimization algorithms on a suite of test landscapes,
measuring convergence speed, sensitivity to hyperparameters, robustness to noise, and
the practical advantage of second-order curvature information. The target metrics are:
- **Convergence**: iterations or wall-clock time to reach ‖∇f‖ < 1e-6 (or 1e-8).
- **Final gap**: f(x_T) - f* after a fixed iteration budget.
- **Robustness**: width (in decades) of the lr range giving final gap < 1e-3.
- **Accuracy**: test accuracy of the MLP classifier trained on make_moons.

## Baseline
Plain Gradient Descent with a fixed step size lr = 1/L (where L is the Lipschitz
constant of the gradient) serves as the baseline for all first-order comparisons.
For the logistic regression and NLS problems, Newton's method (converging in ~5-20
iterations) sets the gold standard for second-order comparisons.

## Formulation
Decision variables: continuous parameter vectors x in R^d.

Objective functions minimised (f : R^d -> R):
- **Quadratic**: f(x) = 0.5 x^T A x - b^T x, A PD with controlled condition number.
- **Rosenbrock**: f(x) = sum_i [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2], non-convex.
- **Sum-quartic**: f(x) = sum_i x_i^4, convex but not strongly convex.
- **Saddle function**: f(x,y) = x^2 - y^2 + 0.25 y^4, non-convex with saddle at origin.
- **Double-well**: f(x,y) = x^4 - 3x^2 + x + y^2, two local minima.
- **Logistic regression**: f(w) = mean log-loss + 0.5 l2 ||w||^2, strongly convex for l2>0.
- **NLS (exp decay)**: f(x) = 0.5 ||r(x)||^2, nonlinear least squares.
- **MLP (BCE + L2)**: binary cross-entropy plus L2 penalty on network weights.

No constraints are imposed; all problems are unconstrained.

## Methods
**First-order** (implemented in `first_order.py`):
- GradientDescent, Momentum (heavy-ball), NAG (Nesterov), AdaGrad, RMSProp, Adam
- Stochastic / mini-batch variants via `minimize_stochastic`
- Decaying step-size GD: lr_t = lr0 / (1 + 0.01*t)  (defined in exp4)

**Second-order** (implemented in `second_order.py`):
- Newton's method (plain and damped, with optional Armijo line search)
- Gauss-Newton and Levenberg-Marquardt (for nonlinear least squares)
- Diagonal-preconditioned GD (empirical Fisher diagonal as preconditioner)

**Analysis utilities**:
- Hessian-vector products (exact analytic and finite-difference)
- Power-iteration top-k eigenpairs, low-rank Hessian approximation
- Empirical Fisher diagonal; stationary-point classifier (Hessian eigenvalues)

**MLP** (implemented in `mlp_backprop.py`):
- TwoLayerMLP (tanh hidden, sigmoid output, manual backpropagation)
- Numerical gradient check via central finite differences

## How to run

Run all experiments (full mode, ~5 min on a laptop):
```bash
python tracks/track_a_continuous/run_all.py
```

Run in quick mode (all experiments in < 30 s):
```bash
python tracks/track_a_continuous/run_all.py --quick
```

Run only selected experiments and/or change output directory:
```bash
python tracks/track_a_continuous/run_all.py --only exp1 exp3 --out my_results/
```

Run all unit and smoke tests:
```bash
pytest tracks/track_a_continuous/tests/
```

Run only the experiment smoke tests:
```bash
pytest tracks/track_a_continuous/tests/test_experiments_smoke.py -v
```

## Results
<!-- Experimental outcomes, comparison tables, and plots saved in results/ -->

## Correctness check
<!-- Verification steps, assertions, unit tests, and theoretical invariants checked -->

## Limitations
<!-- Computational bottlenecks, approximations, assumptions, and failure modes -->
