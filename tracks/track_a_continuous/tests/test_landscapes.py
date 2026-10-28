"""Tests for optimization landscapes in Track A.

Verifies:
- Analytic gradient matches central finite-difference gradient for every Objective.
- Analytic Hessian matches finite-difference of the gradient for every Objective.
- classify_stationary_point returns "saddle" at the origin of saddle_function.
- classify_stationary_point returns "minimum" at quadratic's x_star.
"""

from __future__ import annotations

import numpy as np
import pytest

from landscapes import (
    Objective,
    classify_stationary_point,
    double_well,
    make_exp_decay_fit,
    make_logistic_regression,
    quadratic,
    rosenbrock,
    saddle_function,
    sum_quartic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def finite_diff_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference gradient estimate."""
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return g


def finite_diff_hess(grad_fn, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian estimate via grad differences."""
    d = len(x)
    H = np.zeros((d, d))
    for i in range(d):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        H[:, i] = (grad_fn(xp) - grad_fn(xm)) / (2.0 * eps)
    return H


# ---------------------------------------------------------------------------
# Collect all Objectives
# ---------------------------------------------------------------------------

def _all_objectives():
    """Return a list of (name, Objective, test_point) tuples."""
    items = []

    # quadratic
    obj = quadratic(cond=10.0, dim=5, seed=1)
    x = np.linspace(-1, 1, 5)
    items.append(("quadratic", obj, x))

    # rosenbrock
    obj = rosenbrock(dim=2)
    x = np.array([-0.5, 0.7])
    items.append(("rosenbrock_2d", obj, x))

    obj = rosenbrock(dim=4)
    x = np.array([0.2, -0.3, 0.5, -0.1])
    items.append(("rosenbrock_4d", obj, x))

    # sum_quartic
    obj = sum_quartic(dim=3)
    x = np.array([1.0, -0.5, 0.3])
    items.append(("sum_quartic", obj, x))

    # saddle_function
    obj = saddle_function()
    x = np.array([0.4, -0.8])
    items.append(("saddle_function", obj, x))

    # double_well
    obj = double_well()
    x = np.array([1.0, 0.5])
    items.append(("double_well", obj, x))

    # exp_decay_fit
    obj, _ = make_exp_decay_fit(seed=0)
    x = np.array([1.8, 0.6, 0.25])
    items.append(("exp_decay_fit", obj, x))

    return items


@pytest.mark.parametrize("name,obj,x", _all_objectives(), ids=[t[0] for t in _all_objectives()])
def test_grad_matches_finite_diff(name: str, obj: Objective, x: np.ndarray) -> None:
    """Analytic gradient agrees with central finite-difference within tolerance."""
    analytic = obj.grad(x)
    numeric = finite_diff_grad(obj.f, x)
    np.testing.assert_allclose(
        analytic, numeric, rtol=1e-4, atol=1e-6,
        err_msg=f"{name}: grad mismatch",
    )


@pytest.mark.parametrize(
    "name,obj,x",
    [(n, o, p) for n, o, p in _all_objectives() if o.hess is not None],
    ids=[t[0] for t in _all_objectives() if t[1].hess is not None],
)
def test_hess_matches_finite_diff(name: str, obj: Objective, x: np.ndarray) -> None:
    """Analytic Hessian agrees with finite-difference of grad within tolerance."""
    H_analytic = obj.hess(x)
    H_numeric = finite_diff_hess(obj.grad, x, eps=1e-4)
    np.testing.assert_allclose(
        H_analytic, H_numeric, rtol=1e-3, atol=1e-5,
        err_msg=f"{name}: Hessian mismatch",
    )


def test_classify_saddle_at_origin() -> None:
    """classify_stationary_point returns 'saddle' at the origin of saddle_function."""
    obj = saddle_function()
    origin = np.zeros(2)
    result = classify_stationary_point(obj, origin)
    assert result == "saddle", f"Expected 'saddle', got '{result}'"


def test_classify_minimum_at_quadratic_x_star() -> None:
    """classify_stationary_point returns 'minimum' at quadratic's x_star."""
    obj = quadratic(cond=5.0, dim=4, seed=7)
    result = classify_stationary_point(obj, obj.x_star)
    assert result == "minimum", f"Expected 'minimum', got '{result}'"


def test_logistic_grad_matches_finite_diff() -> None:
    """Logistic regression analytic gradient matches finite differences."""
    obj, _ = make_logistic_regression(n=50, d=5, l2=0.1, seed=3)
    w = np.random.default_rng(42).standard_normal(5) * 0.1
    analytic = obj.grad(w)
    numeric = finite_diff_grad(obj.f, w)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6)
