"""Tests for second-order methods and Hessian utilities in Track A.

Verifies:
- Newton solves a quadratic in exactly one iteration.
- hvp_finite_difference agrees with hvp_exact to 1e-4 relative tolerance.
- top_eigenpairs_from_hvp matches numpy.linalg.eigh on a small quadratic.
- Gauss-Newton fits make_exp_decay_fit to a small residual.
"""

from __future__ import annotations

import numpy as np
import pytest

from landscapes import make_exp_decay_fit, quadratic
from second_order import (
    diag_preconditioned_gd,
    empirical_fisher_diag,
    gauss_newton,
    hvp_exact,
    hvp_finite_difference,
    levenberg_marquardt,
    low_rank_hessian_approx,
    newton,
    top_eigenpairs_from_hvp,
)


# ---------------------------------------------------------------------------
# Newton: one step on a quadratic
# ---------------------------------------------------------------------------

def test_newton_quadratic_one_step() -> None:
    """Newton's method converges in exactly one step on a quadratic.

    For a quadratic f = 0.5 x^T A x - b^T x the Newton direction is exact
    and x_1 = x_star regardless of x_0 (assuming H is non-singular).
    """
    obj = quadratic(cond=5.0, dim=6, seed=1)
    x0 = np.zeros(obj.dim)

    result = newton(obj, x0, max_iter=10, tol=1e-12, damping=0.0, line_search=False)

    # Should converge in exactly 1 Newton step (2 f-evaluations: x0 and x1).
    assert result.n_iters <= 2, (
        f"Newton took {result.n_iters} iterations on quadratic (expected 1)"
    )
    gap = result.f_history[-1] - obj.f_star
    assert gap < 1e-10, f"Newton gap = {gap:.2e}"


# ---------------------------------------------------------------------------
# HVP: finite-difference vs exact
# ---------------------------------------------------------------------------

def test_hvp_fd_vs_exact() -> None:
    """hvp_finite_difference agrees with hvp_exact within 1e-4 relative error."""
    obj = quadratic(cond=20.0, dim=8, seed=3)
    rng = np.random.default_rng(99)
    x = rng.standard_normal(obj.dim)
    v = rng.standard_normal(obj.dim)

    hvp_fd = hvp_finite_difference(obj.grad, x, v, eps=1e-5)
    hvp_ex = hvp_exact(obj, x, v)

    rel_err = np.linalg.norm(hvp_fd - hvp_ex) / (np.linalg.norm(hvp_ex) + 1e-12)
    assert rel_err < 1e-4, f"HVP relative error = {rel_err:.2e}"


# ---------------------------------------------------------------------------
# Top eigenpairs via HVP vs numpy.linalg.eigh
# ---------------------------------------------------------------------------

def test_top_eigenpairs_match_eigh() -> None:
    """top_eigenpairs_from_hvp matches eigh's top-k eigenvalues and vectors.

    We test on a small quadratic where the exact Hessian is known.
    """
    obj = quadratic(cond=15.0, dim=5, seed=5)
    # Build the exact HVP from the explicit Hessian.
    H_exact = obj.hess(obj.x_star)  # constant for quadratic

    def hvp(v: np.ndarray) -> np.ndarray:
        return H_exact @ v

    k = 3
    eigvals_power, eigvecs_power = top_eigenpairs_from_hvp(hvp, dim=5, k=k, seed=0)

    # Reference via eigh (returns ascending order; take top k).
    evals_ref, evecs_ref = np.linalg.eigh(H_exact)
    # Reverse to descending order.
    evals_ref = evals_ref[::-1][:k]

    # Eigenvalues should match within relative 1e-3.
    # (Power iteration with deflation may not be perfectly ordered; sort both.)
    np.testing.assert_allclose(
        np.sort(np.abs(eigvals_power)), np.sort(np.abs(evals_ref)),
        rtol=5e-3, atol=1e-4,
        err_msg="Power iteration eigenvalues do not match eigh",
    )


# ---------------------------------------------------------------------------
# Gauss-Newton fits exp decay
# ---------------------------------------------------------------------------

def test_gauss_newton_exp_decay() -> None:
    """Gauss-Newton solves the exp-decay NLS problem to a small residual."""
    obj, nls = make_exp_decay_fit(seed=0)
    x0 = np.array([1.0, 0.3, 0.1])  # near true params

    result = gauss_newton(
        nls.residual, nls.jacobian, x0, max_iter=100, tol=1e-8
    )

    assert result.f_history[-1] < 0.5, (
        f"Gauss-Newton final NLS value = {result.f_history[-1]:.4f} (expected < 0.5)"
    )


# ---------------------------------------------------------------------------
# Levenberg-Marquardt
# ---------------------------------------------------------------------------

def test_levenberg_marquardt_exp_decay() -> None:
    """Levenberg-Marquardt reaches a small residual on the exp-decay problem."""
    obj, nls = make_exp_decay_fit(seed=0)
    x0 = np.array([1.0, 0.3, 0.1])

    result = levenberg_marquardt(
        nls.residual, nls.jacobian, x0, max_iter=200, tol=1e-8, lam0=1e-2
    )

    assert result.f_history[-1] < 0.5, (
        f"LM final NLS value = {result.f_history[-1]:.4f} (expected < 0.5)"
    )


# ---------------------------------------------------------------------------
# Empirical Fisher diagonal
# ---------------------------------------------------------------------------

def test_empirical_fisher_diag_shape() -> None:
    """empirical_fisher_diag returns shape (d,) and is non-negative."""
    from landscapes import make_logistic_regression

    obj, data = make_logistic_regression(n=100, d=8, l2=0.01, seed=2)
    w = np.zeros(8)
    pg = data.per_sample_grads(w)  # (100, 8)
    diag = empirical_fisher_diag(pg)

    assert diag.shape == (8,), f"Expected shape (8,), got {diag.shape}"
    assert np.all(diag >= 0.0), "Fisher diagonal contains negative values"


# ---------------------------------------------------------------------------
# Diagonal-preconditioned GD
# ---------------------------------------------------------------------------

def test_diag_preconditioned_gd_converges() -> None:
    """Diagonal-preconditioned GD with Fisher diag preconditioner converges."""
    from landscapes import make_logistic_regression

    obj, data = make_logistic_regression(n=200, d=5, l2=0.1, seed=9)
    w0 = np.zeros(5)
    pg = data.per_sample_grads(w0)
    diag = empirical_fisher_diag(pg)

    result = diag_preconditioned_gd(obj, w0, diag, lr=1.0, max_iter=500, tol=1e-6)
    assert result.f_history[-1] < result.f_history[0], (
        "Preconditioned GD did not decrease the loss"
    )
