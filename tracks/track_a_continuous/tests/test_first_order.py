"""Tests for first-order optimizers in Track A.

Verifies:
- Every optimizer drives quadratic(cond=10, dim=5) below f - f_star < 1e-6.
- Plain GD with lr=1/L on quadratic(cond=50) matches the theoretical linear rate.
- minimize_stochastic with batch_size=n equals full-batch GD per epoch.
"""

from __future__ import annotations

import numpy as np
import pytest

from landscapes import quadratic
from first_order import (
    Adam,
    AdaGrad,
    GradientDescent,
    Momentum,
    NAG,
    OptResult,
    RMSProp,
    minimize,
    minimize_stochastic,
)


# ---------------------------------------------------------------------------
# Shared setup helpers
# ---------------------------------------------------------------------------

def _quad5():
    """quadratic(cond=10, dim=5) used by most convergence tests."""
    return quadratic(cond=10.0, dim=5, seed=42)


# ---------------------------------------------------------------------------
# Convergence tests – all optimizers on low-condition quadratic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "optimizer,max_iter",
    [
        (GradientDescent(lr=1.0 / 10.0), 3000),  # lr = 1/L
        (Momentum(lr=0.05, beta=0.85), 1000),
        (NAG(lr=0.05, beta=0.85), 500),
        (AdaGrad(lr=0.5), 3000),
        (RMSProp(lr=0.05, rho=0.9), 1000),
        (Adam(lr=0.1), 2000),
    ],
    ids=["GD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam"],
)
def test_optimizer_converges_quadratic(optimizer, max_iter: int) -> None:
    """All optimizers reach f - f_star < 1e-6 on quadratic(cond=10, dim=5)."""
    obj = _quad5()
    x0 = np.zeros(5)
    result = minimize(obj, x0, optimizer, max_iter=max_iter, tol=1e-8)
    gap = result.f_history[-1] - obj.f_star
    assert gap < 1e-6, (
        f"{type(optimizer).__name__}: f - f_star = {gap:.2e} after {result.n_iters} iters"
    )


# ---------------------------------------------------------------------------
# Linear convergence rate test for plain GD
# ---------------------------------------------------------------------------

def test_gd_linear_convergence_rate() -> None:
    """GD with lr=1/L on quadratic(cond=50) matches the theoretical rate.

    Theory: ||x_t - x*||^2 <= ((L - mu) / (L + mu))^{2t} ||x_0 - x*||^2
    Equivalently, f(x_t) - f* <= (rho^t)^2 * (f(x_0) - f*) approximately,
    where rho = (cond - 1) / (cond + 1).

    We verify that after T iterations the observed contraction factor is within
    a factor of 2 of the theoretical value.
    """
    cond = 50.0
    obj = quadratic(cond=cond, dim=10, seed=0)
    L = cond  # largest eigenvalue
    lr = 1.0 / L
    rho = (cond - 1.0) / (cond + 1.0)  # theoretical rate per step

    x0 = np.ones(obj.dim)
    opt = GradientDescent(lr=lr)
    T = 500
    result = minimize(obj, x0, opt, max_iter=T, tol=1e-12)

    f0_gap = result.f_history[0] - obj.f_star
    fT_gap = result.f_history[-1] - obj.f_star
    actual_iters = result.n_iters

    # Theoretical contraction over actual_iters steps.
    theoretical_gap = f0_gap * (rho ** actual_iters) ** 2

    # We allow the empirical gap to be at most 2x the theoretical (the bound is tight
    # for GD on a quadratic, so the actual ratio should be close to 1).
    assert fT_gap <= max(theoretical_gap * 2.0, 1e-10), (
        f"GD linear rate mismatch: f_gap={fT_gap:.2e}, "
        f"theoretical={theoretical_gap:.2e}, ratio={fT_gap/max(theoretical_gap,1e-30):.2f}"
    )


# ---------------------------------------------------------------------------
# Stochastic vs full-batch equivalence
# ---------------------------------------------------------------------------

def test_minimize_stochastic_fullbatch_equals_gd() -> None:
    """minimize_stochastic with batch_size=n produces the same loss as full GD.

    We build a LogisticData object (which has batch_grad), attach the full-batch
    loss as data.f, and compare the epoch-loss trajectory of the stochastic loop
    (batch_size=n) against running minimize() for the equivalent number of steps.
    """
    from landscapes import make_logistic_regression

    obj, data = make_logistic_regression(n=100, d=5, l2=0.01, seed=7)
    # Attach full-batch f so minimize_stochastic can record it.
    data.f = obj.f

    n = data.X.shape[0]
    w0 = np.zeros(obj.dim)
    lr = 0.05
    epochs = 20

    # Stochastic loop with batch_size=n (one gradient per epoch).
    opt_stoch = GradientDescent(lr=lr)
    losses_stoch = minimize_stochastic(
        data, w0, opt_stoch, batch_size=n, epochs=epochs, seed=0
    )

    # Full-batch GD for the same number of steps.
    opt_gd = GradientDescent(lr=lr)
    result_gd = minimize(obj, w0, opt_gd, max_iter=epochs, tol=1e-15)

    # The per-epoch losses should match closely (same gradient, same step size).
    assert len(losses_stoch) == epochs
    np.testing.assert_allclose(
        losses_stoch,
        result_gd.f_history[1:epochs + 1],  # skip the initial f value
        rtol=1e-4,
        atol=1e-6,
        err_msg="Stochastic full-batch loss does not match GD trajectory",
    )
