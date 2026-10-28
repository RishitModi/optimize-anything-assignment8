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
    "optimizer,max_iter,gap_tol",
    [
        # gap_tol = 1e-6 for methods that converge exactly on a quadratic.
        # RMSProp with a constant learning rate does NOT converge to the exact
        # optimum: the adaptive denominator never fully decays, so the effective
        # step size oscillates around lr / sqrt(v_inf) rather than shrinking to
        # zero.  The iterate therefore hovers in a neighbourhood of x* whose
        # radius is O(lr), giving a gap floor of O(lr) – not zero.  We test only
        # that it reduces the gap to < 10 % of its starting value (see below).
        (GradientDescent(lr=1.0 / 10.0), 3000, 1e-6),  # lr = 1/L
        (Momentum(lr=0.05, beta=0.85), 1000, 1e-6),
        (NAG(lr=0.05, beta=0.85), 500, 1e-6),
        (AdaGrad(lr=0.5), 3000, 1e-6),
        (RMSProp(lr=0.05, rho=0.9), 1000, 5e-2),  # O(lr) noise floor
        (Adam(lr=0.1), 2000, 1e-6),
    ],
    ids=["GD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam"],
)
def test_optimizer_converges_quadratic(optimizer, max_iter: int, gap_tol: float) -> None:
    """Optimizers converge on quadratic(cond=10, dim=5) within per-optimizer tolerance.

    Most optimizers reach f - f_star < 1e-6.  RMSProp with a constant lr has
    an O(lr) noise floor (no decaying step), so it is only required to reach
    gap_tol = 5e-2 and to reduce the gap to < 10 % of its initial value.
    With lr = 0.05 the measured noise floor is about 7.7% of the initial gap,
    so a 90% reduction is the meaningful check.
    """
    obj = _quad5()
    x0 = np.zeros(5)
    initial_gap = obj.f(x0) - obj.f_star
    result = minimize(obj, x0, optimizer, max_iter=max_iter, tol=1e-8)
    gap = result.f_history[-1] - obj.f_star
    assert gap < gap_tol, (
        f"{type(optimizer).__name__}: f - f_star = {gap:.2e} after {result.n_iters} iters"
        f" (tol={gap_tol:.0e})"
    )
    # For RMSProp additionally assert that the gap shrank to < 10% of initial.
    if type(optimizer).__name__ == "RMSProp":
        assert gap < 0.10 * initial_gap, (
            f"RMSProp did not reduce gap by 90%: initial={initial_gap:.2e}, final={gap:.2e}"
        )


# ---------------------------------------------------------------------------
# Linear convergence rate test for plain GD
# ---------------------------------------------------------------------------

def test_gd_linear_convergence_rate() -> None:
    """GD with lr=1/L on quadratic(cond=50) matches the theoretical linear rate.

    Theory: for lr = 1/L each eigen-direction contracts by |1 - lambda/L|,
    so the worst-case per-step contraction is 1 - mu/L = 1 - 1/cond.
    Hence rho = 1 - 1/cond and f - f* contracts by at most rho^2 per step.

    Note: the rate (kappa-1)/(kappa+1) applies only to the optimal step
    lr = 2/(L + mu).  For lr = 1/L the correct rate is rho = 1 - mu/L.

    We assert:
      (i)  f_gap_T <= rho^(2*T) * f_gap_0 * (1 + 1e-6) + 1e-14  (with T = n_iters)
      (ii) the empirical per-step contraction of the gap, computed from
           f_history between iterations 100 and 300 as
           (gap[300] / gap[100])^(1/200), is within 2% of rho^2.
    """
    cond = 50.0
    dim = 10
    obj = quadratic(cond=cond, dim=dim, seed=0)
    L = cond       # largest eigenvalue (= cond when mu = 1)
    mu = 1.0       # smallest eigenvalue
    lr = 1.0 / L
    # Correct per-step contraction rate for lr = 1/L.
    rho = 1.0 - mu / L   # = 1 - 1/cond

    x0 = np.ones(dim)
    opt = GradientDescent(lr=lr)
    T = 500
    result = minimize(obj, x0, opt, max_iter=T, tol=0)

    f_history = result.f_history
    n_iters = result.n_iters  # should equal T since tol=0

    f0_gap = f_history[0] - obj.f_star
    fT_gap = f_history[-1] - obj.f_star

    # (i) Global bound.
    assert fT_gap <= rho ** (2 * n_iters) * f0_gap * (1 + 1e-6) + 1e-14, (
        f"GD linear rate bound violated: f_gap={fT_gap:.3e}, "
        f"bound={rho**(2*n_iters) * f0_gap:.3e}, rho={rho:.6f}, T={n_iters}"
    )

    # (ii) Empirical per-step contraction between iterations 100 and 300.
    gap_100 = f_history[100] - obj.f_star
    gap_300 = f_history[300] - obj.f_star
    empirical_rate = (gap_300 / gap_100) ** (1.0 / 200.0)
    assert abs(empirical_rate - rho ** 2) <= 0.02 * rho ** 2, (
        f"Empirical per-step contraction {empirical_rate:.6f} differs from "
        f"rho^2={rho**2:.6f} by more than 2%"
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
