"""Quantum optimization algorithms for VQE and QAOA parameter training."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Ensure repository root is on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _TRACK_ROOT.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import scipy.optimize


def gd_param_shift(
    params0: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    lr: float,
    n_steps: int,
    cost_per_step: int,
    exact_energy_fn: Callable[[np.ndarray], float],
) -> list[tuple[int, float]]:
    """Gradient descent using parameter-shift gradients.

    Args:
        params0: Initial parameter array.
        grad_fn: Function returning analytic gradient vector.
        lr: Fixed learning rate / step size.
        n_steps: Total number of descent iterations.
        cost_per_step: Number of circuit evaluations per gradient step (typically 2 * P).
        exact_energy_fn: Evaluation function for tracking true energy (not counted as budget).

    Returns:
        History list of (cumulative_circuit_evaluations, exact_energy).
    """
    theta = np.array(params0, dtype=np.float64).copy()
    evals = 0
    history: list[tuple[int, float]] = [(evals, exact_energy_fn(theta))]

    for _ in range(n_steps):
        grad = grad_fn(theta)
        evals += cost_per_step
        theta = theta - lr * grad
        history.append((evals, exact_energy_fn(theta)))

    return history


def spsa(
    params0: np.ndarray,
    loss_fn: Callable[[np.ndarray], float],
    a: float,
    c: float,
    n_steps: int,
    seed: int,
    exact_energy_fn: Callable[[np.ndarray], float],
    A: float | None = None,
) -> list[tuple[int, float]]:
    """Simultaneous Perturbation Stochastic Approximation (SPSA).

    Evaluates 2 circuit queries per iteration regardless of parameter dimension P.

    Args:
        params0: Initial parameter array.
        loss_fn: Objective evaluation function (exact or noisy/finite-shot).
        a: Initial learning rate scale.
        c: Initial perturbation radius scale.
        n_steps: Total number of iterations.
        seed: Random seed for Rademacher direction sampling.
        exact_energy_fn: Evaluation function for tracking true energy.
        A: Stability constant; defaults to 0.1 * n_steps.

    Returns:
        History list of (cumulative_circuit_evaluations, exact_energy).
    """
    rng = np.random.default_rng(seed)
    theta = np.array(params0, dtype=np.float64).copy()
    P = len(theta)

    if A is None:
        A = 0.1 * float(n_steps)

    evals = 0
    history: list[tuple[int, float]] = [(evals, exact_energy_fn(theta))]

    for k in range(n_steps):
        # Asymptotically optimal gain decay schedules
        a_k = a / ((float(k) + 1.0 + A) ** 0.602)
        c_k = c / ((float(k) + 1.0) ** 0.101)

        # Rademacher perturbation vector (+1 or -1 with probability 0.5)
        delta = rng.choice([-1.0, 1.0], size=P)

        # Simultaneous forward and backward evaluations
        f_plus = loss_fn(theta + c_k * delta)
        f_minus = loss_fn(theta - c_k * delta)
        evals += 2

        # Stochastic gradient estimate: (f+ - f-) / (2 * c_k * delta)
        g_hat = (f_plus - f_minus) / (2.0 * c_k) * delta
        theta = theta - a_k * g_hat

        history.append((evals, exact_energy_fn(theta)))

    return history


def qng(
    params0: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    metric_fn: Callable[[np.ndarray], np.ndarray],
    lr: float,
    n_steps: int,
    exact_energy_fn: Callable[[np.ndarray], float],
    reg: float = 1e-3,
    cost_per_step: int = 49,
) -> list[tuple[int, float]]:
    """Quantum Natural Gradient (QNG) optimizer using Fubini-Study metric tensor.

    Update rule:
        theta <- theta - lr * (g + reg * I)^(-1) * grad
    """
    theta = np.array(params0, dtype=np.float64).copy()
    P = len(theta)
    evals = 0
    history: list[tuple[int, float]] = [(evals, exact_energy_fn(theta))]

    eye_reg = reg * np.eye(P, dtype=np.float64)

    for _ in range(n_steps):
        grad = grad_fn(theta)
        g_mat = metric_fn(theta)
        evals += cost_per_step

        # Regularized natural gradient solve
        nat_grad = np.linalg.solve(g_mat + eye_reg, grad)
        theta = theta - lr * nat_grad

        history.append((evals, exact_energy_fn(theta)))

    return history


def cobyla(
    params0: np.ndarray,
    loss_fn: Callable[[np.ndarray], float],
    maxiter: int,
    exact_energy_fn: Callable[[np.ndarray], float],
) -> list[tuple[int, float]]:
    """Constrained Optimization BY Linear Approximation (COBYLA) with iteration tracking."""
    theta = np.array(params0, dtype=np.float64).copy()
    evals = 0
    history: list[tuple[int, float]] = [(evals, exact_energy_fn(theta))]

    def wrapped_loss(th: np.ndarray) -> float:
        nonlocal evals
        evals += 1
        return float(loss_fn(th))

    def callback(xk: np.ndarray) -> None:
        history.append((evals, exact_energy_fn(xk)))

    res = scipy.optimize.minimize(
        wrapped_loss,
        theta,
        method="COBYLA",
        callback=callback,
        options={"maxiter": maxiter},
    )

    # Ensure final state is recorded if not captured by callback
    if history[-1][0] != evals:
        history.append((evals, exact_energy_fn(res.x)))

    return history

