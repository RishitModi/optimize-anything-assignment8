"""Second-order optimization methods and Hessian analysis utilities for Track A.

Includes Newton's method, Gauss-Newton, Levenberg-Marquardt, HVP utilities,
Fisher information approximation, and diagonal preconditioning.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

# Ensure "from common..." imports resolve against the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve as scipy_solve

from first_order import OptResult


# ---------------------------------------------------------------------------
# Newton's method
# ---------------------------------------------------------------------------

def newton(
    obj,  # Objective
    x0: NDArray,
    max_iter: int = 50,
    tol: float = 1e-8,
    damping: float = 0.0,
    line_search: bool = False,
) -> OptResult:
    """Damped Newton's method with optional Armijo backtracking line search.

    Solves ``(H + damping * I) p = -g`` at each step.  If ``line_search`` is
    True an Armijo backtracking condition is used to choose the step length.

    Args:
        obj: Objective with ``f``, ``grad``, and ``hess`` defined.
        x0: Initial point.
        max_iter: Maximum number of Newton steps.
        tol: Stop when ``||grad||_2 < tol``.
        damping: Levenberg-style diagonal regularisation of the Hessian.
        line_search: Whether to use Armijo backtracking.

    Returns:
        OptResult with per-iteration history.

    Raises:
        ValueError: If ``obj.hess`` is None.
    """
    import time

    if obj.hess is None:
        raise ValueError(f"Newton requires a Hessian; '{obj.name}' has none.")

    x = np.array(x0, dtype=float)
    f_history = []
    gnorm_history = []
    converged = False
    diverged = False
    t0 = time.perf_counter()

    for _ in range(max_iter):
        fval = obj.f(x)
        g = obj.grad(x)
        H = obj.hess(x)

        if not np.isfinite(fval) or fval > 1e12:
            diverged = True
            f_history.append(float(fval))
            gnorm_history.append(float("nan"))
            break

        f_history.append(float(fval))
        gnorm = float(np.linalg.norm(g))
        gnorm_history.append(gnorm)

        if gnorm < tol:
            converged = True
            break

        # Regularise Hessian for numerical safety.
        H_reg = H + damping * np.eye(len(x))
        # Solve for Newton direction p such that H p = -g.
        try:
            p = scipy_solve(H_reg, -g, assume_a="sym")
        except np.linalg.LinAlgError:
            # Fall back to gradient descent if H is singular.
            p = -g

        if line_search:
            # Armijo backtracking: reduce step until sufficient decrease is met.
            alpha = 1.0
            c_armijo = 1e-4
            slope = g @ p  # should be negative for a descent direction
            while alpha > 1e-12 and obj.f(x + alpha * p) > fval + c_armijo * alpha * slope:
                alpha *= 0.5
        else:
            alpha = 1.0

        x = x + alpha * p

    runtime_s = time.perf_counter() - t0
    return OptResult(
        x=x,
        f_history=f_history,
        gradnorm_history=gnorm_history,
        n_iters=len(f_history),
        runtime_s=runtime_s,
        converged=converged,
        diverged=diverged,
    )


# ---------------------------------------------------------------------------
# Gauss-Newton method
# ---------------------------------------------------------------------------

def gauss_newton(
    residual: Callable[[NDArray], NDArray],
    jacobian: Callable[[NDArray], NDArray],
    x0: NDArray,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> OptResult:
    """Gauss-Newton method for nonlinear least squares.

    Solves ``(J^T J) p = -J^T r`` at each step (no damping).

    Args:
        residual: ``r(x)`` of shape ``(m,)``.
        jacobian: ``J(x)`` of shape ``(m, d)``.
        x0: Initial parameter vector.
        max_iter: Maximum iterations.
        tol: Convergence on gradient norm.

    Returns:
        OptResult.
    """
    import time

    x = np.array(x0, dtype=float)
    f_history = []
    gnorm_history = []
    converged = False
    diverged = False
    t0 = time.perf_counter()

    for _ in range(max_iter):
        r = residual(x)
        J = jacobian(x)
        fval = 0.5 * float(np.dot(r, r))
        g = J.T @ r  # gradient of 0.5 ||r||^2

        if not np.isfinite(fval) or fval > 1e12:
            diverged = True
            f_history.append(float(fval))
            gnorm_history.append(float("nan"))
            break

        f_history.append(float(fval))
        gnorm = float(np.linalg.norm(g))
        gnorm_history.append(gnorm)

        if gnorm < tol:
            converged = True
            break

        # Gauss-Newton normal equations: J^T J p = -J^T r.
        JTJ = J.T @ J
        try:
            p = scipy_solve(JTJ, -g, assume_a="pos")
        except np.linalg.LinAlgError:
            p = -g

        x = x + p

    runtime_s = time.perf_counter() - t0
    return OptResult(
        x=x,
        f_history=f_history,
        gradnorm_history=gnorm_history,
        n_iters=len(f_history),
        runtime_s=runtime_s,
        converged=converged,
        diverged=diverged,
    )


# ---------------------------------------------------------------------------
# Levenberg-Marquardt
# ---------------------------------------------------------------------------

def levenberg_marquardt(
    residual: Callable[[NDArray], NDArray],
    jacobian: Callable[[NDArray], NDArray],
    x0: NDArray,
    max_iter: int = 100,
    tol: float = 1e-8,
    lam0: float = 1e-2,
) -> OptResult:
    """Levenberg-Marquardt with adaptive regularisation parameter lambda.

    Solves ``(J^T J + lam * I) p = -J^T r``.  Increases ``lam`` when the
    step is rejected and decreases it on acceptance.

    Args:
        residual: ``r(x)`` of shape ``(m,)``.
        jacobian: ``J(x)`` of shape ``(m, d)``.
        x0: Initial parameter vector.
        max_iter: Maximum iterations.
        tol: Convergence on gradient norm.
        lam0: Initial regularisation parameter.

    Returns:
        OptResult.
    """
    import time

    x = np.array(x0, dtype=float)
    lam = float(lam0)
    f_history = []
    gnorm_history = []
    converged = False
    diverged = False
    t0 = time.perf_counter()

    for _ in range(max_iter):
        r = residual(x)
        J = jacobian(x)
        fval = 0.5 * float(np.dot(r, r))
        g = J.T @ r

        if not np.isfinite(fval) or fval > 1e12:
            diverged = True
            f_history.append(float(fval))
            gnorm_history.append(float("nan"))
            break

        f_history.append(float(fval))
        gnorm = float(np.linalg.norm(g))
        gnorm_history.append(gnorm)

        if gnorm < tol:
            converged = True
            break

        JTJ = J.T @ J
        d = len(x)
        # Augmented normal equations.
        try:
            p = scipy_solve(JTJ + lam * np.eye(d), -g, assume_a="pos")
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        x_new = x + p
        f_new = 0.5 * float(np.dot(residual(x_new), residual(x_new)))

        if f_new < fval:
            # Accepted: reduce lam (move toward Gauss-Newton).
            x = x_new
            lam = max(lam / 3.0, 1e-12)
        else:
            # Rejected: increase lam (move toward gradient descent).
            lam = min(lam * 10.0, 1e12)

    runtime_s = time.perf_counter() - t0
    return OptResult(
        x=x,
        f_history=f_history,
        gradnorm_history=gnorm_history,
        n_iters=len(f_history),
        runtime_s=runtime_s,
        converged=converged,
        diverged=diverged,
    )


# ---------------------------------------------------------------------------
# Hessian-vector products
# ---------------------------------------------------------------------------

def hvp_finite_difference(
    grad: Callable[[NDArray], NDArray],
    x: NDArray,
    v: NDArray,
    eps: float = 1e-5,
) -> NDArray:
    """Estimate Hessian-vector product using central finite differences.

    ``hvp(x, v) ≈ (grad(x + eps v) - grad(x - eps v)) / (2 eps)``

    Args:
        grad: Gradient function.
        x: Point at which to evaluate the HVP.
        v: Direction vector (same shape as x).
        eps: Finite difference step size.

    Returns:
        Approximate HVP of shape ``(d,)``.
    """
    v = np.asarray(v, dtype=float)
    return (grad(x + eps * v) - grad(x - eps * v)) / (2.0 * eps)


def hvp_exact(obj, x: NDArray, v: NDArray) -> NDArray:
    """Exact Hessian-vector product using the analytic Hessian.

    Args:
        obj: Objective with ``hess`` defined.
        x: Point of evaluation.
        v: Direction vector.

    Returns:
        HVP ``H(x) @ v``.
    """
    H = np.asarray(obj.hess(x), dtype=float)
    return H @ np.asarray(v, dtype=float)


# ---------------------------------------------------------------------------
# Top-k eigenpairs via power iteration with deflation
# ---------------------------------------------------------------------------

def top_eigenpairs_from_hvp(
    hvp: Callable[[NDArray], NDArray],
    dim: int,
    k: int,
    seed: int = 0,
    n_iter: int = 200,
    tol: float = 1e-8,
) -> Tuple[NDArray, NDArray]:
    """Compute the top-k eigenpairs using power iteration with deflation.

    Uses only matrix-free HVPs (no explicit Hessian construction).

    Args:
        hvp: Callable ``v -> H v`` (Hessian-vector product).
        dim: Ambient dimension.
        k: Number of eigenpairs to compute.
        seed: RNG seed.
        n_iter: Maximum power-iteration steps per eigenpair.
        tol: Convergence tolerance on eigenvalue change.

    Returns:
        ``(eigvals, eigvecs)`` where ``eigvals`` has shape ``(k,)`` and
        ``eigvecs`` has shape ``(dim, k)``.
    """
    rng = np.random.default_rng(seed)
    eigvals = np.zeros(k)
    eigvecs = np.zeros((dim, k))

    for i in range(k):
        # Initialise a random unit vector.
        v = rng.standard_normal(dim)
        v /= np.linalg.norm(v)

        lam_prev = None
        for _ in range(n_iter):
            # Apply H and deflate previously found eigenvectors.
            Hv = hvp(v)
            for j in range(i):
                # Deflation: remove projection onto already-found eigenvectors.
                Hv -= eigvals[j] * (eigvecs[:, j] @ v) * eigvecs[:, j]

            lam = float(v @ Hv)  # Rayleigh quotient
            v_new = Hv / (np.linalg.norm(Hv) + 1e-30)

            if lam_prev is not None and abs(lam - lam_prev) < tol:
                break
            lam_prev = lam
            v = v_new

        eigvals[i] = float(v @ hvp(v))
        eigvecs[:, i] = v

    return eigvals, eigvecs


# ---------------------------------------------------------------------------
# Empirical Fisher diagonal
# ---------------------------------------------------------------------------

def empirical_fisher_diag(per_sample_grads: NDArray) -> NDArray:
    """Compute the diagonal of the empirical Fisher information matrix.

    The empirical Fisher diagonal is the mean of the element-wise squared
    per-sample gradients: ``diag(F) = mean_i( g_i^2 )``.

    Args:
        per_sample_grads: NDArray of shape ``(n, d)`` – each row is the
            gradient of the loss for sample ``i``.

    Returns:
        NDArray of shape ``(d,)`` – the diagonal of the empirical Fisher.
    """
    # Average squared gradient across samples.
    return np.mean(per_sample_grads ** 2, axis=0)


# ---------------------------------------------------------------------------
# Low-rank Hessian approximation
# ---------------------------------------------------------------------------

def low_rank_hessian_approx(
    hvp: Callable[[NDArray], NDArray],
    dim: int,
    k: int,
    seed: int = 0,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Low-rank approximation of the Hessian from the top-k eigenpairs.

    ``H_approx = sum_{i=1}^{k} lambda_i * q_i q_i^T``

    Args:
        hvp: Hessian-vector product callable.
        dim: Ambient dimension.
        k: Rank of the approximation.
        seed: RNG seed.

    Returns:
        ``(eigvals, eigvecs, H_approx)`` where ``H_approx`` has shape ``(dim, dim)``.
    """
    eigvals, eigvecs = top_eigenpairs_from_hvp(hvp, dim, k, seed=seed)
    # Build low-rank approximation.
    H_approx = (eigvecs * eigvals[np.newaxis, :]) @ eigvecs.T
    return eigvals, eigvecs, H_approx


# ---------------------------------------------------------------------------
# Diagonal-preconditioned gradient descent
# ---------------------------------------------------------------------------

def diag_preconditioned_gd(
    obj,  # Objective
    x0: NDArray,
    diag: NDArray,
    lr: float,
    max_iter: int,
    damping: float = 1e-8,
    tol: float = 1e-8,
) -> OptResult:
    """Gradient descent with a fixed diagonal preconditioner.

    The effective step is ``x = x - lr * g / (diag + damping)`` applied
    element-wise, which rescales each coordinate by its preconditioner.

    Args:
        obj: Objective.
        x0: Initial parameter vector.
        diag: Diagonal preconditioner vector (same length as x0).
        lr: Learning rate multiplier.
        max_iter: Maximum iterations.
        damping: Added to ``diag`` for numerical stability.
        tol: Gradient-norm convergence threshold.

    Returns:
        OptResult.
    """
    import time

    x = np.array(x0, dtype=float)
    d = np.asarray(diag, dtype=float) + damping
    f_history = []
    gnorm_history = []
    converged = False
    diverged = False
    t0 = time.perf_counter()

    for _ in range(max_iter):
        fval = obj.f(x)

        if not np.isfinite(fval) or fval > 1e12:
            diverged = True
            f_history.append(float(fval))
            gnorm_history.append(float("nan"))
            break

        f_history.append(float(fval))
        g = obj.grad(x)
        gnorm = float(np.linalg.norm(g))
        gnorm_history.append(gnorm)

        if gnorm < tol:
            converged = True
            break

        # Preconditioned step: rescale gradient by inverse diagonal.
        x = x - lr * g / d

    runtime_s = time.perf_counter() - t0
    return OptResult(
        x=x,
        f_history=f_history,
        gradnorm_history=gnorm_history,
        n_iters=len(f_history),
        runtime_s=runtime_s,
        converged=converged,
        diverged=diverged,
    )
