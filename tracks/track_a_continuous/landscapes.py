"""Optimization landscape definitions for Track A – Continuous Optimization.

Each landscape is an :class:`Objective` dataclass exposing ``f``, ``grad``,
``hess``, and optional ground-truth solution fields.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Allow "from common..." imports regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass
class Objective:
    """Container for a scalar optimization problem.

    Attributes:
        name: Human-readable identifier.
        dim: Dimension of the input space.
        f: Scalar objective function ``f(x) -> float``.
        grad: Gradient function ``grad(x) -> NDArray[float64]`` of shape ``(dim,)``.
        hess: Optional Hessian ``hess(x) -> NDArray`` of shape ``(dim, dim)``.
        x_star: Known minimiser (if available).
        f_star: Known minimum value (if available).
        description: Short prose description of the landscape.
    """

    name: str
    dim: int
    f: Callable[[NDArray], float]
    grad: Callable[[NDArray], NDArray]
    hess: Optional[Callable[[NDArray], NDArray]] = None
    x_star: Optional[NDArray] = None
    f_star: Optional[float] = None
    description: str = ""


# ---------------------------------------------------------------------------
# Quadratic landscape
# ---------------------------------------------------------------------------

def quadratic(cond: float = 50.0, dim: int = 10, seed: int = 0) -> Objective:
    """Create a strongly-convex quadratic with controlled condition number.

    The Hessian is ``A = Q diag(linspace(1, cond, dim)) Q^T`` where ``Q`` is a
    random orthogonal matrix.  This gives ``mu = 1`` (smallest eigenvalue) and
    ``L = cond`` (largest eigenvalue / Lipschitz constant of grad).

    Args:
        cond: Condition number (ratio L/mu). Defaults to 50.0.
        dim: Problem dimension. Defaults to 10.
        seed: RNG seed for reproducibility. Defaults to 0.

    Returns:
        Objective with exact ``x_star`` and ``f_star``.
    """
    rng = np.random.default_rng(seed)

    # Build random orthogonal Q via QR decomposition of a Gaussian matrix.
    G = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(G)

    # Eigenvalues spread log-uniformly from 1 to cond.
    eigvals = np.linspace(1.0, cond, dim)
    D = np.diag(eigvals)

    # A is symmetric positive-definite; x_star = A^{-1} b.
    A = Q @ D @ Q.T
    b = rng.standard_normal(dim)

    # Exact solution and minimum value.
    x_star = np.linalg.solve(A, b)
    f_star = float(-0.5 * b @ x_star)  # = -0.5 b^T A^{-1} b

    def f(x: NDArray) -> float:
        """Evaluate f = 0.5 x^T A x - b^T x."""
        return float(0.5 * x @ A @ x - b @ x)

    def grad(x: NDArray) -> NDArray:
        """Gradient: A x - b."""
        return A @ x - b

    def hess(x: NDArray) -> NDArray:
        """Hessian is constant A."""
        return A.copy()

    return Objective(
        name="quadratic",
        dim=dim,
        f=f,
        grad=grad,
        hess=hess,
        x_star=x_star,
        f_star=f_star,
        description=(
            f"Strongly-convex quadratic in R^{dim}; "
            f"mu=1, L={cond:.1f} (cond={cond:.1f})."
        ),
    )


# ---------------------------------------------------------------------------
# Rosenbrock landscape
# ---------------------------------------------------------------------------

def rosenbrock(dim: int = 2) -> Objective:
    """Rosenbrock (banana) function in arbitrary dimension.

    ``f(x) = sum_{i=0}^{d-2} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]``

    Global minimum is at ``x_star = (1, ..., 1)`` with ``f_star = 0``.

    Args:
        dim: Problem dimension (>= 2). Defaults to 2.

    Returns:
        Objective with known x_star and f_star.
    """

    def f(x: NDArray) -> float:
        """Rosenbrock value – sum of valley and slope terms."""
        x = np.asarray(x, dtype=float)
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

    def grad(x: NDArray) -> NDArray:
        """Analytic Rosenbrock gradient."""
        x = np.asarray(x, dtype=float)
        g = np.zeros_like(x)
        # Contribution from term i (x[i] appears in two terms when 0 < i < d-1).
        g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
        return g

    def hess(x: NDArray) -> NDArray:
        """Analytic Rosenbrock Hessian (tridiagonal structure)."""
        x = np.asarray(x, dtype=float)
        d = len(x)
        H = np.zeros((d, d))
        for i in range(d - 1):
            H[i, i] += -400.0 * x[i + 1] + 1200.0 * x[i] ** 2 + 2.0
            H[i + 1, i + 1] += 200.0
            H[i, i + 1] += -400.0 * x[i]
            H[i + 1, i] += -400.0 * x[i]
        return H

    x_star = np.ones(dim)
    return Objective(
        name="rosenbrock",
        dim=dim,
        f=f,
        grad=grad,
        hess=hess,
        x_star=x_star,
        f_star=0.0,
        description=f"Rosenbrock banana function in R^{dim}; non-convex, narrow curved valley.",
    )


# ---------------------------------------------------------------------------
# Sum-quartic landscape
# ---------------------------------------------------------------------------

def sum_quartic(dim: int = 4) -> Objective:
    """Convex (but not strongly convex) sum-of-fourth-powers.

    ``f(x) = sum_i x_i^4``;  minimum at origin with f_star = 0.
    The Hessian ``diag(12 x_i^2)`` is singular at x=0, so it is not strongly convex.

    Args:
        dim: Problem dimension. Defaults to 4.

    Returns:
        Objective with exact x_star and f_star.
    """

    def f(x: NDArray) -> float:
        """Sum of fourth powers."""
        return float(np.sum(np.asarray(x, dtype=float) ** 4))

    def grad(x: NDArray) -> NDArray:
        """Gradient: 4 x^3 element-wise."""
        return 4.0 * np.asarray(x, dtype=float) ** 3

    def hess(x: NDArray) -> NDArray:
        """Diagonal Hessian: 12 x^2 element-wise."""
        return np.diag(12.0 * np.asarray(x, dtype=float) ** 2)

    return Objective(
        name="sum_quartic",
        dim=dim,
        f=f,
        grad=grad,
        hess=hess,
        x_star=np.zeros(dim),
        f_star=0.0,
        description=(
            f"Convex (not strongly convex) sum x_i^4 in R^{dim}; "
            "Hessian singular at global minimum."
        ),
    )


# ---------------------------------------------------------------------------
# Saddle function
# ---------------------------------------------------------------------------

def saddle_function() -> Objective:
    """2-D function with a saddle at the origin and two global minima.

    ``f(x, y) = x^2 - y^2 + 0.25 y^4``

    * Gradient vanishes at (0, 0) – this is a saddle point.
    * Minima at (0, ±sqrt(2)) where f = -1.

    Returns:
        2-D Objective.
    """

    def f(x: NDArray) -> float:
        """Saddle value."""
        x = np.asarray(x, dtype=float)
        return float(x[0] ** 2 - x[1] ** 2 + 0.25 * x[1] ** 4)

    def grad(x: NDArray) -> NDArray:
        """Analytic gradient of saddle function."""
        x = np.asarray(x, dtype=float)
        return np.array([2.0 * x[0], -2.0 * x[1] + x[1] ** 3])

    def hess(x: NDArray) -> NDArray:
        """Analytic Hessian of saddle function."""
        x = np.asarray(x, dtype=float)
        return np.array([[2.0, 0.0], [0.0, -2.0 + 3.0 * x[1] ** 2]])

    y_min = float(np.sqrt(2.0))
    f_star = -(y_min ** 2) + 0.25 * y_min ** 4  # = -1

    return Objective(
        name="saddle_function",
        dim=2,
        f=f,
        grad=grad,
        hess=hess,
        x_star=np.array([0.0, y_min]),
        f_star=f_star,
        description=(
            "2-D: f = x^2 - y^2 + 0.25 y^4.  "
            "Saddle at origin; minima at (0, ±√2)."
        ),
    )


# ---------------------------------------------------------------------------
# Double-well landscape
# ---------------------------------------------------------------------------

def double_well() -> Objective:
    """2-D double-well with one local and one global minimum in x.

    ``f(x, y) = x^4 - 3 x^2 + x + y^2``

    The y-direction is a simple quadratic (minimum at y=0).  In x the
    quartic has two wells; one is the global minimum and one is local.

    Returns:
        2-D Objective.  x_star is set to the global minimum numerically.
    """

    def f(x: NDArray) -> float:
        """Double-well value."""
        x = np.asarray(x, dtype=float)
        return float(x[0] ** 4 - 3.0 * x[0] ** 2 + x[0] + x[1] ** 2)

    def grad(x: NDArray) -> NDArray:
        """Analytic gradient."""
        x = np.asarray(x, dtype=float)
        return np.array([
            4.0 * x[0] ** 3 - 6.0 * x[0] + 1.0,
            2.0 * x[1],
        ])

    def hess(x: NDArray) -> NDArray:
        """Analytic Hessian."""
        x = np.asarray(x, dtype=float)
        return np.array([
            [12.0 * x[0] ** 2 - 6.0, 0.0],
            [0.0, 2.0],
        ])

    # Locate global minimum numerically (solve 4x^3 - 6x + 1 = 0 in [-2, 0]).
    from scipy.optimize import brentq
    dx = lambda x0: 4 * x0 ** 3 - 6 * x0 + 1  # noqa: E731
    x_global = float(brentq(dx, -2.0, -1.0))
    x_star = np.array([x_global, 0.0])
    f_star = f(x_star)

    return Objective(
        name="double_well",
        dim=2,
        f=f,
        grad=grad,
        hess=hess,
        x_star=x_star,
        f_star=float(f_star),
        description=(
            "2-D: f = x^4 - 3x^2 + x + y^2.  "
            "Double-well in x; global minimum and one local minimum."
        ),
    )


# ---------------------------------------------------------------------------
# Stationary-point classifier
# ---------------------------------------------------------------------------

def classify_stationary_point(
    obj: Objective,
    x: NDArray,
    tol: float = 1e-6,
) -> str:
    """Classify a stationary point using eigenvalues of the Hessian.

    Args:
        obj: Objective whose ``hess`` is used.
        x: Candidate stationary point.
        tol: Tolerance for treating eigenvalues as zero.

    Returns:
        One of ``"minimum"``, ``"maximum"``, ``"saddle"``, or ``"degenerate"``.

    Raises:
        ValueError: If the objective does not provide a Hessian.
    """
    if obj.hess is None:
        raise ValueError(f"Objective '{obj.name}' has no Hessian defined.")

    H = np.asarray(obj.hess(x), dtype=float)
    # Use eigvalsh for symmetric matrices (faster, real output).
    eigvals = np.linalg.eigvalsh(H)

    pos = np.all(eigvals > tol)
    neg = np.all(eigvals < -tol)
    mixed = np.any(eigvals > tol) and np.any(eigvals < -tol)

    if pos:
        return "minimum"
    if neg:
        return "maximum"
    if mixed:
        return "saddle"
    return "degenerate"  # some eigenvalue is within [-tol, tol]


# ---------------------------------------------------------------------------
# Logistic regression objective
# ---------------------------------------------------------------------------

@dataclass
class LogisticData:
    """Auxiliary data object for the logistic regression objective.

    Attributes:
        X: Feature matrix of shape ``(n, d)``.
        y: Binary label vector of shape ``(n,)``, values in {0, 1}.
        l2: L2-regularisation coefficient.
    """

    X: NDArray
    y: NDArray
    l2: float

    def per_sample_grads(self, w: NDArray) -> NDArray:
        """Per-sample gradient matrix of shape ``(n, d)``.

        Each row is the gradient of the i-th loss term w.r.t. ``w``
        (without the regularisation, which is the same for all samples).

        Args:
            w: Weight vector of shape ``(d,)``.

        Returns:
            NDArray of shape ``(n, d)``.
        """
        # Sigmoid predictions.
        logits = self.X @ w
        probs = _sigmoid(logits)
        # Residuals: sigma(logit) - y_i.
        residuals = probs - self.y  # (n,)
        # Outer product: residual_i * x_i.
        return residuals[:, None] * self.X  # (n, d)

    def batch_grad(self, w: NDArray, idx: NDArray) -> NDArray:
        """Mini-batch gradient for sample indices ``idx``.

        Returns the full gradient (averaged over the mini-batch, then scaled
        to n) so that step sizes are independent of batch size.

        Args:
            w: Weight vector of shape ``(d,)``.
            idx: Integer indices of the mini-batch.

        Returns:
            Gradient vector of shape ``(d,)``.
        """
        n = self.X.shape[0]
        X_b = self.X[idx]
        y_b = self.y[idx]
        logits = X_b @ w
        probs = _sigmoid(logits)
        # Average per-sample grad, scaled to n so it approximates the full grad.
        return n / len(idx) * (X_b.T @ (probs - y_b)) / n + self.l2 * w


def _sigmoid(z: NDArray) -> NDArray:
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def make_logistic_regression(
    n: int = 2000,
    d: int = 20,
    l2: float = 1e-2,
    seed: int = 0,
) -> tuple[Objective, LogisticData]:
    """Create a synthetic binary logistic-regression objective.

    Generates a linearly separable dataset with slight label noise and returns
    both the :class:`Objective` (full-batch f / grad / hess) and a
    :class:`LogisticData` object for stochastic methods.

    Args:
        n: Number of data points. Defaults to 2000.
        d: Feature dimension. Defaults to 20.
        l2: L2-regularisation strength. Defaults to 1e-2.
        seed: RNG seed. Defaults to 0.

    Returns:
        ``(obj, data)`` tuple.
    """
    rng = np.random.default_rng(seed)

    # Generate synthetic data: X ~ N(0, I), labels from a linear model.
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d) / np.sqrt(d)
    logits_true = X @ w_true
    probs_true = _sigmoid(logits_true)
    y = (rng.uniform(size=n) < probs_true).astype(float)

    data = LogisticData(X=X, y=y, l2=l2)

    def f(w: NDArray) -> float:
        """Negative log-likelihood + L2 penalty."""
        logits = X @ w
        # log-sum-exp form for numerical stability.
        nll = -np.mean(
            y * logits - np.log1p(np.exp(-np.abs(logits))) - np.maximum(0.0, -logits)
        )
        return float(nll + 0.5 * l2 * np.dot(w, w))

    def grad(w: NDArray) -> NDArray:
        """Full-batch gradient."""
        probs = _sigmoid(X @ w)
        return (X.T @ (probs - y)) / n + l2 * w

    def hess(w: NDArray) -> NDArray:
        """Full-batch Hessian (d x d)."""
        probs = _sigmoid(X @ w)
        S = probs * (1.0 - probs)  # (n,) – diagonal of the weight matrix
        # H = X^T diag(S) X / n + l2 I
        return (X.T * S) @ X / n + l2 * np.eye(d)

    obj = Objective(
        name="logistic_regression",
        dim=d,
        f=f,
        grad=grad,
        hess=hess,
        description=(
            f"L2-regularised logistic regression; n={n}, d={d}, l2={l2}."
        ),
    )
    return obj, data


# ---------------------------------------------------------------------------
# Nonlinear least-squares: exponential decay
# ---------------------------------------------------------------------------

@dataclass
class NLSData:
    """Auxiliary object for the nonlinear least-squares exponential decay fit.

    Attributes:
        t: Time vector of shape ``(m,)``.
        y_noisy: Noisy observations of shape ``(m,)``.
    """

    t: NDArray
    y_noisy: NDArray

    def residual(self, x: NDArray) -> NDArray:
        """Residual vector r(x) = model(t; x) - y_noisy.

        Args:
            x: Parameter vector ``[a, b, c]``.

        Returns:
            NDArray of shape ``(m,)``.
        """
        a, b, c = float(x[0]), float(x[1]), float(x[2])
        return a * np.exp(-b * self.t) + c - self.y_noisy

    def jacobian(self, x: NDArray) -> NDArray:
        """Jacobian of the residual w.r.t. x, shape ``(m, 3)``.

        Args:
            x: Parameter vector ``[a, b, c]``.

        Returns:
            NDArray of shape ``(m, 3)``.
        """
        a, b, c = float(x[0]), float(x[1]), float(x[2])
        exp_term = np.exp(-b * self.t)
        J = np.column_stack([
            exp_term,                    # dr/da
            -a * self.t * exp_term,      # dr/db
            np.ones_like(self.t),        # dr/dc
        ])
        return J


def make_exp_decay_fit(seed: int = 0) -> tuple[Objective, NLSData]:
    """Create a nonlinear least-squares problem for fitting y = a exp(-b t) + c.

    Adds Gaussian noise to synthetic observations so that the residual at the
    true parameters is small but non-zero.

    Args:
        seed: RNG seed. Defaults to 0.

    Returns:
        ``(obj, nls_data)`` where ``obj.f`` is ``0.5 * ||r||^2``,
        ``obj.grad`` is ``J^T r``, and ``nls_data`` exposes ``residual``
        and ``jacobian``.
    """
    rng = np.random.default_rng(seed)

    # True parameters.
    a_true, b_true, c_true = 2.0, 0.5, 0.3
    m = 80
    t = np.linspace(0.0, 6.0, m)
    y_clean = a_true * np.exp(-b_true * t) + c_true
    y_noisy = y_clean + rng.standard_normal(m) * 0.05

    nls = NLSData(t=t, y_noisy=y_noisy)

    def f(x: NDArray) -> float:
        """Least-squares objective: 0.5 ||r||^2."""
        r = nls.residual(x)
        return float(0.5 * np.dot(r, r))

    def grad(x: NDArray) -> NDArray:
        """Gradient: J^T r."""
        r = nls.residual(x)
        J = nls.jacobian(x)
        return J.T @ r

    def hess(x: NDArray) -> NDArray:
        """Gauss-Newton approximation of Hessian: J^T J (ignores second-order terms)."""
        J = nls.jacobian(x)
        return J.T @ J

    obj = Objective(
        name="exp_decay_fit",
        dim=3,
        f=f,
        grad=grad,
        hess=hess,
        x_star=np.array([a_true, b_true, c_true]),
        description=(
            "NLS: fit y = a exp(-b t) + c to noisy data; "
            f"true params=({a_true}, {b_true}, {c_true})."
        ),
    )
    return obj, nls
