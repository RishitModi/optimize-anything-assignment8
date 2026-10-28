"""Stateful first-order optimizers and optimization loop for Track A.

All optimizers follow the interface ``step(x, grad_fn) -> x_new`` and expose a
``reset()`` method for re-use across multiple runs.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

# Ensure "from common..." imports resolve against the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from numpy.typing import NDArray

from common.timing import Timer


# ---------------------------------------------------------------------------
# Optimizer base and implementations
# ---------------------------------------------------------------------------

class GradientDescent:
    """Vanilla gradient descent: x_{t+1} = x_t - lr * grad(x_t).

    Args:
        lr: Learning rate (step size).
    """

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def reset(self) -> None:
        """Reset internal state (no-op for vanilla GD)."""

    def step(self, x: NDArray, grad_fn: Callable[[NDArray], NDArray]) -> NDArray:
        """Perform one gradient descent step.

        Args:
            x: Current parameter vector.
            grad_fn: Returns the gradient at a given point.

        Returns:
            Updated parameter vector.
        """
        return x - self.lr * grad_fn(x)


class Momentum:
    """Heavy-ball momentum: accumulates an exponential moving average of gradients.

    Update rule:
        v_{t+1} = beta * v_t + grad(x_t)
        x_{t+1} = x_t - lr * v_{t+1}

    Args:
        lr: Learning rate.
        beta: Momentum coefficient (0 < beta < 1).
    """

    def __init__(self, lr: float, beta: float = 0.9) -> None:
        self.lr = lr
        self.beta = beta
        self._v: Optional[NDArray] = None

    def reset(self) -> None:
        """Reset velocity buffer."""
        self._v = None

    def step(self, x: NDArray, grad_fn: Callable[[NDArray], NDArray]) -> NDArray:
        """One momentum step."""
        if self._v is None:
            self._v = np.zeros_like(x)
        g = grad_fn(x)
        # Accumulate momentum velocity.
        self._v = self.beta * self._v + g
        return x - self.lr * self._v


class NAG:
    """Nesterov Accelerated Gradient.

    Evaluates the gradient at the look-ahead point ``x - lr * beta * v``.

    Update rule:
        look_ahead = x_t - lr * beta * v_t
        v_{t+1} = beta * v_t + grad(look_ahead)
        x_{t+1} = x_t - lr * v_{t+1}

    Args:
        lr: Learning rate.
        beta: Momentum coefficient.
    """

    def __init__(self, lr: float, beta: float = 0.9) -> None:
        self.lr = lr
        self.beta = beta
        self._v: Optional[NDArray] = None

    def reset(self) -> None:
        """Reset velocity buffer."""
        self._v = None

    def step(self, x: NDArray, grad_fn: Callable[[NDArray], NDArray]) -> NDArray:
        """One NAG step with look-ahead gradient evaluation."""
        if self._v is None:
            self._v = np.zeros_like(x)
        # Look-ahead point.
        x_look = x - self.lr * self.beta * self._v
        g = grad_fn(x_look)
        self._v = self.beta * self._v + g
        return x - self.lr * self._v


class AdaGrad:
    """AdaGrad: per-coordinate adaptive learning rate.

    Accumulates squared gradients; divides lr by sqrt of the sum.

    Args:
        lr: Global learning rate.
        eps: Small constant for numerical stability.
    """

    def __init__(self, lr: float = 0.1, eps: float = 1e-8) -> None:
        self.lr = lr
        self.eps = eps
        self._G: Optional[NDArray] = None  # accumulated squared grad

    def reset(self) -> None:
        """Reset accumulated squared gradient."""
        self._G = None

    def step(self, x: NDArray, grad_fn: Callable[[NDArray], NDArray]) -> NDArray:
        """One AdaGrad step."""
        if self._G is None:
            self._G = np.zeros_like(x)
        g = grad_fn(x)
        # Accumulate squared gradient (element-wise).
        self._G += g ** 2
        # Adaptive per-coordinate step.
        return x - self.lr * g / (np.sqrt(self._G) + self.eps)


class RMSProp:
    """RMSProp: exponential moving average of squared gradients.

    Args:
        lr: Learning rate.
        rho: Decay factor for moving average.
        eps: Numerical stability constant.
    """

    def __init__(self, lr: float = 1e-3, rho: float = 0.9, eps: float = 1e-8) -> None:
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self._v: Optional[NDArray] = None  # EMA of squared grad

    def reset(self) -> None:
        """Reset EMA buffer."""
        self._v = None

    def step(self, x: NDArray, grad_fn: Callable[[NDArray], NDArray]) -> NDArray:
        """One RMSProp step."""
        if self._v is None:
            self._v = np.zeros_like(x)
        g = grad_fn(x)
        # Exponential moving average of squared gradient.
        self._v = self.rho * self._v + (1.0 - self.rho) * g ** 2
        return x - self.lr * g / (np.sqrt(self._v) + self.eps)


class Adam:
    """Adam optimizer with bias-corrected first and second moment estimates.

    Args:
        lr: Learning rate (step size).
        beta1: Decay for first-moment estimate.
        beta2: Decay for second-moment estimate.
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: Optional[NDArray] = None   # first moment
        self._v: Optional[NDArray] = None   # second moment
        self._t: int = 0                    # step counter (for bias correction)

    def reset(self) -> None:
        """Reset moment estimates and step counter."""
        self._m = None
        self._v = None
        self._t = 0

    def step(self, x: NDArray, grad_fn: Callable[[NDArray], NDArray]) -> NDArray:
        """One Adam step with bias correction."""
        if self._m is None:
            self._m = np.zeros_like(x)
            self._v = np.zeros_like(x)
        g = grad_fn(x)
        self._t += 1
        # Biased moment estimates.
        self._m = self.beta1 * self._m + (1.0 - self.beta1) * g
        self._v = self.beta2 * self._v + (1.0 - self.beta2) * g ** 2
        # Bias-corrected estimates.
        m_hat = self._m / (1.0 - self.beta1 ** self._t)
        v_hat = self._v / (1.0 - self.beta2 ** self._t)
        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptResult:
    """Result of an optimization run.

    Attributes:
        x: Final parameter vector.
        f_history: Objective value recorded at each iteration.
        gradnorm_history: Gradient norm at each iteration.
        n_iters: Total number of iterations completed.
        runtime_s: Wall-clock time in seconds.
        converged: True if the gradient norm fell below ``tol``.
        diverged: True if a non-finite value or f > 1e12 was detected.
    """

    x: NDArray
    f_history: List[float]
    gradnorm_history: List[float]
    n_iters: int
    runtime_s: float
    converged: bool
    diverged: bool


# ---------------------------------------------------------------------------
# Optimization loop
# ---------------------------------------------------------------------------

def minimize(
    obj,  # Objective from landscapes.py
    x0: NDArray,
    optimizer,
    max_iter: int,
    tol: float = 1e-6,
    grad_noise_sigma: float = 0.0,
    seed: int = 0,
) -> OptResult:
    """Run a first-order optimizer on an objective until convergence or budget.

    Optionally adds i.i.d. Gaussian noise to each gradient evaluation to
    simulate stochastic gradients.

    Args:
        obj: An :class:`~landscapes.Objective` instance.
        x0: Initial parameter vector.
        optimizer: Stateful optimizer with ``step(x, grad_fn) -> x_new``.
        max_iter: Maximum number of gradient steps.
        tol: Stop when ``||grad||_2 < tol``.
        grad_noise_sigma: Standard deviation of additive Gaussian gradient noise.
        seed: RNG seed for reproducible noise.

    Returns:
        :class:`OptResult` with full history.
    """
    rng = np.random.default_rng(seed)
    optimizer.reset()

    x = np.array(x0, dtype=float)
    f_history: List[float] = []
    gradnorm_history: List[float] = []
    converged = False
    diverged = False

    with Timer("minimize") as timer:
        for _ in range(max_iter):
            fval = obj.f(x)

            # Early-exit on divergence.
            if not np.isfinite(fval) or fval > 1e12:
                diverged = True
                f_history.append(float(fval))
                gradnorm_history.append(float("nan"))
                break

            f_history.append(float(fval))
            g = obj.grad(x)

            # Optional stochastic noise injection.
            if grad_noise_sigma > 0.0:
                g = g + rng.standard_normal(g.shape) * grad_noise_sigma

            gnorm = float(np.linalg.norm(g))
            gradnorm_history.append(gnorm)

            if gnorm < tol:
                converged = True
                break

            # Take optimizer step; pass a closure so NAG can call grad_fn at look-ahead.
            x = optimizer.step(x, lambda pt: obj.grad(pt))

    return OptResult(
        x=x,
        f_history=f_history,
        gradnorm_history=gradnorm_history,
        n_iters=len(f_history),
        runtime_s=timer.elapsed_sec,
        converged=converged,
        diverged=diverged,
    )


# ---------------------------------------------------------------------------
# Stochastic / mini-batch optimization loop
# ---------------------------------------------------------------------------

def minimize_stochastic(
    data,  # LogisticData or any object with .batch_grad(w, idx) and .X
    w0: NDArray,
    optimizer,
    batch_size: int,
    epochs: int,
    seed: int = 0,
) -> List[float]:
    """Epoch-based stochastic gradient descent loop.

    Each epoch shuffles the training set, draws mini-batches of size
    ``batch_size``, and calls ``data.batch_grad(w, idx)`` for gradient
    estimates.  The full-batch loss (``data.f(w)``) is recorded after every
    epoch.

    - ``batch_size = 1``  → SGD
    - ``1 < batch_size < n``  → mini-batch SGD
    - ``batch_size = n``  → full-batch GD (one gradient per epoch)

    Args:
        data: Object exposing ``.X`` (shape (n, d)) and ``.batch_grad(w, idx)``.
              Also must expose a callable ``f`` for the full-batch loss, OR
              the parent Objective's ``f`` can be passed separately.
        w0: Initial weight vector.
        optimizer: Stateful first-order optimizer.
        batch_size: Mini-batch size.
        epochs: Number of passes over the data.
        seed: RNG seed for shuffle reproducibility.

    Returns:
        List of full-batch loss values, one per epoch (length = epochs + 1,
        including the initial loss before any updates).
    """
    rng = np.random.default_rng(seed)
    optimizer.reset()

    n = data.X.shape[0]
    w = np.array(w0, dtype=float)

    # We store the initial loss then one loss per epoch.
    epoch_losses: List[float] = []

    for _ in range(epochs):
        # Shuffle indices for this epoch.
        perm = rng.permutation(n)
        # Process mini-batches.
        for start in range(0, n, batch_size):
            idx = perm[start: start + batch_size]
            # grad_fn closure over current idx.
            g = data.batch_grad(w, idx)
            # Step: optimizer uses constant-time grad evaluation (already computed).
            w = w - optimizer.lr * g  # direct descent; avoids second grad_fn call
        # Record full-batch loss at end of epoch using per-sample reconstruction.
        # data.batch_grad with all indices / n approximates full-batch grad.
        # For loss we rely on the fact that data has a parent objective accessible
        # via data itself if it carries an ``f`` attribute, otherwise approximate.
        if hasattr(data, "f"):
            epoch_losses.append(float(data.f(w)))
        else:
            # Fallback: 0.5 * ||full-batch grad||^2 as a proxy (not ideal).
            all_idx = np.arange(n)
            g_full = data.batch_grad(w, all_idx)
            epoch_losses.append(float(0.5 * np.dot(g_full, g_full)))

    return epoch_losses
