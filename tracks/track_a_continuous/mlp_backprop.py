"""Two-layer MLP with manual backpropagation for Track A.

Implements tanh hidden activation and sigmoid output with binary cross-entropy
loss.  No autograd is used; all derivatives are computed by hand with chain-rule
comments at each step.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Ensure "from common..." imports resolve against the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def _sigmoid(z: NDArray) -> NDArray:
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _tanh(z: NDArray) -> NDArray:
    """Standard hyperbolic tangent (wraps np.tanh for clarity)."""
    return np.tanh(z)


# ---------------------------------------------------------------------------
# Two-layer MLP
# ---------------------------------------------------------------------------

class TwoLayerMLP:
    """Two-layer MLP: linear -> tanh -> linear -> sigmoid -> BCE loss.

    Architecture::

        x  (n, d_in)
        z1 = x @ W1 + b1           shape (n, d_hidden)
        a1 = tanh(z1)              shape (n, d_hidden)
        z2 = a1 @ W2 + b2          shape (n, d_out)
        a2 = sigmoid(z2)           shape (n, d_out)
        loss = BCE(a2, y) + L2 term

    Parameters are stored flattened for compatibility with first-order
    optimizers.

    Args:
        d_in: Input dimension.
        d_hidden: Hidden layer width.
        d_out: Output dimension (default 1 for binary classification).
        l2: L2-regularisation coefficient.
        seed: RNG seed for weight initialisation.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int = 1,
        l2: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.l2 = l2

        rng = np.random.default_rng(seed)
        # Xavier initialisation for tanh hidden layer.
        scale1 = np.sqrt(2.0 / (d_in + d_hidden))
        scale2 = np.sqrt(2.0 / (d_hidden + d_out))

        self.W1: NDArray = rng.standard_normal((d_in, d_hidden)) * scale1
        self.b1: NDArray = np.zeros(d_hidden)
        self.W2: NDArray = rng.standard_normal((d_hidden, d_out)) * scale2
        self.b2: NDArray = np.zeros(d_out)

        # Cache for forward/backward pass.
        self._cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Flat parameter interface (for first-order optimizers)
    # ------------------------------------------------------------------

    @property
    def _shapes(self) -> List[Tuple[Tuple[int, ...], int]]:
        """List of (shape, size) for each parameter in order W1,b1,W2,b2."""
        params = [self.W1, self.b1, self.W2, self.b2]
        return [(p.shape, p.size) for p in params]

    def get_flat_params(self) -> NDArray:
        """Return a flat copy of all parameters concatenated."""
        return np.concatenate([
            self.W1.ravel(), self.b1.ravel(),
            self.W2.ravel(), self.b2.ravel(),
        ])

    def set_flat_params(self, theta: NDArray) -> None:
        """Set parameters from a flat vector (same order as get_flat_params).

        Args:
            theta: Flat parameter vector of length (d_in*d_hidden + d_hidden +
                d_hidden*d_out + d_out).
        """
        idx = 0
        for arr, (shape, size) in zip(
            [self.W1, self.b1, self.W2, self.b2], self._shapes
        ):
            flat = theta[idx: idx + size]
            arr[:] = flat.reshape(shape)
            idx += size

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: NDArray) -> NDArray:
        """Run forward pass and cache intermediate activations.

        Args:
            X: Input matrix of shape ``(n, d_in)``.

        Returns:
            Output probabilities of shape ``(n, d_out)``.
        """
        z1 = X @ self.W1 + self.b1          # (n, d_hidden)
        a1 = _tanh(z1)                       # tanh hidden activation
        z2 = a1 @ self.W2 + self.b2          # (n, d_out)
        a2 = _sigmoid(z2)                    # sigmoid output

        # Cache for backward pass.
        self._cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return a2

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, X: NDArray, y: NDArray) -> float:
        """Binary cross-entropy loss plus optional L2 penalty.

        Args:
            X: Input of shape ``(n, d_in)``.
            y: Binary labels of shape ``(n,)`` or ``(n, 1)``.

        Returns:
            Scalar loss value.
        """
        n = X.shape[0]
        a2 = self.forward(X)                 # runs and caches forward pass
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        # BCE: -mean[ y log(p) + (1-y) log(1-p) ]; clip for numerical stability.
        eps = 1e-12
        bce = -np.mean(y * np.log(a2 + eps) + (1.0 - y) * np.log(1.0 - a2 + eps))

        # L2 regularisation on W1 and W2 only (not biases).
        l2_term = 0.5 * self.l2 * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return float(bce + l2_term)

    # ------------------------------------------------------------------
    # Manual backpropagation
    # ------------------------------------------------------------------

    def backward(self, X: NDArray, y: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Compute gradients via manual backpropagation.

        Args:
            X: Input of shape ``(n, d_in)``.
            y: Binary labels of shape ``(n,)`` or ``(n, 1)``.

        Returns:
            Tuple ``(dW1, db1, dW2, db2)`` of gradient arrays.
        """
        assert self._cache is not None, "Call loss() or forward() before backward()."
        n = X.shape[0]
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        a2 = self._cache["a2"]   # (n, d_out)
        a1 = self._cache["a1"]   # (n, d_hidden)

        # ---- Output layer ----
        # d(BCE)/d(a2) = -(y/a2 - (1-y)/(1-a2)) / n  [chain: loss -> a2]
        # d(a2)/d(z2) = a2 * (1 - a2)                [chain: a2 -> z2; sigmoid deriv]
        # Combined: dL/dz2 = (a2 - y) / n            [standard result for BCE + sigmoid]
        dz2 = (a2 - y) / n                  # (n, d_out)

        # d(z2)/d(W2) = a1^T                          [chain: z2 -> W2]
        dW2 = a1.T @ dz2                    # (d_hidden, d_out)
        dW2 += self.l2 * self.W2            # L2 gradient contribution

        # d(z2)/d(b2) = 1 (sum over batch)            [chain: z2 -> b2]
        db2 = dz2.sum(axis=0)              # (d_out,)

        # ---- Hidden layer ----
        # d(z2)/d(a1) = W2^T                          [chain: z2 -> a1]
        da1 = dz2 @ self.W2.T              # (n, d_hidden)

        # d(a1)/d(z1) = 1 - tanh^2(z1)               [chain: a1 -> z1; tanh deriv]
        dtanh = 1.0 - self._cache["a1"] ** 2
        dz1 = da1 * dtanh                  # (n, d_hidden)

        # d(z1)/d(W1) = X^T                            [chain: z1 -> W1]
        dW1 = self._cache["X"].T @ dz1    # (d_in, d_hidden)
        dW1 += self.l2 * self.W1          # L2 gradient contribution

        # d(z1)/d(b1) = 1                              [chain: z1 -> b1]
        db1 = dz1.sum(axis=0)             # (d_hidden,)

        return dW1, db1, dW2, db2

    def grad_flat(self, X: NDArray, y: NDArray) -> NDArray:
        """Return the flattened gradient vector (same order as get_flat_params).

        Also runs forward pass internally via loss().

        Args:
            X: Input matrix.
            y: Binary labels.

        Returns:
            Flat gradient NDArray.
        """
        _ = self.loss(X, y)  # populate cache
        dW1, db1, dW2, db2 = self.backward(X, y)
        return np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()])

    def predict(self, X: NDArray, threshold: float = 0.5) -> NDArray:
        """Binary predictions (0 or 1).

        Args:
            X: Input matrix.
            threshold: Decision threshold.

        Returns:
            Integer NDArray of shape ``(n,)``.
        """
        probs = self.forward(X).ravel()
        return (probs >= threshold).astype(int)

    def accuracy(self, X: NDArray, y: NDArray) -> float:
        """Fraction of correctly classified examples.

        Args:
            X: Input matrix.
            y: True binary labels.

        Returns:
            Accuracy in [0, 1].
        """
        preds = self.predict(X)
        return float(np.mean(preds == np.asarray(y, dtype=int).ravel()))


# ---------------------------------------------------------------------------
# Numerical gradient check
# ---------------------------------------------------------------------------

def numerical_gradient_check(
    model: TwoLayerMLP,
    X: NDArray,
    y: NDArray,
    eps: float = 1e-5,
) -> float:
    """Compare analytic gradient to central finite differences.

    Args:
        model: TwoLayerMLP instance.
        X: Input matrix.
        y: Binary labels.
        eps: Finite-difference step size.

    Returns:
        Maximum relative error across all parameters.
    """
    theta = model.get_flat_params().copy()
    n_params = len(theta)

    # Analytic gradient.
    analytic_grad = model.grad_flat(X, y).copy()

    # Numerical gradient via central differences.
    numeric_grad = np.zeros(n_params)
    for i in range(n_params):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        theta_minus = theta.copy()
        theta_minus[i] -= eps

        model.set_flat_params(theta_plus)
        f_plus = model.loss(X, y)
        model.set_flat_params(theta_minus)
        f_minus = model.loss(X, y)

        numeric_grad[i] = (f_plus - f_minus) / (2.0 * eps)

    # Restore original parameters.
    model.set_flat_params(theta)

    # Maximum relative error: |analytic - numeric| / (|analytic| + |numeric| + 1e-8).
    diffs = np.abs(analytic_grad - numeric_grad)
    denom = np.abs(analytic_grad) + np.abs(numeric_grad) + 1e-8
    return float(np.max(diffs / denom))


# ---------------------------------------------------------------------------
# MLP training loop using first-order optimizers
# ---------------------------------------------------------------------------

def train_mlp(
    model: TwoLayerMLP,
    X: NDArray,
    y: NDArray,
    optimizer,  # any first-order optimizer from first_order.py
    batch_size: int,
    epochs: int,
    seed: int = 0,
) -> List[float]:
    """Train the MLP using a first-order optimizer on the flattened parameters.

    Each epoch shuffles the data, processes mini-batches, and records the
    full-dataset loss after each epoch.

    Args:
        model: TwoLayerMLP instance (modified in-place).
        X: Training input of shape ``(n, d_in)``.
        y: Binary labels of shape ``(n,)``.
        optimizer: First-order optimizer (must expose ``.lr`` and ``.step``).
        batch_size: Mini-batch size.
        epochs: Number of passes over the training set.
        seed: RNG seed for mini-batch shuffling.

    Returns:
        List of per-epoch training loss values (length = epochs).
    """
    rng = np.random.default_rng(seed)
    optimizer.reset()

    n = X.shape[0]
    epoch_losses: List[float] = []

    for _ in range(epochs):
        perm = rng.permutation(n)
        X_shuf = X[perm]
        y_shuf = y[perm]

        for start in range(0, n, batch_size):
            X_b = X_shuf[start: start + batch_size]
            y_b = y_shuf[start: start + batch_size]

            # Build a gradient closure over the current batch.
            def grad_fn(theta: NDArray, _X=X_b, _y=y_b) -> NDArray:
                model.set_flat_params(theta)
                return model.grad_flat(_X, _y)

            theta = model.get_flat_params()
            theta_new = optimizer.step(theta, grad_fn)
            model.set_flat_params(theta_new)

        # Record full-dataset loss at end of epoch.
        epoch_losses.append(model.loss(X, y))

    return epoch_losses
