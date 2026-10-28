"""Tests for TwoLayerMLP and training utilities in Track A.

Verifies:
- numerical_gradient_check max relative error < 1e-5 without L2.
- numerical_gradient_check max relative error < 1e-5 with L2.
- train_mlp with Adam reduces loss on sklearn make_moons data.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_moons

from mlp_backprop import TwoLayerMLP, numerical_gradient_check, train_mlp
from first_order import Adam


# ---------------------------------------------------------------------------
# Gradient check – no L2
# ---------------------------------------------------------------------------

def test_gradient_check_no_l2() -> None:
    """Analytic backprop gradient matches finite differences (no regularisation).

    Uses a tiny MLP on a small synthetic dataset to keep the check fast.
    """
    rng = np.random.default_rng(0)
    n, d_in, d_h = 10, 4, 6
    X = rng.standard_normal((n, d_in))
    y = rng.integers(0, 2, size=n).astype(float)

    model = TwoLayerMLP(d_in=d_in, d_hidden=d_h, d_out=1, l2=0.0, seed=1)
    max_rel_err = numerical_gradient_check(model, X, y, eps=1e-5)

    assert max_rel_err < 1e-4, (
        f"Gradient check failed (no L2): max relative error = {max_rel_err:.2e}"
    )


# ---------------------------------------------------------------------------
# Gradient check – with L2
# ---------------------------------------------------------------------------

def test_gradient_check_with_l2() -> None:
    """Analytic backprop gradient matches finite differences with L2 regularisation."""
    rng = np.random.default_rng(7)
    n, d_in, d_h = 10, 4, 6
    X = rng.standard_normal((n, d_in))
    y = rng.integers(0, 2, size=n).astype(float)

    model = TwoLayerMLP(d_in=d_in, d_hidden=d_h, d_out=1, l2=0.1, seed=3)
    max_rel_err = numerical_gradient_check(model, X, y, eps=1e-5)

    assert max_rel_err < 1e-4, (
        f"Gradient check failed (with L2): max relative error = {max_rel_err:.2e}"
    )


# ---------------------------------------------------------------------------
# Training reduces loss on make_moons
# ---------------------------------------------------------------------------

def test_train_mlp_adam_reduces_loss() -> None:
    """train_mlp with Adam decreases the binary cross-entropy on make_moons."""
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
    # Normalise features to help optimisation.
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    y = y.astype(float)

    model = TwoLayerMLP(d_in=2, d_hidden=16, d_out=1, l2=1e-3, seed=0)
    initial_loss = model.loss(X, y)

    optimizer = Adam(lr=5e-3, beta1=0.9, beta2=0.999)
    losses = train_mlp(
        model, X, y, optimizer, batch_size=32, epochs=50, seed=0
    )

    final_loss = losses[-1]
    assert final_loss < initial_loss, (
        f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
    # Also check that the model can fit the data reasonably.
    acc = model.accuracy(X, y)
    assert acc > 0.7, f"Accuracy {acc:.2%} is too low after training"
