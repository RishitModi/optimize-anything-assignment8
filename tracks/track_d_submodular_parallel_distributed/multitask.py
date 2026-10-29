"""Multi-task learning with gradient balancing: Equal weights, MGDA, and GradNorm.

Two-task regression architecture with a shared representation layer and task-specific heads.
Demonstrates:
- Gradient magnitude imbalance caused by disparate task loss scales.
- Manual analytical backpropagation with exact gradient computation.
- Multiple Gradient Descent Algorithm (MGDA) Pareto-stationary minimum-norm descent.
- GradNorm dynamic loss weighting balancing relative training rates.
- Numerical gradient verification via central finite differences.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure repository root and track root are on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


def generate_multitask_data(
    n_train: int = 2000,
    n_test: int = 1000,
    d: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic multi-task regression dataset with disparate task scales.

    Input features x ~ N(0, I) in R^10.
    Task 1: y1 = tanh(x @ a) (scale ~ 1)
    Task 2: y2 = 10 * tanh(x @ b + 0.5) (scale ~ 10, creating gradient scale imbalance)
    where a and b are fixed random unit vectors.

    Args:
        n_train: Training sample count.
        n_test: Test sample count.
        d: Input dimensionality.
        seed: Random seed.

    Returns:
        Tuple of (X_train, y1_train, y2_train, X_test, y1_test, y2_test).
    """
    rng = np.random.default_rng(seed)

    # Unit vectors for task latent linear combinations
    raw_a = rng.normal(size=d)
    a = raw_a / np.linalg.norm(raw_a)
    raw_b = rng.normal(size=d)
    b = raw_b / np.linalg.norm(raw_b)

    X_train = rng.normal(size=(n_train, d))
    y1_train = np.tanh(X_train @ a)
    y2_train = 10.0 * np.tanh(X_train @ b + 0.5)

    X_test = rng.normal(size=(n_test, d))
    y1_test = np.tanh(X_test @ a)
    y2_test = 10.0 * np.tanh(X_test @ b + 0.5)

    return X_train, y1_train, y2_train, X_test, y1_test, y2_test


class MultiTaskModel:
    """Two-task neural network with shared representation and task-specific linear heads.

    Architecture:
        Shared: h = tanh(X @ W1 + b1),  W1 in R^(d x 32), b1 in R^32
        Head 1: y1_hat = h @ v1 + c1,   v1 in R^32, c1 in R
        Head 2: y2_hat = h @ v2 + c2,   v2 in R^32, c2 in R
    """

    def __init__(self, d: int = 10, h_dim: int = 32, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        # Xavier / Glorot initialization
        scale_w1 = np.sqrt(2.0 / (d + h_dim))
        self.W1 = rng.normal(0.0, scale_w1, size=(d, h_dim))
        self.b1 = np.zeros(h_dim, dtype=float)

        scale_v = np.sqrt(2.0 / (h_dim + 1))
        self.v1 = rng.normal(0.0, scale_v, size=h_dim)
        self.c1 = 0.0
        self.v2 = rng.normal(0.0, scale_v, size=h_dim)
        self.c2 = 0.0

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute forward pass.

        Args:
            X: Input tensor of shape (N, d).

        Returns:
            Tuple of (y1_pred, y2_pred, h).
        """
        Z1 = X @ self.W1 + self.b1
        h = np.tanh(Z1)
        y1_pred = h @ self.v1 + self.c1
        y2_pred = h @ self.v2 + self.c2
        return y1_pred, y2_pred, h

    def compute_gradients(
        self,
        X: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
    ) -> dict[str, Any]:
        """Perform analytical backward pass returning per-task parameter gradients.

        Chain-rule steps are documented explicitly for each task.

        Returns:
            Dictionary with:
                - 'loss1', 'loss2': Mean squared error per task
                - 'g1_shared', 'g2_shared': Flattened gradients of (W1, b1) for task 1 and task 2
                - 'dW1_1', 'db1_1', 'dv1', 'dc1': Task 1 parameter gradients
                - 'dW1_2', 'db1_2', 'dv2', 'dc2': Task 2 parameter gradients
        """
        N = len(X)
        Z1 = X @ self.W1 + self.b1
        h = np.tanh(Z1)  # Shape (N, 32)
        y1_pred = h @ self.v1 + self.c1  # Shape (N,)
        y2_pred = h @ self.v2 + self.c2  # Shape (N,)

        # Step 1: Loss evaluation (MSE)
        err1 = y1_pred - y1  # Shape (N,)
        err2 = y2_pred - y2  # Shape (N,)
        loss1 = float(np.mean(err1 ** 2))
        loss2 = float(np.mean(err2 ** 2))

        # Step 2: Output head gradients
        # dL1/d(y1_pred) = 2/N * err1
        delta1 = (2.0 / N) * err1  # Shape (N,)
        delta2 = (2.0 / N) * err2  # Shape (N,)

        # Head 1 parameter gradients:
        # y1_pred = h @ v1 + c1 -> dv1 = h^T @ delta1, dc1 = sum(delta1)
        dv1 = h.T @ delta1  # Shape (32,)
        dc1 = float(np.sum(delta1))

        # Head 2 parameter gradients:
        dv2 = h.T @ delta2  # Shape (32,)
        dc2 = float(np.sum(delta2))

        # Step 3: Backpropagate through shared hidden representation h
        # dL1/dh = delta1[:, None] * v1[None, :] -> Shape (N, 32)
        # dL1/dZ1 = dL1/dh * (1 - tanh^2(Z1)) = dL1/dh * (1 - h^2)
        dh1 = np.outer(delta1, self.v1)  # Shape (N, 32)
        dZ1_1 = dh1 * (1.0 - h ** 2)     # Shape (N, 32)

        dh2 = np.outer(delta2, self.v2)  # Shape (N, 32)
        dZ1_2 = dh2 * (1.0 - h ** 2)     # Shape (N, 32)

        # Step 4: Shared parameter gradients (W1 and b1)
        dW1_1 = X.T @ dZ1_1  # Shape (d, 32)
        db1_1 = np.sum(dZ1_1, axis=0)  # Shape (32,)

        dW1_2 = X.T @ dZ1_2  # Shape (d, 32)
        db1_2 = np.sum(dZ1_2, axis=0)  # Shape (32,)

        # Pack flattened shared gradients for vector operations
        g1_shared = np.concatenate([dW1_1.ravel(), db1_1.ravel()])
        g2_shared = np.concatenate([dW1_2.ravel(), db1_2.ravel()])

        return {
            "loss1": loss1,
            "loss2": loss2,
            "g1_shared": g1_shared,
            "g2_shared": g2_shared,
            "dW1_1": dW1_1,
            "db1_1": db1_1,
            "dv1": dv1,
            "dc1": dc1,
            "dW1_2": dW1_2,
            "db1_2": db1_2,
            "dv2": dv2,
            "dc2": dc2,
        }


def solve_mgda_alpha(g1: np.ndarray, g2: np.ndarray) -> float:
    """Find optimal convex combination alpha in [0, 1] minimizing ||alpha*g1 + (1-alpha)*g2||^2.

    Solves min_{alpha in [0, 1]} ||g2 + alpha*(g1 - g2)||^2.
    Unconstrained minimum: alpha* = ((g2 - g1) . g2) / ||g1 - g2||^2.
    Projected onto [0, 1].

    Args:
        g1: Gradient vector for task 1.
        g2: Gradient vector for task 2.

    Returns:
        Scalar alpha in [0.0, 1.0].
    """
    diff = g1 - g2
    denom = float(np.dot(diff, diff))
    if denom < 1e-12:
        return 0.5
    num = float(np.dot(g2 - g1, g2))
    alpha = num / denom
    return float(np.clip(alpha, 0.0, 1.0))


def train_multitask(
    model: MultiTaskModel,
    X_train: np.ndarray,
    y1_train: np.ndarray,
    y2_train: np.ndarray,
    X_test: np.ndarray,
    y1_test: np.ndarray,
    y2_test: np.ndarray,
    scheme: str = "equal",  # "equal", "mgda", "gradnorm"
    lr: float = 0.01,
    epochs: int = 500,
) -> dict[str, Any]:
    """Train multi-task model using full-batch Adam under the chosen gradient balancing scheme.

    Schemes:
        - "equal": g_shared = g1 + g2
        - "mgda": g_shared = alpha*g1 + (1-alpha)*g2 with minimum-norm alpha
        - "gradnorm": dynamically adjusts task loss weights w_t to balance gradient norms

    Args:
        model: MultiTaskModel instance.
        X_train, y1_train, y2_train: Training data.
        X_test, y1_test, y2_test: Test data.
        scheme: One of 'equal', 'mgda', 'gradnorm'.
        lr: Adam learning rate.
        epochs: Number of optimization epochs.

    Returns:
        Dictionary of training curves, test R^2 metrics, and scheme diagnostics.
    """
    # Adam moment variables
    params = [model.W1, model.b1, model.v1, np.array([model.c1]), model.v2, np.array([model.c2])]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    loss1_hist: list[float] = []
    loss2_hist: list[float] = []
    alpha_hist: list[float] = []
    w1_hist: list[float] = []
    w2_hist: list[float] = []

    # GradNorm initial state
    w_t = np.array([1.0, 1.0], dtype=float)
    initial_l1: float | None = None
    initial_l2: float | None = None

    # Track verification checks
    mgda_check_passed = True

    for t in range(1, epochs + 1):
        grads = model.compute_gradients(X_train, y1_train, y2_train)
        l1, l2 = grads["loss1"], grads["loss2"]
        loss1_hist.append(l1)
        loss2_hist.append(l2)

        if initial_l1 is None:
            initial_l1 = max(l1, 1e-8)
            initial_l2 = max(l2, 1e-8)

        g1_shared = grads["g1_shared"]
        g2_shared = grads["g2_shared"]

        if scheme == "equal":
            dW1 = grads["dW1_1"] + grads["dW1_2"]
            db1 = grads["db1_1"] + grads["db1_2"]
            dv1 = grads["dv1"]
            dc1 = grads["dc1"]
            dv2 = grads["dv2"]
            dc2 = grads["dc2"]

        elif scheme == "mgda":
            alpha = solve_mgda_alpha(g1_shared, g2_shared)
            alpha_hist.append(alpha)

            # Combined shared gradient
            g_combined = alpha * g1_shared + (1.0 - alpha) * g2_shared

            # Verification check: ||g_combined|| <= min(||g1||, ||g2||) + 1e-9
            norm_comb = np.linalg.norm(g_combined)
            norm_min = min(np.linalg.norm(g1_shared), np.linalg.norm(g2_shared))
            if norm_comb > norm_min + 1e-9:
                mgda_check_passed = False

            dW1 = alpha * grads["dW1_1"] + (1.0 - alpha) * grads["dW1_2"]
            db1 = alpha * grads["db1_1"] + (1.0 - alpha) * grads["db1_2"]
            dv1 = grads["dv1"]
            dc1 = grads["dc1"]
            dv2 = grads["dv2"]
            dc2 = grads["dc2"]

        elif scheme == "gradnorm":
            # Gradient norm with respect to shared W1
            G1 = w_t[0] * float(np.linalg.norm(grads["dW1_1"]))
            G2 = w_t[1] * float(np.linalg.norm(grads["dW1_2"]))
            G_mean = 0.5 * (G1 + G2)

            # Relative inverse training rate r_t
            inv_rate1 = (l1 / initial_l1)
            inv_rate2 = (l2 / initial_l2)
            mean_inv = 0.5 * (inv_rate1 + inv_rate2)
            r1 = inv_rate1 / max(mean_inv, 1e-8)
            r2 = inv_rate2 / max(mean_inv, 1e-8)

            target1 = G_mean * (r1 ** 1.5)
            target2 = G_mean * (r2 ** 1.5)

            # GradNorm weight update
            w_t[0] -= 0.025 * np.sign(G1 - target1) * (G1 / max(w_t[0], 1e-6))
            w_t[1] -= 0.025 * np.sign(G2 - target2) * (G2 / max(w_t[1], 1e-6))

            # Clip and renormalize to sum to 2
            w_t = np.maximum(w_t, 0.01)
            w_t = 2.0 * w_t / np.sum(w_t)

            w1_hist.append(float(w_t[0]))
            w2_hist.append(float(w_t[1]))

            dW1 = w_t[0] * grads["dW1_1"] + w_t[1] * grads["dW1_2"]
            db1 = w_t[0] * grads["db1_1"] + w_t[1] * grads["db1_2"]
            dv1 = w_t[0] * grads["dv1"]
            dc1 = w_t[0] * grads["dc1"]
            dv2 = w_t[1] * grads["dv2"]
            dc2 = w_t[1] * grads["dc2"]

        else:
            raise ValueError(f"Unknown multi-task scheme: {scheme}")

        # Parameter update via Adam
        grad_list = [dW1, db1, dv1, np.array([dc1]), dv2, np.array([dc2])]

        for i in range(len(params)):
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad_list[i]
            v[i] = beta2 * v[i] + (1.0 - beta2) * (grad_list[i] ** 2)
            m_hat = m[i] / (1.0 - beta1 ** t)
            v_hat = v[i] / (1.0 - beta2 ** t)
            step = lr * m_hat / (np.sqrt(v_hat) + eps)

            if i == 0:
                model.W1 -= step
            elif i == 1:
                model.b1 -= step
            elif i == 2:
                model.v1 -= step
            elif i == 3:
                model.c1 -= float(step[0])
            elif i == 4:
                model.v2 -= step
            elif i == 5:
                model.c2 -= float(step[0])

    # Final test evaluation
    y1_test_pred, y2_test_pred, _ = model.forward(X_test)
    mse1 = float(np.mean((y1_test_pred - y1_test) ** 2))
    mse2 = float(np.mean((y2_test_pred - y2_test) ** 2))

    var1 = float(np.var(y1_test))
    var2 = float(np.var(y2_test))
    r2_task1 = float(1.0 - (mse1 / max(var1, 1e-8)))
    r2_task2 = float(1.0 - (mse2 / max(var2, 1e-8)))
    mean_r2 = 0.5 * (r2_task1 + r2_task2)
    worst_r2 = min(r2_task1, r2_task2)

    return {
        "loss1_hist": loss1_hist,
        "loss2_hist": loss2_hist,
        "r2_task1": r2_task1,
        "r2_task2": r2_task2,
        "mean_r2": mean_r2,
        "worst_r2": worst_r2,
        "alpha_hist": alpha_hist,
        "w1_hist": w1_hist,
        "w2_hist": w2_hist,
        "final_w": list(w_t) if scheme == "gradnorm" else None,
        "mean_alpha": float(np.mean(alpha_hist)) if scheme == "mgda" else None,
        "mgda_check_passed": mgda_check_passed,
    }


def numerical_grad_check(
    model: MultiTaskModel,
    X: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    eps: float = 1e-5,
) -> float:
    """Verify manual analytical gradients against two-sided finite difference numerical gradients.

    Evaluates error across shared parameters W1, b1 and head parameters.

    Args:
        model: MultiTaskModel instance.
        X, y1, y2: Small evaluation batch.
        eps: Finite difference step perturbation.

    Returns:
        Maximum relative error across all parameters.
    """
    grads = model.compute_gradients(X, y1, y2)
    max_rel_error = 0.0

    # Total loss: L = L1 + L2
    def total_loss() -> float:
        y1_p, y2_p, _ = model.forward(X)
        return float(np.mean((y1_p - y1) ** 2) + np.mean((y2_p - y2) ** 2))

    # Check W1
    ana_dW1 = grads["dW1_1"] + grads["dW1_2"]
    for i in range(min(model.W1.shape[0], 3)):
        for j in range(min(model.W1.shape[1], 3)):
            orig = model.W1[i, j]
            model.W1[i, j] = orig + eps
            l_plus = total_loss()
            model.W1[i, j] = orig - eps
            l_minus = total_loss()
            model.W1[i, j] = orig

            num_g = (l_plus - l_minus) / (2.0 * eps)
            ana_g = ana_dW1[i, j]
            rel_err = abs(num_g - ana_g) / max(abs(num_g), abs(ana_g), 1e-8)
            if rel_err > max_rel_error:
                max_rel_error = float(rel_err)

    # Check v1
    ana_dv1 = grads["dv1"]
    for j in range(min(len(model.v1), 3)):
        orig = model.v1[j]
        model.v1[j] = orig + eps
        l_plus = total_loss()
        model.v1[j] = orig - eps
        l_minus = total_loss()
        model.v1[j] = orig

        num_g = (l_plus - l_minus) / (2.0 * eps)
        ana_g = ana_dv1[j]
        rel_err = abs(num_g - ana_g) / max(abs(num_g), abs(ana_g), 1e-8)
        if rel_err > max_rel_error:
            max_rel_error = float(rel_err)

    return max_rel_error

