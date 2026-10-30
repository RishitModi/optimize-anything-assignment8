"""Quantum Machine Learning (QML) algorithms and classical baselines for Track I.

Features:
---------
1. Synthetic classification datasets (moons, circles) with standardized scaling and 50/50 split.
2. 2-qubit Variational Quantum Classifier (VQC) with data re-uploading and 12 trainable parameters.
3. Exact gradient calculation via the analytical Parameter-Shift Rule.
4. Adam optimizer for variational quantum circuit training.
5. 2-qubit Quantum Kernel Support Vector Classifier (exact and shot-based).
6. Classical ML baselines: Logistic Regression, RBF Support Vector Classifier, and MLP Classifier.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root and track root are on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qi_sim import (
    H,
    apply_1q,
    apply_cnot,
    expectation_z,
    rx,
    ry,
    rz,
    zero_state,
)


def make_dataset(
    name: str,
    n: int = 240,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate and preprocess 2D classification dataset with a fixed stratified 50/50 split.

    Datasets:
      - 'moons': scikit-learn make_moons with noise=0.2.
      - 'circles': scikit-learn make_circles with noise=0.1, factor=0.5.

    Preprocessing:
      1. Standardized with StandardScaler.
      2. Scaled into [-1.5, 1.5].
      3. Stratified 50/50 split into train/test using random_state=0.
      4. Binary labels converted to {-1, +1}.

    Args:
        name: Name of dataset ('moons' or 'circles').
        n: Total number of samples.
        seed: Random seed for generating dataset points.

    Returns:
        tuple (X_train, X_test, y_train, y_test).
    """
    if name == "moons":
        X, y = make_moons(n_samples=n, noise=0.2, random_state=seed)
    elif name == "circles":
        X, y = make_circles(n_samples=n, noise=0.1, factor=0.5, random_state=seed)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    # Standardize and scale to [-1.5, 1.5]
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    max_val = np.max(np.abs(X_std))
    if max_val > 0.0:
        X_scaled = 1.5 * (X_std / max_val)
    else:
        X_scaled = X_std

    # Stratified 50/50 train/test split with fixed seed 0
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.5,
        stratify=y,
        random_state=0,
    )

    # Convert {0, 1} to {-1, +1}
    y_train = np.where(y_train == 0, -1.0, 1.0)
    y_test = np.where(y_test == 0, -1.0, 1.0)

    return X_train, X_test, y_train, y_test


def variational_forward(params: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Evaluate batched forward pass for the 2-qubit data re-uploading variational classifier.

    Architecture (L = 3 layers, P = 12 parameters):
      For layer l in 0..2:
        1. Data re-uploading: RX(x_0) on qubit 0, RX(x_1) on qubit 1 (batched U per sample).
        2. Trainable rotations: RY(theta_{l, 0, 0}) then RZ(theta_{l, 0, 1}) on qubit 0;
                               RY(theta_{l, 1, 0}) then RZ(theta_{l, 1, 1}) on qubit 1.
        3. Entangler: CNOT(0, 1).

    Output:
      Observable expectation value f(x) = <Z_0> in [-1, 1].

    Args:
        params: 1D array of 12 real parameters.
        X: Input features of shape (B, 2).

    Returns:
        1D array of model predictions f(x) of shape (B,).
    """
    B = X.shape[0]
    p = params.reshape((3, 2, 2))  # (layer, qubit, [ry, rz])

    x0 = X[:, 0]
    x1 = X[:, 1]
    u_x0 = rx(x0)  # shape (B, 2, 2)
    u_x1 = rx(x1)  # shape (B, 2, 2)

    state = zero_state(2, batch_size=B)

    for l in range(3):
        # 1. Data re-uploading
        state = apply_1q(state, u_x0, 0, 2)
        state = apply_1q(state, u_x1, 1, 2)

        # 2. Trainable rotations
        state = apply_1q(state, ry(p[l, 0, 0]), 0, 2)
        state = apply_1q(state, rz(p[l, 0, 1]), 0, 2)
        state = apply_1q(state, ry(p[l, 1, 0]), 1, 2)
        state = apply_1q(state, rz(p[l, 1, 1]), 1, 2)

        # 3. Entangling CNOT
        state = apply_cnot(state, 0, 1, 2)

    f = expectation_z(state, 0, 2)
    return np.asarray(f, dtype=np.float64)


def param_shift_grad(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute exact analytical loss gradient with respect to params via Parameter-Shift Rule.

    Loss function: L(theta) = (1/B) sum_i (f(x_i; theta) - y_i)^2
    Derivative: d L / d theta_j = (2/B) sum_i (f(x_i) - y_i) * (d f(x_i) / d theta_j)
    where: d f(x_i) / d theta_j = 0.5 * (f(theta + pi/2 e_j) - f(theta - pi/2 e_j)).

    Args:
        params: Parameter vector of length 12.
        X: Batch of input points of shape (B, 2).
        y: Target labels {-1, +1} of shape (B,).

    Returns:
        Gradient vector of length 12.
    """
    P = len(params)
    f0 = variational_forward(params, X)
    diff = 2.0 * (f0 - y)  # shape (B,)
    grad = np.empty(P, dtype=np.float64)
    shift = np.pi / 2.0

    for j in range(P):
        p_plus = params.copy()
        p_plus[j] += shift
        f_plus = variational_forward(p_plus, X)

        p_minus = params.copy()
        p_minus[j] -= shift
        f_minus = variational_forward(p_minus, X)

        df = 0.5 * (f_plus - f_minus)
        grad[j] = float(np.mean(diff * df))

    return grad


def train_classifier(
    dataset_name: str,
    seed: int,
    epochs: int = 30,
    batch_size: int = 20,
    lr: float = 0.1,
) -> dict[str, Any]:
    """Train the 2-qubit variational classifier using mini-batch Adam optimization.

    Args:
        dataset_name: 'moons' or 'circles'.
        seed: Random seed for parameter initialization and mini-batch shuffling.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for Adam.

    Returns:
        Dictionary containing:
          - loss_history: list of MSE loss values per optimization step
          - train_accuracy: float accuracy on training set
          - test_accuracy: float accuracy on test set
          - wall_time_s: training duration in seconds
          - parameter_count: 12
          - circuit_evaluations: total quantum circuit executions
    """
    X_train, X_test, y_train, y_test = make_dataset(dataset_name, n=240, seed=0)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    rng = np.random.default_rng(seed)
    # Initialize 12 parameters uniformly in [-0.2, 0.2]
    params = rng.uniform(-0.2, 0.2, size=12)

    # Adam state
    m = np.zeros(12, dtype=np.float64)
    v = np.zeros(12, dtype=np.float64)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    loss_history: list[float] = []
    step_count = 0

    t0 = time.perf_counter()

    for epoch in range(epochs):
        indices = rng.permutation(n_train)
        for start_idx in range(0, n_train, batch_size):
            end_idx = min(start_idx + batch_size, n_train)
            b_idx = indices[start_idx:end_idx]
            X_b = X_train[b_idx]
            y_b = y_train[b_idx]

            # Forward pass & loss
            f_b = variational_forward(params, X_b)
            loss_val = float(np.mean((f_b - y_b) ** 2))
            loss_history.append(loss_val)

            # Parameter shift gradient
            grad = param_shift_grad(params, X_b, y_b)

            # Adam update
            step_count += 1
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad ** 2)
            m_hat = m / (1.0 - beta1 ** step_count)
            v_hat = v / (1.0 - beta2 ** step_count)
            params -= lr * m_hat / (np.sqrt(v_hat) + eps)

    wall_time = time.perf_counter() - t0

    # Predictions: sign(f(x)) with sign(0) = +1
    f_train = variational_forward(params, X_train)
    pred_train = np.where(f_train >= 0.0, 1.0, -1.0)
    train_acc = float(np.mean(pred_train == y_train))

    f_test = variational_forward(params, X_test)
    pred_test = np.where(f_test >= 0.0, 1.0, -1.0)
    test_acc = float(np.mean(pred_test == y_test))

    # Circuit evaluations count:
    # Each parameter shift step on a batch of size B evaluates:
    # 1 forward pass + 2 * P shifted forward passes = (2 * P + 1) * B circuits.
    # Prediction evaluates n_train + n_test circuits.
    P = 12
    # Count circuits during training:
    n_batches = math.ceil(n_train / float(batch_size))
    total_training_circuits = epochs * n_train * (2 * P + 1)
    total_circuits = total_training_circuits + n_train + n_test

    return {
        "loss_history": loss_history,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "wall_time_s": wall_time,
        "parameter_count": P,
        "circuit_evaluations": total_circuits,
        "final_params": params,
    }


# =============================================================================
# Quantum Kernel Methods
# =============================================================================

def compute_quantum_feature_states(X: np.ndarray) -> np.ndarray:
    """Compute statevectors |phi(x)> under the 2-qubit repeated quantum feature map.

    Feature map on 2 qubits repeated twice:
      1. H on both qubits.
      2. RZ(x_0) on qubit 0, RZ(x_1) on qubit 1.
      3. CNOT(0, 1).
      4. RZ((pi - x_0) * (pi - x_1)) on qubit 1.
      5. CNOT(0, 1).

    Args:
        X: Input data points of shape (N, 2).

    Returns:
        Array of statevectors of shape (N, 4).
    """
    N = X.shape[0]
    x0 = X[:, 0]
    x1 = X[:, 1]
    phi_prod = (np.pi - x0) * (np.pi - x1)

    state = zero_state(2, batch_size=N)

    for _ in range(2):
        state = apply_1q(state, H, 0, 2)
        state = apply_1q(state, H, 1, 2)
        state = apply_1q(state, rz(x0), 0, 2)
        state = apply_1q(state, rz(x1), 1, 2)
        state = apply_cnot(state, 0, 1, 2)
        state = apply_1q(state, rz(phi_prod), 1, 2)
        state = apply_cnot(state, 0, 1, 2)

    return state


def compute_kernel_matrix(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Compute exact quantum kernel Gram matrix K[i, j] = |<phi(x_1_i) | phi(x_2_j)>|^2.

    Args:
        X1: First set of data points of shape (N1, 2).
        X2: Second set of data points of shape (N2, 2).

    Returns:
        Kernel matrix of shape (N1, N2).
    """
    psi1 = compute_quantum_feature_states(X1)  # (N1, 4)
    psi2 = compute_quantum_feature_states(X2)  # (N2, 4)

    # Inner products <psi1_i | psi2_j> = sum_k psi1_ik^* psi2_jk
    inner = psi1 @ psi2.conj().T  # shape (N1, N2)
    K = np.abs(inner) ** 2

    # Symmetrize if self-kernel
    if X1 is X2 or (X1.shape == X2.shape and np.allclose(X1, X2)):
        K = 0.5 * (K + K.T)
        np.fill_diagonal(K, 1.0)

    return np.asarray(K, dtype=np.float64)


def evaluate_quantum_kernel(
    dataset_name: str,
    shot_based: bool = False,
    shots: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate Support Vector Classifier with exact or shot-based quantum kernel.

    Args:
        dataset_name: 'moons' or 'circles'.
        shot_based: If True, simulates finite-shot sampling for Gram matrix entries.
        shots: Number of shots per entry (default: 1000).
        seed: Random seed for finite-shot binomial noise.

    Returns:
        Dictionary of accuracy, wall time, and evaluation count.
    """
    X_train, X_test, y_train, y_test = make_dataset(dataset_name, n=240, seed=0)
    n_train = len(X_train)
    n_test = len(X_test)

    t0 = time.perf_counter()
    K_train = compute_kernel_matrix(X_train, X_train)
    K_test = compute_kernel_matrix(X_test, X_train)

    if shot_based:
        rng = np.random.default_rng(seed)
        # Sample binomial counts for each off-diagonal entry
        # Clip probabilities to [0, 1]
        p_train = np.clip(K_train, 0.0, 1.0)
        K_train = rng.binomial(shots, p_train) / float(shots)
        K_train = 0.5 * (K_train + K_train.T)
        np.fill_diagonal(K_train, 1.0)

        p_test = np.clip(K_test, 0.0, 1.0)
        K_test = rng.binomial(shots, p_test) / float(shots)

    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(K_train, y_train)
    wall_time = time.perf_counter() - t0

    pred_train = clf.predict(K_train)
    pred_test = clf.predict(K_test)

    train_acc = float(np.mean(pred_train == y_train))
    test_acc = float(np.mean(pred_test == y_test))

    # Circuit evaluations on physical hardware:
    # Training evaluates pairs (i < j): n_train * (n_train - 1) / 2
    # Prediction evaluates test-train pairs: n_test * n_train
    circuits_train = n_train * (n_train - 1) // 2
    circuits_pred = n_test * n_train

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "wall_time_s": wall_time,
        "parameter_count": len(clf.support_),  # number of dual support vectors
        "circuit_evaluations_train": circuits_train,
        "circuit_evaluations_pred": circuits_pred,
        "n_qubits": 2,
    }


def evaluate_classical_baselines(dataset_name: str, seed: int = 42) -> dict[str, dict[str, Any]]:
    """Train and evaluate classical ML baselines on the exact same dataset split.

    Models:
      1. Logistic Regression (Linear baseline, parameters = n_features + 1 = 3).
      2. RBF Support Vector Machine (C=1.0).
      3. Multi-Layer Perceptron (1 hidden layer of 4 neurons, max_iter=2000).

    Args:
        dataset_name: 'moons' or 'circles'.
        seed: Random seed for initialization and solver determinism.

    Returns:
        Dictionary mapping model names to evaluation results.
    """
    X_train, X_test, y_train, y_test = make_dataset(dataset_name, n=240, seed=0)

    results: dict[str, dict[str, Any]] = {}

    # 1. Logistic Regression
    t0 = time.perf_counter()
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_train, y_train)
    t_lr = time.perf_counter() - t0
    results["logistic_regression"] = {
        "train_accuracy": float(np.mean(lr.predict(X_train) == y_train)),
        "test_accuracy": float(np.mean(lr.predict(X_test) == y_test)),
        "wall_time_s": t_lr,
        "parameter_count": int(lr.coef_.size + lr.intercept_.size),  # 3
        "circuit_evaluations": 0,
        "n_qubits": 0,
    }

    # 2. RBF SVM
    t0 = time.perf_counter()
    rbf = SVC(kernel="rbf", C=1.0, random_state=seed)
    rbf.fit(X_train, y_train)
    t_rbf = time.perf_counter() - t0
    results["svm_rbf"] = {
        "train_accuracy": float(np.mean(rbf.predict(X_train) == y_train)),
        "test_accuracy": float(np.mean(rbf.predict(X_test) == y_test)),
        "wall_time_s": t_rbf,
        "parameter_count": int(len(rbf.support_)),  # support vector count
        "circuit_evaluations": 0,
        "n_qubits": 0,
    }

    # 3. MLP Classifier (hidden_layer_sizes=(4,))
    t0 = time.perf_counter()
    mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000, random_state=seed)
    mlp.fit(X_train, y_train)
    t_mlp = time.perf_counter() - t0
    mlp_params = int(sum(c.size for c in mlp.coefs_) + sum(i.size for i in mlp.intercepts_))
    results["mlp_hidden4"] = {
        "train_accuracy": float(np.mean(mlp.predict(X_train) == y_train)),
        "test_accuracy": float(np.mean(mlp.predict(X_test) == y_test)),
        "wall_time_s": t_mlp,
        "parameter_count": mlp_params,  # 2*4 + 4 + 4*1 + 1 = 17
        "circuit_evaluations": 0,
        "n_qubits": 0,
    }

    return results

