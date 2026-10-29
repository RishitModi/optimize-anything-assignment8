"""Federated Learning: FedAvg simulation, data partitioning, and centralized baseline.

Features:
- Standardized binary classification dataset generation (sklearn make_classification).
- IID and non-IID (Dirichlet label skew) dataset partitioning across K clients.
- Federated Averaging (FedAvg) with local mini-batch SGD and sample-weighted server aggregation.
- Centralized mini-batch SGD baseline matching effective sample throughput.
- Communication tracking (floats transferred per round: 2 * K * d).
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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_federated_dataset(
    n_samples: int = 6000,
    n_features: int = 20,
    n_informative: int = 10,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate and standardize binary classification dataset split into 80/20 train/test.

    Args:
        n_samples: Total number of samples. Defaults to 6000.
        n_features: Feature dimension d. Defaults to 20.
        n_informative: Informative features. Defaults to 10.
        random_state: Seed for dataset generation. Defaults to 0.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_classes=2,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train.astype(float), X_test, y_test.astype(float)


def logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, l2_reg: float = 1e-3) -> float:
    """Compute binary cross-entropy loss with L2 regularization.

    Loss: -1/N sum [y log(sigma(Xw)) + (1-y) log(1 - sigma(Xw))] + 0.5 * l2_reg * ||w||^2
    """
    z = X @ w
    # Stable softplus formulation: log(1 + exp(z)) - y*z
    # log1pexp(z) = log(1 + exp(-|z|)) + max(z, 0)
    loss_i = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0) - y * z
    return float(np.mean(loss_i) + 0.5 * l2_reg * np.dot(w, w))


def logistic_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray, l2_reg: float = 1e-3) -> np.ndarray:
    """Compute analytical gradient of logistic loss with L2 regularization."""
    z = X @ w
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    return (X.T @ (p - y)) / len(y) + l2_reg * w


def evaluate_accuracy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Compute binary classification accuracy."""
    preds = (X @ w >= 0.0).astype(float)
    return float(np.mean(preds == y))


def partition_iid(n: int, K: int, seed: int = 42) -> list[np.ndarray]:
    """Partition n sample indices into K equal IID shards via random permutation.

    Args:
        n: Total training sample count.
        K: Number of clients.
        seed: Random seed.

    Returns:
        List of K index arrays.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n)
    return list(np.array_split(shuffled, K))


def partition_label_skew(
    y: np.ndarray,
    K: int,
    alpha: float = 0.3,
    seed: int = 42,
) -> list[np.ndarray]:
    """Partition dataset across K clients with non-IID label skew via Dirichlet distribution.

    For each class c, client proportions are sampled from Dirichlet(alpha * 1_K).
    Smaller alpha creates more extreme non-IID skew.

    Args:
        y: Target label array (shape (n,)).
        K: Number of clients.
        alpha: Dirichlet concentration parameter.
        seed: Random seed.

    Returns:
        List of K index arrays.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    client_indices: list[list[int]] = [[] for _ in range(K)]

    for c in classes:
        c_indices = np.where(y == c)[0]
        rng.shuffle(c_indices)
        proportions = rng.dirichlet(np.repeat(alpha, K))
        # Ensure all samples are assigned by cumsum counts
        counts = np.round(proportions * len(c_indices)).astype(int)
        # Fix rounding mismatches
        diff = len(c_indices) - np.sum(counts)
        counts[rng.integers(0, K)] += diff

        splits = np.split(c_indices, np.cumsum(counts)[:-1])
        for k in range(K):
            client_indices[k].extend(splits[k])

    return [np.array(idx, dtype=int) for idx in client_indices]


def fedavg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    partitions: list[np.ndarray],
    rounds: int = 100,
    local_steps: int = 5,
    lr: float = 0.1,
    batch_size: int = 32,
    l2_reg: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute Federated Averaging (FedAvg) over K clients.

    Every round, each client initializes its local model with the global model,
    performs local_steps mini-batch SGD updates, and sends the updated weights to the server.
    The server aggregates weights using sample-count weighting:
        w_{global} = sum_{k=1}^K (n_k / n) w_k

    Args:
        X_train, y_train: Training features and targets.
        X_test, y_test: Evaluation features and targets.
        partitions: List of sample index arrays, one per client.
        rounds: Number of communication rounds.
        local_steps: Number of local mini-batch SGD steps per client per round.
        lr: Learning rate.
        batch_size: Mini-batch size for local SGD.
        l2_reg: L2 regularization coefficient.
        seed: Random seed.

    Returns:
        Dictionary containing:
            - 'accuracies': test accuracy per round (length rounds)
            - 'losses': test loss per round (length rounds)
            - 'communicated_floats': cumulative floats transferred per round
            - 'final_weights': model weights at end of training
    """
    rng = np.random.default_rng(seed)
    K = len(partitions)
    d = X_train.shape[1]
    w_global = np.zeros(d, dtype=float)

    n_total = sum(len(p) for p in partitions)
    client_weights = [len(p) / n_total for p in partitions]

    test_accs: list[float] = []
    test_losses: list[float] = []
    comm_floats: list[int] = []

    # Per round communication: server broadcasts w_global (K*d), clients send w_k (K*d)
    floats_per_round = 2 * K * d

    for r in range(1, rounds + 1):
        local_models: list[np.ndarray] = []

        for k in range(K):
            idx_k = partitions[k]
            n_k = len(idx_k)
            X_k = X_train[idx_k]
            y_k = y_train[idx_k]

            w_local = w_global.copy()

            for _ in range(local_steps):
                # Sample mini-batch
                b_size = min(batch_size, n_k)
                batch_idx = rng.choice(n_k, size=b_size, replace=False)
                X_b = X_k[batch_idx]
                y_b = y_k[batch_idx]

                grad = logistic_grad(w_local, X_b, y_b, l2_reg=l2_reg)
                w_local -= lr * grad

            local_models.append(w_local)

        # Server sample-weighted aggregation
        w_global = np.zeros(d, dtype=float)
        for k in range(K):
            w_global += client_weights[k] * local_models[k]

        acc = evaluate_accuracy(w_global, X_test, y_test)
        loss = logistic_loss(w_global, X_test, y_test, l2_reg=l2_reg)

        test_accs.append(acc)
        test_losses.append(loss)
        comm_floats.append(r * floats_per_round)

    return {
        "accuracies": test_accs,
        "losses": test_losses,
        "communicated_floats": comm_floats,
        "final_weights": w_global,
    }


def centralized_sgd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int = 100,
    local_steps: int = 5,
    lr: float = 0.1,
    batch_size: int = 320,  # Matches K * 32
    l2_reg: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute centralized mini-batch SGD baseline matching FedAvg effective sample throughput.

    Takes local_steps mini-batch steps of size batch_size per 'round'.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        rounds: Number of evaluation rounds.
        local_steps: Steps per round.
        lr: Learning rate.
        batch_size: Batch size per step (typically K * local_batch_size).
        l2_reg: L2 regularization parameter.
        seed: Random seed.

    Returns:
        Dictionary with 'accuracies', 'losses', and 'final_weights'.
    """
    rng = np.random.default_rng(seed)
    n = len(y_train)
    d = X_train.shape[1]
    w = np.zeros(d, dtype=float)

    test_accs: list[float] = []
    test_losses: list[float] = []

    for _ in range(rounds):
        for _ in range(local_steps):
            b_size = min(batch_size, n)
            batch_idx = rng.choice(n, size=b_size, replace=False)
            X_b = X_train[batch_idx]
            y_b = y_train[batch_idx]

            grad = logistic_grad(w, X_b, y_b, l2_reg=l2_reg)
            w -= lr * grad

        acc = evaluate_accuracy(w, X_test, y_test)
        loss = logistic_loss(w, X_test, y_test, l2_reg=l2_reg)
        test_accs.append(acc)
        test_losses.append(loss)

    return {
        "accuracies": test_accs,
        "losses": test_losses,
        "final_weights": w,
    }

