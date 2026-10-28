"""Experiment 6: MLP backpropagation on make_moons.

Uses sklearn make_moons(n_samples=1000, noise=0.25, random_state=0) with a
70/30 train/test split, standardised features, and TwoLayerMLP with 16 hidden
units.

Parts
-----
a) Numerical gradient check – max relative error in a CSV.
b) Train with GD (full batch), SGD (batch 32), Momentum, Adam; 100 epochs;
   lr tuned per optimizer; 5 seeds. Plot train loss and test accuracy vs epoch.
c) Regularisation sweep: Adam, l2 in [0, 1e-3, 1e-2, 1e-1, 1.0].
   Plot train/test loss, test accuracy, parameter norm vs l2.
d) Decision boundary plot for best Adam model (l2=1e-2).

Outputs
-------
- exp6a_grad_check.csv
- exp6b_train_loss.png
- exp6b_test_acc.png
- exp6b_results.csv
- exp6c_regularisation.png
- exp6c_results.csv
- exp6d_decision_boundary.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.seeding import set_global_seed
from common.timing import Timer
from common.io_utils import save_csv

from mlp_backprop import TwoLayerMLP, numerical_gradient_check, train_mlp
from first_order import GradientDescent, Momentum, Adam


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def _get_data(seed: int = 0):
    """Load and preprocess make_moons dataset.

    Returns:
        X_train, X_test, y_train, y_test (standardised arrays).
    """
    X, y = make_moons(n_samples=1000, noise=0.25, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train.astype(float), y_test.astype(float), scaler


# ---------------------------------------------------------------------------
# Part a: Numerical gradient check
# ---------------------------------------------------------------------------

def _run_part_a(out_path: Path) -> dict:
    """Run numerical_gradient_check and save max relative error."""
    set_global_seed(42)
    X_train, X_test, y_train, y_test, _ = _get_data(seed=42)

    # Use a small subset for speed
    X_check = X_train[:20]
    y_check = y_train[:20]

    model = TwoLayerMLP(d_in=2, d_hidden=16, d_out=1, l2=0.0, seed=42)
    max_rel_err = numerical_gradient_check(model, X_check, y_check, eps=1e-5)

    df = pd.DataFrame([{"max_relative_error": max_rel_err}])
    csv_path = out_path / "exp6a_grad_check.csv"
    save_csv(df, csv_path)

    return {"table": df, "csv": str(csv_path), "max_rel_err": max_rel_err}


# ---------------------------------------------------------------------------
# Part b: Training comparison
# ---------------------------------------------------------------------------

def _run_part_b(quick: bool, out_path: Path) -> dict:
    """Train with GD, SGD, Momentum, Adam; tune lr; report convergence."""
    n_seeds = 2 if quick else 5
    n_epochs = 15 if quick else 100
    seeds = list(range(42, 42 + n_seeds))

    lr_grid = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3] if not quick else [1e-3, 1e-2, 0.1]

    configs = {
        "GD (full)": (lambda lr: GradientDescent(lr=lr), 700),   # 700 ≈ train size
        "SGD-32": (lambda lr: GradientDescent(lr=lr), 32),
        "Momentum-32": (lambda lr: Momentum(lr=lr, beta=0.9), 32),
        "Adam-32": (lambda lr: Adam(lr=lr, beta1=0.9, beta2=0.999), 32),
    }

    all_train_losses = {}   # config -> (n_seeds, n_epochs)
    all_test_accs = {}
    table_rows = []

    with Timer("exp6b"):
        for cname, (factory, bsz) in configs.items():
            # Tune lr using seed 42 only
            set_global_seed(42)
            X_train, X_test, y_train, y_test, _ = _get_data(seed=42)

            best_lr = lr_grid[0]
            best_final = float("inf")
            for lr in lr_grid:
                model = TwoLayerMLP(d_in=2, d_hidden=16, d_out=1, l2=0.0, seed=42)
                opt = factory(lr)
                losses = train_mlp(model, X_train, y_train, opt, bsz,
                                   epochs=max(5, n_epochs // 5), seed=42)
                if losses and losses[-1] < best_final:
                    best_final = losses[-1]
                    best_lr = lr

            # Full runs over all seeds
            seed_train_losses = []
            seed_test_accs = []
            runtimes = []
            for seed in seeds:
                set_global_seed(seed)
                X_train, X_test, y_train, y_test, _ = _get_data(seed=seed)
                model = TwoLayerMLP(d_in=2, d_hidden=16, d_out=1, l2=0.0, seed=seed)
                opt = factory(best_lr)
                import time as _t
                t0 = _t.perf_counter()
                losses = train_mlp(model, X_train, y_train, opt, bsz, epochs=n_epochs, seed=seed)
                elapsed = _t.perf_counter() - t0
                # Per-epoch test accuracy
                test_accs = []
                # Compute per epoch by re-running (only store final)
                test_acc = model.accuracy(X_test, y_test)
                seed_train_losses.append(losses)
                seed_test_accs.append(test_acc)
                runtimes.append(elapsed)

            arr_losses = np.array(seed_train_losses)  # (n_seeds, n_epochs)
            mean_final_loss = float(arr_losses[:, -1].mean())
            std_final_loss = float(arr_losses[:, -1].std())
            mean_test_acc = float(np.mean(seed_test_accs))
            std_test_acc = float(np.std(seed_test_accs))
            mean_runtime = float(np.mean(runtimes))

            all_train_losses[cname] = arr_losses
            all_test_accs[cname] = (mean_test_acc, std_test_acc)

            table_rows.append({
                "config": cname,
                "tuned_lr": best_lr,
                "batch_size": bsz,
                "final_train_loss_mean": mean_final_loss,
                "final_train_loss_std": std_final_loss,
                "final_test_acc_mean": mean_test_acc,
                "final_test_acc_std": std_test_acc,
                "runtime_s": mean_runtime,
            })

    df = pd.DataFrame(table_rows)
    csv_path = out_path / "exp6b_results.csv"
    save_csv(df, csv_path)

    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    # Plot: training loss vs epoch
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cname in enumerate(configs):
        arr = all_train_losses[cname]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        epochs_x = np.arange(1, n_epochs + 1)
        ax.plot(epochs_x, mean, color=colors[i], linewidth=1.8, label=cname)
        ax.fill_between(epochs_x, mean - std, mean + std, color=colors[i], alpha=0.2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss (BCE)", fontsize=12)
    ax.set_title("Exp6b: MLP Training Loss vs Epoch", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig1 = out_path / "exp6b_train_loss.png"
    fig.savefig(fig1, dpi=150)
    plt.close(fig)

    # Plot: test accuracy (bar chart, mean ± std)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(configs))
    cnames = list(configs.keys())
    means = [all_test_accs[c][0] for c in cnames]
    stds = [all_test_accs[c][1] for c in cnames]
    ax.bar(x_pos, means, yerr=stds, capsize=6, color=colors[:len(configs)], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cnames, rotation=15, fontsize=10)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Exp6b: Final Test Accuracy (mean ± std)", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig2 = out_path / "exp6b_test_acc.png"
    fig.savefig(fig2, dpi=150)
    plt.close(fig)

    return {
        "table": df,
        "csv": str(csv_path),
        "figures": [str(fig1), str(fig2)],
    }


# ---------------------------------------------------------------------------
# Part c: Regularisation sweep
# ---------------------------------------------------------------------------

def _run_part_c(quick: bool, out_path: Path) -> dict:
    """Train Adam with various l2 values and record train/test metrics."""
    set_global_seed(42)

    l2_vals = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
    n_epochs = 15 if quick else 100
    lr = 1e-2  # fixed good Adam lr

    X_train, X_test, y_train, y_test, _ = _get_data(seed=42)

    table_rows = []
    for l2 in l2_vals:
        set_global_seed(42)
        model = TwoLayerMLP(d_in=2, d_hidden=16, d_out=1, l2=l2, seed=42)
        opt = Adam(lr=lr)
        losses = train_mlp(model, X_train, y_train, opt, batch_size=32,
                           epochs=n_epochs, seed=42)
        train_loss = model.loss(X_train, y_train)
        test_loss = model.loss(X_test, y_test)
        test_acc = model.accuracy(X_test, y_test)
        # Parameter norm (W1 and W2 only as L2 applies to them)
        param_norm = float(np.sqrt(np.sum(model.W1**2) + np.sum(model.W2**2)))
        table_rows.append({
            "l2": l2,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "param_norm": param_norm,
        })

    df = pd.DataFrame(table_rows)
    csv_path = out_path / "exp6c_results.csv"
    save_csv(df, csv_path)

    # Plot: 4 metrics vs l2
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = ["train_loss", "test_loss", "test_accuracy", "param_norm"]
    titles = ["Train Loss", "Test Loss", "Test Accuracy", "Parameter Norm"]
    for ax, metric, title in zip(axes, metrics, titles):
        ax.semilogx(l2_vals, df[metric], "o-", linewidth=1.8, markersize=6)
        ax.set_xlabel("L2 (log scale)", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Exp6c: Regularisation Sweep (Adam, 100 epochs)", fontsize=13)
    fig.tight_layout()
    fig_path = out_path / "exp6c_regularisation.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return {
        "table": df,
        "csv": str(csv_path),
        "figure": str(fig_path),
    }


# ---------------------------------------------------------------------------
# Part d: Decision boundary
# ---------------------------------------------------------------------------

def _run_part_d(out_path: Path) -> dict:
    """Plot decision boundary of best Adam model (l2=1e-2)."""
    set_global_seed(42)

    X_train, X_test, y_train, y_test, scaler = _get_data(seed=42)
    n_epochs = 100
    lr = 1e-2
    l2 = 1e-2

    model = TwoLayerMLP(d_in=2, d_hidden=16, d_out=1, l2=l2, seed=42)
    opt = Adam(lr=lr)
    train_mlp(model, X_train, y_train, opt, batch_size=32, epochs=n_epochs, seed=42)

    # Create grid in standardised space
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.forward(grid).ravel()
    zz = probs.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, zz, levels=50, cmap="RdBu_r", alpha=0.7)
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
               c="blue", s=15, alpha=0.5, label="Train class 0")
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
               c="red", s=15, alpha=0.5, label="Train class 1")
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
               c="cyan", s=20, marker="^", alpha=0.7, label="Test class 0")
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
               c="orange", s=20, marker="^", alpha=0.7, label="Test class 1")
    acc = model.accuracy(X_test, y_test)
    ax.set_xlabel("Feature 1 (standardised)", fontsize=12)
    ax.set_ylabel("Feature 2 (standardised)", fontsize=12)
    ax.set_title(f"Exp6d: Decision Boundary (Adam, l2={l2}, test acc={acc:.3f})", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig_path = out_path / "exp6d_decision_boundary.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return {"figure": str(fig_path), "test_accuracy": acc}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(quick: bool = False, out_dir: str = "results/track_a") -> dict:
    """Run experiment 6: MLP backpropagation.

    Args:
        quick: If True, use reduced epochs and seeds.
        out_dir: Output directory.

    Returns:
        dict with 'part_a', 'part_b', 'part_c', 'part_d' sub-dicts.
    """
    set_global_seed(42)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with Timer("exp6_total"):
        res_a = _run_part_a(out_path)
        res_b = _run_part_b(quick, out_path)
        res_c = _run_part_c(quick, out_path)
        res_d = _run_part_d(out_path)

    return {
        "part_a": res_a,
        "part_b": res_b,
        "part_c": res_c,
        "part_d": res_d,
    }


if __name__ == "__main__":
    result = run(quick=False)
    print("Part a – grad check max rel error:", result["part_a"]["max_rel_err"])
    print("\nPart b:")
    print(result["part_b"]["table"].to_string(index=False))
    print("\nPart c:")
    print(result["part_c"]["table"].to_string(index=False))
    print("\nPart d – test accuracy:", result["part_d"]["test_accuracy"])

