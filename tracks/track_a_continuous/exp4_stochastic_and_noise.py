"""Experiment 4: Stochastic optimization and noisy objectives.

Part a: Mini-batch study on logistic regression (n=2000, d=20, l2=1e-2).
  Configurations: full-batch GD, mini-batch SGD (batch 256, 32), SGD (batch 1),
  Momentum and Adam (batch 32). 30 epochs, lr tuned per configuration, 5 seeds.
  Plots: full-batch loss vs epoch and vs wall-clock time.

Part b: Noisy objective on quadratic(cond=20, dim=10).
  Methods: GD, Momentum, Adam, GD with decaying step size.
  Noise sigmas: [0, 0.1, 0.5, 1.0]. 10 seeds, 1000 iters.
  Plot: final f - f_star vs sigma (log y) per method.

Outputs
-------
- exp4a_loss_vs_epoch.png
- exp4a_loss_vs_time.png
- exp4a_results.csv
- exp4b_noise_sensitivity.png
- exp4b_results.csv
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.seeding import set_global_seed
from common.timing import Timer
from common.io_utils import save_csv

from landscapes import make_logistic_regression, quadratic
from first_order import GradientDescent, Momentum, Adam, minimize, minimize_stochastic


# ---------------------------------------------------------------------------
# Decaying step-size GD wrapper (defined here per instructions)
# ---------------------------------------------------------------------------

class DecayingStepGD:
    """Gradient descent with a decaying step: lr_t = lr0 / (1 + 0.01 * t).

    Implements the same step(x, grad_fn) interface as the standard optimizers.

    Args:
        lr0: Initial learning rate.
    """

    def __init__(self, lr: float) -> None:
        self.lr = lr          # stored as .lr for compatibility
        self.lr0 = lr
        self._t: int = 0

    def reset(self) -> None:
        """Reset step counter."""
        self._t = 0

    def step(self, x: np.ndarray, grad_fn) -> np.ndarray:
        """One decaying-step GD step.

        Args:
            x: Current parameter vector.
            grad_fn: Callable returning the gradient at a point.

        Returns:
            Updated parameter vector.
        """
        g = grad_fn(x)
        lr_t = self.lr0 / (1.0 + 0.01 * self._t)
        self._t += 1
        return x - lr_t * g


# ---------------------------------------------------------------------------
# Part a helpers
# ---------------------------------------------------------------------------

def _tune_lr_stochastic(data, obj, x0, opt_factory, batch_size, epochs, lr_grid, seeds):
    """Grid-search best lr for stochastic training by lowest final full-batch loss.

    Args:
        data: LogisticData object (with .f attached).
        obj: Full-batch Objective.
        x0: Initial weights.
        opt_factory: Callable(lr) -> optimizer instance.
        batch_size: Mini-batch size.
        epochs: Number of epochs per trial.
        lr_grid: Array of lr values to try.
        seeds: List of seeds.

    Returns:
        (best_lr, losses_per_seed) where losses_per_seed is a list of lists.
    """
    best_lr = lr_grid[0]
    best_final = float("inf")

    for lr in lr_grid:
        finals = []
        for seed in seeds[:2]:  # quick lr search uses just 2 seeds
            set_global_seed(seed)
            opt = opt_factory(lr)
            losses = minimize_stochastic(data, x0.copy(), opt, batch_size, epochs, seed=seed)
            finals.append(losses[-1] if losses else float("inf"))
        mean_final = np.mean(finals)
        if mean_final < best_final:
            best_final = mean_final
            best_lr = lr

    # Run with best_lr on all seeds and collect timing
    losses_per_seed = []
    times_per_seed = []
    for seed in seeds:
        set_global_seed(seed)
        opt = opt_factory(best_lr)
        t0 = _time.perf_counter()
        losses = minimize_stochastic(data, x0.copy(), opt, batch_size, epochs, seed=seed)
        elapsed = _time.perf_counter() - t0
        losses_per_seed.append(losses)
        times_per_seed.append(elapsed)

    return best_lr, losses_per_seed, times_per_seed


def _run_part_a(quick: bool, out_path: Path) -> dict:
    """Mini-batch study on logistic regression."""
    set_global_seed(42)

    n, d, l2 = 2000, 20, 1e-2
    obj, data = make_logistic_regression(n=n, d=d, l2=l2, seed=0)
    # Attach full-batch f as tests/minimize_stochastic expect
    data.f = obj.f

    x0 = np.zeros(d)

    n_seeds = 2 if quick else 5
    n_epochs = 5 if quick else 30
    seeds = list(range(42, 42 + n_seeds))
    lr_grid = np.logspace(-3, 0, 5 if quick else 7)

    configs = {
        "GD (n)": (lambda lr: GradientDescent(lr=lr), n),
        "SGD-256": (lambda lr: GradientDescent(lr=lr), 256),
        "SGD-32": (lambda lr: GradientDescent(lr=lr), 32),
        "SGD-1": (lambda lr: GradientDescent(lr=lr), 1),
        "Momentum-32": (lambda lr: Momentum(lr=lr, beta=0.9), 32),
        "Adam-32": (lambda lr: Adam(lr=lr, beta1=0.9, beta2=0.999), 32),
    }

    all_losses = {}     # config_name -> list of per-seed epoch-loss lists
    all_times = {}      # config_name -> list of per-seed total times
    table_rows = []

    with Timer("exp4a"):
        for cname, (factory, bsz) in configs.items():
            best_lr, losses_per_seed, times_per_seed = _tune_lr_stochastic(
                data, obj, x0, factory, bsz, n_epochs, lr_grid, seeds
            )
            all_losses[cname] = losses_per_seed
            all_times[cname] = times_per_seed

            arr = np.array(losses_per_seed)  # (n_seeds, n_epochs)
            mean_final = float(arr[:, -1].mean())
            std_final = float(arr[:, -1].std())
            mean_time_per_epoch = float(np.mean(times_per_seed)) / n_epochs

            # Gradient evaluations per epoch
            n_per_epoch = int(np.ceil(n / bsz))
            table_rows.append({
                "config": cname,
                "batch_size": bsz,
                "tuned_lr": best_lr,
                "final_loss_mean": mean_final,
                "final_loss_std": std_final,
                "time_per_epoch_s": mean_time_per_epoch,
                "grad_evals_per_epoch": n_per_epoch,
            })

    df_a = pd.DataFrame(table_rows)
    csv_path = out_path / "exp4a_results.csv"
    save_csv(df_a, csv_path)

    # ----------------------------------------------------------------
    # Plot 1: loss vs epoch (mean ± std band)
    # ----------------------------------------------------------------
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cname in enumerate(configs):
        arr = np.array(all_losses[cname])  # (n_seeds, n_epochs)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        epochs_x = np.arange(1, n_epochs + 1)
        ax.plot(epochs_x, mean, color=colors[i], linewidth=1.8, label=cname)
        ax.fill_between(epochs_x, mean - std, mean + std, color=colors[i], alpha=0.2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Full-batch loss", fontsize=12)
    ax.set_title("Exp4a: Mini-batch Study – Loss vs Epoch", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig1 = out_path / "exp4a_loss_vs_epoch.png"
    fig.savefig(fig1, dpi=150)
    plt.close(fig)

    # ----------------------------------------------------------------
    # Plot 2: loss vs wall-clock time (approximate with time-per-epoch)
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cname in enumerate(configs):
        arr = np.array(all_losses[cname])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        # Time axis: cumulative time per epoch (use mean total time)
        mean_total = float(np.mean(all_times[cname]))
        t_per_ep = mean_total / n_epochs
        t_axis = np.arange(1, n_epochs + 1) * t_per_ep
        ax.plot(t_axis, mean, color=colors[i], linewidth=1.8, label=cname)
        ax.fill_between(t_axis, mean - std, mean + std, color=colors[i], alpha=0.2)
    ax.set_xlabel("Wall-clock time (s)", fontsize=12)
    ax.set_ylabel("Full-batch loss", fontsize=12)
    ax.set_title("Exp4a: Mini-batch Study – Loss vs Time", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig2 = out_path / "exp4a_loss_vs_time.png"
    fig.savefig(fig2, dpi=150)
    plt.close(fig)

    return {
        "table": df_a,
        "csv": str(csv_path),
        "figures": [str(fig1), str(fig2)],
    }


# ---------------------------------------------------------------------------
# Part b: Noisy objective
# ---------------------------------------------------------------------------

def _run_part_b(quick: bool, out_path: Path) -> dict:
    """Evaluate optimizers under gradient noise on quadratic(cond=20, dim=10)."""
    set_global_seed(42)

    obj = quadratic(cond=20, dim=10, seed=0)
    x0 = np.zeros(10)
    f_star = obj.f_star
    L = 20.0  # largest eigenvalue (= cond when mu=1)
    lr_gd = 1.0 / L  # optimal GD step for this quadratic

    sigmas = [0.0, 0.1, 0.5, 1.0]
    n_seeds = 3 if quick else 10
    max_iter = 200 if quick else 1000
    seeds = list(range(42, 42 + n_seeds))

    methods = {
        "GD (lr=1/L)": (lambda: GradientDescent(lr=lr_gd),),
        "Momentum": (lambda: Momentum(lr=lr_gd * 0.5, beta=0.9),),
        "Adam": (lambda: Adam(lr=1e-2),),
        "GD-Decay": (lambda: DecayingStepGD(lr=lr_gd),),
    }

    table_rows = []
    # Collect (method, sigma) -> array of final gaps across seeds
    results_map = {}

    with Timer("exp4b"):
        for mname, (factory,) in methods.items():
            for sigma in sigmas:
                gaps = []
                for seed in seeds:
                    set_global_seed(seed)
                    opt = factory()
                    result = minimize(obj, x0, opt, max_iter=max_iter, tol=0.0,
                                      grad_noise_sigma=sigma, seed=seed)
                    # Use actual f (not noisy) for the gap
                    f_final = obj.f(result.x)
                    gap = max(f_final - f_star, 0.0)
                    gaps.append(gap)

                mean_gap = float(np.mean(gaps))
                std_gap = float(np.std(gaps))
                results_map[(mname, sigma)] = gaps
                table_rows.append({
                    "method": mname,
                    "sigma": sigma,
                    "mean_gap": mean_gap,
                    "std_gap": std_gap,
                })

    df_b = pd.DataFrame(table_rows)
    csv_path = out_path / "exp4b_results.csv"
    save_csv(df_b, csv_path)

    # ----------------------------------------------------------------
    # Plot: final gap vs sigma (log y) per method with std bars
    # ----------------------------------------------------------------
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, mname in enumerate(methods):
        means = [float(df_b[(df_b["method"] == mname) & (df_b["sigma"] == s)]["mean_gap"].values[0])
                 for s in sigmas]
        stds = [float(df_b[(df_b["method"] == mname) & (df_b["sigma"] == s)]["std_gap"].values[0])
                for s in sigmas]
        ax.errorbar(sigmas, means, yerr=stds, label=mname, color=colors[i],
                    marker="o", linewidth=1.8, capsize=4)

    ax.set_yscale("log")
    ax.set_xlabel("Gradient noise sigma", fontsize=12)
    ax.set_ylabel("Final f - f* (log scale)", fontsize=12)
    ax.set_title("Exp4b: Noisy Objective – Final Gap vs Noise Level", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig_path = out_path / "exp4b_noise_sensitivity.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return {
        "table": df_b,
        "csv": str(csv_path),
        "figure": str(fig_path),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(quick: bool = False, out_dir: str = "results/track_a") -> dict:
    """Run experiment 4: stochastic and noisy optimization.

    Args:
        quick: If True, use reduced seeds, epochs, and iterations.
        out_dir: Output directory.

    Returns:
        dict with 'part_a' and 'part_b' sub-dicts.
    """
    set_global_seed(42)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with Timer("exp4_total"):
        res_a = _run_part_a(quick, out_path)
        res_b = _run_part_b(quick, out_path)

    return {
        "part_a": res_a,
        "part_b": res_b,
    }


if __name__ == "__main__":
    result = run(quick=False)
    print("Part a table:")
    print(result["part_a"]["table"].to_string(index=False))
    print("\nPart b table:")
    print(result["part_b"]["table"].to_string(index=False))

