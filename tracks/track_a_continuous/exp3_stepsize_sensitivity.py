"""Experiment 3: Step-size sensitivity sweep.

Sweeps lr over a log-spaced grid for six optimizers on
quadratic(cond=50, dim=10) with a fixed iteration budget.
Records final f - f_star and marks diverged runs as NaN.

Outputs
-------
- exp3_stepsize_sensitivity.png  – log-log plot of final gap vs lr
- exp3_results.csv               – per-optimizer per-lr table
- exp3_robustness.csv            – best lr and robustness width per optimizer
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

from common.seeding import set_global_seed
from common.timing import Timer
from common.io_utils import save_csv

from landscapes import quadratic
from first_order import GradientDescent, Momentum, NAG, AdaGrad, RMSProp, Adam, minimize


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(quick: bool = False, out_dir: str = "results/track_a") -> dict:
    """Run experiment 3: step-size sensitivity analysis.

    For each optimizer and each lr in the sweep grid, run for a fixed budget
    and record the final gap. Diverged runs are stored as NaN.

    Args:
        quick: If True, use fewer lr points and fewer iterations.
        out_dir: Output directory.

    Returns:
        dict with 'results_table', 'robustness_table', and figure/csv paths.
    """
    set_global_seed(42)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Problem setup
    obj = quadratic(cond=50, dim=10, seed=0)
    x0 = np.zeros(10)
    f_star = obj.f_star

    # Budget and grid
    if quick:
        max_iter = 100
        n_lr = 8
    else:
        max_iter = 500
        n_lr = 20

    lr_grid = np.logspace(-4, 0.5, n_lr)

    # Optimizer constructors: name -> callable(lr) -> optimizer instance
    optimizer_factories = {
        "GradientDescent": lambda lr: GradientDescent(lr=lr),
        "Momentum": lambda lr: Momentum(lr=lr, beta=0.9),
        "NAG": lambda lr: NAG(lr=lr, beta=0.9),
        "AdaGrad": lambda lr: AdaGrad(lr=lr),
        "RMSProp": lambda lr: RMSProp(lr=lr, rho=0.9),
        "Adam": lambda lr: Adam(lr=lr, beta1=0.9, beta2=0.999),
    }

    results_rows = []

    with Timer("exp3_sweep"):
        for oname, factory in optimizer_factories.items():
            for lr in lr_grid:
                opt = factory(lr)
                result = minimize(obj, x0, opt, max_iter=max_iter, tol=0.0)
                if result.diverged:
                    final_gap = float("nan")
                else:
                    final_gap = float(result.f_history[-1] - f_star)
                    if not np.isfinite(final_gap):
                        final_gap = float("nan")

                results_rows.append({
                    "optimizer": oname,
                    "lr": lr,
                    "final_gap": final_gap,
                    "diverged": result.diverged,
                })

    results_df = pd.DataFrame(results_rows)
    res_csv = out_path / "exp3_results.csv"
    save_csv(results_df, res_csv)

    # ----------------------------------------------------------------
    # Robustness table: best lr and width of stable range
    # ----------------------------------------------------------------
    gap_threshold = 1e-3  # define "good" as gap < 1e-3

    robustness_rows = []
    for oname in optimizer_factories:
        sub = results_df[results_df["optimizer"] == oname].copy()
        sub = sub.dropna(subset=["final_gap"])
        if sub.empty:
            robustness_rows.append({
                "optimizer": oname,
                "best_lr": float("nan"),
                "best_gap": float("nan"),
                "range_width_decades": 0.0,
            })
            continue

        # Best lr by lowest final gap
        idx_best = sub["final_gap"].idxmin()
        best_lr = float(sub.loc[idx_best, "lr"])
        best_gap = float(sub.loc[idx_best, "final_gap"])

        # Width of lr range (in decades) where gap < threshold
        good = sub[sub["final_gap"] < gap_threshold]
        if good.empty:
            width = 0.0
        else:
            lr_min = good["lr"].min()
            lr_max = good["lr"].max()
            width = float(np.log10(lr_max) - np.log10(lr_min))

        robustness_rows.append({
            "optimizer": oname,
            "best_lr": best_lr,
            "best_gap": best_gap,
            "range_width_decades": width,
        })

    rob_df = pd.DataFrame(robustness_rows)
    rob_csv = out_path / "exp3_robustness.csv"
    save_csv(rob_df, rob_csv)

    # ----------------------------------------------------------------
    # Plot: final gap vs lr (log-log), one line per optimizer
    # ----------------------------------------------------------------
    colors = plt.cm.tab10(np.linspace(0, 1, len(optimizer_factories)))
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, oname in enumerate(optimizer_factories):
        sub = results_df[results_df["optimizer"] == oname].sort_values("lr")
        lrs = sub["lr"].values
        gaps = sub["final_gap"].values

        # Plot connected segments, break at NaN (diverged)
        valid = np.isfinite(gaps)
        # Plot only valid points connected; diverged points shown as X
        # Use masked plotting for breaks
        ax.plot(lrs[valid], gaps[valid], "o-", color=colors[i], label=oname,
                linewidth=1.8, markersize=4)
        # Mark diverged
        ax.plot(lrs[~valid], np.full(np.sum(~valid), np.nan), "x",
                color=colors[i], markersize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning Rate (lr)", fontsize=12)
    ax.set_ylabel("Final f - f* (log scale)", fontsize=12)
    ax.set_title(f"Exp3: Step-Size Sensitivity – quadratic(cond=50, dim=10), {max_iter} iters",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(gap_threshold, color="gray", linestyle="--", alpha=0.5,
               label=f"gap={gap_threshold:.0e}")
    fig.tight_layout()
    fig_path = out_path / "exp3_stepsize_sensitivity.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return {
        "results_table": results_df,
        "robustness_table": rob_df,
        "results_csv": str(res_csv),
        "robustness_csv": str(rob_csv),
        "figure": str(fig_path),
    }


if __name__ == "__main__":
    result = run(quick=False)
    print("Robustness table:")
    print(result["robustness_table"].to_string(index=False))

