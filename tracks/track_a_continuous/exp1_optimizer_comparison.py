"""Experiment 1: Optimizer comparison on quadratic and Rosenbrock problems.

Compares GradientDescent, Momentum, NAG, AdaGrad, RMSProp, Adam, and Newton
on two problems:
  - quadratic(cond=100, dim=10): strongly convex benchmark.
  - rosenbrock(dim=2): non-convex banana function.

For each first-order method the learning rate is tuned over a log-spaced grid;
the best lr (lowest final f among non-diverged runs) is selected. Newton uses
line search on Rosenbrock and no tuning.

Outputs
-------
- results/track_a/exp1_convergence_quadratic.png
- results/track_a/exp1_convergence_rosenbrock.png
- results/track_a/exp1_tuned_lr.csv
- results/track_a/exp1_results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on the path for common.* imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import time as _time

from common.seeding import set_global_seed
from common.timing import Timer
from common.io_utils import save_csv

from landscapes import quadratic, rosenbrock
from first_order import GradientDescent, Momentum, NAG, AdaGrad, RMSProp, Adam, minimize
from second_order import newton


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_first_order_optimizers(lr: float, beta: float = 0.9) -> dict:
    """Build a fresh dict of first-order optimizer instances for a given lr."""
    return {
        "GradientDescent": GradientDescent(lr=lr),
        "Momentum": Momentum(lr=lr, beta=beta),
        "NAG": NAG(lr=lr, beta=beta),
        "AdaGrad": AdaGrad(lr=lr),
        "RMSProp": RMSProp(lr=lr, rho=0.9),
        "Adam": Adam(lr=lr, beta1=0.9, beta2=0.999),
    }


def _tune_lr(obj, x0, method_name, lr_grid, max_iter, tol=1e-6, beta=0.9):
    """Grid-search the best lr for a single first-order method.

    Selects the lr giving the lowest final f value among non-diverged runs.

    Args:
        obj: Objective landscape.
        x0: Starting point.
        method_name: Name key for _make_first_order_optimizers.
        lr_grid: Array of learning rates to try.
        max_iter: Maximum iterations per trial.
        tol: Gradient-norm tolerance.
        beta: Momentum coefficient.

    Returns:
        (best_lr, best_result) where best_result is the OptResult at best_lr.
    """
    best_lr = lr_grid[0]
    best_f = float("inf")
    best_result = None

    for lr in lr_grid:
        opts = _make_first_order_optimizers(lr, beta=beta)
        opt = opts[method_name]
        result = minimize(obj, x0, opt, max_iter=max_iter, tol=tol)
        if not result.diverged:
            final_f = result.f_history[-1]
            if final_f < best_f:
                best_f = final_f
                best_lr = lr
                best_result = result

    if best_result is None:
        # All diverged – pick first lr and return its (diverged) result
        opts = _make_first_order_optimizers(lr_grid[0], beta=beta)
        opt = opts[method_name]
        best_result = minimize(obj, x0, opts[method_name], max_iter=max_iter, tol=tol)
        best_lr = lr_grid[0]

    return best_lr, best_result


def _iters_to_tol(result, f_star, tol_f=1e-6):
    """Return the iteration index when f - f_star first drops below tol_f, or 'not reached'."""
    for i, fval in enumerate(result.f_history):
        if fval - f_star < tol_f:
            return i
    return "not reached"


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(quick: bool = False, out_dir: str = "results/track_a") -> dict:
    """Run experiment 1: optimizer comparison.

    Args:
        quick: If True, use reduced budget and grid for fast smoke-testing.
        out_dir: Directory where figures and tables are saved.

    Returns:
        dict with keys 'tuned_lr_table', 'results_table', and figure paths.
    """
    set_global_seed(42)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Budgets and grid sizes
    # ----------------------------------------------------------------
    if quick:
        max_iter_quad = 200
        max_iter_rosen = 2000
        n_lr_points = 5
    else:
        max_iter_quad = 2000
        max_iter_rosen = 20000
        n_lr_points = 9

    lr_grid = np.logspace(-4, 0, n_lr_points)
    tol = 1e-6
    beta = 0.9

    # ----------------------------------------------------------------
    # Problem definitions
    # ----------------------------------------------------------------
    problems = {
        "quadratic": {
            "obj": quadratic(cond=100, dim=10, seed=0),
            "x0": np.zeros(10),
            "max_iter": max_iter_quad,
            "f_star": None,  # filled below
            "use_line_search": False,
        },
        "rosenbrock": {
            "obj": rosenbrock(dim=2),
            "x0": np.array([-1.2, 1.0]),
            "max_iter": max_iter_rosen,
            "f_star": 0.0,
            "use_line_search": True,  # for Newton
        },
    }
    for name, cfg in problems.items():
        if cfg["f_star"] is None:
            cfg["f_star"] = cfg["obj"].f_star

    method_names = ["GradientDescent", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam", "Newton"]

    tuned_lr_rows = []
    results_rows = []
    histories = {pname: {} for pname in problems}

    with Timer("exp1_total") as total_timer:
        for pname, cfg in problems.items():
            obj = cfg["obj"]
            x0 = cfg["x0"]
            max_iter = cfg["max_iter"]
            f_star = cfg["f_star"]

            for mname in method_names:
                if mname == "Newton":
                    # Newton: no lr tuning; run directly with line search on Rosenbrock
                    with Timer(f"newton_{pname}") as t:
                        result = newton(
                            obj, x0,
                            max_iter=min(max_iter, 200),  # Newton converges fast
                            tol=tol,
                            damping=0.0,
                            line_search=cfg["use_line_search"],
                        )
                    chosen_lr = "N/A"
                else:
                    # Tune lr by grid search
                    with Timer(f"{mname}_{pname}_tune") as t:
                        chosen_lr, result = _tune_lr(
                            obj, x0, mname, lr_grid, max_iter, tol=tol, beta=beta
                        )

                final_f = result.f_history[-1]
                final_gnorm = result.gradnorm_history[-1] if result.gradnorm_history else float("nan")
                iters = _iters_to_tol(result, f_star, tol_f=tol)
                runtime = result.runtime_s

                tuned_lr_rows.append({
                    "problem": pname,
                    "method": mname,
                    "tuned_lr": chosen_lr,
                })
                results_rows.append({
                    "problem": pname,
                    "method": mname,
                    "tuned_lr": chosen_lr,
                    "final_f": final_f,
                    "final_gradnorm": final_gnorm,
                    "iters_to_tol": iters,
                    "runtime_s": runtime,
                })
                histories[pname][mname] = result.f_history

    # ----------------------------------------------------------------
    # Save tables
    # ----------------------------------------------------------------
    tuned_lr_df = pd.DataFrame(tuned_lr_rows)
    results_df = pd.DataFrame(results_rows)
    lr_csv_path = out_path / "exp1_tuned_lr.csv"
    res_csv_path = out_path / "exp1_results.csv"
    save_csv(tuned_lr_df, lr_csv_path)
    save_csv(results_df, res_csv_path)

    # ----------------------------------------------------------------
    # Convergence plots
    # ----------------------------------------------------------------
    fig_paths = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))

    for pname, cfg in problems.items():
        f_star = cfg["f_star"]
        fig, ax = plt.subplots(figsize=(9, 5))

        for i, mname in enumerate(method_names):
            hist = histories[pname].get(mname, [])
            if not hist:
                continue
            gap = np.array(hist, dtype=float) - f_star
            # Clip to positive (tiny negatives from floating point)
            gap = np.maximum(gap, 1e-16)
            ax.semilogy(gap, label=mname, color=colors[i], linewidth=1.8)

        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("f - f* (log scale)", fontsize=12)
        ax.set_title(f"Exp1: Optimizer Comparison – {pname}", fontsize=13)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()

        fname = f"exp1_convergence_{pname}.png"
        fpath = out_path / fname
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        fig_paths.append(str(fpath))

    return {
        "tuned_lr_table": tuned_lr_df,
        "results_table": results_df,
        "figures": fig_paths,
        "tuned_lr_csv": str(lr_csv_path),
        "results_csv": str(res_csv_path),
    }


if __name__ == "__main__":
    result = run(quick=False)
    print("Results table:")
    print(result["results_table"].to_string(index=False))

