"""Experiment 2: Landscape analysis – convexity, saddle points, local minima.

Parts
-----
a) Convexity classes: GradientDescent on strongly-convex, convex (not SC),
   and non-convex objectives. Fits the convergence slope for the first two.
b) Saddle point: four methods on saddle_function(); trajectories on contour.
c) Local vs global minimum: double_well from three starting points.

Outputs
-------
- exp2a_convexity.png
- exp2a_table.csv
- exp2b_saddle_trajectories.png
- exp2b_table.csv
- exp2c_double_well_trajectories.png
- exp2c_table.csv
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

from landscapes import (
    quadratic,
    sum_quartic,
    rosenbrock,
    saddle_function,
    double_well,
    classify_stationary_point,
)
from first_order import GradientDescent, Momentum, Adam, minimize
from second_order import newton


# ---------------------------------------------------------------------------
# Part a: Convexity classes
# ---------------------------------------------------------------------------

def _run_part_a(quick: bool, out_path: Path) -> dict:
    """Convergence of GD on three objectives with different convexity classes.

    Strongly convex (quadratic cond=10) converges linearly.
    Convex but not SC (sum_quartic) converges sublinearly.
    Non-convex (rosenbrock) has no convergence guarantee.
    """
    set_global_seed(42)

    max_iter = 500 if quick else 5000
    tol = 0.0  # run to full budget to see rate

    # Strongly convex quadratic: lr = 1/L = 1/cond = 0.1
    obj_sc = quadratic(cond=10, dim=5, seed=0)
    x0_sc = np.ones(5)
    lr_sc = 1.0 / 10.0
    res_sc = minimize(obj_sc, x0_sc, GradientDescent(lr=lr_sc), max_iter=max_iter, tol=tol)

    # Convex but not SC: sum_quartic dim=5, start at ones
    obj_cx = sum_quartic(dim=5)
    x0_cx = np.ones(5)
    # Hessian at x0 is diag(12); lr < 1/12 ≈ 0.083; use 0.05
    lr_cx = 0.05
    res_cx = minimize(obj_cx, x0_cx, GradientDescent(lr=lr_cx), max_iter=max_iter, tol=tol)

    # Non-convex rosenbrock
    obj_nc = rosenbrock(dim=2)
    x0_nc = np.array([-1.2, 1.0])
    # Small lr for Rosenbrock's large curvature
    lr_nc = 1e-3
    res_nc = minimize(obj_nc, x0_nc, GradientDescent(lr=lr_nc), max_iter=max_iter, tol=tol)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5))

    f_star_sc = obj_sc.f_star
    gap_sc = np.maximum(np.array(res_sc.f_history) - f_star_sc, 1e-16)

    f_star_cx = obj_cx.f_star
    gap_cx = np.maximum(np.array(res_cx.f_history) - f_star_cx, 1e-16)

    f_star_nc = 0.0  # rosenbrock f_star = 0
    gap_nc = np.maximum(np.array(res_nc.f_history) - f_star_nc, 1e-16)

    ax.semilogy(gap_sc, label="Strongly convex (linear rate)", color="blue", linewidth=1.8)
    ax.semilogy(gap_cx, label="Convex, not SC (sublinear rate)", color="orange", linewidth=1.8)
    ax.semilogy(gap_nc, label="Non-convex Rosenbrock (none guaranteed)", color="red",
                linewidth=1.8, linestyle="--")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("f - f* (log scale)", fontsize=12)
    ax.set_title("Exp2a: Convergence by Convexity Class", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig_path = out_path / "exp2a_convexity.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # --- Slope fitting on second half ---
    def _fit_slope(gap_arr):
        """Fit log(gap) vs iteration on second half of run (linear in log space)."""
        n = len(gap_arr)
        half = n // 2
        y = np.log(gap_arr[half:] + 1e-16)
        x = np.arange(half, n)
        # Linear regression: y = slope * x + intercept
        x_mean, y_mean = x.mean(), y.mean()
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-30))
        return slope

    slope_sc = _fit_slope(gap_sc)
    slope_cx = _fit_slope(gap_cx)

    table_rows = [
        {
            "objective": "quadratic(cond=10)",
            "convexity": "strongly convex",
            "expected_rate": "linear",
            "log_gap_slope_2nd_half": slope_sc,
            "final_gap": float(gap_sc[-1]),
        },
        {
            "objective": "sum_quartic(dim=5)",
            "convexity": "convex (not SC)",
            "expected_rate": "sublinear",
            "log_gap_slope_2nd_half": slope_cx,
            "final_gap": float(gap_cx[-1]),
        },
        {
            "objective": "rosenbrock(dim=2)",
            "convexity": "non-convex",
            "expected_rate": "none guaranteed",
            "log_gap_slope_2nd_half": float("nan"),
            "final_gap": float(gap_nc[-1]),
        },
    ]
    df = pd.DataFrame(table_rows)
    csv_path = out_path / "exp2a_table.csv"
    save_csv(df, csv_path)

    return {
        "table": df,
        "csv": str(csv_path),
        "figure": str(fig_path),
    }


# ---------------------------------------------------------------------------
# Part b: Saddle point
# ---------------------------------------------------------------------------

def _run_part_b(quick: bool, out_path: Path) -> dict:
    """Run four methods on saddle_function(); record trajectories and classify final points."""
    set_global_seed(42)

    obj = saddle_function()
    # Small perturbation away from saddle at origin to break exact symmetry
    x0 = np.array([1e-3, 1e-3])

    max_iter = 500 if quick else 3000
    lr_gd = 0.05
    lr_mom = 0.02
    lr_adam = 0.05

    # Compute damping for Newton so H + damping*I is PD
    H0 = obj.hess(x0)
    eigvals0 = np.linalg.eigvalsh(H0)
    damping_val = max(0.0, -eigvals0.min() + 0.1)

    methods = {
        "GradientDescent": (GradientDescent(lr=lr_gd), False),
        "Momentum": (Momentum(lr=lr_mom, beta=0.9), False),
        "Adam": (Adam(lr=lr_adam), False),
        "Newton (undamped)": (None, False),    # special-cased below
        "Newton (damped)": (None, True),       # uses computed damping
    }

    trajectories = {}
    table_rows = []

    for name, (opt, use_damping) in methods.items():
        if "Newton" in name:
            damp = damping_val if use_damping else 0.0
            result = newton(obj, x0, max_iter=min(max_iter, 100), tol=1e-10,
                            damping=damp, line_search=False)
        else:
            result = minimize(obj, x0, opt, max_iter=max_iter, tol=1e-10)

        # Recover trajectory by re-running with stored steps (or just final)
        # We'll track trajectory by re-running with step capture
        x_final = result.x
        classification = classify_stationary_point(obj, x_final)
        f_final = result.f_history[-1] if result.f_history else float("nan")

        table_rows.append({
            "method": name,
            "final_x0": float(x_final[0]),
            "final_x1": float(x_final[1]),
            "final_f": f_final,
            "classification": classification,
        })

    # Re-run with trajectory capture for plot
    def _run_with_traj(opt_or_none, is_newton, damping_v, n_iter):
        """Re-run optimizer, capturing trajectory at each step."""
        traj = [x0.copy()]
        x = x0.copy()
        if is_newton:
            # Step manually
            for _ in range(min(n_iter, 100)):
                g = obj.grad(x)
                H = obj.hess(x)
                H_reg = H + damping_v * np.eye(2)
                try:
                    from scipy.linalg import solve as sp_solve
                    p = sp_solve(H_reg, -g, assume_a="sym")
                except Exception:
                    p = -g
                x = x + p
                traj.append(x.copy())
                if np.linalg.norm(g) < 1e-10:
                    break
        else:
            opt_or_none.reset()
            for _ in range(n_iter):
                g = obj.grad(x)
                if np.linalg.norm(g) < 1e-10:
                    break
                x = opt_or_none.step(x, lambda pt: obj.grad(pt))
                traj.append(x.copy())
        return np.array(traj)

    n_traj = 300 if quick else 1000
    trajectories = {
        "GradientDescent": _run_with_traj(GradientDescent(lr=lr_gd), False, 0.0, n_traj),
        "Momentum": _run_with_traj(Momentum(lr=lr_mom, beta=0.9), False, 0.0, n_traj),
        "Adam": _run_with_traj(Adam(lr=lr_adam), False, 0.0, n_traj),
        "Newton (undamped)": _run_with_traj(None, True, 0.0, n_traj),
        "Newton (damped)": _run_with_traj(None, True, damping_val, n_traj),
    }

    # --- Contour plot with trajectories ---
    xv = np.linspace(-3, 3, 200)
    yv = np.linspace(-3, 3, 200)
    XX, YY = np.meshgrid(xv, yv)
    ZZ = XX**2 - YY**2 + 0.25 * YY**4

    fig, ax = plt.subplots(figsize=(8, 6))
    cset = ax.contourf(XX, YY, ZZ, levels=30, cmap="RdBu_r", alpha=0.6)
    plt.colorbar(cset, ax=ax)

    colors_traj = {"GradientDescent": "blue", "Momentum": "orange",
                   "Adam": "green", "Newton (undamped)": "red", "Newton (damped)": "purple"}
    for name, traj in trajectories.items():
        ax.plot(traj[:, 0], traj[:, 1], "o-", markersize=2, linewidth=1.2,
                label=name, color=colors_traj[name], alpha=0.85)
        ax.plot(traj[0, 0], traj[0, 1], "k*", markersize=10)

    ax.set_xlabel("x[0]", fontsize=12)
    ax.set_ylabel("x[1]", fontsize=12)
    ax.set_title("Exp2b: Saddle Function Trajectories", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig_path = out_path / "exp2b_saddle_trajectories.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    df = pd.DataFrame(table_rows)
    csv_path = out_path / "exp2b_table.csv"
    save_csv(df, csv_path)

    return {
        "table": df,
        "csv": str(csv_path),
        "figure": str(fig_path),
    }


# ---------------------------------------------------------------------------
# Part c: Double-well – local vs global minimum
# ---------------------------------------------------------------------------

def _run_part_c(quick: bool, out_path: Path) -> dict:
    """Run GD and Adam on double_well from three starting points.

    Starting points:
      - negative x: (-2.0, 0.0) – attracted to local minimum in left well
      - near zero:  (0.0, 0.0)  – near saddle between wells
      - positive x: (1.5, 0.0)  – attracted to global minimum in right well
    """
    set_global_seed(42)

    obj = double_well()
    max_iter = 300 if quick else 2000

    # Find global minimum via fine grid search (vectorised for speed)
    xs = np.linspace(-2.5, 2.5, 500)
    ys = np.linspace(-2.5, 2.5, 500)
    XX, YY = np.meshgrid(xs, ys)
    # double_well: f(x,y) = x^4 - 3x^2 + x + y^2
    ZZ_grid = XX**4 - 3.0 * XX**2 + XX + YY**2
    idx_flat = np.argmin(ZZ_grid)
    iy, ix = np.unravel_index(idx_flat, ZZ_grid.shape)
    f_grid_min = float(ZZ_grid[iy, ix])
    x_global = np.array([float(xs[ix]), float(ys[iy])])

    starts = [
        ("negative_x", np.array([-2.0, 0.0])),
        ("near_zero",  np.array([0.0, 0.0])),
        ("positive_x", np.array([1.5, 0.0])),
    ]

    methods_cfg = {
        "GradientDescent": GradientDescent(lr=0.01),
        "Adam": Adam(lr=0.05),
    }

    table_rows = []
    all_trajs = {}

    for start_name, x0 in starts:
        for mname, opt in methods_cfg.items():
            result = minimize(obj, x0, opt, max_iter=max_iter, tol=1e-10)
            x_final = result.x
            f_final = result.f_history[-1]
            is_global = bool(np.abs(f_final - f_grid_min) < 0.1)
            table_rows.append({
                "start": start_name,
                "method": mname,
                "final_x0": float(x_final[0]),
                "final_x1": float(x_final[1]),
                "final_f": f_final,
                "is_global_minimum": is_global,
            })
            all_trajs[(start_name, mname)] = result

    # Recompute trajectories for plot
    def _traj(x0, opt):
        """Collect trajectory by stepping manually."""
        traj = [x0.copy()]
        x = x0.copy()
        opt.reset()
        for _ in range(max_iter):
            g = obj.grad(x)
            if np.linalg.norm(g) < 1e-10:
                break
            x = opt.step(x, lambda pt: obj.grad(pt))
            traj.append(x.copy())
        return np.array(traj)

    # Contour plot
    xv = np.linspace(-2.5, 2.5, 200)
    yv = np.linspace(-1.5, 1.5, 200)
    XX, YY = np.meshgrid(xv, yv)
    ZZ = XX**4 - 3 * XX**2 + XX + YY**2

    fig, ax = plt.subplots(figsize=(9, 6))
    cset = ax.contourf(XX, YY, ZZ, levels=30, cmap="viridis", alpha=0.65)
    plt.colorbar(cset, ax=ax)

    colors_start = {"negative_x": "blue", "near_zero": "orange", "positive_x": "green"}
    ls_method = {"GradientDescent": "-", "Adam": "--"}
    markers_method = {"GradientDescent": "o", "Adam": "s"}

    for start_name, x0 in starts:
        for mname, opt in {
            "GradientDescent": GradientDescent(lr=0.01),
            "Adam": Adam(lr=0.05),
        }.items():
            traj = _traj(x0, opt)
            ax.plot(traj[:, 0], traj[:, 1],
                    linestyle=ls_method[mname],
                    marker=markers_method[mname],
                    markersize=2, linewidth=1.2,
                    color=colors_start[start_name],
                    label=f"{start_name}/{mname}",
                    alpha=0.85)
        ax.plot(x0[0], x0[1], "k*", markersize=12)

    ax.set_xlabel("x[0]", fontsize=12)
    ax.set_ylabel("x[1]", fontsize=12)
    ax.set_title("Exp2c: Double-Well – Local vs Global Minimum", fontsize=13)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig_path = out_path / "exp2c_double_well_trajectories.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    df = pd.DataFrame(table_rows)
    csv_path = out_path / "exp2c_table.csv"
    save_csv(df, csv_path)

    return {
        "table": df,
        "csv": str(csv_path),
        "figure": str(fig_path),
        "global_f_min": f_grid_min,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(quick: bool = False, out_dir: str = "results/track_a") -> dict:
    """Run all parts of experiment 2.

    Args:
        quick: If True, use reduced budgets.
        out_dir: Output directory for figures and tables.

    Returns:
        dict with keys 'part_a', 'part_b', 'part_c', each a sub-dict of
        table, csv path, figure path.
    """
    set_global_seed(42)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with Timer("exp2_total"):
        res_a = _run_part_a(quick, out_path)
        res_b = _run_part_b(quick, out_path)
        res_c = _run_part_c(quick, out_path)

    return {
        "part_a": res_a,
        "part_b": res_b,
        "part_c": res_c,
    }


if __name__ == "__main__":
    result = run(quick=False)
    print("Part a table:")
    print(result["part_a"]["table"].to_string(index=False))
    print("\nPart b table:")
    print(result["part_b"]["table"].to_string(index=False))
    print("\nPart c table:")
    print(result["part_c"]["table"].to_string(index=False))
