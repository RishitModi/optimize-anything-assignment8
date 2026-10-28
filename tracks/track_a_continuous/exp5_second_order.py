"""Experiment 5: Second-order methods and Hessian analysis.

Parts
-----
a) Newton vs GradientDescent on logistic regression (n=2000, d=20, l2=1e-2):
   iterations to gradient norm 1e-8, runtime, and f history; convergence plot.

b) Gauss-Newton and Levenberg-Marquardt vs GD and Adam on make_exp_decay_fit
   from a poor start (a=1, b=2, c=0): convergence plot, parameter recovery table.

c) Curvature approximations on the logistic regression solution:
   (i)  hvp_finite_difference vs hvp_exact – max relative error for 5 vectors.
   (ii) top-5 eigenvalues via top_eigenpairs_from_hvp vs numpy.linalg.eigh.
   (iii) low-rank approximation Frobenius error for k in {1,2,5,10,20}.
   (iv) empirical Fisher diagonal vs true Hessian diagonal (bar chart + corr).
   (v)  diag_preconditioned_gd (Fisher diagonal) vs plain GD, same budget.

Outputs
-------
- exp5a_convergence.png, exp5a_results.csv
- exp5b_convergence.png, exp5b_params.csv
- exp5c_hvp_error.csv
- exp5c_eigvals.csv
- exp5c_lowrank_error.png, exp5c_lowrank_error.csv
- exp5c_fisher_diag.png, exp5c_fisher_diag.csv
- exp5c_diag_precond.png
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

from landscapes import make_logistic_regression, make_exp_decay_fit
from first_order import GradientDescent, Adam, minimize
from second_order import (
    newton,
    gauss_newton,
    levenberg_marquardt,
    hvp_finite_difference,
    hvp_exact,
    top_eigenpairs_from_hvp,
    low_rank_hessian_approx,
    empirical_fisher_diag,
    diag_preconditioned_gd,
)


# ---------------------------------------------------------------------------
# Part a: Newton vs GD on logistic regression
# ---------------------------------------------------------------------------

def _run_part_a(quick: bool, out_path: Path) -> dict:
    """Newton vs GD on logistic regression."""
    set_global_seed(42)

    obj, data = make_logistic_regression(n=2000, d=20, l2=1e-2, seed=0)
    x0 = np.zeros(obj.dim)
    tol = 1e-8

    # First get a long Newton run to establish f_star
    res_newton_long = newton(obj, x0, max_iter=500, tol=1e-12, line_search=True)
    f_star = res_newton_long.f_history[-1]

    # GD with sensible lr
    L_approx = 0.25 + 1e-2  # approximate: max(p(1-p)) = 0.25 for logistic + l2
    lr_gd = 1.0 / (L_approx * 1.5)
    max_iter_gd = 50 if quick else 500

    res_gd = minimize(obj, x0, GradientDescent(lr=lr_gd), max_iter=max_iter_gd, tol=tol)
    max_iter_newton = 20 if quick else 100
    res_newton = newton(obj, x0, max_iter=max_iter_newton, tol=tol, line_search=True)

    table_rows = [
        {
            "method": "GradientDescent",
            "iters": res_gd.n_iters,
            "converged": res_gd.converged,
            "final_f": res_gd.f_history[-1],
            "final_gradnorm": res_gd.gradnorm_history[-1],
            "runtime_s": res_gd.runtime_s,
        },
        {
            "method": "Newton (line search)",
            "iters": res_newton.n_iters,
            "converged": res_newton.converged,
            "final_f": res_newton.f_history[-1],
            "final_gradnorm": res_newton.gradnorm_history[-1],
            "runtime_s": res_newton.runtime_s,
        },
    ]
    df = pd.DataFrame(table_rows)
    csv_path = out_path / "exp5a_results.csv"
    save_csv(df, csv_path)

    # Convergence plot
    fig, ax = plt.subplots(figsize=(9, 5))
    gap_gd = np.maximum(np.array(res_gd.f_history) - f_star, 1e-16)
    gap_newton = np.maximum(np.array(res_newton.f_history) - f_star, 1e-16)
    ax.semilogy(gap_gd, label="GradientDescent", color="blue", linewidth=1.8)
    ax.semilogy(gap_newton, label="Newton (line search)", color="red", linewidth=1.8)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("f - f* (log scale)", fontsize=12)
    ax.set_title("Exp5a: Newton vs GD on Logistic Regression", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig_path = out_path / "exp5a_convergence.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return {
        "table": df,
        "csv": str(csv_path),
        "figure": str(fig_path),
        "newton_solution": res_newton_long.x,
    }


# ---------------------------------------------------------------------------
# Part b: Gauss-Newton and LM on exp_decay_fit
# ---------------------------------------------------------------------------

def _run_part_b(quick: bool, out_path: Path) -> dict:
    """Gauss-Newton and LM vs GD and Adam on exponential decay fitting."""
    set_global_seed(42)

    obj, nls = make_exp_decay_fit(seed=0)
    x0_poor = np.array([1.0, 2.0, 0.0])  # poor start (task spec)
    true_params = np.array([2.0, 0.5, 0.3])

    max_iter = 20 if quick else 100

    res_gn = gauss_newton(nls.residual, nls.jacobian, x0_poor, max_iter=max_iter, tol=1e-8)
    res_lm = levenberg_marquardt(nls.residual, nls.jacobian, x0_poor,
                                 max_iter=max_iter * 3, tol=1e-8, lam0=1e-2)
    res_gd = minimize(obj, x0_poor, GradientDescent(lr=1e-3), max_iter=max_iter * 10, tol=1e-8)
    res_adam = minimize(obj, x0_poor, Adam(lr=1e-2), max_iter=max_iter * 10, tol=1e-8)

    # Parameter recovery table
    param_rows = []
    for name, res in [("Gauss-Newton", res_gn), ("LM", res_lm),
                      ("GD", res_gd), ("Adam", res_adam)]:
        param_rows.append({
            "method": name,
            "a_recovered": float(res.x[0]),
            "b_recovered": float(res.x[1]),
            "c_recovered": float(res.x[2]),
            "a_true": true_params[0],
            "b_true": true_params[1],
            "c_true": true_params[2],
            "final_f": res.f_history[-1],
            "iters": res.n_iters,
        })
    df_params = pd.DataFrame(param_rows)
    csv_params = out_path / "exp5b_params.csv"
    save_csv(df_params, csv_params)

    # Convergence plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, res, color, ls in [
        ("Gauss-Newton", res_gn, "blue", "-"),
        ("LM", res_lm, "red", "--"),
        ("GD (lr=1e-3)", res_gd, "orange", "-."),
        ("Adam (lr=1e-2)", res_adam, "green", ":"),
    ]:
        ax.semilogy(res.f_history, label=name, color=color, linestyle=ls, linewidth=1.8)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("f = 0.5||r||² (log scale)", fontsize=12)
    ax.set_title("Exp5b: Exp Decay Fitting – Gauss-Newton / LM vs GD / Adam", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig_path = out_path / "exp5b_convergence.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return {
        "params_table": df_params,
        "params_csv": str(csv_params),
        "figure": str(fig_path),
    }


# ---------------------------------------------------------------------------
# Part c: Curvature approximations
# ---------------------------------------------------------------------------

def _run_part_c(quick: bool, out_path: Path, x_sol: np.ndarray) -> dict:
    """Hessian analysis at the logistic regression solution."""
    set_global_seed(42)

    obj, data = make_logistic_regression(n=2000, d=20, l2=1e-2, seed=0)
    d = obj.dim
    x = x_sol

    # Build exact Hessian
    H_exact = obj.hess(x)

    # HVP closures
    def hvp_fn(v):
        return hvp_exact(obj, x, v)

    def hvp_fd_fn(v):
        return hvp_finite_difference(obj.grad, x, v)

    # ----------------------------------------------------------------
    # (i) HVP finite-difference vs exact – 5 random vectors
    # ----------------------------------------------------------------
    set_global_seed(42)
    rng = np.random.default_rng(42)
    hvp_rows = []
    for k_v in range(5):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        hvp_ex = hvp_fn(v)
        hvp_fd = hvp_fd_fn(v)
        diff = np.abs(hvp_ex - hvp_fd)
        denom = np.abs(hvp_ex) + 1e-8
        rel_err = float(np.max(diff / denom))
        hvp_rows.append({"vector_idx": k_v, "max_relative_error": rel_err})
    df_hvp = pd.DataFrame(hvp_rows)
    csv_hvp = out_path / "exp5c_hvp_error.csv"
    save_csv(df_hvp, csv_hvp)

    # ----------------------------------------------------------------
    # (ii) Top-5 eigenvalues: power iteration vs numpy.linalg.eigh
    # ----------------------------------------------------------------
    k_eig = 5
    eigvals_power, _ = top_eigenpairs_from_hvp(hvp_fn, d, k=k_eig, seed=0, n_iter=300)
    eigvals_exact_all, _ = np.linalg.eigh(H_exact)
    eigvals_exact_top = eigvals_exact_all[-k_eig:][::-1]  # top-k, descending

    eig_rows = []
    for i in range(k_eig):
        eig_rows.append({
            "rank": i + 1,
            "power_iter_eigval": float(eigvals_power[i]),
            "exact_eigval": float(eigvals_exact_top[i]),
            "abs_error": float(abs(eigvals_power[i] - eigvals_exact_top[i])),
        })
    df_eig = pd.DataFrame(eig_rows)
    csv_eig = out_path / "exp5c_eigvals.csv"
    save_csv(df_eig, csv_eig)

    # ----------------------------------------------------------------
    # (iii) Low-rank approximation Frobenius error
    # ----------------------------------------------------------------
    k_vals = [1, 2, 5, 10, 20] if not quick else [1, 2, 5, 10]
    H_frob = float(np.linalg.norm(H_exact, "fro"))
    lr_rows = []
    for k_rank in k_vals:
        _, _, H_approx = low_rank_hessian_approx(hvp_fn, d, k=k_rank, seed=0)
        frob_err = float(np.linalg.norm(H_exact - H_approx, "fro")) / (H_frob + 1e-30)
        lr_rows.append({"k": k_rank, "rel_frob_error": frob_err})
    df_lr = pd.DataFrame(lr_rows)
    csv_lr = out_path / "exp5c_lowrank_error.csv"
    save_csv(df_lr, csv_lr)

    fig_lr, ax_lr = plt.subplots(figsize=(7, 4))
    ax_lr.plot(df_lr["k"], df_lr["rel_frob_error"], "o-", linewidth=1.8, markersize=6)
    ax_lr.set_xlabel("Rank k", fontsize=12)
    ax_lr.set_ylabel("Relative Frobenius Error", fontsize=12)
    ax_lr.set_title("Exp5c(iii): Low-Rank Hessian Approximation Error vs Rank", fontsize=12)
    ax_lr.grid(True, alpha=0.3)
    fig_lr.tight_layout()
    fig_lr_path = out_path / "exp5c_lowrank_error.png"
    fig_lr.savefig(fig_lr_path, dpi=150)
    plt.close(fig_lr)

    # ----------------------------------------------------------------
    # (iv) Empirical Fisher diagonal vs true Hessian diagonal
    # ----------------------------------------------------------------
    per_sample_grads = data.per_sample_grads(x)  # (n, d)
    fisher_diag = empirical_fisher_diag(per_sample_grads)
    hess_diag = np.diag(H_exact)

    # Correlation coefficient
    corr = float(np.corrcoef(fisher_diag, hess_diag)[0, 1])

    fisher_rows = [
        {
            "coord": i,
            "fisher_diag": float(fisher_diag[i]),
            "hess_diag": float(hess_diag[i]),
        }
        for i in range(d)
    ]
    fisher_rows.append({
        "coord": "correlation",
        "fisher_diag": corr,
        "hess_diag": corr,
    })
    df_fisher = pd.DataFrame(fisher_rows)
    csv_fisher = out_path / "exp5c_fisher_diag.csv"
    save_csv(df_fisher, csv_fisher)

    fig_f, ax_f = plt.subplots(figsize=(10, 5))
    coords = np.arange(d)
    width = 0.4
    ax_f.bar(coords - width / 2, hess_diag, width, label="True Hessian diag", color="blue",
             alpha=0.7)
    ax_f.bar(coords + width / 2, fisher_diag, width, label="Empirical Fisher diag", color="orange",
             alpha=0.7)
    ax_f.set_xlabel("Coordinate index", fontsize=12)
    ax_f.set_ylabel("Value", fontsize=12)
    ax_f.set_title(f"Exp5c(iv): Fisher Diagonal vs Hessian Diagonal (corr={corr:.4f})", fontsize=12)
    ax_f.legend(fontsize=10)
    ax_f.grid(True, alpha=0.3)
    fig_f.tight_layout()
    fig_f_path = out_path / "exp5c_fisher_diag.png"
    fig_f.savefig(fig_f_path, dpi=150)
    plt.close(fig_f)

    # ----------------------------------------------------------------
    # (v) Diag-preconditioned GD vs plain GD
    # ----------------------------------------------------------------
    x0 = np.zeros(d)
    max_iter_v = 50 if quick else 300

    # Fisher-preconditioned GD
    res_prec = diag_preconditioned_gd(
        obj, x0, diag=fisher_diag, lr=1.0,
        max_iter=max_iter_v, tol=1e-8,
    )
    # Plain GD with small lr
    lr_gd = 1.0 / (0.25 + 1e-2)
    res_plain = minimize(obj, x0, GradientDescent(lr=lr_gd * 0.5), max_iter=max_iter_v, tol=1e-8)

    f_star_approx = obj.f(x)  # solution found earlier

    fig_v, ax_v = plt.subplots(figsize=(9, 5))
    gap_prec = np.maximum(np.array(res_prec.f_history) - f_star_approx, 1e-16)
    gap_plain = np.maximum(np.array(res_plain.f_history) - f_star_approx, 1e-16)
    ax_v.semilogy(gap_prec, label="Fisher-preconditioned GD", color="blue", linewidth=1.8)
    ax_v.semilogy(gap_plain, label="Plain GD", color="orange", linewidth=1.8)
    ax_v.set_xlabel("Iteration", fontsize=12)
    ax_v.set_ylabel("f - f* (log scale)", fontsize=12)
    ax_v.set_title("Exp5c(v): Diag-Preconditioned vs Plain GD", fontsize=13)
    ax_v.legend(fontsize=10)
    ax_v.grid(True, which="both", alpha=0.3)
    fig_v.tight_layout()
    fig_v_path = out_path / "exp5c_diag_precond.png"
    fig_v.savefig(fig_v_path, dpi=150)
    plt.close(fig_v)

    return {
        "hvp_table": df_hvp,
        "hvp_csv": str(csv_hvp),
        "eigvals_table": df_eig,
        "eigvals_csv": str(csv_eig),
        "lowrank_table": df_lr,
        "lowrank_csv": str(csv_lr),
        "lowrank_figure": str(fig_lr_path),
        "fisher_table": df_fisher,
        "fisher_csv": str(csv_fisher),
        "fisher_figure": str(fig_f_path),
        "diag_precond_figure": str(fig_v_path),
        "fisher_hess_corr": corr,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(quick: bool = False, out_dir: str = "results/track_a") -> dict:
    """Run experiment 5: second-order methods and Hessian analysis.

    Args:
        quick: If True, use reduced budgets.
        out_dir: Output directory.

    Returns:
        dict with 'part_a', 'part_b', 'part_c' sub-dicts.
    """
    set_global_seed(42)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with Timer("exp5_total"):
        res_a = _run_part_a(quick, out_path)
        res_b = _run_part_b(quick, out_path)
        # Use Newton solution from part a for curvature analysis
        res_c = _run_part_c(quick, out_path, x_sol=res_a["newton_solution"])

    return {
        "part_a": res_a,
        "part_b": res_b,
        "part_c": res_c,
    }


if __name__ == "__main__":
    result = run(quick=False)
    print("Part a:")
    print(result["part_a"]["table"].to_string(index=False))
    print("\nPart b (params):")
    print(result["part_b"]["params_table"].to_string(index=False))
    print("\nPart c HVP errors:")
    print(result["part_c"]["hvp_table"].to_string(index=False))

