"""run_all.py – Run all Track A experiments in order.

Usage
-----
    python tracks/track_a_continuous/run_all.py [--quick] [--only exp1 ...] [--out DIR]

Arguments
---------
--quick         Run each experiment in quick mode (reduced budget, finishes in <30 s).
--only          Space-separated list of experiment names to run, e.g. ``exp1 exp3``.
--out           Output directory (default: ``results/track_a``).

Exit code 1 if any experiment fails; 0 if all succeed.

At the end, writes ``results/track_a/SUMMARY.md`` with environment info, setup
descriptions and tables for every experiment.
"""

from __future__ import annotations

import argparse
import importlib
import io
import os
import platform
import sys
import time
import traceback
from pathlib import Path

# Add repo root and track root to path.
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _TRACK_ROOT.parents[1]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import scipy
import sklearn


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

EXPERIMENTS = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]

EXPERIMENT_DESCRIPTIONS = {
    "exp1": (
        "**Optimizer Comparison.** "
        "Problems: quadratic(cond=100, dim=10) and rosenbrock(dim=2, x0=(-1.2,1)). "
        "Methods: GradientDescent, Momentum, NAG, AdaGrad, RMSProp, Adam (all first-order) "
        "and Newton (second-order, line_search=True for Rosenbrock). "
        "Budget: 2000 iters (quadratic), 20000 iters (Rosenbrock) in full mode. "
        "Learning rate grid: np.logspace(-4, 0, 9) per method per problem; "
        "best lr selected by lowest final f among non-diverged runs. "
        "Momentum beta=0.9, Adam betas=(0.9,0.999), RMSProp rho=0.9. "
        "No seeds (deterministic problems). "
        "Outputs: convergence plots (log y) and results table."
    ),
    "exp2": (
        "**Landscape Analysis.** "
        "(a) GradientDescent on three convexity classes: strongly convex quadratic(cond=10), "
        "convex-not-SC sum_quartic(dim=5), non-convex Rosenbrock. Log-scale convergence plot; "
        "slope of log(gap) fitted on second half. "
        "(b) Saddle function: GD, Momentum, Adam, Newton (undamped and damped) from a small "
        "perturbation (1e-3, 1e-3); 2-D contour plot with trajectories; "
        "stationary-point classification via Hessian eigenvalues. "
        "(c) Double-well: GD and Adam from three starting points; contour plot with trajectories; "
        "table of final points and whether they reach the global minimum (checked via grid search)."
    ),
    "exp3": (
        "**Step-Size Sensitivity.** "
        "Problem: quadratic(cond=50, dim=10). "
        "Methods: GradientDescent, Momentum, NAG, AdaGrad, RMSProp, Adam. "
        "Budget: 500 iterations (fixed). "
        "LR sweep: np.logspace(-4, 0.5, 20) in full mode. "
        "Diverged runs recorded as NaN. "
        "Robustness metric: width of lr range (in decades) where final gap < 1e-3."
    ),
    "exp4": (
        "**Stochastic and Noisy Optimization.** "
        "(a) Mini-batch study: logistic regression (n=2000, d=20, l2=1e-2). "
        "Configurations: GD (full batch), SGD-256, SGD-32, SGD-1, Momentum-32, Adam-32. "
        "30 epochs, lr tuned per configuration from np.logspace(-3,0,7), 5 seeds. "
        "Plots: full-batch loss vs epoch and vs wall-clock time (mean ± std band). "
        "(b) Noisy objective: quadratic(cond=20, dim=10) with gradient noise sigma in "
        "[0, 0.1, 0.5, 1.0]; methods: GD (lr=1/L), Momentum, Adam, GD-Decay "
        "(lr_t = lr0/(1+0.01*t)); 10 seeds, 1000 iters. "
        "Plot: final gap vs sigma (log y) with std bars."
    ),
    "exp5": (
        "**Second-Order Methods and Hessian Analysis.** "
        "(a) Newton (line search) vs GD on logistic regression (n=2000, d=20, l2=1e-2); "
        "f_star from a long Newton run; convergence plot (log y). "
        "(b) Gauss-Newton, Levenberg-Marquardt, GD, Adam on make_exp_decay_fit from a "
        "poor start (a=1, b=2, c=0); convergence plot and parameter recovery table. "
        "(c) Curvature approximations at the Newton solution: "
        "HVP finite-difference vs exact (5 vectors, max relative error); "
        "top-5 eigenvalues power iteration vs numpy.linalg.eigh; "
        "low-rank Frobenius error for k in {1,2,5,10,20}; "
        "empirical Fisher diagonal vs Hessian diagonal (bar chart + correlation); "
        "diagonal-preconditioned GD vs plain GD (same budget, convergence plot)."
    ),
    "exp6": (
        "**MLP Backpropagation.** "
        "Dataset: make_moons(n=1000, noise=0.25, seed=0), 70/30 split, standardised, "
        "TwoLayerMLP with 16 hidden units. "
        "(a) Numerical gradient check – max relative error CSV. "
        "(b) GD (full batch), SGD-32, Momentum-32, Adam-32; 100 epochs; "
        "lr tuned over a small grid; 5 seeds; "
        "plots: train loss vs epoch and test accuracy vs epoch. "
        "(c) Adam with l2 in {0, 1e-3, 1e-2, 1e-1, 1.0}; "
        "table and 4-panel plot of train loss, test loss, test accuracy, param norm vs l2. "
        "(d) Decision boundary of Adam model (l2=1e-2) overlaid on data."
    ),
}


# ---------------------------------------------------------------------------
# Small markdown table renderer (no tabulate dependency)
# ---------------------------------------------------------------------------

def _df_to_markdown(df: "pd.DataFrame") -> str:
    """Render a pandas DataFrame as a GitHub-Flavoured Markdown table.

    Args:
        df: DataFrame to render.

    Returns:
        Multi-line Markdown table string.
    """
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.6g}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# SUMMARY.md writer
# ---------------------------------------------------------------------------

def _write_summary(
    out_dir: Path,
    experiment_results: dict,
    quick: bool,
) -> Path:
    """Write SUMMARY.md from actual run data.

    Args:
        out_dir: Output directory.
        experiment_results: dict mapping exp_name -> (result_dict, elapsed_s, success).
        quick: Whether quick mode was used.

    Returns:
        Path to the written SUMMARY.md.
    """
    lines = []

    # ---- Environment ----
    lines.append("# Track A Continuous Optimization – Summary\n")
    lines.append("## Environment\n")
    lines.append(f"- **Python version**: {sys.version}")
    lines.append(f"- **NumPy version**: {np.__version__}")
    lines.append(f"- **SciPy version**: {scipy.__version__}")
    lines.append(f"- **scikit-learn version**: {sklearn.__version__}")
    lines.append(f"- **Platform**: {platform.platform()}")
    lines.append(f"- **CPU count**: {os.cpu_count()}")
    lines.append(f"- **Quick mode**: {quick}")
    lines.append(f"- **Fixed base seed**: 42")
    lines.append(f"- **Stochastic experiments seeds**: 5 (2 in quick mode)\n")

    # ---- Per-experiment sections ----
    for exp_name in EXPERIMENTS:
        if exp_name not in experiment_results:
            continue
        result, elapsed, success = experiment_results[exp_name]
        lines.append(f"---\n\n## {exp_name.upper()}\n")
        lines.append(f"**Status**: {'SUCCESS' if success else 'FAILED'}")
        lines.append(f"**Runtime**: {elapsed:.2f} s\n")
        lines.append(f"**Setup**: {EXPERIMENT_DESCRIPTIONS.get(exp_name, '')}\n")

        if not success or result is None:
            lines.append("*(Experiment failed – no output available.)*\n")
            continue

        # Render tables found in result
        def _render_tables(d, prefix=""):
            sub_lines = []
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, pd.DataFrame) and not v.empty:
                        sub_lines.append(f"\n### {prefix}{k}\n")
                        sub_lines.append(_df_to_markdown(v) + "\n")
                    elif isinstance(v, dict):
                        sub_lines.extend(_render_tables(v, prefix=f"{k}/"))
            return sub_lines

        lines.extend(_render_tables(result))

        # List figures
        fig_paths = []
        def _collect_figs(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, str) and v.endswith(".png"):
                        fig_paths.append(Path(v).name)
                    elif isinstance(v, dict):
                        _collect_figs(v)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and item.endswith(".png"):
                                fig_paths.append(Path(item).name)
        _collect_figs(result)
        if fig_paths:
            lines.append("\n**Figures**: " + ", ".join(fig_paths) + "\n")

    summary_path = out_dir / "SUMMARY.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track A continuous optimization experiments.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: reduced budgets, finishes in <30 s.")
    parser.add_argument("--only", nargs="+", default=None,
                        metavar="EXP",
                        help="Run only named experiments, e.g. --only exp1 exp3.")
    parser.add_argument("--out", default="results/track_a",
                        help="Output directory (default: results/track_a).")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    to_run = args.only if args.only else EXPERIMENTS

    experiment_results: dict = {}
    any_failed = False

    print(f"\n{'='*60}")
    print(f"  Track A – Continuous Optimization")
    print(f"  Quick={args.quick}  Out={out_dir}")
    print(f"{'='*60}\n")

    for exp_name in EXPERIMENTS:
        if exp_name not in to_run:
            continue

        # Map exp_name -> module filename stem
        module_map = {
            "exp1": "exp1_optimizer_comparison",
            "exp2": "exp2_landscapes",
            "exp3": "exp3_stepsize_sensitivity",
            "exp4": "exp4_stochastic_and_noise",
            "exp5": "exp5_second_order",
            "exp6": "exp6_backprop_mlp",
        }
        module_file = module_map[exp_name]

        t0 = time.perf_counter()
        result = None
        success = False
        try:
            mod = importlib.import_module(module_file)
            result = mod.run(quick=args.quick, out_dir=str(out_dir))
            success = True
            elapsed = time.perf_counter() - t0
            print(f"  [OK]  {exp_name:6s}  {elapsed:7.2f} s")
        except Exception:
            elapsed = time.perf_counter() - t0
            any_failed = True
            print(f"  [FAIL]{exp_name:6s}  {elapsed:7.2f} s")
            traceback.print_exc()

        experiment_results[exp_name] = (result, elapsed, success)

    # Write SUMMARY.md
    summary_path = _write_summary(out_dir, experiment_results, quick=args.quick)
    print(f"\nSummary written to: {summary_path}")

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
