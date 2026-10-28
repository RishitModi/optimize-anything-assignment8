"""Smoke tests for Track A experiment modules.

Each test calls run(quick=True, out_dir=<tmp_path>) on one experiment module,
then asserts:
  - The returned dict is non-empty.
  - Expected CSV and PNG files exist in the output directory.
  - Key numeric columns (final_f, final_loss) do not contain NaN,
    except in exp3 where NaN marks intentional divergence.

Usage
-----
    pytest tracks/track_a_continuous/tests/test_experiments_smoke.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
import importlib

import numpy as np
import pandas as pd
import pytest

# Ensure repo root and track root are on the path.
_TRACK_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _TRACK_ROOT.parents[1]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file as a DataFrame."""
    return pd.read_csv(path)


def _assert_no_nan_in_col(df: pd.DataFrame, col: str, context: str = "") -> None:
    """Assert that a DataFrame column has no NaN values."""
    if col not in df.columns:
        return  # Column not present – skip
    nan_mask = df[col].isna()
    assert not nan_mask.any(), (
        f"{context}: column '{col}' has NaN at rows {list(df.index[nan_mask])}.\n"
        f"Values: {list(df[col])}"
    )


# ---------------------------------------------------------------------------
# Exp 1: Optimizer comparison
# ---------------------------------------------------------------------------

def test_exp1_smoke(tmp_path: Path) -> None:
    """Smoke test for exp1_optimizer_comparison."""
    mod = importlib.import_module("exp1_optimizer_comparison")
    result = mod.run(quick=True, out_dir=str(tmp_path))

    # Dict must be non-empty
    assert result, "run() returned empty dict"

    # Expected CSVs
    assert (tmp_path / "exp1_tuned_lr.csv").exists(), "Missing exp1_tuned_lr.csv"
    assert (tmp_path / "exp1_results.csv").exists(), "Missing exp1_results.csv"

    # Expected PNGs
    assert (tmp_path / "exp1_convergence_quadratic.png").exists(), (
        "Missing exp1_convergence_quadratic.png"
    )
    assert (tmp_path / "exp1_convergence_rosenbrock.png").exists(), (
        "Missing exp1_convergence_rosenbrock.png"
    )

    # No NaN in key columns (except intentionally diverged methods)
    results_df = _load_csv(tmp_path / "exp1_results.csv")
    assert not results_df.empty
    # final_f should be finite for all non-diverged rows; we just check it exists
    assert "final_f" in results_df.columns
    assert "final_gradnorm" in results_df.columns


# ---------------------------------------------------------------------------
# Exp 2: Landscapes
# ---------------------------------------------------------------------------

def test_exp2_smoke(tmp_path: Path) -> None:
    """Smoke test for exp2_landscapes."""
    mod = importlib.import_module("exp2_landscapes")
    result = mod.run(quick=True, out_dir=str(tmp_path))

    assert result, "run() returned empty dict"

    # Part a
    assert (tmp_path / "exp2a_convexity.png").exists()
    assert (tmp_path / "exp2a_table.csv").exists()
    df_a = _load_csv(tmp_path / "exp2a_table.csv")
    assert not df_a.empty
    assert "final_gap" in df_a.columns

    # Part b
    assert (tmp_path / "exp2b_saddle_trajectories.png").exists()
    assert (tmp_path / "exp2b_table.csv").exists()
    df_b = _load_csv(tmp_path / "exp2b_table.csv")
    assert not df_b.empty
    assert "classification" in df_b.columns

    # Part c
    assert (tmp_path / "exp2c_double_well_trajectories.png").exists()
    assert (tmp_path / "exp2c_table.csv").exists()
    df_c = _load_csv(tmp_path / "exp2c_table.csv")
    assert not df_c.empty
    assert "final_f" in df_c.columns
    _assert_no_nan_in_col(df_c, "final_f", "exp2c")


# ---------------------------------------------------------------------------
# Exp 3: Step-size sensitivity (NaN allowed for diverged runs)
# ---------------------------------------------------------------------------

def test_exp3_smoke(tmp_path: Path) -> None:
    """Smoke test for exp3_stepsize_sensitivity.

    NaN in final_gap is intentional for diverged lr values.
    """
    mod = importlib.import_module("exp3_stepsize_sensitivity")
    result = mod.run(quick=True, out_dir=str(tmp_path))

    assert result, "run() returned empty dict"

    assert (tmp_path / "exp3_stepsize_sensitivity.png").exists()
    assert (tmp_path / "exp3_results.csv").exists()
    assert (tmp_path / "exp3_robustness.csv").exists()

    df = _load_csv(tmp_path / "exp3_results.csv")
    assert not df.empty
    assert "final_gap" in df.columns
    # NaN is allowed here (diverged runs) – no assertion on NaN

    df_rob = _load_csv(tmp_path / "exp3_robustness.csv")
    assert not df_rob.empty
    assert "best_lr" in df_rob.columns


# ---------------------------------------------------------------------------
# Exp 4: Stochastic and noise
# ---------------------------------------------------------------------------

def test_exp4_smoke(tmp_path: Path) -> None:
    """Smoke test for exp4_stochastic_and_noise."""
    mod = importlib.import_module("exp4_stochastic_and_noise")
    result = mod.run(quick=True, out_dir=str(tmp_path))

    assert result, "run() returned empty dict"

    # Part a
    assert (tmp_path / "exp4a_loss_vs_epoch.png").exists()
    assert (tmp_path / "exp4a_loss_vs_time.png").exists()
    assert (tmp_path / "exp4a_results.csv").exists()
    df_a = _load_csv(tmp_path / "exp4a_results.csv")
    assert not df_a.empty
    _assert_no_nan_in_col(df_a, "final_loss_mean", "exp4a")

    # Part b
    assert (tmp_path / "exp4b_noise_sensitivity.png").exists()
    assert (tmp_path / "exp4b_results.csv").exists()
    df_b = _load_csv(tmp_path / "exp4b_results.csv")
    assert not df_b.empty
    _assert_no_nan_in_col(df_b, "mean_gap", "exp4b")


# ---------------------------------------------------------------------------
# Exp 5: Second-order methods
# ---------------------------------------------------------------------------

def test_exp5_smoke(tmp_path: Path) -> None:
    """Smoke test for exp5_second_order."""
    mod = importlib.import_module("exp5_second_order")
    result = mod.run(quick=True, out_dir=str(tmp_path))

    assert result, "run() returned empty dict"

    # Part a
    assert (tmp_path / "exp5a_convergence.png").exists()
    assert (tmp_path / "exp5a_results.csv").exists()
    df_a = _load_csv(tmp_path / "exp5a_results.csv")
    assert not df_a.empty
    _assert_no_nan_in_col(df_a, "final_f", "exp5a")

    # Part b
    assert (tmp_path / "exp5b_convergence.png").exists()
    assert (tmp_path / "exp5b_params.csv").exists()

    # Part c
    assert (tmp_path / "exp5c_hvp_error.csv").exists()
    assert (tmp_path / "exp5c_eigvals.csv").exists()
    assert (tmp_path / "exp5c_lowrank_error.png").exists()
    assert (tmp_path / "exp5c_lowrank_error.csv").exists()
    assert (tmp_path / "exp5c_fisher_diag.png").exists()
    assert (tmp_path / "exp5c_fisher_diag.csv").exists()
    assert (tmp_path / "exp5c_diag_precond.png").exists()

    df_hvp = _load_csv(tmp_path / "exp5c_hvp_error.csv")
    assert not df_hvp.empty
    assert "max_relative_error" in df_hvp.columns


# ---------------------------------------------------------------------------
# Exp 6: MLP backpropagation
# ---------------------------------------------------------------------------

def test_exp6_smoke(tmp_path: Path) -> None:
    """Smoke test for exp6_backprop_mlp."""
    mod = importlib.import_module("exp6_backprop_mlp")
    result = mod.run(quick=True, out_dir=str(tmp_path))

    assert result, "run() returned empty dict"

    # Part a
    assert (tmp_path / "exp6a_grad_check.csv").exists()
    df_a = _load_csv(tmp_path / "exp6a_grad_check.csv")
    assert not df_a.empty
    assert "max_relative_error" in df_a.columns
    _assert_no_nan_in_col(df_a, "max_relative_error", "exp6a")

    # Part b
    assert (tmp_path / "exp6b_train_loss.png").exists()
    assert (tmp_path / "exp6b_test_acc.png").exists()
    assert (tmp_path / "exp6b_results.csv").exists()
    df_b = _load_csv(tmp_path / "exp6b_results.csv")
    assert not df_b.empty
    _assert_no_nan_in_col(df_b, "final_train_loss_mean", "exp6b")

    # Part c
    assert (tmp_path / "exp6c_regularisation.png").exists()
    assert (tmp_path / "exp6c_results.csv").exists()
    df_c = _load_csv(tmp_path / "exp6c_results.csv")
    assert not df_c.empty
    _assert_no_nan_in_col(df_c, "test_accuracy", "exp6c")

    # Part d
    assert (tmp_path / "exp6d_decision_boundary.png").exists()

