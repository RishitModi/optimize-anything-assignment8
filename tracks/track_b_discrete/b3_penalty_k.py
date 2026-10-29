"""Track B Addendum: Feature selection QUBO penalty sweep with target cardinality k = 7.

Solves the QUBO formulation of feature selection across varying penalty multipliers lambda,
demonstrating enforcement of the cardinality constraint when the unconstrained optimum
differs from target cardinality k = 7.

Outputs:
    - results/track_b/b3_penalty_k7.csv
    - results/track_b/b3_penalty_k7.png
    - results/track_b/b3_penalty_k7.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure repository root and track root are on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _TRACK_ROOT.parents[1]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from common.io_utils import save_csv
from qubo import brute_force_qubo, feature_selection_qubo


def df_to_markdown(df: pd.DataFrame) -> str:
    """Render a pandas DataFrame as a GitHub-Flavored Markdown table without external dependencies."""
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    rows: list[str] = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            val = row[c]
            if isinstance(val, (float, np.floating)):
                cells.append(f"{val:.4g}")
            elif val is None or pd.isna(val):
                cells.append("")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_line, sep_line] + rows)


def main() -> None:
    """Execute feature selection QUBO cardinality penalty sweep for k = 7."""
    out_dir = _REPO_ROOT / "results" / "track_b"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset loading and standardization
    data = load_breast_cancer()
    X_full = data.data
    y = data.target
    X_scaled = StandardScaler().fit_transform(X_full)

    # 2. Select 12 features deterministically
    rng_pick = np.random.default_rng(0)
    chosen_indices = sorted(rng_pick.choice(30, 12, replace=False).tolist())
    X_sub = X_scaled[:, chosen_indices]

    # 3. Compute relevance and redundancy matrices
    relevance = np.zeros(12, dtype=np.float64)
    for i in range(12):
        relevance[i] = abs(float(np.corrcoef(X_sub[:, i], y)[0, 1]))
    redundancy = np.abs(np.corrcoef(X_sub, rowvar=False))

    k = 7
    alpha = 0.5
    lam_values = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    def compute_raw_objective(x_vec: np.ndarray) -> float:
        """Compute objective value without the penalty term: -sum r_i x_i + alpha * sum_{i<j} R_ij x_i x_j."""
        obj = -float(np.sum(relevance * x_vec))
        quad = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                quad += redundancy[i, j] * x_vec[i] * x_vec[j]
        return obj + alpha * quad

    records: list[dict[str, Any]] = []

    # 4. Sweep penalty multiplier lambda
    for lam in lam_values:
        Q, offset = feature_selection_qubo(relevance, redundancy, k=k, alpha=alpha, lam=lam)
        bf_res = brute_force_qubo(Q, offset=offset)
        x_opt = bf_res["best_x"]

        selected_indices = [i for i in range(12) if x_opt[i] == 1]
        n_selected = len(selected_indices)
        equals_k = (n_selected == k)
        raw_obj = compute_raw_objective(x_opt)

        records.append({
            "lam": lam,
            "selected_count": n_selected,
            "equals_k": equals_k,
            "selected_indices": str(selected_indices),
            "objective": raw_obj,
        })

    df = pd.DataFrame(records)

    # Save CSV
    csv_path = out_dir / "b3_penalty_k7.csv"
    save_csv(df, csv_path)

    # Save Markdown table
    md_content = "# Feature Selection Penalty Sweep (k = 7)\n\n" + df_to_markdown(df) + "\n"
    md_path = out_dir / "b3_penalty_k7.md"
    md_path.write_text(md_content, encoding="utf-8")

    # Save plot with log/symlog scale for lambda
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df["lam"],
        df["selected_count"],
        "o-",
        color="#2b5c8f",
        linewidth=2,
        markersize=6,
        label="Selected Features Count",
    )
    ax.axhline(k, color="red", linestyle="--", linewidth=1.5, label=f"Target Cardinality (k = {k})")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("Penalty Multiplier lambda (symlog scale)", fontsize=10)
    ax.set_ylabel("Selected Feature Count", fontsize=10)
    ax.set_title("Feature Selection Cardinality vs Penalty Multiplier (Target k = 7)", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()

    png_path = out_dir / "b3_penalty_k7.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

