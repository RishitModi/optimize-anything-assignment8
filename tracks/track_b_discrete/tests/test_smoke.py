"""Smoke test verifying execution of run_all in quick mode and output artifact generation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from run_all import main, run_all


def test_smoke_run_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute run_all in quick mode into tmp_path and assert all required artifacts exist."""
    out_dir = tmp_path / "track_b_results"
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--quick", "--out", str(out_dir)])

    # Execute main runner
    main()

    # Verify CSV artifact creation
    expected_csvs = [
        "b1_knapsack_runs.csv",
        "b1_knapsack_summary.csv",
        "b2_maxcut_summary.csv",
        "b3_penalty_sweep.csv",
        "b3_method_comparison.csv",
    ]
    for csv_name in expected_csvs:
        csv_file = out_dir / csv_name
        assert csv_file.exists(), f"Expected CSV file not found: {csv_name}"
        assert csv_file.stat().st_size > 0, f"CSV file is empty: {csv_name}"

    # Verify PNG plot generation
    expected_pngs = [
        "b1_runtime_vs_n.png",
        "b1_greedy_gap_vs_n.png",
        "b2_runtime_vs_n.png",
        "b2_cut_vs_n.png",
        "b3_penalty_sweep.png",
    ]
    for png_name in expected_pngs:
        png_file = out_dir / png_name
        assert png_file.exists(), f"Expected PNG figure not found: {png_name}"
        assert png_file.stat().st_size > 0, f"PNG file is empty: {png_name}"

    # Verify SUMMARY.md report
    summary_path = out_dir / "SUMMARY.md"
    assert summary_path.exists(), "Expected SUMMARY.md not found"
    assert summary_path.stat().st_size > 0, "SUMMARY.md is empty"

    # Verify qubo_instances.json export
    instances_path = out_dir / "qubo_instances.json"
    assert instances_path.exists(), "Expected qubo_instances.json not found"
    assert instances_path.stat().st_size > 0, "qubo_instances.json is empty"

