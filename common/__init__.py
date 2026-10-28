"""Shared utilities package for optimize-anything-assignment8.

Provides reusable components across optimization tracks:
- Deterministic random seed management (`common.seeding`)
- Wall-clock benchmarking and profiling (`common.timing`)
- Standardized scientific plotting (`common.plotting`)
- Multi-seed experiment aggregation and statistics (`common.stats`)
- JSON and CSV persistence utilities (`common.io_utils`)
"""

from common.seeding import set_global_seed
from common.timing import Timer, timed
from common.plotting import save_line_plot, save_bar_chart
from common.stats import run_multiseed_experiment, calculate_summary_stats
from common.io_utils import save_json, load_json, save_csv, load_csv

__all__ = [
    "set_global_seed",
    "Timer",
    "timed",
    "save_line_plot",
    "save_bar_chart",
    "run_multiseed_experiment",
    "calculate_summary_stats",
    "save_json",
    "load_json",
    "save_csv",
    "load_csv",
]
