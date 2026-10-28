"""Deterministic global seed management across scientific and ML libraries."""

from __future__ import annotations

import os
import random
import sys
from typing import Optional


def set_global_seed(seed: int = 42) -> None:
    """Set the random seed across Python, NumPy, and PyTorch (if installed).

    Ensures repeatable stochastic behavior for benchmarks, initialization,
    and optimization trajectory comparisons.

    Args:
        seed: The integer seed value to configure. Defaults to 42.

    Returns:
        None
    """
    # Python environment hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Standard library random
    random.seed(seed)

    # NumPy (if installed and loadable)
    try:
        import numpy as np

        np.random.seed(seed)
    except (ImportError, Exception):
        pass

    # PyTorch (if imported or installed)
    torch_mod = sys.modules.get("torch")
    if torch_mod is None and os.environ.get("DISABLE_TORCH_SEEDING", "").lower() not in ("1", "true", "yes"):
        try:
            import torch as torch_mod
        except (ImportError, Exception):
            torch_mod = None

    if torch_mod is not None:
        try:
            torch_mod.manual_seed(seed)
            if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                torch_mod.cuda.manual_seed_all(seed)
                torch_mod.backends.cudnn.deterministic = True
                torch_mod.backends.cudnn.benchmark = False
        except (AttributeError, Exception):
            pass
