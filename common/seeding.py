"""Deterministic global seed management across scientific and ML libraries."""

from __future__ import annotations

import random
import sys
from typing import Optional


def set_global_seed(seed: int = 42) -> None:
    """Set the random seed across Python, NumPy, and optionally PyTorch.

    Seeds Python's built-in ``random`` module and NumPy unconditionally.

    For PyTorch: this module **never** imports torch itself.  If the caller
    has already imported torch (i.e. ``"torch"`` is present in
    ``sys.modules``), ``torch.manual_seed`` and
    ``torch.cuda.manual_seed_all`` are called inside a ``try/except`` so
    that any attribute errors are silently ignored.  If torch has not been
    imported by the caller, it is left untouched.

    Args:
        seed: The integer seed value to configure. Defaults to 42.

    Returns:
        None
    """
    # Standard library random
    random.seed(seed)

    # NumPy (if installed and loadable)
    try:
        import numpy as np

        np.random.seed(seed)
    except (ImportError, Exception):
        pass

    # PyTorch — only if already present in sys.modules (imported by caller).
    # This module never imports torch on its own.
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        try:
            torch_mod.manual_seed(seed)
            torch_mod.cuda.manual_seed_all(seed)
        except (AttributeError, Exception):
            pass
