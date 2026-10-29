"""Pytest configuration: add repository root and track root to sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

# Repository root (two levels up from tracks/track_b_discrete/tests/)
_REPO_ROOT = Path(__file__).resolve().parents[3]
# Track root (one level up from tests/)
_TRACK_ROOT = Path(__file__).resolve().parents[1]

for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

