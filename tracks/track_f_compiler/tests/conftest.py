"""Pytest configuration and sys.path setup for Track F test suite."""

from __future__ import annotations

import sys
from pathlib import Path

_TRACK_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]

for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

