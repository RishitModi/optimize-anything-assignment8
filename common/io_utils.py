"""Input/output utilities for storing and loading benchmark data in JSON and CSV formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


def _json_serial_default(obj: Any) -> Any:
    """Fallback JSON serializer for NumPy scalar types and objects."""
    # NumPy data types
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except (ImportError, Exception):
        pass

    if isinstance(obj, Path):
        return str(obj)

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> Path:
    """Serialize data to a JSON file, creating parent directories if necessary.

    Args:
        data: Python object or data structure to serialize.
        filepath: Target JSON file path.
        indent: Indentation level for pretty-printing. Defaults to 2.

    Returns:
        Path to the saved JSON file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_json_serial_default)

    return path


def load_json(filepath: Union[str, Path]) -> Any:
    """Load and parse data from a JSON file.

    Args:
        filepath: Target JSON file path.

    Returns:
        Parsed JSON structure (dict, list, etc.).
    """
    path = Path(filepath)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(
    data: Union[List[Dict[str, Any]], Sequence[Sequence[Any]], Any],
    filepath: Union[str, Path],
    fieldnames: Optional[Sequence[str]] = None,
) -> Path:
    """Save tabular records or a DataFrame to a CSV file.

    Supports:
        - List of dictionaries (fieldnames inferred from keys if not provided).
        - Sequence of rows (sequence of sequences) with optional header fieldnames.
        - Pandas DataFrame (if installed).

    Args:
        data: Tabular records to write.
        filepath: Target CSV file path.
        fieldnames: Optional column headers.

    Returns:
        Path to the saved CSV file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Check if data is a pandas DataFrame
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
            return path
    except (ImportError, Exception):
        pass

    if isinstance(data, list) and data and isinstance(data[0], dict):
        cols = list(fieldnames) if fieldnames is not None else list(data[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
    else:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if fieldnames is not None:
                writer.writerow(fieldnames)
            for row in data:
                writer.writerow(row)

    return path


def load_csv(
    filepath: Union[str, Path],
    as_dicts: bool = True,
) -> Union[List[Dict[str, str]], List[List[str]]]:
    """Read records from a CSV file.

    Args:
        filepath: Target CSV file path.
        as_dicts: If True, returns list of dicts (using header row as keys).
            If False, returns list of string rows. Defaults to True.

    Returns:
        List of row dicts or list of row lists.
    """
    path = Path(filepath)
    with path.open("r", encoding="utf-8", newline="") as f:
        if as_dicts:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
        else:
            raw_reader = csv.reader(f)
            return [row for row in raw_reader]
