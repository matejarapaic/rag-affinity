"""
Tracks which Google Drive files have already been processed,
using a local JSON file to persist state across runs.
"""

import json
import os
from typing import Optional

TRACKER_FILE = "processed_files.json"


def _load() -> dict:
    if not os.path.exists(TRACKER_FILE):
        return {}
    with open(TRACKER_FILE, "r") as f:
        return json.load(f)


def _save(data: dict) -> None:
    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2)


def is_processed(file_id: str, modified_time: str) -> bool:
    """Returns True if this file version has already been uploaded."""
    data = _load()
    return data.get(file_id) == modified_time


def mark_processed(file_id: str, modified_time: str) -> None:
    """Record that this file version has been successfully processed."""
    data = _load()
    data[file_id] = modified_time
    _save(data)


def get_last_modified(file_id: str) -> Optional[str]:
    return _load().get(file_id)
