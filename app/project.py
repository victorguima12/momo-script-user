"""
Momo Script - Project save/load utilities (.mscript files).
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict


PROJECT_VERSION = 1


def save_project(path: str, state: Dict[str, Any]) -> None:
    """Serialize project state to a .mscript JSON file (atomic write)."""
    state["version"] = PROJECT_VERSION
    data = json.dumps(state, indent=2, ensure_ascii=False)

    # Atomic write: write to temp file in same directory, then rename
    dir_path = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        # On Windows, target must not exist for os.rename
        if os.path.exists(path):
            os.replace(tmp_path, path)
        else:
            os.rename(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_project(path: str) -> Dict[str, Any]:
    """Read a .mscript JSON file, validate version, return state dict."""
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)

    version = state.get("version", 0)
    if version < 1:
        raise ValueError(f"Unknown project version: {version}")

    return state


def autosave_dir() -> Path:
    """Return the autosave directory (~/.momo_script), creating it if needed."""
    d = Path.home() / ".momo_script"
    d.mkdir(exist_ok=True)
    return d


def autosave_path() -> str:
    """Return the path for the autosave file."""
    return str(autosave_dir() / "autosave.mscript")
