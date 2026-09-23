"""Crash-safe result saving for the Phase 1 run scripts.

Results files double as resume checkpoints, so a process killed mid-write must
never leave a truncated file behind. The JSON is written to a temporary file in
the same directory and then moved over the target with os.replace, which is
atomic on the same filesystem: the file on disk is always either the previous
complete checkpoint or the new one.
"""
import json
import os
from pathlib import Path


def save_json_atomic(path, obj) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
