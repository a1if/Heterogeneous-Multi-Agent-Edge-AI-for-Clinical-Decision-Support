"""Shared paths, selection constants, and small helpers used across the project.

WHY THIS EXISTS
---------------
Before this module, the same literals were re-typed in file after file:
``data/processed/ds2_test.npz`` in 19 places, ``perception/checkpoints/cnn_lstm.pt``
in 17, the headline adapter checkpoint in 10. Three helper functions were also
duplicated verbatim across file pairs: ``select_events``, ``measure_vram`` and
``wilson_ci``.

The ``select_events`` duplication was the one that actually mattered. Two
independent copies existed -- one in ``day6_run_comparison.py``, one in
``day7_auditability_probe.py`` -- and six other scripts imported the day7 copy
while day6 used its own. Every one of those scripts documents that it evaluates
"the same 80 events" as Day 6, but nothing enforced it: an edit to either copy
would have silently broken that guarantee with no error. Both copies were
verified to return byte-identical output before being merged here, so this
consolidation preserves current behaviour exactly while removing the drift risk.

SCOPE -- deliberately limited
-----------------------------
Only root-level scripts and ``diagnostics/`` import from here. The
``perception/`` and ``reasoning/`` packages are left alone: a package importing
a repo-root module would invert the dependency direction and break if those
packages were ever imported from outside this checkout. ``reasoning/baseline_arm.py``
and ``reasoning/prompt_template.py`` are additionally frozen by design (see their
docstrings) and are not touched.

``AAMI_CLASSES`` is deliberately NOT re-exported here. It already lives in
``perception.model``; importing it into this module would pull ``torch`` into
every importer, including scripts that document themselves as needing neither a
GPU nor a model load (e.g. ``check_s_class_headroom.py``). Files that need the
class list import it from ``perception.model`` directly -- still one source of
truth, without the dependency cost.

Import as ``from project_config import DS2_PATH, select_events`` -- this file
sits at the repo root alongside ``perception/`` and ``reasoning/``, so it
resolves for scripts run from the repo root and for ``diagnostics/`` scripts
(which already insert the repo root into ``sys.path``), exactly like those
packages do. See PROJECT_LAYOUT.md.
"""
import math
from collections import defaultdict

import numpy as np

__all__ = [
    "DS1_PATH", "DS2_PATH", "PERCEPTION_CHECKPOINT", "ADAPTER_CHECKPOINT",
    "DAY6_RESULTS", "LEDGER_PATH", "PER_CLASS", "MAX_PER_RECORD",
    "select_events", "measure_vram", "wilson_ci",
]

# --- Data and checkpoints ---------------------------------------------------
DS1_PATH = "data/processed/ds1_train.npz"
DS2_PATH = "data/processed/ds2_test.npz"
PERCEPTION_CHECKPOINT = "perception/checkpoints/cnn_lstm.pt"
ADAPTER_CHECKPOINT = "reasoning/checkpoints/virtual_adapter_day5_larger.pt"
DAY6_RESULTS = "results/day6_results.json"
LEDGER_PATH = "results_ledger.json"

# --- Evaluation-set selection ----------------------------------------------
# 20 per class x 4 classes (N/S/V/F; Q is absent from DS2) = the 80-event set.
PER_CLASS = 20
MAX_PER_RECORD = 5   # cap so one patient cannot dominate a class


def select_events(y, record_ids, per_class=PER_CLASS, max_per_record=MAX_PER_RECORD):
    """Deterministic first-occurrence selection of ``per_class`` events per class.

    This is THE definition of the 80-event evaluation set. Day 6, Day 7, the E1/E2
    ablations and the S-class checks must all score the identical events for their
    numbers to be comparable, which is why there is now exactly one copy.

    Selection is first-occurrence within a ``max_per_record``-per-patient cap --
    documented and reproducible, not random, so no seed is involved. Body is kept
    byte-faithful to the two implementations it replaces (same ``np.flatnonzero``
    ordering, same cap semantics); only the error message wording is unified.
    """
    selected = []
    for class_id in range(4):                      # N/S/V/F -- Q absent in DS2
        per_record_count = defaultdict(int)
        class_indices = np.flatnonzero(y == class_id)
        picked = []
        for idx in class_indices:
            rec = int(record_ids[idx])
            if per_record_count[rec] >= max_per_record:
                continue
            picked.append(int(idx))
            per_record_count[rec] += 1
            if len(picked) == per_class:
                break
        if len(picked) < per_class:
            from perception.model import AAMI_CLASSES   # lazy: keeps torch off the happy path
            raise RuntimeError(
                f"Class {AAMI_CLASSES[class_id]}: only found {len(picked)}/{per_class} "
                f"events under the {max_per_record}-per-record cap. "
                f"Loosen max_per_record or per_class."
            )
        selected.extend(picked)
    return selected


def measure_vram(fn, *args, **kwargs):
    """Runs fn, returns (result, peak_vram_mb) isolated to this call.

    ``torch`` is imported lazily so that importing this module stays cheap for the
    CPU-only scripts that only want the path constants.
    """
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = fn(*args, **kwargs)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
    return result, peak_mb


def wilson_ci(correct, total, z=1.96):
    """Wilson score 95% CI for a binomial proportion. Returns (lower, upper) as percentages."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return (100 * (center - margin), 100 * (center + margin))
