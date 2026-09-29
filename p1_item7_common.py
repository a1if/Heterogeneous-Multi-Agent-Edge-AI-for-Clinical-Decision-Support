"""Shared data plumbing for item 7 (Deviation 9): RR-encoder replay of a whole split,
cached on disk, plus the per-beat reference tiers derived from it.

A full chronological replay of DS1 or DS2 through PerceptionAgent takes several
minutes; the baseline, training and evaluation scripts all need the same events,
so the result is cached under cache/item7/, keyed by the split file's and the
encoder checkpoint's SHA-256 (any change to either invalidates the cache).
"""
import hashlib
import pickle
from pathlib import Path

import numpy as np

from perception.perception_agent import PerceptionAgent, replay_selected
from reasoning.training_targets import urgency_tier_from_event

RR_ENCODER = "perception/checkpoints/cnn_lstm_rr_seed0.pt"
SPLITS = {"ds1": "data/processed/ds1_train.npz", "ds2": "data/processed/ds2_test.npz"}
CACHE_DIR = Path("cache/item7")
RR_GAP_CAP_MS = 11999.0  # HealthEventJSON rejects heart rates < 5 bpm (RR > 12 s gaps)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_split(split):
    with np.load(SPLITS[split]) as z:
        data = {k: z[k] for k in ("features", "labels", "rr_interval_ms", "record_ids")}
    # DS1 stores 3 annotation gaps (> 12 s) as RR intervals; they are gaps, not beats.
    data["rr_interval_ms"] = np.minimum(data["rr_interval_ms"], RR_GAP_CAP_MS)
    return data


def replay_split(split, checkpoint=RR_ENCODER):
    """-> dict with events (list of HealthEvent dicts), vectors (n, 32), tiers (list),
    classes (predicted label per beat), record_ids, labels. Cached on disk."""
    key = f"{split}_{_sha(SPLITS[split])}_{_sha(checkpoint)}"
    path = CACHE_DIR / f"replay_{key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    data = load_split(split)
    n = len(data["labels"])
    agent = PerceptionAgent(checkpoint_path=checkpoint)
    out = replay_selected(agent, data["features"], data["rr_interval_ms"], data["record_ids"], list(range(n)))
    events = [out[i][0] for i in range(n)]
    result = {
        "events": events,
        "vectors": np.stack([out[i][1] for i in range(n)]).astype(np.float32),
        "tiers": [urgency_tier_from_event(e) for e in events],
        "classes": [e["classification"]["label"] for e in events],
        "record_ids": data["record_ids"],
        "labels": data["labels"],
        "checkpoint": checkpoint,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(result, f)
    tmp.replace(path)
    return result


# Deviation 18 (recipe r4): per-event side inputs appended to the 32-d vector.
SIDE_FEATURES = ("hr_z", "rr_z", "run_z")


def side_features(events):
    """(n, 3) float32: heart rate, RR interval and run length from the perception
    agent's events (the same fields the A-compact text shows), scaled to O(1)."""
    out = np.empty((len(events), 3), dtype=np.float32)
    for i, e in enumerate(events):
        sf, cf = e["signal_features"], e["clinical_flags"]
        out[i, 0] = np.clip((sf["heart_rate_bpm"] - 80.0) / 30.0, -3, 3)
        out[i, 1] = np.clip((sf["rr_interval_ms"] - 750.0) / 250.0, -3, 3)
        out[i, 2] = np.log1p(min(cf["consecutive_abnormal_beats"], 20)) / np.log(4.0)  # run 3 -> 1.0
    return out


def window_vectors(r, start, n, input_dim=32):
    """Adapter input for one window: the 32-d vectors, plus side inputs when input_dim == 35."""
    v = np.asarray(r["vectors"][start:start + n], dtype=np.float32)
    if input_dim == 32:
        return v
    return np.concatenate([v, side_features(r["events"][start:start + n])], axis=1)
