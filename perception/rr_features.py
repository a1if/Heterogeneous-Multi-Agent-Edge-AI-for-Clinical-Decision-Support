"""RR-interval features for the RR-branch encoder (Phase 1, encoder sweep).

Supraventricular ectopic (S) beats usually look like normal beats. What sets them
apart is timing: a short pre-RR, a longer post-RR, relative to the patient's own
rhythm. The reference CNN-LSTM sees one 1-s window and no RR input, which is the
main reason its DS2 S recall is 8.2%. These are the de Chazal-style interval
features, normalised against the patient's local rhythm so they carry across
patients (inter-patient split).

Features per beat, computed within each record only (never across records):
    pre_s       pre-RR (s)
    post_s      post-RR (s): the next beat's pre-RR. Needs the next beat, so an
                online agent emits each beat one beat late. The last beat in a
                record falls back to its pre-RR.
    local_s     mean pre-RR of the previous LOCAL_WINDOW beats (causal, excludes
                the current beat); falls back to pre-RR for a record's first beat
    pre_ratio   pre_s / local_s
    post_ratio  post_s / local_s

RR values are clipped to [RR_MIN_MS, RR_MAX_MS] first: DS1 contains gaps up to
~100 s where annotations or edge windows were skipped (data_prep.py).
"""
import numpy as np

RR_MIN_MS = 200.0
RR_MAX_MS = 3000.0
LOCAL_WINDOW = 10
FEATURE_NAMES = ("pre_s", "post_s", "local_s", "pre_ratio", "post_ratio")
N_RR_FEATURES = len(FEATURE_NAMES)


def compute_rr_features(rr_interval_ms: np.ndarray, record_ids: np.ndarray) -> np.ndarray:
    """(n,) pre-RR in ms + (n,) record ids, in chronological order within each
    record -> (n, 5) float32 features. Records must be contiguous."""
    rr = np.clip(np.asarray(rr_interval_ms, dtype=np.float64), RR_MIN_MS, RR_MAX_MS) / 1000.0
    record_ids = np.asarray(record_ids)
    boundaries = np.flatnonzero(np.diff(record_ids) != 0) + 1
    if len(np.unique(record_ids)) != len(boundaries) + 1:
        raise ValueError("record_ids must be contiguous (one run per record)")

    out = np.empty((len(rr), N_RR_FEATURES), dtype=np.float32)
    for seg in np.split(np.arange(len(rr)), boundaries):
        pre = rr[seg]
        post = np.append(pre[1:], pre[-1])
        csum = np.concatenate([[0.0], np.cumsum(pre)])
        local = np.empty_like(pre)
        for j in range(len(pre)):
            lo = max(0, j - LOCAL_WINDOW)
            local[j] = (csum[j] - csum[lo]) / (j - lo) if j > lo else pre[j]
        out[seg] = np.stack([pre, post, local, pre / local, post / local], axis=1)
    return out


def fit_standardizer(features: np.ndarray) -> dict:
    """Per-feature mean/std from training data only; saved with the checkpoint."""
    return {"mean": features.mean(axis=0).tolist(), "std": (features.std(axis=0) + 1e-6).tolist()}


def standardize(features: np.ndarray, stats: dict) -> np.ndarray:
    return ((features - np.asarray(stats["mean"])) / np.asarray(stats["std"])).astype(np.float32)
