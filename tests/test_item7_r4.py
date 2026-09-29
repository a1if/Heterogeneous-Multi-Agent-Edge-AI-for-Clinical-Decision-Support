"""Recipe r4 (Deviation 18): side inputs and hard-negative windows (CPU only)."""
import numpy as np
import torch

import p1_item7_train as t7
from p1_item7_common import side_features, window_vectors
from reasoning.multi_event_adapter import MultiEventVirtualAdapter


def ev(label="N", conf=0.99, hr=80.0, rr=750.0, run=0):
    return {"classification": {"label": label, "confidence": conf},
            "signal_features": {"heart_rate_bpm": hr, "rr_interval_ms": rr},
            "clinical_flags": {"consecutive_abnormal_beats": run, "requires_urgent_review": False}}


def test_side_features_scaling():
    f = side_features([ev(), ev(hr=170, rr=5000, run=3), ev(run=100)])
    assert f.shape == (3, 3) and f.dtype == np.float32
    assert np.allclose(f[0], [0.0, 0.0, 0.0])
    assert np.isclose(f[1, 0], 3.0) and np.isclose(f[1, 1], 3.0)   # clipped
    assert np.isclose(f[1, 2], 1.0)                                  # run 3 -> 1.0
    assert np.isclose(f[2, 2], np.log1p(20) / np.log(4.0))           # run capped at 20


def test_window_vectors_dims():
    r = {"vectors": np.random.default_rng(0).standard_normal((10, 32)).astype(np.float32),
         "events": [ev(run=i) for i in range(10)]}
    assert window_vectors(r, 2, 5, 32).shape == (5, 32)
    v = window_vectors(r, 2, 5, 35)
    assert v.shape == (5, 35) and np.allclose(v[:, :32], r["vectors"][2:7])


def test_adapter_accepts_35d_input():
    a = MultiEventVirtualAdapter(16, num_tokens=4, input_dim=35, max_events=50, init_scale=2.0)
    out = a(torch.randn(50, 35))
    assert out.shape == (1, 200, 16)


def test_hard_negatives_are_routine_low_confidence_and_capped():
    rng = np.random.default_rng(0)
    n_beats = 400
    record_ids = np.repeat([1, 2, 3, 4], 100)
    conf = np.where(rng.random(n_beats) < 0.05, 0.6, 0.99)
    tiers = ["routine"] * n_beats
    for i in (50, 150):
        tiers[i] = "priority"
    r = {"record_ids": record_ids, "tiers": tiers, "events": [ev(conf=c) for c in conf]}
    hn = t7.hard_negative_windows(r, {1, 2, 3, 4}, per_n=10, rng=np.random.default_rng(2), ns=(1, 5), max_per_record=3)
    assert hn and all(w["hard_negative"] and w["tier"] == "routine" for w in hn)
    for w in hn:
        idx = range(w["start"], w["start"] + w["n"])
        assert all(tiers[i] == "routine" for i in idx) and min(conf[i] for i in idx) < 0.8
    for n in (1, 5):
        per_rec = {}
        for w in hn:
            if w["n"] == n:
                per_rec[int(record_ids[w["start"]])] = per_rec.get(int(record_ids[w["start"]]), 0) + 1
        assert max(per_rec.values()) <= 3
    excluded = t7.hard_negative_windows(r, {1, 2, 3, 4}, 10, np.random.default_rng(2), (1,), exclude={(w["n"], w["start"]) for w in hn})
    assert not {(w["n"], w["start"]) for w in excluded} & {(w["n"], w["start"]) for w in hn}
