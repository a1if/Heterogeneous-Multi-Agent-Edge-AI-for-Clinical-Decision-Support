"""Tier-balanced adapter training examples (balance_by="tier"): same total as
true-class balancing, split evenly over the reference tiers, first qualifying
beats in chronological order. CPU only; the perception model is stubbed."""
import numpy as np

import reasoning.adapter_training as at


class FakeAgent:
    """Predicts a tier pattern keyed on the beat index: every 5th beat is urgent,
    every 7th (not 5th) is priority, the rest routine."""

    def __init__(self, checkpoint_path=None):
        self.last = None

    def reset_state(self):
        pass

    def predict(self, x, rr_interval_ms=None, event_seq=0):
        i = event_seq
        urgent, priority = i % 5 == 0, i % 7 == 0 and i % 5 != 0
        self.last = np.full(32, i, np.float32)
        return {"classification": {"label": "N" if not (urgent or priority) else "V"},
                "clinical_flags": {"requires_urgent_review": urgent}, "i": i}

    def get_last_context_vector(self):
        return self.last


def test_tier_quotas_and_order(tmp_path, monkeypatch):
    n = 400
    path = tmp_path / "ds1.npz"
    np.savez(path, features=np.zeros((n, 360), np.float32), labels=np.arange(n) % 4,
             rr_interval_ms=np.full(n, 800.0, np.float32), record_ids=np.zeros(n, int))
    monkeypatch.setattr(at, "PerceptionAgent", FakeAgent)

    by_class = at._build_real_training_examples_uncached(path, per_class=16)
    by_tier = at._build_real_training_examples_uncached(path, per_class=16, balance_by="tier")

    assert len(by_class) == len(by_tier) == 64
    tiers = [at.urgency_tier_from_event(e["health_event"]) for e in by_tier]
    assert {t: tiers.count(t) for t in set(tiers)} == {"routine": 22, "priority": 21, "urgent": 21}
    idx = [e["source_index"] for e in by_tier]
    assert idx == sorted(idx)  # chronological, first qualifying beats
    assert [e["source_index"] for e in by_tier if e["source_index"] % 5 == 0][:3] == [0, 5, 10]
