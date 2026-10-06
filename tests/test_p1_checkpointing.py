"""Checkpoint / resume behaviour of the Phase 1 run scripts (CPU only, generation stubbed)."""
import json

import numpy as np

import ablation_common
from p1_io import save_json_atomic


def test_save_json_atomic_replaces_and_leaves_no_temp(tmp_path):
    path = tmp_path / "state.json"
    save_json_atomic(path, {"rows": [1]})
    save_json_atomic(path, {"rows": [1, 2]})
    assert json.loads(path.read_text(encoding="utf-8")) == {"rows": [1, 2]}
    assert not (tmp_path / "state.json.tmp").exists()


def _prepared(n):
    return [{"idx": i, "true_class": "N", "predicted_class": "N", "record_id": 100,
             "reference_tier": "routine", "health_event": {}, "context_vector": np.zeros(32, np.float32)}
            for i in range(n)]


def test_run_arm_b_eval_resumes_and_checkpoints_every_event(monkeypatch):
    calls = []

    def fake_arm(health_event, context_vector, model, processor, adapter):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("parse failure")  # a failed event must be checkpointed too
        return {"result": {"urgency_tier": "routine", "justification": "j", "referenced_guideline_fact": "f"},
                "prompt_tokens": 486, "output_tokens": 50, "generation_duration_ms": 1.0,
                "time_to_first_token_ms": 1.0, "parse_attempts": 1}

    monkeypatch.setattr(ablation_common, "run_adapter_arm_timed", fake_arm)
    monkeypatch.setattr(ablation_common, "measure_vram", lambda fn, *a, **k: (fn(*a, **k), 0.0))

    saved = []
    already = [{"idx": 0, "correct": True}]  # event 0 finished before the interruption
    rows = ablation_common.run_arm_b_eval(_prepared(4), None, None, None, label="t",
                                          resume_from=already, on_event=lambda r: saved.append(len(r)))
    assert len(calls) == 3                      # events 1-3 run, event 0 skipped
    assert [r["idx"] for r in rows] == [0, 1, 2, 3]
    assert saved == [2, 3, 4]                   # a checkpoint after every event, failure included
    assert rows[2]["generation_failed"] is True
