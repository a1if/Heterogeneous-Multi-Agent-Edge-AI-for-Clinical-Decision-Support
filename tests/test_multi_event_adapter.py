"""Item 7: MultiEventVirtualAdapter and the window target (CPU only)."""
import json

import pytest
import torch
import torch.nn as nn

from reasoning.multi_event_adapter import MultiEventVirtualAdapter
from reasoning.output_schema import ReasoningOutput
from reasoning.training_targets import canonical_window_target, most_urgent_index


def test_shape_norm_and_slot_positions():
    torch.manual_seed(0)
    a = MultiEventVirtualAdapter(16, num_tokens=4, max_events=20, init_scale=3.0)
    ctx = torch.randn(7, 32)
    out = a(ctx)
    assert out.shape == (1, 28, 16)
    assert torch.allclose(out.norm(dim=-1), torch.full((1, 28), 3.0), atol=1e-4)  # every token at the scale
    # identical events in different slots differ only through their position embeddings
    same = a(torch.stack([ctx[0], ctx[0]]))
    assert not torch.allclose(same[0, :4], same[0, 4:8])


def test_slot_depends_only_on_its_own_event():
    torch.manual_seed(0)
    a = MultiEventVirtualAdapter(16, num_tokens=2, init_scale=1.0)
    ctx = torch.randn(5, 32)
    changed = ctx.clone()
    changed[3] += 10.0
    base, new = a(ctx), a(changed)
    assert torch.allclose(base[0, :6], new[0, :6])          # slots 0-2 untouched
    assert not torch.allclose(base[0, 6:8], new[0, 6:8])    # slot 3 moved


def test_too_many_events_rejected():
    with pytest.raises(ValueError):
        MultiEventVirtualAdapter(8, max_events=3)(torch.randn(4, 32))


def test_for_model_scale_uses_embedding_outputs():
    class Scaled(nn.Embedding):
        def forward(self, ids):
            return super().forward(ids) * 10.0  # like Gemma's scaled word embeddings

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.e = Scaled(100, 8)
            nn.init.normal_(self.e.weight)

        def get_input_embeddings(self):
            return self.e

    m = M()
    a = MultiEventVirtualAdapter.for_model(m, num_tokens=2, probe_tokens=512)
    raw = float(m.e.weight.norm(dim=-1).median())
    assert a.embedding_dim == 8
    assert a.scale.item() == pytest.approx(10 * raw, rel=0.2)


def ev(label, conf=0.9, run=1, urgent=False):
    return {"classification": {"label": label, "confidence": conf},
            "clinical_flags": {"requires_urgent_review": urgent, "consecutive_abnormal_beats": run}}


def test_window_target_names_most_urgent_beat_and_parses():
    events = [ev("N"), ev("S"), ev("V", urgent=True), ev("V", urgent=True), ev("N")]
    assert most_urgent_index(events) == 2  # earliest of the tied urgent beats
    t = json.loads(canonical_window_target(events))
    assert t["urgency_tier"] == "urgent" and t["justification"].startswith("Beat 3 of 5")
    ReasoningOutput(**t)
    assert json.loads(canonical_window_target([ev("N"), ev("N")]))["urgency_tier"] == "routine"
    assert json.loads(canonical_window_target([ev("N"), ev("S")]))["urgency_tier"] == "priority"
