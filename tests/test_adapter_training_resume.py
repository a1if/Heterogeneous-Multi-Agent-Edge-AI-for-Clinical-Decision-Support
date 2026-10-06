"""train_adapter's mid-training checkpoints: a run interrupted and resumed must end
with exactly the weights and losses of an uninterrupted run. CPU only: Gemma and
the DS1 examples are replaced by a tiny deterministic stand-in."""
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn as nn

import reasoning.adapter_training as at
from reasoning.virtual_adapter import VirtualTokenAdapter

EMB, VOCAB, N_EXAMPLES = 8, 11, 12


class FakeModel(nn.Module):
    """Stands in for frozen Gemma: fixed embedding table and a fixed linear LM head."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(123)
        self.embed = nn.Embedding(VOCAB, EMB)
        self.embed.weight.data = torch.randn(VOCAB, EMB, generator=g)
        self.head = nn.Linear(EMB, VOCAB)
        self.head.weight.data = torch.randn(VOCAB, EMB, generator=g)
        self.head.bias.data.zero_()
        for p in self.parameters():
            p.requires_grad_(False)
        self.calls, self.fail_on_call = 0, None

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask=None, per_layer_inputs=None, use_cache=False):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise KeyboardInterrupt("simulated stop")
        return type("Out", (), {"logits": self.head(inputs_embeds)})()


@dataclass
class FakeInputs:
    inputs_embeds: torch.Tensor
    sequence_length: int


@pytest.fixture
def stub(monkeypatch):
    model = FakeModel()
    rng = np.random.default_rng(0)
    examples = [{"source_index": i, "context_vector": rng.normal(size=32).astype(np.float32),
                 "health_event": {"i": i}} for i in range(N_EXAMPLES)]
    monkeypatch.setattr(at, "build_real_training_examples", lambda *a, **k: examples)
    monkeypatch.setattr(at, "load_model", lambda: (model, None))
    monkeypatch.setattr(at, "freeze_language_model", lambda m: None)
    monkeypatch.setattr(at.VirtualTokenAdapter, "for_model",
                        classmethod(lambda cls, m, num_tokens=2: VirtualTokenAdapter(EMB, num_tokens=num_tokens)))
    monkeypatch.setattr(at, "prepare_adapter_inputs",
                        lambda m, p, adapter, ev, ctx: FakeInputs(adapter(ctx), adapter.num_tokens))
    monkeypatch.setattr(at, "_target_text", lambda ex, mode: "t")
    monkeypatch.setattr(at, "urgency_tier_from_event", lambda ev: "routine")
    monkeypatch.setattr(at, "_target_ids_and_weights",
                        lambda proc, text, tier, w: (torch.tensor([[1, 2, 3]]), torch.ones(1, 3)))

    def append(m, ai, target_ids):
        embeds = torch.cat((ai.inputs_embeds, m.embed(target_ids)), dim=1)
        return embeds, torch.ones(embeds.shape[:2], dtype=torch.long), None
    monkeypatch.setattr(at, "_append_target_for_teacher_forcing", append)
    monkeypatch.setattr(at, "RESUME_EVERY_STEPS", 4)
    return model


def config():
    return at.TrainingConfig(per_class=1, epochs=3, learning_rate=0.01, num_tokens=2, seed=7)


def weights(path):
    return torch.load(path, weights_only=False)


@pytest.mark.parametrize("fail_on_call", [7, 13])  # mid-epoch 1 (after the step-4 save) / first step of epoch 2
def test_resumed_run_matches_uninterrupted(stub, tmp_path, fail_on_call):
    ref = tmp_path / "ref.pt"
    at.train_adapter(config(), output_path=ref)

    out = tmp_path / "resumed.pt"
    stub.calls, stub.fail_on_call = 0, fail_on_call
    with pytest.raises(KeyboardInterrupt):
        at.train_adapter(config(), output_path=out)
    assert at.resume_path_for(out).exists() and not out.exists()

    stub.calls, stub.fail_on_call = 0, None
    at.train_adapter(config(), output_path=out)

    a, b = weights(ref), weights(out)
    for k in a["adapter_state_dict"]:
        assert torch.equal(a["adapter_state_dict"][k], b["adapter_state_dict"][k]), k
    assert a["losses"] == b["losses"]
    assert not at.resume_path_for(out).exists()  # cleaned up once training finishes


def test_resume_refuses_a_different_config(stub, tmp_path):
    out = tmp_path / "x.pt"
    stub.fail_on_call = 7
    with pytest.raises(KeyboardInterrupt):
        at.train_adapter(config(), output_path=out)
    stub.calls, stub.fail_on_call = 0, None
    other = at.TrainingConfig(per_class=1, epochs=3, learning_rate=0.02, num_tokens=2, seed=7)
    with pytest.raises(RuntimeError, match="different config"):
        at.train_adapter(other, output_path=out)
