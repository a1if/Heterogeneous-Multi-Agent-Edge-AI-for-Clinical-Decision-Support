"""Day 3 adapter unit tests: no Gemma download/GPU execution required."""
import torch
from torch import nn

from reasoning.virtual_adapter import (
    NUM_VIRTUAL_TOKENS,
    PERCEPTION_FEATURE_DIM,
    VirtualTokenAdapter,
    build_adapter_prompt_parts,
    compose_adapter_inputs,
)


class _TinyLanguageModel(nn.Module):
    def __init__(self, embedding_dim=12):
        super().__init__()
        self.embedding = nn.Embedding(64, embedding_dim)
        self.model = _TinyGemmaModel()
        self.config = type("Config", (), {"text_config": type("TextConfig", (), {"pad_token_id": 0})()})()

    def get_input_embeddings(self):
        return self.embedding


class _TinyPerLayerInputs(nn.Module):
    def get_per_layer_inputs(self, input_ids, _inputs_embeds):
        # The production Gemma method returns (batch, seq, layers, width).
        return input_ids.to(torch.float32).unsqueeze(-1).unsqueeze(-1)


class _TinyGemmaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _TinyPerLayerInputs()


def test_adapter_projects_32d_context_to_four_native_embedding_tokens():
    adapter = VirtualTokenAdapter(embedding_dim=2560)
    output = adapter(torch.zeros(3, PERCEPTION_FEATURE_DIM))
    assert output.shape == (3, NUM_VIRTUAL_TOKENS, 2560)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 337_920


def test_adapter_rejects_incorrect_context_width():
    adapter = VirtualTokenAdapter(embedding_dim=12)
    try:
        adapter(torch.zeros(1, 31))
    except ValueError as exc:
        assert "32" in str(exc)
    else:
        raise AssertionError("Expected an invalid context width to fail")


def test_composition_replaces_payload_with_four_tokens_and_preserves_gradients():
    model = _TinyLanguageModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = VirtualTokenAdapter(embedding_dim=12)

    inputs = compose_adapter_inputs(
        model,
        adapter,
        torch.randn(1, PERCEPTION_FEATURE_DIM),
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[4, 5]]),
    )
    assert inputs.inputs_embeds.shape == (1, 3 + NUM_VIRTUAL_TOKENS + 2, 12)
    assert inputs.attention_mask.tolist() == [[1] * (3 + NUM_VIRTUAL_TOKENS + 2)]
    assert inputs.per_layer_inputs.shape == (1, 3 + NUM_VIRTUAL_TOKENS + 2, 1, 1)
    assert inputs.surrogate_input_ids.shape == (1, 3 + NUM_VIRTUAL_TOKENS + 2)

    inputs.inputs_embeds.sum().backward()
    assert adapter.projection.weight.grad is not None
    assert model.embedding.weight.grad is None


def test_adapter_prompt_parts_keep_the_frozen_scaffold_and_drop_event_json():
    health_event = {
        "event_id": "evt_0000000000_0000",
        "timestamp": "2026-01-01T00:00:00.000Z",
        "classification": {
            "label": "V", "description": "Ventricular ectopic beat", "confidence": 0.9,
            "top_3": [{"label": "V", "confidence": 0.9}, {"label": "N", "confidence": 0.07}, {"label": "S", "confidence": 0.03}],
        },
        "signal_features": {"rr_interval_ms": 400.0, "qrs_duration_ms": 140.0, "heart_rate_bpm": 150.0, "beat_morphology": "wide_complex"},
        "segment_metadata": {"window_samples": 360, "sample_rate_hz": 360, "lead": "MLII", "signal_quality_index": 0.9},
        "clinical_flags": {"requires_urgent_review": True, "flag_reason": "test fixture", "consecutive_abnormal_beats": 1},
    }
    prefix, suffix = build_adapter_prompt_parts(health_event)
    assert prefix.endswith("--- Event data ---\n")
    assert suffix.startswith("\n\n--- Instructions ---\n")
    assert '"classification"' not in prefix + suffix
    assert "Respond with a JSON object" in suffix


def test_adapter_prompt_does_not_leak_predicted_class_through_the_text_scaffold():
    """Regression test for a real bug found during review: the original
    implementation reused build_baseline_prompt's class-specific background
    heading ("Background context for beat class 'X'"), which is real-token
    text never replaced by virtual tokens. Since that heading's class comes
    from the same predicted-label field the training target is built from,
    the heading alone was sufficient to solve the task -- making the
    virtual tokens' actual content irrelevant to the loss. This test proves
    the prompt scaffold is now identical regardless of which class was
    predicted, so any signal the adapter learns must come from the virtual
    tokens themselves, not a text side-channel."""

    def make_event(label):
        return {
            "event_id": "evt_0000000000_0000", "timestamp": "2026-01-01T00:00:00.000Z",
            "classification": {
                "label": label, "description": "x", "confidence": 0.9,
                "top_3": [{"label": label, "confidence": 0.9},
                          {"label": "N", "confidence": 0.07}, {"label": "S", "confidence": 0.03}],
            },
            "signal_features": {"rr_interval_ms": 400.0, "qrs_duration_ms": 140.0,
                                 "heart_rate_bpm": 150.0, "beat_morphology": "wide_complex"},
            "segment_metadata": {"window_samples": 360, "sample_rate_hz": 360,
                                  "lead": "MLII", "signal_quality_index": 0.9},
            "clinical_flags": {"requires_urgent_review": True, "flag_reason": "x", "consecutive_abnormal_beats": 1},
        }

    results = [build_adapter_prompt_parts(make_event(label)) for label in ["N", "S", "V", "F", "Q"]]
    prefixes = [r[0] for r in results]
    suffixes = [r[1] for r in results]

    assert all(p == prefixes[0] for p in prefixes), (
        "Arm B's prompt scaffold varies by predicted class -- this reintroduces "
        "the class-leak bug (a real-token heading revealing the class, bypassing "
        "the virtual tokens entirely). The scaffold must be class-neutral."
    )
    assert all(s == suffixes[0] for s in suffixes)
