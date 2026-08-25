"""Day 3 virtual-token adapter utilities.

Arm B is deliberately limited to a learned projection from the Perception
Agent's frozen 32-dimensional context vector into Gemma's *existing* input
embedding space.  The language model remains frozen.  Text before and after
the baseline's ``Event data`` payload is embedded through Gemma's own token
embedding table; only the payload span is replaced by virtual tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from reasoning.prompt_template import SYSTEM_PROMPT, OUTPUT_INSTRUCTIONS
from reasoning.fixed_context_neutral import NEUTRAL_CONTEXT


PERCEPTION_FEATURE_DIM = 32
NUM_VIRTUAL_TOKENS = 4
_PLACEHOLDER = "\n[[ADAPTER_EVENT_DATA_REPLACED_BY_VIRTUAL_TOKENS]]\n"


class VirtualTokenAdapter(nn.Module):
    """Projects one 32-D Perception vector to ``num_tokens`` LLM embeddings.

    The output width is supplied from ``model.get_input_embeddings()`` rather
    than hard-coded.  Gemma 4 E4B currently uses 2,560-dimensional text
    embeddings; its similarly named 256-wide per-layer input is *not* a valid
    external embedding width.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        num_tokens: int = NUM_VIRTUAL_TOKENS,
        input_dim: int = PERCEPTION_FEATURE_DIM,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_tokens <= 0 or input_dim <= 0:
            raise ValueError("embedding_dim, num_tokens, and input_dim must be positive")
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, num_tokens * embedding_dim)

        # Match the approximate scale of a normal token embedding at startup;
        # this is preferable to an arbitrary large random soft prompt.
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02 / input_dim**0.5)
        nn.init.zeros_(self.projection.bias)

    @classmethod
    def for_model(cls, model, *, num_tokens: int = NUM_VIRTUAL_TOKENS) -> "VirtualTokenAdapter":
        embeddings = model.get_input_embeddings()
        embedding_dim = getattr(embeddings, "embedding_dim", None)
        if embedding_dim is None:
            embedding_dim = embeddings.weight.shape[-1]
        return cls(int(embedding_dim), num_tokens=num_tokens)

    def forward(self, context_vectors: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, num_tokens, embedding_dim)`` virtual tokens."""
        if context_vectors.ndim == 1:
            context_vectors = context_vectors.unsqueeze(0)
        if context_vectors.ndim != 2 or context_vectors.shape[-1] != self.input_dim:
            raise ValueError(
                "Expected context_vectors with shape "
                f"(batch, {self.input_dim}), got {tuple(context_vectors.shape)}"
            )
        projected = self.projection(context_vectors)
        return projected.reshape(-1, self.num_tokens, self.embedding_dim)


@dataclass(frozen=True)
class AdapterInputs:
    """Inputs ready for ``Gemma4ForConditionalGeneration.forward/generate``."""

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    per_layer_inputs: torch.Tensor
    surrogate_input_ids: torch.Tensor
    prefix_token_count: int
    suffix_token_count: int

    @property
    def sequence_length(self) -> int:
        return self.inputs_embeds.shape[1]


def build_adapter_prompt_parts(health_event: dict) -> tuple[str, str]:
    """Build Arm B's prompt scaffold — deliberately NOT via build_baseline_prompt.

    build_baseline_prompt's background context is class-specific (matches
    the detected label). That's correct for Arm A, where the JSON payload
    legitimately carries the class as content. For Arm B it would leak the
    class through a real-token heading regardless of what the virtual
    tokens encode — confirmed directly by rendering a real Arm B prompt and
    finding "Background context for beat class 'V'" sitting in plain text
    ahead of the (virtual-token) event span. Since that heading is derived
    from the same predicted-label field the training target is built from,
    the heading alone would let the model solve the task with zero
    information from the adapter — invalidating the whole comparison.

    Fix: Arm B uses ONE shared, class-neutral context block (see
    reasoning/fixed_context_neutral.py) instead of the per-class one. The
    system prompt and output instructions are imported — not duplicated —
    from the frozen prompt_template.py, so those two blocks stay
    byte-identical to Arm A. Only the background-context content legitimately
    differs between arms, by design, disclosed explicitly in the design doc.

    health_event is accepted but currently unused — kept for interface
    stability (e.g. if a future non-class-revealing per-event context is
    ever needed) rather than silently changing the call signature.
    """
    del health_event  # intentionally unused — see docstring
    prefix = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Background context ---\n"
        f"{NEUTRAL_CONTEXT}\n"
        f"--- Event data ---\n"
    )
    suffix = (
        f"\n\n--- Instructions ---\n"
        f"{OUTPUT_INSTRUCTIONS}"
    )
    return prefix, suffix


def freeze_language_model(model) -> None:
    """Freeze Gemma explicitly; only ``VirtualTokenAdapter`` may train."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def _render_prompt_with_placeholder(processor, prefix: str, suffix: str) -> tuple[str, int, int]:
    """Render the full chat prompt and return the placeholder character span."""
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": prefix + _PLACEHOLDER + suffix}],
        add_generation_prompt=True,
        tokenize=False,
    )
    start = rendered.find(_PLACEHOLDER)
    if start < 0:
        raise RuntimeError("Adapter placeholder was lost while applying the chat template")
    return rendered, start, start + len(_PLACEHOLDER)


def _tokenize_and_remove_placeholder(processor, rendered: str, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize once, then remove exactly the placeholder-token span.

    Offset mappings make this robust to chat-template control tokens and avoid
    independently tokenizing prefix/suffix, which can change boundary merges.
    Gemma's fast tokenizer supplies these mappings.
    """
    encoded = processor.tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"][0].tolist()
    overlap = [i for i, (token_start, token_end) in enumerate(offsets) if token_start < end and token_end > start]
    if not overlap:
        raise RuntimeError("Tokenizer produced no tokens for the adapter placeholder")
    if overlap != list(range(overlap[0], overlap[-1] + 1)):
        raise RuntimeError("Adapter placeholder tokens are unexpectedly non-contiguous")

    first, last = overlap[0], overlap[-1] + 1
    return input_ids[:, :first], input_ids[:, last:]


def compose_adapter_inputs(
    model,
    adapter: VirtualTokenAdapter,
    context_vectors: torch.Tensor,
    prefix_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
) -> AdapterInputs:
    """Embed text, insert virtual tokens, and create an all-valid mask.

    No ``input_ids`` are passed to Gemma afterwards: the concatenated tensor is
    the complete prompt in its native embedding space.
    """
    if prefix_ids.ndim != 2 or suffix_ids.ndim != 2:
        raise ValueError("prefix_ids and suffix_ids must have shape (batch, tokens)")
    if prefix_ids.shape[0] != suffix_ids.shape[0]:
        raise ValueError("prefix_ids and suffix_ids must have the same batch size")

    embedding_layer = model.get_input_embeddings()
    device = embedding_layer.weight.device
    prefix_embeds = embedding_layer(prefix_ids.to(device))
    suffix_embeds = embedding_layer(suffix_ids.to(device))

    adapter_device = next(adapter.parameters()).device
    virtual_tokens = adapter(context_vectors.to(adapter_device, dtype=torch.float32))
    virtual_tokens = virtual_tokens.to(device=device, dtype=prefix_embeds.dtype)
    if virtual_tokens.shape[0] != prefix_embeds.shape[0]:
        raise ValueError("Context-vector batch size must match text-token batch size")

    inputs_embeds = torch.cat((prefix_embeds, virtual_tokens, suffix_embeds), dim=1)
    attention_mask = torch.ones(
        inputs_embeds.shape[:2], dtype=torch.long, device=device
    )

    # Gemma 4 has Per-Layer Embeddings (PLE).  Supplying only inputs_embeds
    # makes it try to reverse every embedding against its 262k-token vocab,
    # which is both invalid for learned vectors and prohibitive in memory.
    # Build PLE from harmless PAD identities for virtual positions instead.
    # The learned virtual embeddings still provide the content-dependent
    # component through Gemma's PLE projection.
    pad_id = model.config.text_config.pad_token_id
    virtual_ids = torch.full(
        (prefix_ids.shape[0], adapter.num_tokens), pad_id, dtype=prefix_ids.dtype, device=device
    )
    surrogate_ids = torch.cat((prefix_ids.to(device), virtual_ids, suffix_ids.to(device)), dim=1)
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        # Gemma4ForConditionalGeneration holds Gemma4Model at ``.model``;
        # the latter owns the text language model.
        language_model = getattr(getattr(model, "model", None), "language_model", None)
    try:
        per_layer_inputs = language_model.get_per_layer_inputs(surrogate_ids, None)
    except AttributeError as exc:
        raise RuntimeError(
            "Arm B requires Gemma 4's public language_model.get_per_layer_inputs API"
        ) from exc
    return AdapterInputs(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        per_layer_inputs=per_layer_inputs,
        surrogate_input_ids=surrogate_ids,
        prefix_token_count=prefix_embeds.shape[1],
        suffix_token_count=suffix_embeds.shape[1],
    )


def prepare_adapter_inputs(
    model,
    processor,
    adapter: VirtualTokenAdapter,
    health_event: dict,
    context_vectors: torch.Tensor,
) -> AdapterInputs:
    """Construct a complete Arm-B prompt from one HealthEvent and vector."""
    prefix, suffix = build_adapter_prompt_parts(health_event)
    rendered, start, end = _render_prompt_with_placeholder(processor, prefix, suffix)
    prefix_ids, suffix_ids = _tokenize_and_remove_placeholder(processor, rendered, start, end)
    return compose_adapter_inputs(model, adapter, context_vectors, prefix_ids, suffix_ids)
