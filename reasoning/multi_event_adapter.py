"""Multi-event virtual-token adapter (item 7, Deviation 9).

N perception context vectors (32-d each) -> N x k virtual tokens in the LLM's
input-embedding space, one block of k tokens per event, concatenated in event
order. Compared with VirtualTokenAdapter (one event), it adds:
  - learned event-position embeddings, one (k, E) block per event slot, so the
    order of events is encoded without spending text tokens on it;
  - scale control: every virtual token is L2-normalised and multiplied by one
    learnable scale, initialised to the median norm of the embeddings the model
    actually receives for real tokens. The single-event adapter's tokens drifted
    far off that scale, and many of them at once broke generation (pilots 1-2).
Per-event blocks keep per-event auditability: slot i's k tokens are a function of
event i's vector (plus the slot's fixed position embedding) only.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from reasoning.virtual_adapter import PERCEPTION_FEATURE_DIM, compose_adapter_inputs

MAX_EVENTS = 20


class MultiEventVirtualAdapter(nn.Module):
    def __init__(self, embedding_dim: int, *, num_tokens: int = 4, input_dim: int = PERCEPTION_FEATURE_DIM,
                 max_events: int = MAX_EVENTS, init_scale: float = 1.0):
        super().__init__()
        self.embedding_dim, self.num_tokens, self.input_dim = embedding_dim, num_tokens, input_dim
        self.max_events = max_events
        self.projection = nn.Linear(input_dim, num_tokens * embedding_dim)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02 / input_dim ** 0.5)
        nn.init.zeros_(self.projection.bias)
        self.position = nn.Parameter(torch.randn(max_events, num_tokens, embedding_dim) * 0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    @classmethod
    def for_model(cls, model, *, num_tokens: int = 4, max_events: int = MAX_EVENTS, probe_tokens: int = 4096):
        """Size from the model's embedding layer; initial scale = median norm of what
        the embedding layer outputs for real token ids (this includes any embedding
        scaling the model applies, unlike the raw weight rows)."""
        emb = model.get_input_embeddings()
        dim = int(getattr(emb, "embedding_dim", emb.weight.shape[-1]))
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(0, emb.weight.shape[0], (1, probe_tokens), generator=g).to(emb.weight.device)
        with torch.no_grad():
            scale = float(emb(ids).float().norm(dim=-1).median())
        return cls(dim, num_tokens=num_tokens, max_events=max_events, init_scale=scale)

    @property
    def scale(self) -> torch.Tensor:
        return self.log_scale.exp()

    def forward(self, context_vectors: torch.Tensor) -> torch.Tensor:
        """(N, input_dim) or (B, N, input_dim) -> (B, N * num_tokens, embedding_dim)."""
        if context_vectors.ndim == 2:
            context_vectors = context_vectors.unsqueeze(0)
        b, n, d = context_vectors.shape
        if d != self.input_dim or n > self.max_events:
            raise ValueError(f"expected (B, N <= {self.max_events}, {self.input_dim}), got {tuple(context_vectors.shape)}")
        x = self.projection(context_vectors).view(b, n, self.num_tokens, self.embedding_dim)
        x = x + self.position[:n].unsqueeze(0)
        x = F.normalize(x, dim=-1) * self.scale
        return x.reshape(b, n * self.num_tokens, self.embedding_dim)


class _BoundWindow(nn.Module):
    """Presents a MultiEventVirtualAdapter for a fixed N to compose_adapter_inputs,
    which expects num_tokens and a (batch, features) input."""

    def __init__(self, adapter: MultiEventVirtualAdapter, n_events: int):
        super().__init__()
        self.adapter, self.n_events = adapter, n_events
        self.num_tokens = adapter.num_tokens * n_events

    def forward(self, context):  # (1, N * input_dim)
        return self.adapter(context.reshape(1, self.n_events, -1))


def compose_multi_event_inputs(model, adapter: MultiEventVirtualAdapter, context_vectors, prefix_ids, suffix_ids):
    """Full prompt embeddings for one window: text prefix + N x k virtual tokens +
    text suffix, with Gemma's per-layer inputs built from PAD surrogates exactly as
    for the single-event adapter (reuses compose_adapter_inputs)."""
    context_vectors = torch.as_tensor(context_vectors, dtype=torch.float32)
    n = context_vectors.shape[0]
    return compose_adapter_inputs(model, _BoundWindow(adapter, n), context_vectors.reshape(1, -1),
                                  prefix_ids, suffix_ids)
