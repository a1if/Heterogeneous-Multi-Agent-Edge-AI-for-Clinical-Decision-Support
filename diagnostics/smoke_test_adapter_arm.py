"""Day 3 gate: run one untrained Arm-B forward pass on a real MIT-BIH event.

This intentionally checks only wiring, shape, and the frozen-model forward
path.  It does not claim useful performance before Day 4-5 adapter training.
Run from the repository root:
    .\\venv\\Scripts\\python.exe diagnostics\\smoke_test_adapter_arm.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for perception./reasoning.

from perception.perception_agent import PerceptionAgent
from reasoning.model_loader import load_model
from reasoning.virtual_adapter import (
    VirtualTokenAdapter,
    freeze_language_model,
    prepare_adapter_inputs,
)


def main() -> None:
    data = np.load("data/processed/ds2_test.npz")
    raw_segment = data["features"][0]

    perception = PerceptionAgent(checkpoint_path="perception/checkpoints/cnn_lstm.pt")
    health_event = perception.predict(raw_segment)
    context_vector = torch.from_numpy(perception.get_last_context_vector()).unsqueeze(0)

    model, processor = load_model()
    freeze_language_model(model)
    adapter = VirtualTokenAdapter.for_model(model).to(model.get_input_embeddings().weight.device)
    inputs = prepare_adapter_inputs(model, processor, adapter, health_event, context_vector)

    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs.inputs_embeds,
            attention_mask=inputs.attention_mask,
            per_layer_inputs=inputs.per_layer_inputs,
            use_cache=False,
        )

    assert outputs.logits.shape[0] == 1
    assert outputs.logits.shape[1] == inputs.sequence_length
    assert outputs.logits.shape[-1] == model.config.text_config.vocab_size
    print("Day 3 adapter smoke test passed")
    print(f"Virtual tokens: {adapter.num_tokens}")
    print(f"Embedding width: {adapter.embedding_dim}")
    print(f"Adapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")
    print(f"Prompt length after replacement: {inputs.sequence_length} positions")


if __name__ == "__main__":
    main()
