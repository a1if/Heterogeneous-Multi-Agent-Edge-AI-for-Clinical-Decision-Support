"""Day 4 training harness for the virtual-token interface.

The CNN-LSTM and Gemma 4 are explicitly frozen.  Only the 337,920-parameter
``VirtualTokenAdapter`` receives gradients.  Training data comes from a small,
class-balanced DS1 subset; DS2 remains untouched for Day 6 evaluation.
"""
from __future__ import annotations
from tqdm import tqdm
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from perception.perception_agent import PerceptionAgent
from reasoning.model_loader import load_model
from reasoning.training_targets import canonical_reasoning_target, tier_target_prefix, urgency_tier_from_event
from reasoning.virtual_adapter import VirtualTokenAdapter, freeze_language_model, prepare_adapter_inputs


DEFAULT_DATASET = Path("data/processed/ds1_train.npz")
DEFAULT_OUTPUT = Path("reasoning/checkpoints/virtual_adapter_day4.pt")


@dataclass(frozen=True)
class TrainingConfig:
    per_class: int = 4
    epochs: int = 2
    learning_rate: float = 2e-3
    loss_mode: str = "full"  # ``tier`` is only a fast diagnostic; ``full`` is the main experiment.
    tier_weight: float = 4.0
    max_examples: int | None = None


def build_real_training_examples(
    dataset_path: Path,
    *,
    per_class: int,
    max_examples: int | None = None,
) -> list[dict]:
    """Extract real vectors/events before Gemma is loaded onto the GPU."""
    data = np.load(dataset_path)
    perception = PerceptionAgent(checkpoint_path="perception/checkpoints/cnn_lstm.pt")
    remaining = {int(class_id): per_class for class_id in np.unique(data["labels"])}
    examples = []
    active_record_id = None
    for index in range(len(data["labels"])):
        record_id = int(data["record_ids"][index])
        if record_id != active_record_id:
            perception.reset_state()
            active_record_id = record_id

        event = perception.predict(
            data["features"][index], rr_interval_ms=float(data["rr_interval_ms"][index]), event_seq=index
        )
        source_class = int(data["labels"][index])
        if remaining[source_class] <= 0:
            continue
        examples.append(
            {
                "source_index": int(index),
                "source_aami_class": source_class,
                "context_vector": perception.get_last_context_vector().astype(np.float32),
                "health_event": event,
            }
        )
        remaining[source_class] -= 1
        if max_examples is not None and len(examples) >= max_examples:
            break
        if all(count == 0 for count in remaining.values()):
            break

    missing = {class_id: count for class_id, count in remaining.items() if count > 0}
    if missing and max_examples is None:
        raise RuntimeError(f"Could not collect the requested real DS1 class-balanced subset: {missing}")

    # Gemma will need the GPU next.  The context vectors and JSON-compatible
    # events above are now independent of the Perception model.
    del perception
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return examples


def _get_language_model(model):
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        language_model = getattr(getattr(model, "model", None), "language_model", None)
    if language_model is None:
        raise RuntimeError("Gemma 4 language model was not found for PLE construction")
    return language_model


def _append_target_for_teacher_forcing(model, adapter_inputs, target_ids: torch.Tensor):
    """Append target token embeddings and PLE identities to an Arm-B prompt."""
    embedding_layer = model.get_input_embeddings()
    device = embedding_layer.weight.device
    target_ids = target_ids.to(device)
    target_embeds = embedding_layer(target_ids)
    inputs_embeds = torch.cat((adapter_inputs.inputs_embeds, target_embeds), dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    surrogate_ids = torch.cat((adapter_inputs.surrogate_input_ids, target_ids), dim=1)
    per_layer_inputs = _get_language_model(model).get_per_layer_inputs(surrogate_ids, None)
    return inputs_embeds, attention_mask, per_layer_inputs


def _target_text(example: dict, loss_mode: str) -> str:
    if loss_mode == "tier":
        return tier_target_prefix(example["health_event"])
    if loss_mode == "full":
        return canonical_reasoning_target(example["health_event"])
    raise ValueError("loss_mode must be 'tier' or 'full'")


def _labels_for_target(
    prompt_length: int, target_ids: torch.Tensor, target_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask prompt tokens and place target-token weights after the prompt."""
    labels = torch.full((1, prompt_length + target_ids.shape[1]), -100, dtype=torch.long, device=target_ids.device)
    weights = torch.zeros((1, prompt_length + target_ids.shape[1]), dtype=torch.float32, device=target_ids.device)
    labels[:, prompt_length:] = target_ids
    weights[:, prompt_length:] = target_weights
    return labels, weights


def _target_ids_and_weights(processor, text: str, tier: str, tier_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a target and up-weight only its urgency-tier value tokens.

    Anchors to the exact '"urgency_tier":"<tier>"' field pattern, not a bare
    text.index(tier) search. A bare search previously happened to work only
    because every justification template repeats the tier word AFTER the
    urgency_tier field in JSON key order (e.g. "...for urgent review") --
    coincidental, not robust. Editing a justification template to mention the
    tier word earlier would have silently mis-weighted the wrong span with no
    error. Anchoring to the field pattern is immune to that."""
    encoded = processor.tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt"
    )
    target_ids = encoded["input_ids"]
    weights = torch.ones_like(target_ids, dtype=torch.float32)

    tier_field_marker = f'"urgency_tier":"{tier}"'
    marker_pos = text.index(tier_field_marker)  # raises loudly if the field is missing/malformed
    tier_start = marker_pos + len('"urgency_tier":"')
    tier_end = tier_start + len(tier)
    for index, (token_start, token_end) in enumerate(encoded["offset_mapping"][0].tolist()):
        if token_start < tier_end and token_end > tier_start:
            weights[0, index] = tier_weight
    return target_ids, weights


def _weighted_teacher_forcing_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over target completion, emphasizing urgency-tier tokens."""
    shifted_logits = logits[:, :-1, :].float().contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    shifted_weights = weights[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shifted_weights)
    valid_weights = shifted_weights * (shifted_labels != -100)
    return (token_loss * valid_weights).sum() / valid_weights.sum().clamp_min(1.0)


def train_adapter(config: TrainingConfig, *, output_path: Path = DEFAULT_OUTPUT) -> dict:
    if config.loss_mode not in {"tier", "full"}:
        raise ValueError("loss_mode must be 'tier' or 'full'")
    if config.tier_weight < 1.0:
        raise ValueError("tier_weight must be >= 1.0")

    examples = build_real_training_examples(
        DEFAULT_DATASET, per_class=config.per_class, max_examples=config.max_examples
    )
    model, processor = load_model()
    freeze_language_model(model)
    adapter = VirtualTokenAdapter.for_model(model).to(model.get_input_embeddings().weight.device)
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.learning_rate, weight_decay=0.0)

    losses: list[float] = []
    for epoch in range(config.epochs):
        epoch_losses = []
        
        # Wrap the examples list in a tqdm progress bar
        progress_bar = tqdm(examples, desc=f"Epoch {epoch + 1}/{config.epochs}")
        
        for example in progress_bar:
            context = torch.from_numpy(example["context_vector"]).unsqueeze(0)
            adapter_inputs = prepare_adapter_inputs(
                model, processor, adapter, example["health_event"], context
            )
            text = _target_text(example, config.loss_mode)
            tier_text = urgency_tier_from_event(example["health_event"])
            target_ids, target_weights = _target_ids_and_weights(
                processor, text, tier_text, config.tier_weight
            )
            inputs_embeds, attention_mask, per_layer_inputs = _append_target_for_teacher_forcing(
                model, adapter_inputs, target_ids
            )
            labels, loss_weights = _labels_for_target(
                adapter_inputs.sequence_length,
                target_ids.to(inputs_embeds.device),
                target_weights.to(inputs_embeds.device),
            )

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                per_layer_inputs=per_layer_inputs,
                use_cache=False,
            )
            loss = _weighted_teacher_forcing_loss(outputs.logits, labels, loss_weights)
            loss.backward()
            clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Extract loss and update the progress bar display
            step_loss = float(loss.detach().cpu())
            epoch_losses.append(step_loss)
            progress_bar.set_postfix({"loss": f"{step_loss:.4f}"})

        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        # The progress bar will complete, and this prints the final summary for the epoch
        print(f"epoch={epoch + 1}/{config.epochs} mean_loss={mean_loss:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "adapter_state_dict": adapter.state_dict(),
        "embedding_dim": adapter.embedding_dim,
        "num_tokens": adapter.num_tokens,
        "input_dim": adapter.input_dim,
        "config": asdict(config),
        "losses": losses,
        "example_count": len(examples),
        "source_dataset": str(DEFAULT_DATASET),
        "source_indices": [example["source_index"] for example in examples],
    }
    torch.save(checkpoint, output_path)
    summary = {
        "checkpoint": str(output_path),
        "example_count": len(examples),
        "losses": losses,
        "loss_decreased": len(losses) < 2 or losses[-1] < losses[0],
        "loss_mode": config.loss_mode,
    }
    print(json.dumps(summary, indent=2))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Day 4 virtual-token adapter on real DS1 events.")
    parser.add_argument("--per-class", type=int, default=TrainingConfig.per_class)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--loss-mode", choices=("tier", "full"), default=TrainingConfig.loss_mode)
    parser.add_argument("--tier-weight", type=float, default=TrainingConfig.tier_weight)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_adapter(
        TrainingConfig(
            per_class=args.per_class,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            loss_mode=args.loss_mode,
            tier_weight=args.tier_weight,
            max_examples=args.max_examples,
        ),
        output_path=args.output,
    )
