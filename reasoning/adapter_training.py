"""Day 4 training harness for the virtual-token interface.

The CNN-LSTM and Gemma 4 are explicitly frozen.  Only the 337,920-parameter
``VirtualTokenAdapter`` receives gradients.  Training data comes from a small,
class-balanced DS1 subset; DS2 remains untouched for Day 6 evaluation.
"""
from __future__ import annotations
from tqdm import tqdm
import argparse
import hashlib
import json
import os
import pickle
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
    num_tokens: int = 4  # E1 ablation varies this (k in {1,2,4,8}); 4 matches the existing headline checkpoint.
    seed: int | None = None  # E2 seed-variance sets this explicitly; None preserves prior (unseeded) behavior.
    # Phase 1 B-null control: train on all-zero context vectors, so the linear adapter
    # can learn only its bias, i.e. one fixed learned prefix carrying no event information.
    zero_context: bool = False
    # How the training examples are balanced. "true_class" (the dissertation's choice):
    # per_class examples of each true AAMI class. "tier": the same total, split evenly
    # across the three reference urgency tiers. Phase 1 found only 5 of the 64
    # true-class-balanced examples were "priority", the tier behind most Arm B errors.
    balance_by: str = "true_class"


RESUME_EVERY_STEPS = 8  # ~1.5 min of training at ~10 s/step


def resume_path_for(output_path: Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(output_path.stem + ".resume.pt")


def _atomic_torch_save(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    for attempt in range(20):  # Windows may briefly lock the target (antivirus scan): retry, then raise
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            import time
            time.sleep(0.5)


def _save_resume(path, config, adapter, optimizer, losses, *, epoch, step, epoch_losses) -> None:
    """Everything needed to continue training exactly where it stopped: adapter and
    AdamW state, position (epoch, next example index), losses so far and RNG state.
    Example order is fixed and Gemma runs in inference mode, so a resumed run
    follows the same trajectory as an uninterrupted one."""
    _atomic_torch_save({
        "config": asdict(config),
        "adapter_state_dict": adapter.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": list(losses), "epoch": epoch, "step": step, "epoch_losses": list(epoch_losses),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


PERCEPTION_CHECKPOINT_PATH = Path("perception/checkpoints/cnn_lstm.pt")
EXAMPLE_CACHE_DIR = Path("cache/adapter_training_examples")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_real_training_examples(
    dataset_path: Path,
    *,
    per_class: int,
    max_examples: int | None = None,
    balance_by: str = "true_class",
) -> list[dict]:
    """Disk-cached wrapper around the chronological DS1 replay below.

    The replay is deterministic and train_adapter() repeats it for every seed,
    so it is computed once per input state. The cache key covers
    everything the result depends on: dataset bytes, Perception checkpoint bytes,
    per_class and max_examples.
    """
    key_fields = {
        "dataset": _file_sha256(Path(dataset_path)),
        "perception": _file_sha256(PERCEPTION_CHECKPOINT_PATH),
        "per_class": per_class, "max_examples": max_examples,
    }
    if balance_by != "true_class":  # the default keeps its original cache key
        key_fields["balance_by"] = balance_by
    key = hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = EXAMPLE_CACHE_DIR / f"examples_{key}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            examples = pickle.load(f)
        print(f"Loaded {len(examples)} cached training examples from {cache_path}")
        return examples
    examples = _build_real_training_examples_uncached(
        dataset_path, per_class=per_class, max_examples=max_examples, balance_by=balance_by)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(examples, f)
    return examples


def _build_real_training_examples_uncached(
    dataset_path: Path,
    *,
    per_class: int,
    max_examples: int | None = None,
    balance_by: str = "true_class",
) -> list[dict]:
    """Extract real vectors/events before Gemma is loaded onto the GPU."""
    # Materialise the arrays once: indexing an NpzFile (data["features"][i]) re-reads
    # and decompresses the whole array on every access, which made this loop ~70 min
    # (predict() itself is ~6 ms/beat).
    with np.load(dataset_path) as npz:
        data = {key: npz[key] for key in ("features", "labels", "rr_interval_ms", "record_ids")}
    perception = PerceptionAgent(checkpoint_path=str(PERCEPTION_CHECKPOINT_PATH))
    remaining = {int(class_id): per_class for class_id in np.unique(data["labels"])}
    if balance_by == "tier":
        # Same total as true-class balancing, split as evenly as possible over the tiers.
        total, tiers = sum(remaining.values()), ("routine", "priority", "urgent")
        remaining = {t: total // len(tiers) + (i < total % len(tiers)) for i, t in enumerate(tiers)}
    elif balance_by != "true_class":
        raise ValueError(f"balance_by must be 'true_class' or 'tier', got {balance_by!r}")
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
        bucket = urgency_tier_from_event(event) if balance_by == "tier" else source_class
        if remaining[bucket] <= 0:
            continue
        examples.append(
            {
                "source_index": int(index),
                "source_aami_class": source_class,
                "context_vector": perception.get_last_context_vector().astype(np.float32),
                "health_event": event,
            }
        )
        remaining[bucket] -= 1
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
        DEFAULT_DATASET, per_class=config.per_class, max_examples=config.max_examples,
        balance_by=config.balance_by,
    )
    model, processor = load_model()
    freeze_language_model(model)
    if config.seed is not None:
        # Only the adapter's random init depends on RNG state -- example selection
        # and order are already deterministic (see build_real_training_examples).
        torch.manual_seed(config.seed)
    adapter = VirtualTokenAdapter.for_model(model, num_tokens=config.num_tokens).to(
        model.get_input_embeddings().weight.device
    )
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.learning_rate, weight_decay=0.0)

    # Mid-training checkpoint (see _save_resume): resume only if it was written
    # for exactly this config, so two different runs can never be mixed.
    resume_path = resume_path_for(output_path)
    losses: list[float] = []
    start_epoch, start_step, resumed_epoch_losses = 0, 0, []
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["config"] != asdict(config):
            raise RuntimeError(f"{resume_path} was written for a different config: {state['config']} "
                               f"vs {asdict(config)}. Delete it to start this run from scratch.")
        adapter.load_state_dict(state["adapter_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        losses, start_epoch, start_step = state["losses"], state["epoch"], state["step"]
        resumed_epoch_losses = state["epoch_losses"]
        print(f"Resuming training from {resume_path}: epoch {start_epoch + 1}, step {start_step}")

    for epoch in range(start_epoch, config.epochs):
        resuming = epoch == start_epoch and start_step > 0
        epoch_losses = list(resumed_epoch_losses) if resuming else []

        # Wrap the examples list in a tqdm progress bar
        progress_bar = tqdm(examples[start_step:] if resuming else examples,
                            desc=f"Epoch {epoch + 1}/{config.epochs}",
                            initial=start_step if resuming else 0, total=len(examples))

        for example in progress_bar:
            context = torch.from_numpy(example["context_vector"]).unsqueeze(0)
            if config.zero_context:
                context = torch.zeros_like(context)
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

            # Each example has a different prompt length (variable event content), so
            # every step allocates differently-shaped activation/gradient tensors --
            # this fragments the CUDA caching allocator's free blocks over many steps
            # until a later, differently-shaped allocation OOMs despite adequate total
            # free memory. Confirmed happening in practice at step 51/64 of a k=1 run.
            # PYTORCH_CUDA_ALLOC_CONF=expandable_segments is the usual fix but is not
            # supported on Windows (silently ignored, confirmed via the runtime
            # warning) -- clearing the cache periodically here works on every platform.
            if torch.cuda.is_available() and len(epoch_losses) % 10 == 0:
                torch.cuda.empty_cache()

            if len(epoch_losses) % RESUME_EVERY_STEPS == 0 and len(epoch_losses) < len(examples):
                _save_resume(resume_path, config, adapter, optimizer, losses,
                             epoch=epoch, step=len(epoch_losses), epoch_losses=epoch_losses)

        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        if epoch + 1 < config.epochs:
            _save_resume(resume_path, config, adapter, optimizer, losses,
                         epoch=epoch + 1, step=0, epoch_losses=[])
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
    # Atomic: callers treat "output_path exists" as "training finished", so a
    # half-written final checkpoint must never appear under that name.
    _atomic_torch_save(checkpoint, output_path)
    resume_path.unlink(missing_ok=True)
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
    parser.add_argument("--num-tokens", type=int, default=TrainingConfig.num_tokens)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--zero-context", action="store_true")
    parser.add_argument("--balance-by", choices=("true_class", "tier"), default=TrainingConfig.balance_by)
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
            num_tokens=args.num_tokens,
            seed=args.seed,
            zero_context=args.zero_context,
            balance_by=args.balance_by,
        ),
        output_path=args.output,
    )
