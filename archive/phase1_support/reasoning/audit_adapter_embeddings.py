"""Day 4 diagnostic: inspect learned virtual-token separation by AAMI class.

This is a training-subset diagnostic, not the Day 7 held-out linear-probe
auditability result.  It reports class-centroid distances and leave-one-out
nearest-centroid accuracy so representation collapse is observable early.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reasoning.adapter_training import DEFAULT_OUTPUT, TrainingConfig, build_real_training_examples
from reasoning.virtual_adapter import VirtualTokenAdapter


def audit_checkpoint(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TrainingConfig(**checkpoint["config"])
    examples = build_real_training_examples(
        Path(checkpoint["source_dataset"]),
        per_class=config.per_class,
        max_examples=config.max_examples,
    )
    expected_indices = checkpoint["source_indices"]
    actual_indices = [example["source_index"] for example in examples]
    if actual_indices != expected_indices:
        raise RuntimeError("Rebuilt training subset differs from checkpoint provenance")

    adapter = VirtualTokenAdapter(
        checkpoint["embedding_dim"],
        num_tokens=checkpoint["num_tokens"],
        input_dim=checkpoint["input_dim"],
    )
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()

    contexts = torch.stack([torch.from_numpy(example["context_vector"]) for example in examples])
    labels = torch.tensor([example["source_aami_class"] for example in examples])
    with torch.no_grad():
        vectors = adapter(contexts).flatten(start_dim=1)

    class_ids = sorted(labels.unique().tolist())
    centroids = {class_id: vectors[labels == class_id].mean(dim=0) for class_id in class_ids}
    distances = {
        f"{left}-{right}": float(torch.linalg.vector_norm(centroids[left] - centroids[right]))
        for position, left in enumerate(class_ids)
        for right in class_ids[position + 1:]
    }

    correct = 0
    for row in range(len(vectors)):
        candidate_centroids = {}
        for class_id in class_ids:
            members = vectors[labels == class_id]
            if class_id == int(labels[row]):
                members = vectors[(labels == class_id) & (torch.arange(len(labels)) != row)]
            candidate_centroids[class_id] = members.mean(dim=0)
        prediction = min(
            class_ids,
            key=lambda class_id: float(torch.linalg.vector_norm(vectors[row] - candidate_centroids[class_id])),
        )
        correct += prediction == int(labels[row])

    return {
        "checkpoint": str(checkpoint_path),
        "training_example_count": len(examples),
        "class_centroid_l2_distances": distances,
        "leave_one_out_nearest_centroid_accuracy": correct / len(vectors),
        "note": "Training-subset representation diagnostic only; not a held-out accuracy or auditability result.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Day 4 virtual-token separation on its training subset.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=Path("reasoning/checkpoints/virtual_adapter_day4_audit.json"))
    args = parser.parse_args()
    result = audit_checkpoint(args.checkpoint)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
