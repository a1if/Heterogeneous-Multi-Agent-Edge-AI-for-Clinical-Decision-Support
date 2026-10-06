"""Perception Agent (frozen checkpoint) evaluation on DS2 -- for the ledger.

Reuses train_perception_agent.py's evaluate() against the SAME checkpoint
used throughout Day 6/7 (perception/checkpoints/cnn_lstm.pt). No retraining --
this is inference-only re-analysis of an already-trained, frozen artifact,
so it produces a genuine, re-derivable number (unlike the KB's "84.9%/80.1%/
85.4% across 3 independent training runs" note, which describes training-time
variance across separate checkpoints, not this one frozen checkpoint).

Run:
    python perception_eval.py
"""
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from perception.model import CNNLSTM
from train_perception_agent import evaluate, load_split
from project_config import DS2_PATH

CHECKPOINT_PATH = "perception/checkpoints/cnn_lstm.pt"
RESULTS_PATH = "results/perception_eval_results.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNLSTM().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    X_test, y_test, _, _ = load_split(DS2_PATH)
    test_ds = TensorDataset(torch.from_numpy(X_test).unsqueeze(1), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    _, test_acc, per_class_acc = evaluate(model, test_loader, device, nn.CrossEntropyLoss())
    print(f"DS2 overall accuracy: {test_acc:.4f}")
    print(f"Per-class accuracy: {per_class_acc}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": CHECKPOINT_PATH,
            "n_test": len(y_test),
            "overall_accuracy": test_acc,
            "per_class_accuracy": per_class_acc,
        }, f, indent=2)
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
