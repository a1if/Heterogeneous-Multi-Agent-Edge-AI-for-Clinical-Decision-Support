"""
Trains the CNN-LSTM Perception Agent on MIT-BIH DS1, evaluates on DS2.

Per KB Section 3.2: class imbalance (N:V/S/F/Q ~ 15:1 or worse) handled via
WeightedRandomSampler, not oversampling. DS1/DS2 must not be mixed (de Chazal
et al. inter-patient split) — a small validation slice is carved from DS1
only, for early stopping; DS2 is touched exactly once, at the end, for the
final reported accuracy.

Run:
    python train_perception_agent.py

Produces:
    perception/checkpoints/cnn_lstm.pt
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from perception.model import CNNLSTM, AAMI_CLASSES

DATA_DIR = "data/processed"
CHECKPOINT_DIR = "perception/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cnn_lstm.pt")

VAL_FRACTION = 0.1  # target fraction, approximated via whole-record holdout below
# Deliberate, documented patient-level validation holdout — NOT a random beat-level
# split. Spread across the DS1 record list (100-series and 200-series) for some
# diversity. Fixes a real leakage bug: an earlier version used a random per-beat
# shuffle, which let validation beats come from the same patients as training beats,
# artificially inflating validation accuracy (98-99%) versus the true DS2 result
# (~85%). This is the same intra-patient leakage the de Chazal DS1/DS2 split itself
# exists to prevent — it must also be respected for any internal validation carve-out.
VAL_RECORDS = {109, 205, 223}
BATCH_SIZE = 256
MAX_EPOCHS = 30
PATIENCE = 5  # early stopping on val loss
LEARNING_RATE = 3e-4  # reduced from 1e-3 — original LR combined with aggressive
# minority-class resampling caused unstable, oscillating val_loss across epochs
# (best checkpoint landed at epoch 1 essentially by chance, not real convergence)
MAX_SAMPLER_WEIGHT_RATIO = 20.0  # caps how disproportionately minority classes get
# resampled; raw inverse-frequency weighting (F:N ~ 100:1) was likely a major
# contributor to the instability above


def load_split(path):
    data = np.load(path)
    return data["features"], data["labels"], data["rr_interval_ms"], data["record_ids"]


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(labels, minlength=5)
    class_counts = np.maximum(class_counts, 1)  # avoid div-by-zero for absent classes
    class_weights = 1.0 / class_counts
    # Cap the max class weight relative to the majority class's weight — uncapped
    # inverse-frequency weighting (e.g. F:N ~ 100:1 in this dataset) forces extreme,
    # repeated resampling of a tiny example pool, which destabilizes training
    # (see LEARNING_RATE comment above).
    min_weight = class_weights.min()  # majority class (N) gets the smallest raw weight
    class_weights = np.minimum(class_weights, min_weight * MAX_SAMPLER_WEIGHT_RATIO)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    per_class_correct = np.zeros(5)
    per_class_total = np.zeros(5)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            for c in range(5):
                mask = y == c
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += (preds[mask] == c).sum().item()
    model.train()
    per_class_acc = {
        AAMI_CLASSES[c]: (per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else float("nan"))
        for c in range(5)
    }
    return total_loss / total, correct / total, per_class_acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading DS1 (train+val)...")
    X, y, _, record_ids = load_split(os.path.join(DATA_DIR, "ds1_train.npz"))
    print(f"DS1 total: {len(y)} windows. Class distribution: "
          f"{ {AAMI_CLASSES[c]: int((y == c).sum()) for c in range(5)} }")

    # Patient-level split: hold out WHOLE records for validation, never mixing
    # a patient's beats between train and val (see VAL_RECORDS docstring above).
    val_mask = np.isin(record_ids, list(VAL_RECORDS))
    train_mask = ~val_mask
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"Patient-level split: {len(y_train)} train windows "
          f"({len(set(record_ids[train_mask]))} records), "
          f"{len(y_val)} val windows ({sorted(set(record_ids[val_mask]))} records held out)")
    print(f"Val class distribution: { {AAMI_CLASSES[c]: int((y_val == c).sum()) for c in range(5)} }")

    train_ds = TensorDataset(torch.from_numpy(X_train).unsqueeze(1), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val).unsqueeze(1), torch.from_numpy(y_val))

    sampler = make_weighted_sampler(y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\nTraining (max {MAX_EPOCHS} epochs, patience {PATIENCE})...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, batch_y in train_loader:
            x, batch_y = x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_loss, val_acc, per_class_acc = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch:2d} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"per_class={ {k: round(v, 3) for k, v in per_class_acc.items()} }")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> New best, checkpoint saved to {CHECKPOINT_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    print("\nLoading best checkpoint for final DS2 evaluation (touched once, here only)...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    X_test, y_test, _, _ = load_split(os.path.join(DATA_DIR, "ds2_test.npz"))
    test_ds = TensorDataset(torch.from_numpy(X_test).unsqueeze(1), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    test_loss, test_acc, test_per_class = evaluate(model, test_loader, device, criterion)
    print(f"\n=== Final DS2 (test) results ===")
    print(f"Overall accuracy: {test_acc:.4f}")
    print(f"Per-class accuracy: {test_per_class}")
    print(f"\nCheckpoint saved at: {CHECKPOINT_PATH}")
    print("Load it in PerceptionAgent via: PerceptionAgent(checkpoint_path='perception/checkpoints/cnn_lstm.pt')")


if __name__ == "__main__":
    main()
