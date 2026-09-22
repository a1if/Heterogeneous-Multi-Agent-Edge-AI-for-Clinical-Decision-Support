"""Trains the RR-branch CNN-LSTM (perception/model_rr.py) on MIT-BIH DS1 and
evaluates it and the reference CNN-LSTM on DS2 with identical metric code.

Everything except the RR branch matches train_perception_agent.py: the same
whole-record validation holdout, capped WeightedRandomSampler, learning rate,
batch size and early stopping (imported, not copied). DS2 is touched once, at
the end. The seed is recorded (the reference checkpoint's was not).

Run (from repo root):
    python train_perception_agent_rr.py              # full run, GPU
    python train_perception_agent_rr.py --smoke      # a few batches, no checkpoint written
Produces:
    perception/checkpoints/cnn_lstm_rr_seed{seed}.pt
    results/p1_rr_encoder_results.json
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from perception.model import AAMI_CLASSES, CNNLSTM
from perception.model_rr import CNNLSTMRR
from perception.rr_features import FEATURE_NAMES, compute_rr_features, fit_standardizer, standardize
from train_perception_agent import (BATCH_SIZE, DATA_DIR, LEARNING_RATE, MAX_EPOCHS, PATIENCE,
                                    VAL_RECORDS, load_split, make_weighted_sampler)

REFERENCE_CHECKPOINT = "perception/checkpoints/cnn_lstm.pt"
RESULTS_PATH = "results/p1_rr_encoder_results.json"


def load_with_rr(split: str):
    X, y, rr_ms, record_ids = load_split(os.path.join(DATA_DIR, f"{split}.npz"))
    return X, y, compute_rr_features(rr_ms, record_ids), record_ids


def loader(X, rr, y, *, batch_size=BATCH_SIZE, sampler=None):
    ds = TensorDataset(torch.from_numpy(X).unsqueeze(1), torch.from_numpy(rr), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=False)


@torch.no_grad()
def predict(model, data_loader, device, uses_rr: bool):
    model.eval()
    preds, loss_sum, n = [], 0.0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for x, rr, y in data_loader:
        x, rr, y = x.to(device), rr.to(device), y.to(device)
        logits, _ = model(x, rr) if uses_rr else model(x)
        loss_sum += criterion(logits, y).item()
        n += len(y)
        preds.append(logits.argmax(1).cpu().numpy())
    model.train()
    return np.concatenate(preds), loss_sum / n


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Overall accuracy plus per-class sensitivity (Se) and positive predictive
    value (+P), the standard inter-patient reporting pair."""
    out = {"accuracy": float((y_true == y_pred).mean()), "per_class": {}}
    for c, name in enumerate(AAMI_CLASSES):
        support = int((y_true == c).sum())
        if support == 0:
            continue
        tp = int(((y_pred == c) & (y_true == c)).sum())
        predicted = int((y_pred == c).sum())
        out["per_class"][name] = {"support": support, "se": tp / support,
                                  "ppv": tp / predicted if predicted else None}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="3 batches on CPU, no checkpoint or results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cpu" if args.smoke else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = f"perception/checkpoints/cnn_lstm_rr_seed{args.seed}.pt"

    X, y, rr, record_ids = load_with_rr("ds1_train")
    val_mask = np.isin(record_ids, list(VAL_RECORDS))
    stats = fit_standardizer(rr[~val_mask])  # training records only
    rr = standardize(rr, stats)
    train_loader = loader(X[~val_mask], rr[~val_mask], y[~val_mask],
                          sampler=make_weighted_sampler(y[~val_mask]))
    val_loader = loader(X[val_mask], rr[val_mask], y[val_mask])

    model = CNNLSTMRR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_val, stale, history, t0 = float("inf"), 0, [], time.time()

    for epoch in range(1, (1 if args.smoke else MAX_EPOCHS) + 1):
        for step, (xb, rrb, yb) in enumerate(train_loader):
            if args.smoke and step == 3:
                break
            xb, rrb, yb = xb.to(device), rrb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, context = model(xb, rrb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if args.smoke:
            assert context.shape == (len(yb), 32), context.shape
            print(f"smoke OK: 3 batches, loss={loss.item():.4f}, context {tuple(context.shape)}")
            return

        val_pred, val_loss = predict(model, val_loader, device, uses_rr=True)
        val_m = metrics(y[val_mask], val_pred)
        history.append({"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_m["accuracy"],
                        "val_se": {k: round(v["se"], 4) for k, v in val_m["per_class"].items()}})
        print(f"Epoch {epoch:2d} | val_loss={val_loss:.4f} val_acc={val_m['accuracy']:.4f} "
              f"se={history[-1]['val_se']}")
        if val_loss < best_val:
            best_val, stale = val_loss, 0
            torch.save({"state_dict": model.state_dict(), "rr_standardizer": stats,
                        "rr_feature_names": list(FEATURE_NAMES), "seed": args.seed}, checkpoint_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # DS2, touched once: the RR encoder and the reference CNN-LSTM, same metric code.
    X2, y2, rr2, _ = load_with_rr("ds2_test")
    test_loader = loader(X2, standardize(rr2, stats), y2, batch_size=1024)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])
    rr_pred, _ = predict(model, test_loader, device, uses_rr=True)
    reference = CNNLSTM().to(device)
    reference.load_state_dict(torch.load(REFERENCE_CHECKPOINT, map_location=device))
    ref_pred, _ = predict(reference, test_loader, device, uses_rr=False)

    results = {
        "checkpoint": checkpoint_path, "seed": args.seed,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "train_seconds": round(time.time() - t0, 1), "history": history,
        "ds2": {"rr_encoder": metrics(y2, rr_pred), "reference_cnn_lstm": metrics(y2, ref_pred)},
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    for name, m in results["ds2"].items():
        print(f"{name:20s} acc={m['accuracy']:.4f} " +
              " ".join(f"{c}: Se={v['se']:.3f} +P={v['ppv'] if v['ppv'] is None else round(v['ppv'], 3)}"
                       for c, v in m["per_class"].items()))


if __name__ == "__main__":
    main()
