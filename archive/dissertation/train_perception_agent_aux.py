"""
Trains the CNN-LSTM Perception Agent with an added auxiliary reconstruction
objective (L = L_classification + lambda * L_reconstruction), per
aux_training_precommitted_analysis_plan.md. Tests whether a richer training
signal raises the auditability ceiling identified by the context-32d
attribution probe, without touching the encoder architecture or the
classification-only baseline checkpoint (perception/checkpoints/cnn_lstm.pt).

Reuses load_split/make_weighted_sampler/evaluate from train_perception_agent.py
unchanged -- only the loss and the forward pass wiring differ.

Run (single seed):
    python train_perception_agent_aux.py --seed 101 --lam 0.1

Produces:
    perception/checkpoints/cnn_lstm_aux_seed<seed>.pt
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from perception.model import CNNLSTM
from perception.reconstruction_decoder import ReconstructionDecoder, combined_loss
from train_perception_agent import (
    DATA_DIR, VAL_RECORDS, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE,
    load_split, make_weighted_sampler, evaluate,
)

CHECKPOINT_DIR = "perception/checkpoints"


def train_one_seed(seed: int, lam: float, verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y, _, record_ids = load_split(os.path.join(DATA_DIR, "ds1_train.npz"))
    val_mask = np.isin(record_ids, list(VAL_RECORDS))
    train_mask = ~val_mask
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    train_ds = TensorDataset(torch.from_numpy(X_train).unsqueeze(1), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val).unsqueeze(1), torch.from_numpy(y_val))

    sampler = make_weighted_sampler(y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNLSTM().to(device)
    decoder = ReconstructionDecoder(context_dim=32, target_length=360).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"cnn_lstm_aux_seed{seed}.pt")
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    history = []
    if verbose:
        print(f"[seed {seed}] Training on {device}, lam={lam}...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        decoder.train()
        epoch_recon, epoch_cls, n_batches = 0.0, 0.0, 0
        for x, batch_y in train_loader:
            x, batch_y = x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, context_vector = model(x)
            reconstructed = decoder(context_vector)
            loss, components = combined_loss(logits, batch_y, reconstructed, x.squeeze(1), lam=lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(decoder.parameters()), max_norm=1.0
            )
            optimizer.step()
            epoch_recon += components["loss_recon"]
            epoch_cls += components["loss_cls"]
            n_batches += 1

        val_loss, val_acc, per_class_acc = evaluate(model, val_loader, device, criterion)
        mean_recon = epoch_recon / n_batches
        mean_cls = epoch_cls / n_batches
        history.append({
            "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc,
            "train_loss_recon": mean_recon, "train_loss_cls": mean_cls,
        })
        if verbose:
            print(f"[seed {seed}] Epoch {epoch:2d} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                  f"train_loss_recon={mean_recon:.4f} train_loss_cls={mean_cls:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            if verbose:
                print(f"  -> New best, checkpoint saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                if verbose:
                    print(f"[seed {seed}] Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    X_test, y_test, _, _ = load_split(os.path.join(DATA_DIR, "ds2_test.npz"))
    test_ds = TensorDataset(torch.from_numpy(X_test).unsqueeze(1), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loss, test_acc, test_per_class = evaluate(model, test_loader, device, criterion)

    if verbose:
        print(f"[seed {seed}] Final DS2 accuracy: {test_acc:.4f}")
        print(f"[seed {seed}] Final DS2 per-class: {test_per_class}")

    return {
        "seed": seed,
        "lam": lam,
        "checkpoint": checkpoint_path,
        "ds2_accuracy": test_acc,
        "ds2_per_class": test_per_class,
        "final_train_loss_recon": history[-1]["train_loss_recon"],
        "first_train_loss_recon": history[0]["train_loss_recon"],
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--out", type=str, default=None, help="optional path to dump result JSON")
    args = parser.parse_args()

    result = train_one_seed(args.seed, args.lam)
    summary = {k: v for k, v in result.items() if k != "history"}
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, default=float))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"Full result (with per-epoch history) saved to {args.out}")
