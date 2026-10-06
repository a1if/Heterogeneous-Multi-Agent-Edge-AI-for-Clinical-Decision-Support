"""E4: training-set-size sweep at fixed seed and fixed epochs.

Follows E3 (e3_training_compute_ladder.py), which showed accuracy rising with
training budget (20.0% -> 73.8% -> 95.0% across 32/64/192 gradient updates) but
could not isolate the cause: its three rungs differed in training-set size AND
epochs AND initialisation, all three unrecorded.

This experiment removes two of those three confounds. Every run here fixes
seed=101, epochs=3, learning_rate=0.002, tier_weight=4.0, num_tokens=4 and varies
ONLY per_class, so the curve is a controlled comparison of training-set size:

    per_class=16 ->  64 examples   ALREADY TRAINED as virtual_adapter_e2_seed101.pt
                                   (E2 seed-variance run; 81.25% on the 80-event set)
    per_class=32 -> 128 examples   trained here
    per_class=64 -> 256 examples   trained here

The per_class=16 anchor is reused rather than retrained: it is the identical
configuration at the identical seed, so retraining could only introduce drift.

DS1 supplies 45,841 N / 3,788 V / 944 S / 414 F beats, so F is the binding class
and per_class up to 128 is available; 64 is chosen as the top rung to stay well
inside that and inside the evening's compute budget.

WHAT THIS ESTABLISHES AND WHAT IT DOES NOT.
It is a single seed. E2 measured seed noise at fixed k=4 as std 8.7pp / range
16.2pp, so a difference smaller than roughly 16pp between adjacent rungs is not
separable from initialisation noise on one seed alone. What the curve can support
is a direction and an approximate magnitude at seed 101; establishing that
training-set size matters in general would need repeated seeds per rung, which is
the same limitation E1 discloses for k. State this wherever the result is used.

Additive only: writes its own checkpoints and results file, touches nothing that
feeds the headline.

Run (from repo root):
    python e4_training_set_size.py
"""
import json
from pathlib import Path

import torch

from ablation_common import prepare_events, run_arm_b_eval, summarize_arm_b
from reasoning.adapter_arm import load_trained_adapter
from reasoning.adapter_training import TrainingConfig, train_adapter
from reasoning.model_loader import load_model

SEED = 101
EPOCHS = 3
CKPT_DIR = Path("reasoning/checkpoints")
RESULTS_PATH = Path("results/e4_training_set_size_results.json")

# per_class -> checkpoint. 16 is E2's existing seed-101 run, reused as the anchor.
ANCHOR_PER_CLASS = 16
ANCHOR_CKPT = CKPT_DIR / "virtual_adapter_e2_seed101.pt"
NEW_PER_CLASS = [32, 64]


def checkpoint_for(per_class: int) -> Path:
    return CKPT_DIR / f"virtual_adapter_e4_pc{per_class}_seed{SEED}.pt"


def main():
    print("Preparing the 80-event set (chronological replay, same as Day 6/7)...")
    prepared = prepare_events(note="E4 training-set-size sweep")
    print(f"  {len(prepared)} events prepared\n")

    model, processor = load_model()
    rungs = {}

    for per_class in [ANCHOR_PER_CLASS] + NEW_PER_CLASS:
        if per_class == ANCHOR_PER_CLASS:
            ckpt = ANCHOR_CKPT
            print(f"--- per_class={per_class} ({per_class * 4} examples): "
                  f"reusing E2's seed-{SEED} checkpoint, no retraining ---")
        else:
            ckpt = checkpoint_for(per_class)
            if ckpt.exists():
                print(f"--- per_class={per_class}: checkpoint exists, skipping training "
                      f"(delete to force) ---")
            else:
                print(f"--- per_class={per_class} ({per_class * 4} examples): training, "
                      f"seed={SEED}, epochs={EPOCHS} ---")
                cfg = TrainingConfig(
                    per_class=per_class, epochs=EPOCHS, learning_rate=0.002,
                    loss_mode="full", tier_weight=4.0, max_examples=None,
                    num_tokens=4, seed=SEED,
                )
                train_adapter(cfg, output_path=ckpt)

        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        examples = int(saved["example_count"])
        losses = [float(x) for x in saved["losses"]]

        adapter = load_trained_adapter(str(ckpt), model)
        results = run_arm_b_eval(prepared, model, processor, adapter,
                                 label=f"pc={per_class}")
        summary = summarize_arm_b(results)

        rungs[per_class] = {
            "per_class": per_class,
            "examples": examples,
            "epochs": int(saved["config"]["epochs"]),
            "gradient_updates": examples * int(saved["config"]["epochs"]),
            "seed": SEED,
            "checkpoint": str(ckpt),
            "retrained_here": per_class != ANCHOR_PER_CLASS,
            "loss_curve": losses,
            "final_train_loss": losses[-1],
            **summary,
            "events": results,
        }
        print(f"  accuracy {summary.get('accuracy', float('nan')):.4f}\n")

        # Variable-length examples fragment the allocator across sequential runs;
        # e1_ablation.py hit a real OOM this way. Clear between rungs.
        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        RESULTS_PATH.parent.mkdir(exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "question": "does a larger training set raise Arm B accuracy?",
                "controls": {"seed": SEED, "epochs": EPOCHS, "num_tokens": 4,
                             "learning_rate": 0.002, "tier_weight": 4.0},
                "caveat": ("Single seed. E2 seed noise at fixed k=4 is std 8.7pp / "
                           "range 16.2pp, so adjacent-rung differences below ~16pp are "
                           "not separable from initialisation noise here."),
                "rungs": [rungs[k] for k in sorted(rungs)],
            }, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"{'per_class':>10} {'examples':>9} {'updates':>8} {'final loss':>11} {'accuracy':>10}")
    print("=" * 70)
    for k in sorted(rungs):
        r = rungs[k]
        print(f"{r['per_class']:>10} {r['examples']:>9} {r['gradient_updates']:>8} "
              f"{r['final_train_loss']:>11.3f} {r['accuracy'] * 100:>9.2f}%")
    print("=" * 70)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
