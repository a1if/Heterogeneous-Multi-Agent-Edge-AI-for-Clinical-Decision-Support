"""E4b: training-set-size comparison at the largest size the sampler can reach.

E4's per_class=32 run crashed: build_real_training_examples() scans DS1 beat by
beat and calls PerceptionAgent.predict() on every beat, whose HealthEventJSON
schema rejects heart_rate_bpm outside [5, 300]. Three beats in record 207 carry
~15 s RR intervals (indices 33144, 33152, 34550; the first gives 4.07 bpm), so
any scan reaching index 33144 dies. per_class=32 needs depth 34795. It never had
a chance.

per_class=27 needs depth 31756, which stops 1,388 beats short of the first bad
beat, so it is reachable with NO change to shared code. 27 is the largest
per_class available before that wall (only 27 F-class beats precede index 33144).

This gives the controlled comparison E4 was designed for, one rung smaller:

    per_class=16 ->  64 examples   virtual_adapter_e2_seed101.pt   81.25%
    per_class=27 -> 108 examples   trained here

seed=101, epochs=3, learning_rate=0.002, tier_weight=4.0, num_tokens=4 are
identical across both, so training-set size is the only variable. The 64-example
anchor is NOT re-evaluated: E4 already reproduced it at exactly 81.25%, matching
E2's recorded figure, so re-running could only introduce drift.

WHAT THIS CAN AND CANNOT SHOW. One seed, two points, a 1.7x change in training
data. E2 measured seed noise at fixed k=4 as std 8.7pp / range 16.2pp, so a
difference below roughly 16pp is not separable from initialisation noise here.
A rise would be directional evidence at seed 101, not a general claim about
training-set size. Report it that way.

Additive only. Writes its own checkpoint and merges into E4's results file.

Run (from repo root):
    python e4b_per_class_27.py
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
PER_CLASS = 27
FIRST_BAD_INDEX = 33144
EXPECTED_SCAN_DEPTH = 31756

CKPT = Path(f"reasoning/checkpoints/virtual_adapter_e4_pc{PER_CLASS}_seed{SEED}.pt")
RESULTS_PATH = Path("results/e4_training_set_size_results.json")


def main():
    assert EXPECTED_SCAN_DEPTH < FIRST_BAD_INDEX, "scan would hit the record-207 artefact beats"
    print(f"per_class={PER_CLASS}: scan depth {EXPECTED_SCAN_DEPTH} < first bad beat "
          f"{FIRST_BAD_INDEX}, safe to proceed.\n")

    print("Preparing the 80-event evaluation set (chronological replay)...")
    prepared = prepare_events(note="E4b per_class=27")
    print(f"  {len(prepared)} events prepared\n")

    model, processor = load_model()

    if CKPT.exists():
        print(f"checkpoint exists at {CKPT}, skipping training (delete to force)")
    else:
        print(f"Training per_class={PER_CLASS} ({PER_CLASS * 4} examples), "
              f"seed={SEED}, epochs={EPOCHS}...")
        print("  (the DS1 scan is single-threaded and slow -- expect ~50 min before "
              "the first epoch begins)")
        cfg = TrainingConfig(
            per_class=PER_CLASS, epochs=EPOCHS, learning_rate=0.002,
            loss_mode="full", tier_weight=4.0, max_examples=None,
            num_tokens=4, seed=SEED,
        )
        train_adapter(cfg, output_path=CKPT)

    saved = torch.load(CKPT, map_location="cpu", weights_only=False)
    losses = [float(x) for x in saved["losses"]]
    examples = int(saved["example_count"])

    adapter = load_trained_adapter(str(CKPT), model)
    results = run_arm_b_eval(prepared, model, processor, adapter, label=f"pc={PER_CLASS}")
    summary = summarize_arm_b(results)

    rung = {
        "per_class": PER_CLASS,
        "examples": examples,
        "epochs": EPOCHS,
        "gradient_updates": examples * EPOCHS,
        "seed": SEED,
        "checkpoint": str(CKPT),
        "retrained_here": True,
        "loss_curve": losses,
        "final_train_loss": losses[-1],
        **summary,
        "events": results,
    }

    payload = {"rungs": []}
    if RESULTS_PATH.exists():
        payload = json.load(open(RESULTS_PATH, encoding="utf-8"))
    payload["rungs"] = [r for r in payload.get("rungs", [])
                        if r.get("per_class") != PER_CLASS] + [rung]
    payload["rungs"].sort(key=lambda r: r["per_class"])
    payload["sampler_ceiling_note"] = (
        f"per_class is capped at {PER_CLASS} without a code change: "
        f"build_real_training_examples scans DS1 sequentially and "
        f"PerceptionAgent.predict rejects heart_rate_bpm outside [5,300]; three "
        f"beats in record 207 (indices 33144, 33152, 34550) violate it, the first "
        f"at index {FIRST_BAD_INDEX}. Only 27 F-class beats precede it."
    )
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 66)
    print(f"{'per_class':>10} {'examples':>9} {'updates':>8} {'final loss':>11} {'accuracy':>10}")
    print("=" * 66)
    for r in payload["rungs"]:
        print(f"{r['per_class']:>10} {r['examples']:>9} {r['gradient_updates']:>8} "
              f"{r['final_train_loss']:>11.3f} {r['accuracy'] * 100:>9.2f}%")
    print("=" * 66)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
