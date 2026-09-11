"""E3: training-compute ladder -- does more adapter training raise accuracy?

Supervisor question (2026-09-09): "Is there any way to improve the accuracy? Can
you change any parameter to increase the computation cost close to the Arm A and
also see if it increase the accuracy?"

The virtual-token axis is already answered by E1: k=8 is WORSE than k=4 on both
the uncontrolled and the matched-seed curve, so spending more tokens at the
interface does not buy accuracy back.

The training axis was never evaluated, though the checkpoints exist. Three
adapter checkpoints from development share k=4 and every other hyperparameter,
differing only in how much training they received:

    virtual_adapter_day4.pt          16 examples x 2 epochs =  32 updates
    virtual_adapter_day5_epoch1.pt   64 examples x 1 epoch  =  64 updates
    virtual_adapter_day5_larger.pt   64 examples x 3 epochs = 192 updates  <- headline

Only the last was ever evaluated (95.0%, day6_run_comparison.py). This script
evaluates the other two on the IDENTICAL 80-event set, through the identical
Arm B path (ablation_common.run_arm_b_eval), giving a three-point curve.

WHAT THIS ESTABLISHES AND WHAT IT DOES NOT -- read before writing prose.
All three checkpoints predate seed control, so their initialisations are
unrecorded and independent. E2 measured pure seed noise at fixed k=4 as
std 8.7pp / range 16.2pp, which is the same order as any difference this ladder
is likely to show. The two rungs also differ in BOTH training-set size and
epochs, so they do not isolate either. The supportable claim is therefore
"directionally consistent / inconsistent with more training helping", NOT
"accuracy depends on training budget". This is the same limitation E1 already
discloses for k, and it must be stated wherever the result is reported.

Additive only: writes its own results file, retrains nothing, and cannot
perturb the headline.

Run (from repo root):
    python e3_training_compute_ladder.py
"""
import json
from pathlib import Path

import torch

from ablation_common import prepare_events, run_arm_b_eval, summarize_arm_b
from reasoning.adapter_arm import load_trained_adapter
from reasoning.model_loader import load_model

CKPT_DIR = Path("reasoning/checkpoints")
RESULTS_PATH = Path("results/e3_training_ladder_results.json")

# The headline rung is not re-run: its 95.0% comes from day6_run_comparison.py
# and re-evaluating it here could only introduce drift into a published number.
HEADLINE = {
    "checkpoint": "virtual_adapter_day5_larger.pt",
    "accuracy": 0.95,
    "n": 80,
    "source": "day6_run_comparison.py Arm B (not re-run here)",
}

LADDER = [
    "virtual_adapter_day4.pt",
    "virtual_adapter_day5_epoch1.pt",
]


def rung_metadata(name: str) -> dict:
    """Training budget read back from the checkpoint's own saved config."""
    d = torch.load(CKPT_DIR / name, map_location="cpu", weights_only=False)
    cfg = d["config"]
    examples = int(d["example_count"])
    epochs = int(cfg["epochs"])
    return {
        "checkpoint": name,
        "examples": examples,
        "epochs": epochs,
        "gradient_updates": examples * epochs,
        "num_tokens": int(d["num_tokens"]),
        "learning_rate": cfg["learning_rate"],
        "tier_weight": cfg["tier_weight"],
        "final_train_loss": float(d["losses"][-1]),
        "loss_curve": [float(x) for x in d["losses"]],
    }


def main():
    print("Preparing the 80-event set (chronological replay, same as Day 6/7)...")
    prepared = prepare_events(note="E3 training-compute ladder")
    print(f"  {len(prepared)} events prepared\n")

    model, processor = load_model()

    rungs = []
    for name in LADDER:
        meta = rung_metadata(name)
        print(f"--- {name}: {meta['examples']} ex x {meta['epochs']} ep "
              f"= {meta['gradient_updates']} updates, k={meta['num_tokens']} ---")
        if meta["num_tokens"] != 4:
            print(f"  SKIPPED: k={meta['num_tokens']}, not comparable to the k=4 ladder")
            continue

        adapter = load_trained_adapter(str(CKPT_DIR / name), model)
        results = run_arm_b_eval(prepared, model, processor, adapter,
                                 label=f"{meta['gradient_updates']} updates")
        summary = summarize_arm_b(results)
        meta.update(summary)
        meta["per_event"] = results
        rungs.append(meta)
        print(f"  accuracy {summary.get('accuracy', float('nan')):.4f}\n")

        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    headline_meta = rung_metadata(HEADLINE["checkpoint"])
    headline_meta.update(HEADLINE)

    payload = {
        "question": "does more adapter training raise Arm B accuracy?",
        "caveat": ("All rungs predate seed control (initialisations unrecorded and "
                   "independent) and differ in both training-set size and epochs. "
                   "E2 seed noise at fixed k=4 is std 8.7pp / range 16.2pp, the same "
                   "order as any difference here. Directional only."),
        "rungs": sorted(rungs + [headline_meta], key=lambda r: r["gradient_updates"]),
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("=" * 68)
    print(f"{'updates':>9}  {'examples':>8} {'epochs':>7}  {'final loss':>10}  {'accuracy':>9}")
    print("=" * 68)
    for r in payload["rungs"]:
        acc = r.get("accuracy")
        acc_s = f"{acc * 100:8.2f}%" if isinstance(acc, (int, float)) else "       --"
        print(f"{r['gradient_updates']:>9}  {r['examples']:>8} {r['epochs']:>7}  "
              f"{r['final_train_loss']:>10.3f}  {acc_s}")
    print("=" * 68)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
