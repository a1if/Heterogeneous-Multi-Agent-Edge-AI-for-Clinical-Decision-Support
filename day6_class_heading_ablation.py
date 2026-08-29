"""
Ablation: quantify whether Arm A's 100% accuracy on the 80-event set depends
on the class-specific heading in its background-context block -- the
asymmetry disclosed (not measured) in reasoning/fixed_context_neutral.py's
docstring. Pre-committed reporting, same discipline as the rest of this
project: two outcomes, both informative --
  - accuracy drops -> quantifies the asymmetry directly, with the specific
    delta and which events flipped, replacing the current qualitative
    disclosure in Section 3.9/5.9.
  - accuracy holds at 100% -> rules out the criticism with evidence rather
    than just disclosure.

Modified copy of day6_run_comparison.py -- NOT an edit to it, so the
headline 100% Arm A result stays reproducible from the unmodified script.
select_events, measure_vram, PER_CLASS, MAX_PER_RECORD, AAMI_CLASSES are
imported from it unchanged, so the 80-event selection is identical, not
re-derived. Everything about Arm A's evaluation stays identical: same
chronological per-record replay for stateful Perception features, same
deterministic reference tier, same interleaved run order (controls
thermal/caching drift the same way the original A/B comparison did), same
greedy decoding settings, same full text pipeline (JSON prompt -> generate
-> decode -> parse). The ONLY change under test is which prompt builder
constructs Arm A's background-context heading:
  - "A_original": build_baseline_prompt      (class-specific, unmodified)
  - "A_neutral":  build_baseline_prompt_neutral (Arm B's class-neutral
                  block, copied verbatim -- not re-paraphrased)

No adapter and no Arm B run here -- this ablation is entirely within Arm A's
own text pipeline.

Run:
    python day6_class_heading_ablation.py
"""
import json
import time

import numpy as np
import torch

from day6_run_comparison import AAMI_CLASSES, MAX_PER_RECORD, PER_CLASS, measure_vram, select_events
from perception.perception_agent import PerceptionAgent, replay_selected
from reasoning.baseline_arm import run_baseline_arm_timed
from reasoning.baseline_arm_class_heading_ablation import run_baseline_arm_neutral_timed
from reasoning.model_loader import load_model
from reasoning.training_targets import urgency_tier_from_event

PERCEPTION_CHECKPOINT = "perception/checkpoints/cnn_lstm.pt"
DS2_PATH = "data/processed/ds2_test.npz"
RESULTS_PATH = "results/class_heading_ablation_results.json"


def main():
    print("Loading Perception Agent + DS2...")
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)
    data = np.load(DS2_PATH)
    X, y, rr, record_ids = data["features"], data["labels"], data["rr_interval_ms"], data["record_ids"]

    selected = select_events(y, record_ids)
    print(f"Selected {len(selected)} events: {PER_CLASS} per class (N/S/V/F), "
          f"capped at {MAX_PER_RECORD}/record -- identical selection to day6_run_comparison.py")

    print("Building Perception outputs via chronological per-record replay "
          "(same approach as day6_run_comparison.py)...")
    replayed = replay_selected(agent, X, rr, record_ids, selected)
    prepared = []
    for i in selected:  # preserve original class-grouped order
        health_event, _context_vector = replayed[i]
        prepared.append({
            "idx": i,
            "true_class": AAMI_CLASSES[y[i]],
            "predicted_class": health_event["classification"]["label"],
            "record_id": int(record_ids[i]),
            "health_event": health_event,
            "reference_tier": urgency_tier_from_event(health_event),
        })

    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading Gemma 4 E4B (Arm A text pipeline only -- no adapter needed)...")
    load_model()  # cached; both heading variants share the one loaded model

    print(f"\nRunning {len(prepared)} events x 2 heading variants, interleaved "
          f"(same thermal/caching-drift control as the original A/B comparison).\n")

    results = []
    run_start = time.time()
    for i, item in enumerate(prepared):
        health_event = item["health_event"]

        orig_out, orig_vram = measure_vram(run_baseline_arm_timed, health_event, perception_agent=None)
        neutral_out, neutral_vram = measure_vram(
            run_baseline_arm_neutral_timed, health_event, perception_agent=None
        )

        for variant_label, out, vram in [
            ("A_original", orig_out, orig_vram), ("A_neutral", neutral_out, neutral_vram)
        ]:
            results.append({
                "idx": item["idx"], "true_class": item["true_class"],
                "predicted_class": item["predicted_class"], "record_id": item["record_id"],
                "reference_tier": item["reference_tier"], "variant": variant_label,
                "urgency_tier": out["result"]["urgency_tier"],
                "correct": out["result"]["urgency_tier"] == item["reference_tier"],
                "prompt_tokens": out["prompt_tokens"], "output_tokens": out["output_tokens"],
                "generation_duration_ms": out["generation_duration_ms"],
                "time_to_first_token_ms": out["time_to_first_token_ms"],
                "peak_vram_mb": vram, "parse_attempts": out["parse_attempts"],
            })

        elapsed = time.time() - run_start
        print(f"[{i+1}/{len(prepared)}] idx={item['idx']} true={item['true_class']} "
              f"orig={orig_out['result']['urgency_tier']} neutral={neutral_out['result']['urgency_tier']} "
              f"ref={item['reference_tier']} elapsed={elapsed:.0f}s")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to {RESULTS_PATH}")

    print_summary(results)


def print_summary(results: list[dict]) -> None:
    print(f"\n{'=' * 70}\nCLASS-HEADING ABLATION SUMMARY\n{'=' * 70}")
    for variant in ["A_original", "A_neutral"]:
        variant_results = [r for r in results if r["variant"] == variant]
        accuracy = sum(r["correct"] for r in variant_results) / len(variant_results)
        print(f"\n{variant} (n={len(variant_results)}): accuracy vs reference tier = {accuracy:.1%}")

    by_idx_orig = {r["idx"]: r for r in results if r["variant"] == "A_original"}
    by_idx_neutral = {r["idx"]: r for r in results if r["variant"] == "A_neutral"}
    flipped = []
    for idx in by_idx_orig:
        o, n = by_idx_orig[idx], by_idx_neutral[idx]
        if o["correct"] != n["correct"] or o["urgency_tier"] != n["urgency_tier"]:
            flipped.append({
                "idx": idx, "true_class": o["true_class"], "reference_tier": o["reference_tier"],
                "original_prediction": o["urgency_tier"], "original_correct": o["correct"],
                "neutral_prediction": n["urgency_tier"], "neutral_correct": n["correct"],
            })

    orig_acc = sum(r["correct"] for r in results if r["variant"] == "A_original") / \
        sum(1 for r in results if r["variant"] == "A_original")
    neutral_acc = sum(r["correct"] for r in results if r["variant"] == "A_neutral") / \
        sum(1 for r in results if r["variant"] == "A_neutral")
    delta_pp = (neutral_acc - orig_acc) * 100

    print(f"\nAccuracy: original (class-specific heading) {orig_acc:.1%} -> "
          f"neutral (class-neutral heading) {neutral_acc:.1%} ({delta_pp:+.1f}pp)")
    print(f"Events that flipped (prediction or correctness changed): {len(flipped)}")
    for f in flipped:
        print(f"  idx={f['idx']} true={f['true_class']} ref={f['reference_tier']} "
              f"original={f['original_prediction']}({'correct' if f['original_correct'] else 'WRONG'}) "
              f"neutral={f['neutral_prediction']}({'correct' if f['neutral_correct'] else 'WRONG'})")

    if abs(delta_pp) < 1e-9:
        print("\nAccuracy holds at parity -- tested directly, no measurable dependence on "
              "the class-specific heading.")
    else:
        print(f"\nAccuracy changed by {delta_pp:+.1f}pp -- the class-specific heading "
              "measurably contributes to Arm A's reported accuracy; see flipped events above.")


if __name__ == "__main__":
    main()
