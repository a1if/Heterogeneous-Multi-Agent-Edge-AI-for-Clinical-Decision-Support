"""
Day 6: formal Arm A vs Arm B comparison on held-out DS2 events.

PER_CLASS events per class (N/S/V/F -- Q absent in DS2), i.e. 4 x PER_CLASS
events total, shared between both arms, capped at MAX_PER_RECORD events per
patient record per class to avoid one patient dominating a class. Interleaved
run order (Arm A then Arm B per event, alternating through all of them) per the
design doc's control table -- not all-A-then-all-B, which would let
thermal/caching drift apply asymmetrically.

Counts are stated as constants deliberately: PER_CLASS was raised 15 -> 20
after this docstring was first written, and the hardcoded "60 events (15 per
class)" text survived the change and disagreed with both the code and the
select_events docstring below (which separately claimed 30/class). The run
actually scores 4 x PER_CLASS = 80 events at present.

Metrics collected per event, per arm: prompt_tokens, output_tokens,
generation_duration_ms, time_to_first_token_ms, peak_vram_mb (PyTorch's own
allocator stats -- more precise for isolating one call's peak than an
nvidia-smi delta, which includes whole-GPU/driver overhead), and accuracy
against the deterministic reference tier.

Run:
    python day6_run_comparison.py
"""
import json
import time

import numpy as np

from reasoning.adapter_arm import load_trained_adapter, run_adapter_arm_timed
from reasoning.baseline_arm import run_baseline_arm_timed
from reasoning.model_loader import load_model
from ablation_common import prepare_events
from project_config import DAY6_RESULTS, ADAPTER_CHECKPOINT, DS2_PATH, MAX_PER_RECORD, PERCEPTION_CHECKPOINT, PER_CLASS, measure_vram, select_events
from perception.model import AAMI_CLASSES  # single source of truth

RESULTS_PATH = DAY6_RESULTS


def main():
    # Chronological per-record replay lives in ablation_common.prepare_events():
    # consecutive_abnormal_beats is stateful and depends on true beat order within
    # each record, so predicting directly on scattered selected indices would give
    # it arbitrary, incorrect state. Shared with the ablations rather than copied,
    # so "the same 80 events" is enforced by construction rather than by comment.
    prepared = prepare_events(
        note=f" ({PER_CLASS}/class N/S/V/F, capped at {MAX_PER_RECORD}/record)")

    print("Loading Gemma 4 E4B + trained adapter (shared by both arms)...")
    model, processor = load_model()
    adapter = load_trained_adapter(ADAPTER_CHECKPOINT, model)

    print(f"\nRunning {len(prepared)} events x 2 arms, interleaved. "
          f"Estimated ~20-25 minutes -- do not run other GPU workloads concurrently.\n")

    results = []
    run_start = time.time()
    for i, item in enumerate(prepared):
        health_event = item["health_event"]

        arm_a_out, arm_a_vram = measure_vram(
            run_baseline_arm_timed, health_event, perception_agent=None
        )
        arm_b_out, arm_b_vram = measure_vram(
            run_adapter_arm_timed, health_event, item["context_vector"], model, processor, adapter
        )

        for arm_label, out, vram in [("A", arm_a_out, arm_a_vram), ("B", arm_b_out, arm_b_vram)]:
            results.append({
                "idx": item["idx"], "true_class": item["true_class"],
                "predicted_class": item["predicted_class"], "record_id": item["record_id"],
                "reference_tier": item["reference_tier"], "arm": arm_label,
                "urgency_tier": out["result"]["urgency_tier"],
                "correct": out["result"]["urgency_tier"] == item["reference_tier"],
                "prompt_tokens": out["prompt_tokens"], "output_tokens": out["output_tokens"],
                "generation_duration_ms": out["generation_duration_ms"],
                "time_to_first_token_ms": out["time_to_first_token_ms"],
                "peak_vram_mb": vram, "parse_attempts": out["parse_attempts"],
            })

        elapsed = time.time() - run_start
        print(f"[{i+1}/{len(prepared)}] idx={item['idx']} true={item['true_class']} "
              f"A={arm_a_out['result']['urgency_tier']} B={arm_b_out['result']['urgency_tier']} "
              f"ref={item['reference_tier']} elapsed={elapsed:.0f}s")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to {RESULTS_PATH}")

    print_summary(results)


def print_summary(results: list[dict]) -> None:
    print(f"\n{'='*70}\nDAY 6 SUMMARY\n{'='*70}")
    for arm in ["A", "B"]:
        arm_results = [r for r in results if r["arm"] == arm]
        prompt_tokens = [r["prompt_tokens"] for r in arm_results]
        gen_ms = [r["generation_duration_ms"] for r in arm_results]
        ttft_ms = [r["time_to_first_token_ms"] for r in arm_results if r["time_to_first_token_ms"] is not None]
        vram = [r["peak_vram_mb"] for r in arm_results if r["peak_vram_mb"] is not None]
        accuracy = sum(r["correct"] for r in arm_results) / len(arm_results)

        print(f"\nArm {arm} (n={len(arm_results)}):")
        print(f"  Accuracy vs reference tier: {accuracy:.1%}")
        print(f"  Prompt tokens: mean={np.mean(prompt_tokens):.1f} "
              f"median={np.median(prompt_tokens):.1f}")
        print(f"  Generation duration (ms): mean={np.mean(gen_ms):.1f} "
              f"p50={np.percentile(gen_ms, 50):.1f} p95={np.percentile(gen_ms, 95):.1f}")
        if ttft_ms:
            print(f"  Time to first token (ms): mean={np.mean(ttft_ms):.1f} "
                  f"p50={np.percentile(ttft_ms, 50):.1f} p95={np.percentile(ttft_ms, 95):.1f}")
        if vram:
            print(f"  Peak VRAM (MB): mean={np.mean(vram):.1f} max={np.max(vram):.1f}")

    a_tokens = np.mean([r["prompt_tokens"] for r in results if r["arm"] == "A"])
    b_tokens = np.mean([r["prompt_tokens"] for r in results if r["arm"] == "B"])
    print(f"\nToken reduction (A -> B): {a_tokens:.0f} -> {b_tokens:.0f} "
          f"({(1 - b_tokens/a_tokens)*100:.1f}% reduction)")

    a_acc = sum(r["correct"] for r in results if r["arm"] == "A") / sum(1 for r in results if r["arm"] == "A")
    b_acc = sum(r["correct"] for r in results if r["arm"] == "B") / sum(1 for r in results if r["arm"] == "B")
    print(f"Accuracy cost (A -> B): {a_acc:.1%} -> {b_acc:.1%} ({(a_acc-b_acc)*100:+.1f}pp)")


if __name__ == "__main__":
    main()
