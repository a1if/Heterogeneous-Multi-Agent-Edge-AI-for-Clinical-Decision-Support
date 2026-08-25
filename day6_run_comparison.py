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
from collections import defaultdict

import numpy as np
import torch

from perception.perception_agent import PerceptionAgent, replay_selected
from reasoning.adapter_arm import load_trained_adapter, run_adapter_arm_timed
from reasoning.baseline_arm import run_baseline_arm_timed
from reasoning.model_loader import load_model
from reasoning.training_targets import urgency_tier_from_event

ADAPTER_CHECKPOINT = "reasoning/checkpoints/virtual_adapter_day5_larger.pt"
PERCEPTION_CHECKPOINT = "perception/checkpoints/cnn_lstm.pt"
DS2_PATH = "data/processed/ds2_test.npz"
PER_CLASS = 20
MAX_PER_RECORD = 5  # cap so one patient can't dominate a class
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
RESULTS_PATH = "results/day6_results.json"


def select_events(y: np.ndarray, record_ids: np.ndarray) -> list[int]:
    """PER_CLASS events per class (N/S/V/F), capped at MAX_PER_RECORD per patient
    record, deterministic first-occurrence within that cap -- documented, not random."""
    selected = []
    for class_id in range(4):  # N,S,V,F -- Q excluded, zero examples in DS2
        per_record_count = defaultdict(int)
        class_indices = np.flatnonzero(y == class_id)
        picked_for_class = []
        for idx in class_indices:
            rec = int(record_ids[idx])
            if per_record_count[rec] >= MAX_PER_RECORD:
                continue
            picked_for_class.append(int(idx))
            per_record_count[rec] += 1
            if len(picked_for_class) == PER_CLASS:
                break
        if len(picked_for_class) < PER_CLASS:
            raise RuntimeError(
                f"Class {AAMI_CLASSES[class_id]}: only found {len(picked_for_class)}/{PER_CLASS} "
                f"events under the {MAX_PER_RECORD}-per-record cap. Loosen MAX_PER_RECORD or PER_CLASS."
            )
        selected.extend(picked_for_class)
    return selected


def measure_vram(fn, *args, **kwargs):
    """Runs fn, returns (result, peak_vram_mb) isolated to this call."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = fn(*args, **kwargs)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
    return result, peak_mb


def main():
    print("Loading Perception Agent + DS2...")
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)
    data = np.load(DS2_PATH)
    X, y, rr, record_ids = data["features"], data["labels"], data["rr_interval_ms"], data["record_ids"]

    selected = select_events(y, record_ids)
    print(f"Selected {len(selected)} events: {PER_CLASS} per class (N/S/V/F), "
          f"capped at {MAX_PER_RECORD}/record")

    print("Building Perception outputs via chronological per-record replay "
          "(matches adapter_training.py's approach -- consecutive_abnormal_beats "
          "is stateful and depends on true beat order within each record; "
          "predicting directly on scattered selected indices would give it "
          "arbitrary, incorrect state)...")
    replayed = replay_selected(agent, X, rr, record_ids, selected)
    prepared = []
    for i in selected:  # preserve original class-grouped order
        health_event, context_vector = replayed[i]
        prepared.append({
            "idx": i,
            "true_class": AAMI_CLASSES[y[i]],
            "predicted_class": health_event["classification"]["label"],
            "record_id": int(record_ids[i]),
            "health_event": health_event,
            "context_vector": context_vector,
            "reference_tier": urgency_tier_from_event(health_event),
        })

    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
