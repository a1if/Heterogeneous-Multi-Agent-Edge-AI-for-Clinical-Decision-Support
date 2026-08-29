"""
Quantization coupling pilot: does the trained Arm B adapter -- calibrated to
Gemma 4 E4B's 4-bit NF4 embedding geometry -- still work when the SAME
checkpoint is loaded at 8-bit precision instead?

Framing, decided in advance: this bounds the DIRECTION of degradation under
a quantization change of one checkpoint, nothing more. It is explicitly NOT
an attempt to resolve the broader "what happens when the receiver model
itself updates" gap named in Appendix B -- that question covers version
changes, architecture changes, and retraining, none of which this test
touches. Report accordingly regardless of which way the result points: a
directional bound, not a resolved characterization.

Design:
  - Same weights (google/gemma-4-E4B-it), same trained adapter checkpoint,
    unmodified, no retraining -- only the quantization precision changes
    (4-bit NF4 -> 8-bit), via reasoning/model_loader_8bit.py (a separate
    loader/cache, not an edit to model_loader.py -- the study's headline
    4-bit results stay reproducible from the unmodified loader).
  - Pilot scale, not the full 80: a stratified 5-per-class (20 total)
    subset of the SAME 80-event pool day6_run_comparison.py already
    selected (select_events is imported, not re-derived) -- first 5
    events per class in that pool's deterministic order, so this is a
    known subset of an already-validated selection, not a fresh sample.
  - The 4-bit baseline for these same 20 events is NOT re-run here -- it's
    read directly from results/day6_results.json's existing Arm B entries
    (same checkpoint, same adapter, same events, already computed). Keeps
    this a cheap, direction-finding test rather than a second full run.
  - Two things checked per event, not just accuracy: (1) does generation
    stay well-formed (parseable ReasoningOutput JSON within the same
    MAX_PARSE_RETRIES budget used everywhere else), and (2) for the events
    that ARE well-formed, does the predicted urgency tier still match the
    deterministic reference tier.

Run:
    python quantization_coupling_pilot.py
"""
import json
from collections import defaultdict

import numpy as np
import torch

from day6_run_comparison import ADAPTER_CHECKPOINT, DS2_PATH, PERCEPTION_CHECKPOINT, select_events
from perception.perception_agent import PerceptionAgent, replay_selected
from reasoning.adapter_arm import load_trained_adapter, run_adapter_arm_timed
from reasoning.model_loader_8bit import load_model_8bit
from reasoning.training_targets import urgency_tier_from_event

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
PER_CLASS_PILOT = 5
DAY6_RESULTS_PATH = "results/day6_results.json"
RESULTS_PATH = "results/quantization_coupling_pilot_results.json"


def select_pilot_subset(y: np.ndarray, record_ids: np.ndarray) -> list[int]:
    """First PER_CLASS_PILOT events per class, in the SAME deterministic
    order as day6_run_comparison.py's 80-event selection -- a known subset
    of that pool, not a freshly drawn sample."""
    full_selection = select_events(y, record_ids)
    by_class = defaultdict(list)
    for idx in full_selection:
        by_class[int(y[idx])].append(idx)
    subset = []
    for class_id in range(4):  # N,S,V,F -- matches select_events' own scope
        subset.extend(by_class[class_id][:PER_CLASS_PILOT])
    return subset


def load_4bit_baseline(selected: list[int]) -> dict[int, dict]:
    """Reads the already-computed 4-bit Arm B result for each selected idx
    from day6_results.json -- no re-run needed, same checkpoint/adapter/events."""
    with open(DAY6_RESULTS_PATH, "r", encoding="utf-8") as f:
        day6_results = json.load(f)
    by_idx = {r["idx"]: r for r in day6_results if r["arm"] == "B"}
    missing = [idx for idx in selected if idx not in by_idx]
    if missing:
        raise RuntimeError(
            f"Pilot subset includes idx(es) {missing} not found in {DAY6_RESULTS_PATH}'s "
            f"Arm B results -- selection has drifted from the original 80-event pool."
        )
    return {idx: by_idx[idx] for idx in selected}


def main():
    print("Loading Perception Agent + DS2...")
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)
    data = np.load(DS2_PATH)
    X, y, rr, record_ids = data["features"], data["labels"], data["rr_interval_ms"], data["record_ids"]

    selected = select_pilot_subset(y, record_ids)
    print(f"Pilot subset: {len(selected)} events ({PER_CLASS_PILOT}/class, N/S/V/F), "
          f"a subset of day6_run_comparison.py's own 80-event selection.")

    four_bit_baseline = load_4bit_baseline(selected)

    print("Building Perception outputs via chronological per-record replay "
          "(same approach as day6/day7)...")
    replayed = replay_selected(agent, X, rr, record_ids, selected)
    prepared = []
    for idx in selected:
        health_event, context_vector = replayed[idx]
        prepared.append({
            "idx": idx,
            "true_class": AAMI_CLASSES[y[idx]],
            "record_id": int(record_ids[idx]),
            "health_event": health_event,
            "context_vector": context_vector,
            "reference_tier": urgency_tier_from_event(health_event),
        })

    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading Gemma 4 E4B at 8-bit (separate loader/cache from the "
          "study's 4-bit path) + the existing trained adapter, unmodified...")
    model, processor = load_model_8bit()
    adapter = load_trained_adapter(ADAPTER_CHECKPOINT, model)

    print(f"\nRunning {len(prepared)} events through Arm B under 8-bit quantization...\n")

    results = []
    for i, item in enumerate(prepared):
        entry = {
            "idx": item["idx"], "true_class": item["true_class"],
            "record_id": item["record_id"], "reference_tier": item["reference_tier"],
        }
        try:
            out = run_adapter_arm_timed(
                item["health_event"], item["context_vector"], model, processor, adapter
            )
            entry["well_formed"] = True
            entry["urgency_tier_8bit"] = out["result"]["urgency_tier"]
            entry["correct_8bit"] = out["result"]["urgency_tier"] == item["reference_tier"]
            entry["parse_attempts"] = out["parse_attempts"]
            status = f"8bit={entry['urgency_tier_8bit']} ({'correct' if entry['correct_8bit'] else 'WRONG'})"
        except torch.cuda.OutOfMemoryError as e:
            entry["well_formed"] = False
            entry["failure"] = f"CUDA OOM: {e}"
            torch.cuda.empty_cache()
            status = "OOM"
        except RuntimeError as e:
            entry["well_formed"] = False
            entry["failure"] = f"generation/parse failure: {e}"
            status = "MALFORMED (failed to parse after retries)"

        baseline = four_bit_baseline[item["idx"]]
        entry["urgency_tier_4bit"] = baseline["urgency_tier"]
        entry["correct_4bit"] = baseline["correct"]
        results.append(entry)
        print(f"[{i+1}/{len(prepared)}] idx={item['idx']} true={item['true_class']} "
              f"ref={item['reference_tier']} 4bit={entry['urgency_tier_4bit']} "
              f"({'correct' if entry['correct_4bit'] else 'WRONG'}) {status}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to {RESULTS_PATH}")

    print_summary(results)


def print_summary(results: list[dict]) -> None:
    n = len(results)
    well_formed = [r for r in results if r["well_formed"]]
    malformed = [r for r in results if not r["well_formed"]]

    print(f"\n{'=' * 74}\nQUANTIZATION COUPLING PILOT SUMMARY (n={n})\n{'=' * 74}")
    print(f"Well-formed (parseable) generations: {len(well_formed)}/{n} "
          f"({len(well_formed) / n:.1%})")
    if malformed:
        print("Breakdown events:")
        for r in malformed:
            print(f"  idx={r['idx']} true={r['true_class']} -> {r['failure']}")

    if well_formed:
        acc_8bit = sum(r["correct_8bit"] for r in well_formed) / len(well_formed)
        print(f"\nAccuracy on well-formed events, 8-bit: {acc_8bit:.1%} ({len(well_formed)} events)")
    acc_4bit_same_events = sum(r["correct_4bit"] for r in results) / n
    print(f"Accuracy on the SAME {n} events, 4-bit (from day6_results.json, not re-run): "
          f"{acc_4bit_same_events:.1%}")

    if well_formed:
        flipped = [r for r in well_formed if r["correct_4bit"] != r["correct_8bit"]]
        print(f"Events where correctness changed (4-bit vs 8-bit): {len(flipped)}")
        for r in flipped:
            print(f"  idx={r['idx']} true={r['true_class']} ref={r['reference_tier']} "
                  f"4bit={r['urgency_tier_4bit']}({'correct' if r['correct_4bit'] else 'WRONG'}) "
                  f"8bit={r['urgency_tier_8bit']}({'correct' if r['correct_8bit'] else 'WRONG'})")

    print(f"\n{'=' * 74}")
    print("FRAMING (decided in advance): this bounds only the DIRECTION of degradation")
    print("under an NF4 -> 8-bit quantization change of the identical checkpoint and")
    print("adapter. It is a narrower, related question to the broader model-version-")
    print("coupling gap named in Appendix B, and does not resolve that gap. Treat this")
    print("as a directional bound on a pilot-scale sample, not a definitive result --")
    print("regardless of which way the numbers above point.")


if __name__ == "__main__":
    main()
