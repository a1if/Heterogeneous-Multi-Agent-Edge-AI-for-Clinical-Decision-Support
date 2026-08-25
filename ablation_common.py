"""Shared Arm-B-only evaluation harness for E1 (k ablation) and E2 (seed variance).

Both experiments need the exact same 80-event DS2 selection and chronological
Perception replay that day6_run_comparison.py uses, then an Arm-B-only eval
loop per adapter checkpoint (Arm A doesn't depend on the adapter at all, so
re-running it per checkpoint would just waste GPU time -- both experiments
reuse Arm A's existing results/day6_results.json numbers instead). Extracted
here rather than duplicated in e1_ablation.py and e2_seed_variance.py so the
two experiments can't silently drift apart on methodology.
"""
import sys
import time

import numpy as np
import torch

# This system's console/redirected-stdout encoding is not UTF-8. A degenerate
# adapter (seen in practice at low/high k) can generate non-ASCII output (e.g.
# Japanese) that ends up inside a printed failure message below; printing it
# with the platform-default encoding raises UnicodeEncodeError and crashes the
# whole run -- confirmed happening in practice, not theoretical (see the
# separate encoding fix for file I/O in results_ledger.json's history).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from perception.perception_agent import PerceptionAgent, replay_selected
from reasoning.adapter_arm import run_adapter_arm_timed
from reasoning.training_targets import urgency_tier_from_event
from day7_auditability_probe import select_events  # identical 80-event selection as Day 6/7

PERCEPTION_CHECKPOINT = "perception/checkpoints/cnn_lstm.pt"
DS2_PATH = "data/processed/ds2_test.npz"
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]


def measure_vram(fn, *args, **kwargs):
    """Runs fn, returns (result, peak_vram_mb) isolated to this call."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = fn(*args, **kwargs)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
    return result, peak_mb


def prepare_events() -> list[dict]:
    """Chronological per-record replay -> the same 80 events Day 6/7 scored.

    Computed once and reused across every checkpoint evaluated below: Perception
    output does not depend on the adapter, so redoing this per-checkpoint would
    be pure waste.
    """
    print("Loading Perception Agent + DS2, preparing the 80-event selection "
          "(chronological replay, matches day6_run_comparison.py)...")
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)
    data = np.load(DS2_PATH)
    X, y, rr, record_ids = data["features"], data["labels"], data["rr_interval_ms"], data["record_ids"]

    selected = select_events(y, record_ids)
    replayed = replay_selected(agent, X, rr, record_ids, selected)
    prepared = []
    for i in selected:
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
    print(f"Prepared {len(prepared)} events.")
    return prepared


def run_arm_b_eval(prepared: list[dict], model, processor, adapter, *, label: str) -> list[dict]:
    """Runs Arm B (real generate()) over every prepared event for one checkpoint.

    A checkpoint occasionally exhausts run_adapter_arm_timed's parse retries and
    raises (seen in practice at low k: a low-capacity adapter degenerates into
    unparseable output). That's a real, reportable outcome for this ablation --
    not a reason to lose the other 79 events' worth of GPU time. Caught here and
    recorded as a failed event (counts against accuracy, excluded from the
    continuous-metric means) rather than propagating and killing the whole run.
    """
    print(f"\nEvaluating {label}: {len(prepared)} events, Arm B only "
          f"(~10-12 min, do not run other GPU workloads concurrently)...")
    results = []
    run_start = time.time()
    for i, item in enumerate(prepared):
        try:
            out, vram = measure_vram(
                run_adapter_arm_timed, item["health_event"], item["context_vector"], model, processor, adapter
            )
        except RuntimeError as e:
            results.append({
                "idx": item["idx"], "true_class": item["true_class"],
                "predicted_class": item["predicted_class"], "record_id": item["record_id"],
                "reference_tier": item["reference_tier"], "arm": "B",
                "urgency_tier": None, "correct": False,
                "prompt_tokens": None, "output_tokens": None,
                "generation_duration_ms": None, "time_to_first_token_ms": None,
                "peak_vram_mb": None, "parse_attempts": None,
                "generation_failed": True, "error": str(e),
            })
            elapsed = time.time() - run_start
            print(f"[{label}][{i+1}/{len(prepared)}] idx={item['idx']} true={item['true_class']} "
                  f"GENERATION FAILED: {e} elapsed={elapsed:.0f}s")
            continue

        results.append({
            "idx": item["idx"], "true_class": item["true_class"],
            "predicted_class": item["predicted_class"], "record_id": item["record_id"],
            "reference_tier": item["reference_tier"], "arm": "B",
            "urgency_tier": out["result"]["urgency_tier"],
            "correct": out["result"]["urgency_tier"] == item["reference_tier"],
            "prompt_tokens": out["prompt_tokens"], "output_tokens": out["output_tokens"],
            "generation_duration_ms": out["generation_duration_ms"],
            "time_to_first_token_ms": out["time_to_first_token_ms"],
            "peak_vram_mb": vram, "parse_attempts": out["parse_attempts"],
            "generation_failed": False,
        })
        elapsed = time.time() - run_start
        print(f"[{label}][{i+1}/{len(prepared)}] idx={item['idx']} true={item['true_class']} "
              f"B={out['result']['urgency_tier']} ref={item['reference_tier']} elapsed={elapsed:.0f}s")
    return results


def summarize_arm_b(results: list[dict]) -> dict:
    """Mirrors day6_run_comparison.py's print_summary aggregation, as a dict.

    Accuracy/n_failed are computed over ALL events (a failed generation counts
    against accuracy -- it is not a correct answer). Continuous metrics
    (tokens/gen_ms/vram) are means over only the successfully-parsed events,
    since a failed event has no real values for them (existing results, e.g.
    from day6_results.json, have no "generation_failed" key at all -- .get()
    defaults those to False, i.e. successful, which is correct for that data).
    """
    ok = [r for r in results if not r.get("generation_failed", False)]
    n_failed = len(results) - len(ok)
    prompt_tokens = [r["prompt_tokens"] for r in ok]
    gen_ms = [r["generation_duration_ms"] for r in ok]
    vram = [r["peak_vram_mb"] for r in ok if r["peak_vram_mb"] is not None]
    correct = sum(r["correct"] for r in results)
    n = len(results)

    # parse_attempts / output_tokens are recorded per event by run_arm_b_eval but
    # were previously dropped here, leaving the k=1 generation-latency anomaly
    # untestable: the two candidate explanations (retry-driven vs output-length-
    # driven) are exactly what these two fields distinguish. gen_ms measures ONE
    # generate() call, so retries multiply wall time without showing up in it.
    #   - parse_attempts elevated at some k -> retry hypothesis
    #   - output_tokens elevated instead    -> longer-generation hypothesis
    #   - neither                           -> anomaly stays unexplained (reportable)
    # attempts_gt1 is reported alongside the mean because the mean sits near 1.0
    # and hides a handful of 2s/3s that dominate the wall-clock effect.
    parse_attempts = [r["parse_attempts"] for r in ok if r.get("parse_attempts") is not None]
    output_tokens = [r["output_tokens"] for r in ok if r.get("output_tokens") is not None]

    return {
        "n": n,
        "n_failed": n_failed,
        "correct": correct,
        "accuracy": correct / n,
        "tokens_mean": float(np.mean(prompt_tokens)) if prompt_tokens else None,
        "gen_ms_mean": float(np.mean(gen_ms)) if gen_ms else None,
        "vram_mean": float(np.mean(vram)) if vram else None,
        "parse_attempts_mean": float(np.mean(parse_attempts)) if parse_attempts else None,
        "parse_attempts_max": int(np.max(parse_attempts)) if parse_attempts else None,
        "parse_attempts_gt1": int(sum(1 for a in parse_attempts if a > 1)) if parse_attempts else None,
        "output_tokens_mean": float(np.mean(output_tokens)) if output_tokens else None,
        "output_tokens_std": float(np.std(output_tokens, ddof=1)) if len(output_tokens) > 1 else None,
    }
