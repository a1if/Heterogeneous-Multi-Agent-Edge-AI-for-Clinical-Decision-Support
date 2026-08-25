"""
Manual smoke test — NOT pytest. Run directly to see raw output/errors against
your real local transformers + Gemma 4 E4B instance (bitsandbytes 4-bit).

Run (from repo root):
    python diagnostics/smoke_test_baseline_arm.py

This uses perception_agent=None (dummy event, no real Perception Agent yet)
to isolate and test only the Reasoning Agent side: prompt building, HF
generation, JSON extraction/parsing, and timing instrumentation.

First run will be slow (~model load into VRAM). Run it twice in the same
process invocation isn't possible from the CLI (each `python` call is a new
process, so the model reloads every time) — to see true warm-vs-cold timing,
you'd need to call run_baseline_arm_timed twice within one Python session.
This script does exactly that, so you get a real cold vs warm comparison.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for perception./reasoning.

from reasoning.baseline_arm import run_baseline_arm_timed
from reasoning.model_loader import MODEL_ID

DUMMY_EVENT = {
    "event_id": "evt_1750262403_0004",
    "timestamp": "2026-06-18T14:00:03.890Z",
    "classification": {
        "label": "V",
        "description": "Ventricular ectopic beat",
        "confidence": 0.93,
        "top_3": [
            {"label": "V", "confidence": 0.93},
            {"label": "F", "confidence": 0.05},
            {"label": "N", "confidence": 0.02},
        ],
    },
    "signal_features": {
        "rr_interval_ms": 412.0,
        "qrs_duration_ms": 146.0,
        "heart_rate_bpm": 146.0,
        "beat_morphology": "wide_complex",
    },
    "segment_metadata": {
        "window_samples": 360,
        "sample_rate_hz": 360,
        "lead": "MLII",
        "signal_quality_index": 0.91,
    },
    "clinical_flags": {
        "requires_urgent_review": True,
        "flag_reason": "3 consecutive ventricular ectopic beats (V) detected",
        "consecutive_abnormal_beats": 3,
    },
}


def print_run(label, result):
    print(f"\n=== {label} ===")
    print("--- Parsed structured output ---")
    print(json.dumps(result["result"], indent=2))
    print(f"\nprompt_tokens:            {result['prompt_tokens']}")
    print(f"output_tokens:            {result['output_tokens']}")
    print(f"time_to_first_token_ms:   {result['time_to_first_token_ms']:.1f} ms")
    print(f"generation_duration_ms:   {result['generation_duration_ms']:.1f} ms")
    print(f"parse_attempts:           {result['parse_attempts']} "
          f"{'(retried!)' if result['parse_attempts'] > 1 else ''}")


def main():
    print(f"Using MODEL_ID = {MODEL_ID!r} (bitsandbytes 4-bit, via transformers)")

    print("\nRun 1 (cold — includes model load into VRAM)...")
    result_cold = run_baseline_arm_timed(DUMMY_EVENT, perception_agent=None)
    print_run("COLD RUN", result_cold)

    print("\nRun 2 (warm — model already resident)...")
    result_warm = run_baseline_arm_timed(DUMMY_EVENT, perception_agent=None)
    print_run("WARM RUN", result_warm)

    print("\n--- Determinism check ---")
    same_tier = result_cold["result"]["urgency_tier"] == result_warm["result"]["urgency_tier"]
    print(f"urgency_tier identical across runs: {same_tier} "
          f"({result_cold['result']['urgency_tier']!r} vs {result_warm['result']['urgency_tier']!r})")
    if not same_tier:
        print("WARNING: greedy decoding should give identical results — investigate before Day 2 freeze.")


if __name__ == "__main__":
    main()