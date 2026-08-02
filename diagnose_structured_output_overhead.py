"""
Diagnostic only — not part of the pipeline. Tests whether the ~5.8s
unaccounted latency in run_baseline_arm_timed is caused by Ollama's
structured-output (format=<schema>) grammar compilation.

Run: python diagnose_structured_output_overhead.py
"""
import ollama
from reasoning.prompt_template import build_baseline_prompt
from reasoning.output_schema import ReasoningOutput
from reasoning.baseline_arm import MODEL_TAG, GENERATION_CONFIG

DUMMY_EVENT = {
    "event_id": "evt_1750262403_0004",
    "timestamp": "2026-06-18T14:00:03.890Z",
    "classification": {
        "label": "V", "description": "Ventricular ectopic beat", "confidence": 0.93,
        "top_3": [{"label": "V", "confidence": 0.93}, {"label": "F", "confidence": 0.05},
                  {"label": "N", "confidence": 0.02}],
    },
    "signal_features": {"rr_interval_ms": 412.0, "qrs_duration_ms": 146.0,
                         "heart_rate_bpm": 146.0, "beat_morphology": "wide_complex"},
    "segment_metadata": {"window_samples": 360, "sample_rate_hz": 360,
                          "lead": "MLII", "signal_quality_index": 0.91},
    "clinical_flags": {"requires_urgent_review": True,
                        "flag_reason": "3 consecutive ventricular ectopic beats (V) detected",
                        "consecutive_abnormal_beats": 3},
}


def run_with_format(prompt):
    response = ollama.chat(
        model=MODEL_TAG,
        messages=[{"role": "user", "content": prompt}],
        format=ReasoningOutput.model_json_schema(),
        options=GENERATION_CONFIG,
    )
    return response


def run_without_format(prompt):
    response = ollama.chat(
        model=MODEL_TAG,
        messages=[{"role": "user", "content": prompt}],
        options=GENERATION_CONFIG,
        # no format= at all — free text
    )
    return response


def report(label, response):
    print(f"\n--- {label} ---")
    print(f"total_duration:      {response['total_duration'] / 1e6:.1f} ms")
    print(f"load_duration:       {response['load_duration'] / 1e6:.1f} ms")
    print(f"prompt_eval_duration:{response['prompt_eval_duration'] / 1e6:.1f} ms")
    print(f"eval_duration:       {response['eval_duration'] / 1e6:.1f} ms")
    accounted = response['load_duration'] + response['prompt_eval_duration'] + response['eval_duration']
    unaccounted = (response['total_duration'] - accounted) / 1e6
    print(f"unaccounted:         {unaccounted:.1f} ms")


def main():
    prompt = build_baseline_prompt(DUMMY_EVENT)

    print("Run 1: WITH format=<schema> (current pipeline behavior)")
    r1 = run_with_format(prompt)
    report("WITH format", r1)

    print("\nRun 2: WITHOUT format= (free text)")
    r2 = run_without_format(prompt)
    report("WITHOUT format", r2)

    print("\n--- Verdict ---")
    if r2['total_duration'] < r1['total_duration'] * 0.5:
        print("Structured output IS the cause — total_duration dropped sharply without format=.")
    else:
        print("Structured output is NOT the main cause — similar total_duration either way. "
              "Look elsewhere (e.g. OLLAMA_NUM_PARALLEL, model_recommendations network call, "
              "or something in Ollama's request scheduling).")


if __name__ == "__main__":
    main()
