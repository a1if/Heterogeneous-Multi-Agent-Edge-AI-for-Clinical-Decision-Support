"""
Smoke test for Arm B's REAL generate() path — not teacher-forced training.

This is the first time model.generate() gets called with the adapter's
inputs_embeds + per_layer_inputs. Run this BEFORE building the full 60-event
Day 6 comparison harness, since this exact code path has never been
exercised (adapter_training.py only ever calls forward(), never generate()).

Run (from repo root):
    python diagnostics/smoke_test_adapter_generation.py

Uses the Day 5 trained checkpoint (virtual_adapter_day5_larger.pt) and one
real DS2 event, held out from all training so far.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for perception./reasoning.

from perception.perception_agent import PerceptionAgent
from reasoning.adapter_arm import load_trained_adapter, run_adapter_arm_timed
from reasoning.model_loader import load_model
from project_config import DS2_PATH, PERCEPTION_CHECKPOINT

CHECKPOINT_PATH = "reasoning/checkpoints/virtual_adapter_day5_larger.pt"


def main():
    print("Loading Perception Agent (trained checkpoint)...")
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)

    print("Loading DS2, picking one real held-out event (first V-class, index-deterministic)...")
    data = np.load(DS2_PATH)
    X, y, rr = data["features"], data["labels"], data["rr_interval_ms"]
    v_indices = np.flatnonzero(y == 2)  # V is class index 2 (N=0,S=1,V=2,F=3,Q=4)
    idx = int(v_indices[0])
    print(f"Selected DS2 index {idx}, true label V")

    health_event = agent.predict(X[idx], rr_interval_ms=float(rr[idx]), event_seq=idx)
    context_vector = agent.get_last_context_vector()
    print(f"Perception output: label={health_event['classification']['label']} "
          f"confidence={health_event['classification']['confidence']:.3f}")

    # Free the Perception model's GPU memory before Gemma loads, same pattern
    # as build_real_training_examples in adapter_training.py.
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nLoading Gemma 4 E4B + trained adapter...")
    model, processor = load_model()
    adapter = load_trained_adapter(CHECKPOINT_PATH, model)
    print(f"Adapter loaded: {adapter.num_tokens} virtual tokens, "
          f"{sum(p.numel() for p in adapter.parameters())} parameters")

    print("\nCalling run_adapter_arm_timed (REAL generate(), first time this path has run)...")
    try:
        result = run_adapter_arm_timed(health_event, context_vector, model, processor, adapter)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"FAILED: {type(e).__name__}: {e}")
        print(f"{'='*60}")
        print("This is exactly what the smoke test is for -- report this "
              "error back rather than building the full Day 6 harness on "
              "top of an unverified generate() path.")
        raise

    print(f"\n{'='*60}")
    print("SUCCESS -- Arm B generate() path works")
    print(f"{'='*60}")
    print("\n--- Parsed structured output ---")
    print(json.dumps(result["result"], indent=2))
    print(f"\nprompt_tokens={result['prompt_tokens']} "
          f"(includes {adapter.num_tokens} virtual tokens)")
    print(f"output_tokens={result['output_tokens']}")
    print(f"generation_duration_ms={result['generation_duration_ms']:.1f}")
    print(f"time_to_first_token_ms={result['time_to_first_token_ms']:.1f}")
    print(f"parse_attempts={result['parse_attempts']}")


if __name__ == "__main__":
    main()
