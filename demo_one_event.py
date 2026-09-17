"""Viva demonstration: one real held-out event through both arms.

Runs a single DS2 event end to end so the two interfaces can be shown side by
side in a few minutes: the ECG model's output, the JSON Arm A sends, the four
virtual tokens Arm B sends instead, both generations, and the paired cost.
Everything is the production path -- the same functions the 80-event
comparison calls -- on one event, so nothing shown here is a mock-up.

    python demo_one_event.py                 # idx 18566: V beat, both arms correct
    python demo_one_event.py --index 1905    # the Figure 3.2 event: one of Arm B's four misses
    python demo_one_event.py --list          # the 80 headline events

Gemma loads on the first call (~30-60 s on the RTX 5070); run the script
once before the demonstration so the model is warm, then run it live.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from ablation_common import prepare_events
from reasoning.adapter_arm import load_trained_adapter, run_adapter_arm_timed
from reasoning.baseline_arm import run_baseline_arm_timed
from reasoning.model_loader import load_model
from reasoning.prompt_template import build_baseline_prompt
from reasoning.training_targets import urgency_tier_from_event

ADAPTER_CHECKPOINT = "reasoning/checkpoints/virtual_adapter_day5_larger.pt"
DEFAULT_INDEX = 18566            # V beat, reference urgent, both arms correct on the headline pass
FIGURE_3_2_INDEX = 1905          # the Figure 3.2 event; Arm B answered routine against a priority reference
RULE = ("reference tier: urgent if consecutive_abnormal_beats >= 3 OR a V/F beat "
        "with confidence > 0.85; routine only for N; otherwise priority")


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=DEFAULT_INDEX, help="DS2 beat index")
    ap.add_argument("--list", action="store_true", help="print the 80 headline events and exit")
    args = ap.parse_args()

    # --- 1. the event, exactly as the headline run prepares it ----------------
    hr("1. ECG model (frozen CNN-LSTM) reads the beat")
    prepared = prepare_events(note=" (headline set)")
    if args.list:
        for it in prepared:
            print(f"  idx {it['idx']:>6}  record {it['record_id']}  true {it['true_class']}"
                  f"  predicted {it['predicted_class']}  ref {it['reference_tier']}")
        return
    item = next((it for it in prepared if it["idx"] == args.index), None)
    if item is None:
        sys.exit(f"index {args.index} is not one of the 80 headline events (use --list)")

    ev = item["health_event"]
    print(f"  DS2 index {item['idx']}, record {item['record_id']}")
    print(f"  true AAMI class:        {item['true_class']}")
    print(f"  ECG model predicted:    {ev['classification']['label']}  "
          f"(confidence {ev['classification']['confidence']:.3f})")
    print(f"  consecutive abnormal:   {ev['clinical_flags']['consecutive_abnormal_beats']}")
    print(f"  32-dim context vector:  {np.round(item['context_vector'][:6], 3).tolist()} ...")
    print(f"\n  {RULE}")
    print(f"  -> reference tier for this event: {item['reference_tier']}   "
          f"(computed from the ECG model's output, never shown to either arm)")
    if ev["classification"]["label"] != item["true_class"]:
        print(f"\n  NOTE: the ECG model's prediction ({ev['classification']['label']}) differs from the "
              f"MIT-BIH annotation ({item['true_class']}).\n"
              f"  Both arms are scored against the ECG model's output, so a {ev['classification']['label']}-class "
              f"answer is the correct one here (Section 3.7).\n"
              f"  The ECG model agrees with the annotation on 39 of these 80 events (48.8%, Table 4.3); "
              f"the interface is not the source of that error.")

    # --- 2. what each arm sends ------------------------------------------------
    hr("2. Arm A sends JSON text")
    prompt = build_baseline_prompt(ev)
    start = prompt.index("--- Event data ---")
    end = prompt.index("--- Instructions ---")
    print(prompt[start:end].rstrip())
    print(f"\n  [{len(prompt.encode('utf-8'))} bytes of prompt; the event block above is what Arm B replaces]")

    hr("3. Arm B sends 4 virtual tokens instead")
    model, processor = load_model()
    adapter = load_trained_adapter(ADAPTER_CHECKPOINT, model)
    with torch.no_grad():
        ctx = torch.from_numpy(item["context_vector"]).float().unsqueeze(0).to(
            model.get_input_embeddings().weight.device)
        vt = adapter(ctx)[0].float().cpu().numpy()
    print(f"  adapter: Linear(32 -> 4 x 2560), {sum(p.numel() for p in adapter.parameters()):,} parameters")
    print(f"  output shape {vt.shape}; per-token L2 norm {np.round(np.linalg.norm(vt, axis=1), 1).tolist()}")
    print(f"  token 1, first 8 of 2560 values: {np.round(vt[0, :8], 3).tolist()}")
    print("  (these four vectors go where the JSON block was; system prompt and instructions are identical)")

    # --- 3. both generations, timed --------------------------------------------
    hr("4. Gemma 4 E4B answers -- Arm A (JSON)")
    a = run_baseline_arm_timed(ev, perception_agent=None)
    print(json.dumps(a["result"], indent=2))

    hr("5. Gemma 4 E4B answers -- Arm B (virtual tokens)")
    b = run_adapter_arm_timed(ev, item["context_vector"], model, processor, adapter)
    print(json.dumps(b["result"], indent=2))

    # --- 4. the paired comparison ----------------------------------------------
    hr("6. Same event, two interfaces")
    ref = item["reference_tier"]
    rows = [
        ("prompt tokens", a["prompt_tokens"], b["prompt_tokens"]),
        ("output tokens", a["output_tokens"], b["output_tokens"]),
        ("generation latency (ms)", round(a["generation_duration_ms"]), round(b["generation_duration_ms"])),
        ("time to first token (ms)", round(a["time_to_first_token_ms"]), round(b["time_to_first_token_ms"])),
        ("urgency tier", a["result"]["urgency_tier"], b["result"]["urgency_tier"]),
        ("matches reference", a["result"]["urgency_tier"] == ref, b["result"]["urgency_tier"] == ref),
    ]
    print(f"  {'':<26}{'Arm A':>12}{'Arm B':>12}")
    for name, va, vb in rows:
        print(f"  {name:<26}{str(va):>12}{str(vb):>12}")
    saved = a["prompt_tokens"] - b["prompt_tokens"]
    print(f"\n  prompt tokens saved: {saved} ({100 * saved / a['prompt_tokens']:.1f}%); "
          f"headline over 80 events: 645.2 -> 486.0 (24.7%)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[{time.time() - t0:.0f} s wall clock]")
