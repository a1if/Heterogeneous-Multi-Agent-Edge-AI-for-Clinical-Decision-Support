"""Text-interface family for Phase 1 step 4 (docs/analysis_plan.md §3).

A-full is reasoning/prompt_template.build_baseline_prompt, unchanged and frozen
(BASELINE_FROZEN.json). The arms here reuse its scaffold byte for byte (system
prompt, class-specific background context, output instructions) and change only
the "--- Event data ---" payload, so any difference between arms is the payload:

  compact  label, confidence, consecutive_abnormal_beats, signal_quality_index
  label    label only (cannot express the confidence or run criteria of the rule)
  none     empty payload: the scaffold alone. Never generated; used to measure
           each arm's interface-attributable tokens (prompt tokens minus scaffold)
"""
import json

from reasoning.fixed_context import get_fixed_context_for_class
from reasoning.prompt_template import OUTPUT_INSTRUCTIONS, SYSTEM_PROMPT, build_baseline_prompt

PAYLOADS = ("full", "compact", "label", "none")


def _payload(health_event: dict, payload: str) -> dict:
    c = health_event["classification"]
    if payload == "compact":
        return {
            "label": c["label"],
            "confidence": c["confidence"],
            "consecutive_abnormal_beats": health_event["clinical_flags"]["consecutive_abnormal_beats"],
            "signal_quality_index": health_event["segment_metadata"]["signal_quality_index"],
        }
    if payload == "label":
        return {"label": c["label"]}
    raise ValueError(payload)


def build_family_prompt(health_event: dict, payload: str) -> str:
    if payload == "full":
        return build_baseline_prompt(health_event)
    label = health_event["classification"]["label"]
    event_block = "" if payload == "none" else json.dumps(_payload(health_event, payload), indent=2)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Background context for beat class '{label}' ---\n"
        f"{get_fixed_context_for_class(label)}\n\n"
        f"--- Event data ---\n"
        f"{event_block}\n\n"
        f"--- Instructions ---\n"
        f"{OUTPUT_INSTRUCTIONS}"
    )
