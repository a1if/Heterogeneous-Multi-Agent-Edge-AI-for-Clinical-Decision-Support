"""
Baseline arm (Arm A) prompt template.

Deliberate field subset from HealthEventJSON (design decision, agreed during
Reasoning Agent design discussion):

INCLUDED — clinically relevant to the urgency-tier decision:
    classification.label, classification.description, classification.confidence,
    classification.top_3, signal_features.* (all four), segment_metadata.signal_quality_index,
    clinical_flags.consecutive_abnormal_beats

EXCLUDED, deliberately:
    event_id, timestamp                      -> bookkeeping, zero clinical signal
    segment_metadata.window_samples/sample_rate_hz -> always-constant, zero information
    segment_metadata.lead                     -> not decision-relevant here
    clinical_flags.requires_urgent_review,
    clinical_flags.flag_reason                -> THESE ARE THE PRE-COMPUTED ANSWER.
        Including them would let the model copy a rule-based field instead of
        reasoning from clinical features, invalidating the accuracy metric.
        They are used ONLY afterward, as the ground-truth label to score
        both arms against — never shown to the Reasoning Agent.

This is a real, deliberate scope decision affecting the token-count metric
(Day 6) — the brief's "~140-160 tokens" estimate was for the FULL payload
and must be re-measured against this subset once implemented.
"""
import json

from reasoning.fixed_context import get_fixed_context_for_class

SYSTEM_PROMPT = (
    "You are a clinical decision support assistant reviewing a single ECG beat "
    "classification. You will be given structured signal features and fixed "
    "background context for the detected beat class. Based on this information "
    "only, output a structured urgency tier and a brief justification. Do not "
    "invent information not present in the input. This is a research prototype; "
    "your output is not sanctioned clinical advice."
)

OUTPUT_INSTRUCTIONS = (
    "Respond with a JSON object with exactly three fields:\n"
    '  "urgency_tier": one of "routine", "priority", "urgent"\n'
    '  "justification": a one-sentence clinical rationale for the tier\n'
    '  "referenced_guideline_fact": the specific fact from the background '
    "context you relied on, stated in your own words\n"
    "Your justification must explicitly draw on the background context provided."
)


def _extract_prompt_fields(health_event: dict) -> dict:
    """Pulls exactly the deliberate field subset described above."""
    c = health_event["classification"]
    sf = health_event["signal_features"]
    return {
        "classification": {
            "label": c["label"],
            "description": c["description"],
            "confidence": c["confidence"],
            "top_3": c["top_3"],
        },
        "signal_features": {
            "rr_interval_ms": sf["rr_interval_ms"],
            "qrs_duration_ms": sf["qrs_duration_ms"],
            "heart_rate_bpm": sf["heart_rate_bpm"],
            "beat_morphology": sf["beat_morphology"],
        },
        "signal_quality_index": health_event["segment_metadata"]["signal_quality_index"],
        "consecutive_abnormal_beats": health_event["clinical_flags"]["consecutive_abnormal_beats"],
    }


def build_baseline_prompt(health_event: dict) -> str:
    """Pure function: same input -> byte-identical output, every call.
    No timestamps, no randomness, no hidden state — required for the
    baseline-freeze gate (test_baseline_is_frozen)."""
    fields = _extract_prompt_fields(health_event)
    label = health_event["classification"]["label"]
    context = get_fixed_context_for_class(label)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Background context for beat class '{label}' ---\n"
        f"{context}\n\n"
        f"--- Event data ---\n"
        f"{json.dumps(fields, indent=2)}\n\n"
        f"--- Instructions ---\n"
        f"{OUTPUT_INSTRUCTIONS}"
    )
    return prompt
