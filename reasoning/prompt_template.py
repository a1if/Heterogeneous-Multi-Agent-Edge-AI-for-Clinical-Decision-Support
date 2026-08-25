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

Few-shot examples (added post-Day-6, to fix a training-parity confound):
Arm B's adapter was trained toward the escalation rule; Arm A was originally
pure zero-shot, an unfair comparison. Fix: 3 real DS1 examples (never DS2 —
using held-out evaluation data as demonstrations would be a worse leak than
the confound being fixed), one per urgency tier, loaded from the static,
committed reasoning/few_shot_examples.json (see select_few_shot_examples.py).
This is expected to substantially increase Arm A's token count and likely
its latency — a disclosed, honest side effect of making the comparison fair,
not a change to Arm B.
"""
import json
import os

from reasoning.fixed_context import get_fixed_context_for_class

_FEW_SHOT_PATH = os.path.join(os.path.dirname(__file__), "few_shot_examples.json")

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
    '  "urgency_tier": one of "routine", "priority", "urgent". Apply this rule '
    "exactly: use \"routine\" ONLY for a normal (N-class) beat. Use \"urgent\" if "
    "EITHER of these is true on its own, independently — each is sufficient by "
    "itself, do not require both: (a) consecutive_abnormal_beats is 3 or more, "
    "OR (b) this is a ventricular or fusion beat with confidence greater than "
    "0.85. Criterion (b) alone is sufficient for \"urgent\" even when "
    "consecutive_abnormal_beats is only 1 — do not downgrade to \"priority\" "
    "in that case just because the background context describes an isolated "
    "occurrence as usually benign; confidence above 0.85 overrides that framing. "
    "Use \"priority\" only for a non-normal beat that meets NEITHER urgent "
    "criterion above — a non-normal beat is never \"routine\".\n"
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


def _load_few_shot_block() -> str:
    """Loads the 3 static DS1 examples and formats each identically to how
    the real event will be shown (same _extract_prompt_fields subset),
    paired with its correct canonical output. Deterministic: same file
    content -> byte-identical output, required for the freeze gate."""
    if not os.path.exists(_FEW_SHOT_PATH):
        raise RuntimeError(
            f"{_FEW_SHOT_PATH} not found. Run select_few_shot_examples.py first "
            "to generate the static few-shot example file."
        )
    with open(_FEW_SHOT_PATH) as f:
        examples = json.load(f)

    parts = []
    for i, ex in enumerate(examples, 1):
        fields = _extract_prompt_fields(ex["health_event"])
        parts.append(
            f"Example {i} (input):\n{json.dumps(fields, indent=2)}\n"
            f"Example {i} (correct output):\n{json.dumps(ex['canonical_output'], indent=2)}"
        )
    return "\n\n".join(parts)


def build_baseline_prompt(health_event: dict) -> str:
    """Pure function: same input -> byte-identical output, every call.
    No timestamps, no randomness, no hidden state — required for the
    baseline-freeze gate (test_baseline_is_frozen).

    NOTE: _load_few_shot_block() exists but is deliberately NOT called here
    yet. Diagnosis (arm_a_disagreement_diagnosis.json) found the accuracy
    gap traces to two precise, cheap-to-fix content gaps (missing confidence
    criterion in fixed_context.py; underspecified label-mapping rule in
    OUTPUT_INSTRUCTIONS) rather than a fundamental need for worked examples.
    Testing this fix in isolation first — one variable at a time — before
    deciding whether few-shot examples are still needed on top of it."""
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
