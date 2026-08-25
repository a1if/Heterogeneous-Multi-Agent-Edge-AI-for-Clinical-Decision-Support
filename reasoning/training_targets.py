"""Deterministic Day 4 supervision targets for the frozen-LLM adapter.

These are experiment labels, not clinical advice.  They are derived only from
the Perception Agent's existing event fields and its fixed context, so they do
not use an LLM to invent a reference answer or leak held-out DS2 information.
"""
from __future__ import annotations

import json


URGENCY_TIERS = frozenset({"routine", "priority", "urgent"})


def urgency_tier_from_event(health_event: dict) -> str:
    """Map the already-defined prototype escalation rule to the task label.

    ``requires_urgent_review`` is never shown to either reasoning arm.  It is
    used here and later in evaluation as the deterministic reference rule.
    Non-normal, non-escalated events receive ``priority``; normal beats are
    ``routine``.  This supplies all three output classes without adding an
    unrecorded clinical policy.
    """
    if health_event["clinical_flags"]["requires_urgent_review"]:
        return "urgent"
    if health_event["classification"]["label"] == "N":
        return "routine"
    return "priority"


def _guideline_fact(label: str) -> str:
    return {
        "N": "A normal sinus pattern calls for routine ongoing monitoring.",
        "S": "Frequent or sustained supraventricular ectopy warrants confirmation and clinical correlation.",
        "V": "Frequent ventricular ectopy or three or more beats in a row warrants prompt review.",
        "F": "Repeated fusion beats alongside abnormal beats require the same escalation as sustained ventricular ectopy.",
        "Q": "An uncertain classification requires repeat recording or manual review before interpretation.",
    }[label]


def _justification(health_event: dict, tier: str) -> str:
    label = health_event["classification"]["label"]
    count = health_event["clinical_flags"]["consecutive_abnormal_beats"]
    if tier == "urgent":
        if count >= 3:
            return f"{count} consecutive abnormal beats meet the prototype escalation rule for urgent review."
        return f"The {label}-class event meets the prototype escalation rule for urgent review."
    if tier == "priority":
        return f"The isolated {label}-class event needs priority clinical correlation under the fixed background context."
    return "The event is a normal sinus pattern, so routine ongoing monitoring is appropriate."


def canonical_reasoning_target(health_event: dict) -> str:
    """Return a byte-stable structured completion for full teacher forcing."""
    label = health_event["classification"]["label"]
    tier = urgency_tier_from_event(health_event)
    return json.dumps(
        {
            "urgency_tier": tier,
            "justification": _justification(health_event, tier),
            "referenced_guideline_fact": _guideline_fact(label),
        },
        separators=(",", ":"),
    )


def tier_target_prefix(health_event: dict) -> str:
    """Short completion used by the fast, tier-focused Day 4 objective."""
    return '{"urgency_tier":"' + urgency_tier_from_event(health_event) + '"}'
