import json

from reasoning.training_targets import canonical_reasoning_target, tier_target_prefix, urgency_tier_from_event


def _event(label="N", urgent=False, consecutive=0):
    return {
        "classification": {"label": label},
        "clinical_flags": {"requires_urgent_review": urgent, "consecutive_abnormal_beats": consecutive},
    }


def test_reference_rule_has_all_three_urgency_tiers():
    assert urgency_tier_from_event(_event("N")) == "routine"
    assert urgency_tier_from_event(_event("S")) == "priority"
    assert urgency_tier_from_event(_event("V", urgent=True)) == "urgent"


def test_canonical_target_is_stable_structured_json():
    event = _event("V", urgent=True)
    target = canonical_reasoning_target(event)
    assert target == canonical_reasoning_target(event)
    parsed = json.loads(target)
    assert parsed["urgency_tier"] == "urgent"
    assert parsed["justification"]
    assert parsed["referenced_guideline_fact"]


def test_tier_target_is_short_valid_json_completion():
    assert json.loads(tier_target_prefix(_event("S"))) == {"urgency_tier": "priority"}
