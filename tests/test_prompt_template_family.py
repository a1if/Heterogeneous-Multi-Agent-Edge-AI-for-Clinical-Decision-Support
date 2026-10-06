import json

from reasoning.prompt_template import build_baseline_prompt
from reasoning.prompt_template_family import build_family_prompt


def event():
    return {
        "classification": {"label": "V", "description": "Ventricular ectopic beat", "confidence": 0.91,
                           "top_3": [{"label": "V", "confidence": 0.93}, {"label": "N", "confidence": 0.05},
                                     {"label": "F", "confidence": 0.02}]},
        "signal_features": {"rr_interval_ms": 610.0, "qrs_duration_ms": 44.0, "heart_rate_bpm": 98.4,
                            "beat_morphology": "narrow_complex"},
        "segment_metadata": {"signal_quality_index": 0.87},
        "clinical_flags": {"requires_urgent_review": True, "consecutive_abnormal_beats": 1},
    }


def _event_block(prompt):
    return prompt.split("--- Event data ---\n")[1].split("\n\n--- Instructions ---")[0]


def test_full_is_the_frozen_baseline():
    assert build_family_prompt(event(), "full") == build_baseline_prompt(event())


def test_arms_differ_only_in_the_payload():
    full = build_family_prompt(event(), "full")
    for payload in ("compact", "label", "none"):
        p = build_family_prompt(event(), payload)
        assert p.replace(_event_block(p), "") == full.replace(_event_block(full), "")


def test_payload_contents():
    assert json.loads(_event_block(build_family_prompt(event(), "compact"))) == {
        "label": "V", "confidence": 0.91, "consecutive_abnormal_beats": 1, "signal_quality_index": 0.87}
    assert json.loads(_event_block(build_family_prompt(event(), "label"))) == {"label": "V"}
    assert _event_block(build_family_prompt(event(), "none")) == ""
