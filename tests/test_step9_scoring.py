from reasoning.claim_scorer import extract_claims, score_text
from reasoning.multiflag_target import multiflag_target, score


def event(label="V", conf=0.9, top3=(("V", 0.9), ("N", 0.06), ("S", 0.04)), hr=75.0, rr=800.0, run=1, urgent=True):
    return {
        "classification": {"label": label, "confidence": conf,
                           "top_3": [{"label": l, "confidence": c} for l, c in top3]},
        "signal_features": {"heart_rate_bpm": hr, "rr_interval_ms": rr,
                            "qrs_duration_ms": 40.0, "beat_morphology": "wide_complex"},
        "segment_metadata": {"signal_quality_index": 0.87},
        "clinical_flags": {"requires_urgent_review": urgent, "consecutive_abnormal_beats": run},
    }


def test_multiflag_target_flags():
    t = multiflag_target(event(conf=0.55, top3=(("V", 0.55), ("N", 0.40), ("S", 0.05)), hr=120, urgent=False))
    assert t == {"urgency_tier": "priority", "flags": ["confirm_classification", "rate_out_of_range"]}
    assert multiflag_target(event())["flags"] == []


def test_multiflag_score_exact_match_needs_tier_and_flags():
    t = {"urgency_tier": "urgent", "flags": ["rate_out_of_range"]}
    assert score({"urgency_tier": "urgent", "flags": ["rate_out_of_range"]}, t)["exact_match"]
    s = score({"urgency_tier": "urgent", "flags": []}, t)
    assert s["tier_correct"] and not s["exact_match"]


def test_supraventricular_is_not_read_as_ventricular():
    kinds = {(c["kind"], c["value"]) for c in extract_claims("A supraventricular ectopic beat.")}
    assert ("class", "S") in kinds and ("class", "V") not in kinds


def test_supported_and_unsupported_claims():
    ev = event(hr=75.0, rr=800.0, run=1)
    ok = score_text("Ventricular ectopic beat at 76 bpm with RR interval of 790 ms, confidence 0.9.", ev)
    assert ok["n_claims"] == 4 and ok["n_unsupported"] == 0
    bad = score_text("Fusion beat, 3 consecutive abnormal beats, heart rate 120 bpm.", ev)
    assert {c["kind"] for c in bad["unsupported"]} == {"class", "run", "rate"}


def test_no_checkable_claims():
    assert score_text("Routine monitoring is appropriate.", event()) == {
        "n_claims": 0, "n_unsupported": 0, "unsupported": []}


def test_class_notation_is_recognised():
    kinds = {(c["kind"], c["value"]) for c in extract_claims("The isolated F-class event needs review.")}
    assert kinds == {("class", "F")}
