"""Deviation 20: the A-filtered prompt lists exactly the non-normal beats, with 1-based positions (CPU)."""
import json

from p1_item7_filtered import filtered_content


def ev(label, conf=0.9, run=0):
    return {"classification": {"label": label, "confidence": conf},
            "clinical_flags": {"consecutive_abnormal_beats": run, "requires_urgent_review": False},
            "segment_metadata": {"signal_quality_index": 0.9}}


def payload(text):
    block = text.split("--- Event data ---\n", 1)[1].split("\n\n--- Instructions ---", 1)[0]
    return json.loads(block)


def test_lists_only_non_normal_with_positions():
    events = [ev("N"), ev("N"), ev("V", 0.95, 1), ev("N"), ev("S", 0.7, 0), ev("N")]
    p = payload(filtered_content(events, 1, 4))  # beats 1..4 -> N V N S
    assert p["total_beats"] == 4 and p["normal_beats_not_listed"] == 2
    assert [(b["position"], b["label"]) for b in p["non_normal_beats"]] == [(2, "V"), (4, "S")]
    assert set(p["non_normal_beats"][0]) == {"position", "label", "confidence", "consecutive_abnormal_beats",
                                             "signal_quality_index"}


def test_all_normal_window():
    p = payload(filtered_content([ev("N")] * 5, 0, 5))
    assert p == {"total_beats": 5, "normal_beats_not_listed": 5, "non_normal_beats": []}
