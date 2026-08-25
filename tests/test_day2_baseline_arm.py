"""
Day 2 gate: baseline arm (Arm A) built end-to-end and FROZEN.

Per the design doc, Section 4/6: Perception -> HealthEventJSON -> prompt
template -> Gemma 4 E4B -> structured urgency tier + justification.

Critical: once test_baseline_is_frozen passes, do not modify the prompt
template, decoding params, or model checkpoint again for the rest of the
7 days. Arm B (adapter, Day 3+) is compared against this exact frozen
configuration. Re-touching it after freezing silently invalidates the
"one controlled variable" design (Section 3 of the design doc).

Run: pytest tests/test_day2_baseline_arm.py -v
Expected state before Day 2 starts: all tests FAIL/ERROR or collect-skip.
Expected state to move to Day 3: all tests PASS, then hash-freeze (see
test_baseline_is_frozen below).
"""
import hashlib
import json
import os
import pytest

from perception.perception_agent import PerceptionAgent  # built in WP3, carries over unchanged
from perception.health_event_schema import HealthEventJSON


# ---------------------------------------------------------------------------
# Fixtures — reuse the same real-event loading path Arm B will use later,
# so both arms are guaranteed to see identical inputs (Section 3 control:
# "Perception Agent + input events").
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_real_event_sample():
    """A small, fixed subset of real MIT-BIH DS2 events (NOT synthetic),
    held constant across every test in this file and reused unchanged by
    Arm B's tests later (Day 4-6). Deliberate, documented selection: one
    event per available AAMI class in DS2 (N, S, V, F — Q has zero examples
    in this dataset after the standard 44-record paced-beat exclusion, see
    KB Section 3.2), first occurrence of each class by array index. This is
    the exact same selection logic already validated in
    smoke_test_real_baseline_arm.py — kept consistent so results from that
    manual run and this automated suite refer to the same events.

    Returns a list of raw 360-sample ECG windows (np.ndarray), indexable as
    small_real_event_sample[0], [1], etc. — NOT HealthEventJSON dicts; each
    element is fed directly to perception_agent.predict().
    """
    import numpy as np
    data = np.load("data/processed/ds2_test.npz")
    X, y = data["features"], data["labels"]

    picked = {}
    for i in range(len(y)):
        label = int(y[i])
        if label not in picked:
            picked[label] = i
        if len(picked) == 5:  # all 5 AAMI classes found (won't happen — Q is absent)
            break

    AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
    selected_indices = sorted(picked.values())
    print(f"\n[small_real_event_sample] Selected DS2 indices {selected_indices}, "
          f"classes: {[AAMI_CLASSES[y[i]] for i in selected_indices]}")
    return [X[i] for i in selected_indices]


@pytest.fixture(scope="module")
def perception_agent():
    return PerceptionAgent(checkpoint_path="perception/checkpoints/cnn_lstm.pt")


# ---------------------------------------------------------------------------
# Perception -> JSON (this half already has coverage from WP3's
# test_perception_agent.py — these tests only check the JSON is usable as
# an LLM prompt input, not re-test Perception's own correctness).
# ---------------------------------------------------------------------------

def test_perception_output_serializes_to_prompt_text(perception_agent, small_real_event_sample):
    """The HealthEventJSON object for a real event must serialize cleanly
    to the text block that gets inserted into the prompt template."""
    event = perception_agent.predict(small_real_event_sample[0])
    validated = HealthEventJSON(**event)
    text = validated.model_dump_json(indent=2)
    assert '"event_id"' in text
    assert '"classification"' in text


# ---------------------------------------------------------------------------
# Prompt template — frozen once these pass.
# ---------------------------------------------------------------------------

def test_prompt_template_produces_expected_structure():
    """The prompt scaffolding (system prompt + output-format instructions +
    injected fixed NICE-guideline context, per Section 4 of the design doc)
    must be a pure function of the event JSON and the AAMI class — no
    randomness, no hidden state — so it can be byte-identical every run."""
    from reasoning.prompt_template import build_baseline_prompt  # to be built

    dummy_json = _dummy_health_event_json()
    prompt_a = build_baseline_prompt(dummy_json)
    prompt_b = build_baseline_prompt(dummy_json)
    assert prompt_a == prompt_b, "Prompt template is non-deterministic — must be a pure function."
    assert "urgency" in prompt_a.lower(), "Prompt must instruct the model to output an urgency tier."


def test_fixed_context_matches_aami_class():
    """Per KB Section 18: fixed NICE-guideline excerpt block must be selected
    by AAMI class (V/S/F/Q vs N), not dynamically retrieved. Confirms the
    'thin Memory Agent' scope cut is actually implemented as thin."""
    from reasoning.prompt_template import get_fixed_context_for_class

    ctx_v = get_fixed_context_for_class("V")
    ctx_n = get_fixed_context_for_class("N")
    assert ctx_v != ctx_n, "Different AAMI classes must retrieve different fixed context blocks."
    assert isinstance(ctx_v, str) and len(ctx_v) > 0


# ---------------------------------------------------------------------------
# End-to-end baseline arm output.
# ---------------------------------------------------------------------------

def test_baseline_arm_produces_structured_output(perception_agent, small_real_event_sample):
    """Full Arm A path on one real event: Perception -> JSON -> prompt ->
    Gemma -> structured {urgency_tier, justification}."""
    from reasoning.baseline_arm import run_baseline_arm  # to be built

    result = run_baseline_arm(small_real_event_sample[0], perception_agent)
    assert result["urgency_tier"] in {"routine", "priority", "urgent"}, (
        f"Unexpected urgency_tier value: {result.get('urgency_tier')!r} — "
        "must match the fixed vocabulary defined in the design doc's downstream task spec."
    )
    assert isinstance(result["justification"], str) and len(result["justification"]) > 0
    assert "guideline" in result.get("justification", "").lower() or result.get(
        "referenced_guideline_fact"
    ), "Justification must reference at least one injected guideline fact (brief Section 5, downstream task spec)."


def test_baseline_arm_uses_greedy_decoding():
    """Section 3 control: decoding must be greedy for a clean deterministic
    comparison against Arm B. In transformers, do_sample=False is the actual
    greedy switch (temperature is ignored when do_sample=False, but kept in
    the config dict for audit clarity against the original Ollama version)."""
    from reasoning.baseline_arm import GENERATION_CONFIG

    assert GENERATION_CONFIG.get("do_sample") is False, (
        "Baseline arm must use do_sample=False (greedy) per the design doc's "
        "controlled-variable table — sampling would confound the comparison."
    )


def test_baseline_arm_is_deterministic(perception_agent, small_real_event_sample):
    """Same real event run twice through the frozen baseline arm must
    produce the identical urgency_tier (justification text may vary only
    if decoding truly isn't greedy yet — this test should force that gap shut)."""
    from reasoning.baseline_arm import run_baseline_arm

    result_a = run_baseline_arm(small_real_event_sample[0], perception_agent)
    result_b = run_baseline_arm(small_real_event_sample[0], perception_agent)
    assert result_a["urgency_tier"] == result_b["urgency_tier"]


def test_timer_boundaries_match_design_doc():
    """Section 3 control: clock starts when Perception's forward pass
    completes (32-dim vector available) and stops at Reasoning's first
    output token. Confirms the timing harness measures this exact segment,
    not e.g. total wall-clock including Perception's own inference time."""
    from reasoning.baseline_arm import run_baseline_arm_timed

    timing = run_baseline_arm_timed(_dummy_health_event_json(), perception_agent=None)
    assert "perception_complete_ts" in timing
    assert "first_reasoning_token_ts" in timing
    assert timing["first_reasoning_token_ts"] > timing["perception_complete_ts"]


# ---------------------------------------------------------------------------
# The freeze gate itself.
# ---------------------------------------------------------------------------

def test_baseline_is_frozen():
    """Once all the above pass, hash the prompt template file + generation
    config + model checkpoint reference and record it. Re-run this test
    before every later day's work (3 through 7) — if the hash ever changes
    without a deliberate, documented reason, the comparison is compromised."""
    frozen_manifest_path = "reasoning/BASELINE_FROZEN.json"

    def _current_hash():
        parts = []
        for path in ["reasoning/prompt_template.py", "reasoning/baseline_arm.py",
                      "reasoning/model_loader.py", "reasoning/fixed_context.py"]:
            with open(path, "rb") as f:
                parts.append(hashlib.sha256(f.read()).hexdigest())
        return hashlib.sha256("".join(parts).encode()).hexdigest()

    if not os.path.exists(frozen_manifest_path):
        # First run: write the freeze record. Commit this file to version control.
        from reasoning.model_loader import MODEL_ID
        with open(frozen_manifest_path, "w") as f:
            json.dump({"hash": _current_hash(), "model_id": MODEL_ID}, f, indent=2)
        pytest.skip("Baseline hash recorded for the first time — re-run to verify no drift.")
    else:
        with open(frozen_manifest_path) as f:
            recorded = json.load(f)
        assert _current_hash() == recorded["hash"], (
            "Baseline arm has changed since it was frozen! If this is deliberate "
            "(e.g. a genuine bug fix), delete BASELINE_FROZEN.json, re-run this test "
            "to re-freeze, and note the change explicitly in the design doc."
        )


def _dummy_health_event_json():
    """Minimal valid payload matching schema v1.0 (KB Section 3.1), for
    tests that only need structural validity, not a real classifier output."""
    return {
        "event_id": "evt_0000000000_0000",
        "timestamp": "2026-01-01T00:00:00.000Z",
        "classification": {
            "label": "V",
            "description": "Ventricular ectopic beat",
            "confidence": 0.9,
            "top_3": [
                {"label": "V", "confidence": 0.9},
                {"label": "N", "confidence": 0.07},
                {"label": "S", "confidence": 0.03},
            ],
        },
        "signal_features": {
            "rr_interval_ms": 400.0, "qrs_duration_ms": 140.0, "heart_rate_bpm": 150.0,
            "beat_morphology": "wide_complex",
        },
        "segment_metadata": {
            "window_samples": 360, "sample_rate_hz": 360, "lead": "MLII", "signal_quality_index": 0.9,
        },
        "clinical_flags": {
            "requires_urgent_review": True, "flag_reason": "test fixture", "consecutive_abnormal_beats": 1,
        },
    }
