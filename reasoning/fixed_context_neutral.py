"""
Class-NEUTRAL fixed context, used only by Arm B (the adapter arm).

Why this exists, separately from reasoning/fixed_context.py:

Arm A's fixed_context.py is deliberately class-specific — the background
context heading names the detected AAMI class, and that's fine for Arm A,
because the JSON payload legitimately carries the class as its content; the
heading isn't leaking anything the JSON doesn't already say.

For Arm B, the same class-specific heading is a genuine confound, not a
harmless mirror: the heading text is real-token-embedded (never replaced by
virtual tokens), so it sits in the prompt regardless of what the adapter
outputs. Since the heading's class comes from the identical predicted-label
field that the training target (urgency_tier) is derived from, the heading
alone is sufficient to solve the task — the model does not need to decode
anything from the virtual tokens at all. This was caught via direct
inspection of the real rendered Arm B prompt, not by assumption.

Fix: a single, class-neutral context block, covering the same escalation
logic as all five per-class blocks combined, without naming which one
applies to the current event. The model must infer which situation is
relevant from the event representation itself (JSON in Arm A, virtual
tokens in Arm B) — restoring the actual point of the comparison.

This is a deliberate, disclosed asymmetry: Arm A and Arm B's background
context differ in specificity, by design, not by oversight. Arm A's system
prompt and output instructions remain byte-identical (imported from
prompt_template.py, not duplicated) — only the background-context content
differs, and only because Arm A's "class in the JSON" and Arm B's "class in
a side-channel heading" are not equivalent information flows.
"""

NEUTRAL_CONTEXT = (
    "This event was classified using AAMI beat-type criteria (normal, "
    "supraventricular ectopic, ventricular ectopic, fusion, or unclassifiable). "
    "General escalation criteria, across all classes: a normal sinus beat needs "
    "only routine ongoing monitoring. An isolated supraventricular, ventricular, "
    "or fusion beat is usually benign on its own, but frequent occurrences, "
    "three or more consecutive abnormal beats, or a high-confidence ventricular "
    "or fusion classification each independently warrant prompt clinical review. "
    "A beat that cannot be classified due to low signal quality cannot be "
    "interpreted automatically and requires manual review rather than an "
    "automated judgement. Use the event representation below to determine "
    "which of these situations applies to this specific beat."
)
