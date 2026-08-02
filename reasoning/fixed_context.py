"""
Fixed context blocks, one per AAMI EC57 class (N/S/V/F/Q).

Design decision (see project design doc discussion): content is generic,
paraphrased clinical reasoning — NOT sourced from or attributed to any
specific named guideline body (NICE, ESC, AHA, etc.). This sidesteps two
real problems found during sourcing: (1) NICE has no dedicated guideline
for isolated ventricular/fusion ectopy at the beat level — only atrial
fibrillation (NG196) is well-covered and maps to the S-class; (2) ESC's
2022 ventricular arrhythmia guidelines are explicitly copyright-protected
with a stated opt-out against use in training generative AI models.

This is infrastructure supporting the experiment (KB Section 18), not a
measured variable and not a claim requiring formal citation. It must be
described in the design doc / dissertation as "fixed synthetic clinical
context, not sourced from a specific named guideline" — do not relabel
this as "NICE guideline text" anywhere in formal writing.
"""

FIXED_CONTEXT = {
    "N": (
        "This beat shows a normal sinus pattern with no arrhythmic features. "
        "No clinical action is indicated beyond routine, ongoing monitoring."
    ),
    "S": (
        "This beat originates above the ventricles, outside the normal sinus "
        "pathway. Isolated supraventricular ectopy is usually benign, but frequent "
        "or sustained occurrences raise the possibility of an underlying atrial "
        "rhythm disturbance and warrant confirmation with a full ECG trace and "
        "clinical correlation with symptoms such as palpitations, breathlessness, "
        "or dizziness."
    ),
    "V": (
        "This beat originates within the ventricle itself, outside the normal "
        "conduction pathway. An isolated occurrence in a patient without known "
        "structural heart disease is usually benign. Frequent occurrences, or "
        "three or more in a row, raise concern for a more serious ventricular "
        "arrhythmia and warrant prompt clinical review, particularly when "
        "accompanied by a wide QRS duration."
    ),
    "F": (
        "This beat shows features of both a normal and a ventricular-origin "
        "activation occurring together. It is clinically interpreted similarly "
        "to an isolated ventricular ectopic beat — usually benign on its own, "
        "but repeated occurrence alongside other abnormal beats warrants the "
        "same escalation as sustained ventricular ectopy."
    ),
    "Q": (
        "Signal quality was insufficient to confidently classify this beat. "
        "The classification label should not be treated as clinically meaningful "
        "on its own; a repeat recording or manual review is recommended before "
        "any clinical interpretation is made."
    ),
}


def get_fixed_context_for_class(aami_label: str) -> str:
    if aami_label not in FIXED_CONTEXT:
        raise ValueError(
            f"Unknown AAMI class {aami_label!r} — expected one of {list(FIXED_CONTEXT)}."
        )
    return FIXED_CONTEXT[aami_label]
