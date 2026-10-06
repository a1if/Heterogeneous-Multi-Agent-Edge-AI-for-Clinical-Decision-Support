"""Multi-flag decision target (Phase 1 step 9, candidate v1; not yet pre-specified).

The current task target is a single urgency tier that the prompt states as an
exact rule, and Arm A reproduces it perfectly (no headroom). This target keeps
the tier and adds flags whose inputs are computed OUTSIDE the CNN (heart rate
from RR) or need the full classifier output (top-2 margin), so a latent interface
has to carry that information rather than only the class.

Flags, chosen after measuring prevalence on real events (E80 / DS2 sample of 2,000):
  confirm_classification  confidence < 0.60 or top-2 margin < 0.20  (~11% / ~12%)
  rate_out_of_range       instantaneous heart rate < 50 or > 100 bpm (51% / 25%)

Excluded, with reasons (see TASKS.md):
  wide-QRS discordance    estimate_qrs_duration_ms is miscalibrated (median 36-44 ms,
                          physiological ~80-120 ms): the flag never fires on N/S labels
  low signal quality      compute_sqi barely varies on MIT-BIH (median 0.87, 99% < 0.95),
                          so any threshold would be arbitrary
"""
from reasoning.training_targets import urgency_tier_from_event

CONFIDENCE_MIN = 0.60
MARGIN_MIN = 0.20
HR_LOW, HR_HIGH = 50.0, 100.0
FLAGS = ("confirm_classification", "rate_out_of_range")


def multiflag_target(health_event: dict) -> dict:
    c = health_event["classification"]
    top3 = c["top_3"]
    margin = top3[0]["confidence"] - top3[1]["confidence"]
    hr = health_event["signal_features"]["heart_rate_bpm"]
    flags = []
    if c["confidence"] < CONFIDENCE_MIN or margin < MARGIN_MIN:
        flags.append("confirm_classification")
    if hr < HR_LOW or hr > HR_HIGH:
        flags.append("rate_out_of_range")
    return {"urgency_tier": urgency_tier_from_event(health_event), "flags": flags}


def score(predicted: dict, target: dict) -> dict:
    """Exact match on (tier, flag set), plus per-flag correctness for per-flag F1."""
    pred_flags, true_flags = set(predicted.get("flags", [])), set(target["flags"])
    return {
        "tier_correct": predicted.get("urgency_tier") == target["urgency_tier"],
        "exact_match": predicted.get("urgency_tier") == target["urgency_tier"] and pred_flags == true_flags,
        "per_flag": {f: {"predicted": f in pred_flags, "true": f in true_flags} for f in FLAGS},
    }
