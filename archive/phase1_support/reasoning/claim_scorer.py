"""Unsupported-claim scorer for generated justifications (Phase 1 step 9).

Extracts checkable factual claims from free text and compares each with the
ECG model's event (the ground truth for what the perception side produced; for
Arm B it is what the adapter's vector would have had to convey):

  class     a beat-class term (normal, supraventricular, ventricular, fusion,
            unclassifiable, or the "V-class" notation) must be the predicted
            label (strict, default). The lenient variant also accepts any top-3
            label, but "N" is in almost every top-3, so it hides a justification
            that calls a V beat normal (B-null on step 4: 0 vs 30 flagged events)
  rate      a number followed by bpm must be within RATE_TOL of heart_rate_bpm
  rr        a number followed by ms next to "RR" must be within RR_TOL of rr_interval_ms
  run       "N consecutive" / "N beats in a row" must equal consecutive_abnormal_beats
  confidence  a percentage or 0.xx near "confidence" must be within CONF_TOL

QRS duration is deliberately not checked: the estimator is miscalibrated
(see multiflag_target.py), so a "correct" claim against it would be meaningless.
A justification with no checkable claims scores n_claims = 0 (reported, not
counted as supported). Score the ``justification`` field only: the
``referenced_guideline_fact`` paraphrases background context, which legitimately
names other classes ("the same escalation as sustained ventricular ectopy").
"""
import re

RATE_TOL = 5.0      # bpm
RR_TOL = 30.0       # ms
CONF_TOL = 0.05

CLASS_TERMS = {
    "N": r"\bnormal\b|\bN-class\b",
    "S": r"\bsupraventricular\b|\bSVEB?\b|\batrial premature\b|\bS-class\b",
    "V": r"\bventricular\b(?! rate)|\bPVC\b|\bVEB?\b|\bV-class\b",
    "F": r"\bfusion\b|\bF-class\b",
    "Q": r"\bunclassifiable\b|\bunknown beat\b|\bQ-class\b",
}
_NUM = r"(\d+(?:\.\d+)?)"


# Justifications often restate the escalation rule itself ("a ventricular or fusion
# beat with confidence greater than 0.85"); those are statements about the rule,
# not claims about this beat, and are removed before extraction.
RULE_RESTATEMENTS = (
    r"\b(?:not\s+)?(?:a\s+)?ventricular\s+or\s+fusion(?:\s+beat)?",
    r"(?:greater|higher|more|less|lower)\s+than\s+(?:or\s+equal\s+to\s+)?0?\.85",
    r"(?:above|below|over|under|exceeds?|exceeding)\s+(?:the\s+)?(?:threshold\s+(?:of\s+)?)?0?\.85",
    r"[<>]=?\s*0?\.85",
    r"threshold\s+(?:of\s+)?0?\.85",
)


def extract_claims(text: str) -> list[dict]:
    t = text or ""
    for pattern in RULE_RESTATEMENTS:
        t = re.sub(pattern, " ", t, flags=re.I)
    claims = []
    for label, pattern in CLASS_TERMS.items():
        # "supraventricular" contains "ventricular"; strip S terms before looking for V.
        haystack = re.sub(CLASS_TERMS["S"], " ", t, flags=re.I) if label == "V" else t
        if re.search(pattern, haystack, flags=re.I):
            claims.append({"kind": "class", "value": label})
    for m in re.finditer(_NUM + r"\s*(?:bpm|beats per minute)", t, flags=re.I):
        claims.append({"kind": "rate", "value": float(m.group(1))})
    for m in re.finditer(r"\bRR[^.\d]{0,30}?" + _NUM + r"\s*ms", t, flags=re.I):
        claims.append({"kind": "rr", "value": float(m.group(1))})
    for m in re.finditer(_NUM + r"\s+(?:consecutive|beats in a row)", t, flags=re.I):
        claims.append({"kind": "run", "value": int(float(m.group(1)))})
    for m in re.finditer(r"confidence[^.\d]{0,20}?" + _NUM + r"\s*(%)?", t, flags=re.I):
        v = float(m.group(1))
        claims.append({"kind": "confidence", "value": v / 100 if m.group(2) or v > 1 else v})
    return claims


def check_claim(claim: dict, event: dict, strict_class: bool = True) -> bool:
    c, sf = event["classification"], event["signal_features"]
    kind, v = claim["kind"], claim["value"]
    if kind == "class":
        return v == c["label"] or (not strict_class and v in {t["label"] for t in c["top_3"]})
    if kind == "rate":
        return abs(v - sf["heart_rate_bpm"]) <= RATE_TOL
    if kind == "rr":
        return abs(v - sf["rr_interval_ms"]) <= RR_TOL
    if kind == "run":
        return v == event["clinical_flags"]["consecutive_abnormal_beats"]
    if kind == "confidence":
        return abs(v - c["confidence"]) <= CONF_TOL
    raise ValueError(kind)


def score_text(text: str, event: dict, strict_class: bool = True) -> dict:
    claims = extract_claims(text)
    unsupported = [cl for cl in claims if not check_claim(cl, event, strict_class)]
    return {"n_claims": len(claims), "n_unsupported": len(unsupported), "unsupported": unsupported}
