"""
PerceptionAgent — wraps the CNN-LSTM model, turns raw ECG windows into
HealthEventJSON-compliant dicts, and exposes the raw 32-dim context vector
for Arm B's adapter (Day 3+).

Design decisions made while building this (flagged, not silent):
- Signal Quality Index (SQI): simple flatline/clipping heuristic. Not
  specified anywhere in the KB; chosen because MIT-BIH is curated clinical
  data where SQI mainly needs to catch rare corrupted edges, not calibrate
  general quality (classifier accuracy comes from training, not SQI).
- QRS duration: estimated from within-window slope thresholding around the
  R-peak (standard technique, needs no external context).
- RR interval: cannot be derived from a single window alone (needs the
  previous beat's position). Real values come from data_prep.py's
  annotation-timing computation; predict() accepts rr_interval_ms as an
  optional override for real evaluation, defaulting to 800ms for
  synthetic/unit-test use where no real timing exists.
"""
import time

import numpy as np
import torch

from perception.model import CNNLSTM, AAMI_CLASSES
from perception.health_event_schema import HealthEventJSON

WINDOW_LEN = 360
SAMPLE_RATE_HZ = 360
QRS_WIDE_THRESHOLD_MS = 120.0
SQI_LOW_THRESHOLD = 0.5
DEFAULT_RR_INTERVAL_MS = 800.0  # ~75bpm, synthetic/unit-test fallback only


def compute_sqi(window: np.ndarray) -> float:
    """Simple, documented heuristic: penalizes flatline (near-zero local
    variance) and clipping (values pinned at the window's own min/max for
    an extended run). Returns a score in [0, 1]; 1.0 = clean signal.
    NOTE: operates on the raw (pre-normalization) window."""
    if window.std() < 1e-6:
        return 0.0  # fully flat — degenerate

    # Flatline check: fraction of samples with near-zero local derivative
    diffs = np.abs(np.diff(window))
    flat_fraction = float((diffs < 1e-4 * (window.max() - window.min() + 1e-8)).mean())

    # Clipping check: fraction of samples pinned at min or max
    lo, hi = window.min(), window.max()
    clip_fraction = float(((window <= lo + 1e-8) | (window >= hi - 1e-8)).mean())

    penalty = min(1.0, flat_fraction + clip_fraction)
    return max(0.0, 1.0 - penalty)


def estimate_qrs_duration_ms(window: np.ndarray, sample_rate_hz: int = SAMPLE_RATE_HZ) -> float:
    """Slope-threshold QRS width estimate around the window center
    (windows are R-peak-centered by construction, per data_prep.py)."""
    center = len(window) // 2
    abs_signal = np.abs(window - np.median(window))
    threshold = 0.25 * abs_signal.max() if abs_signal.max() > 0 else 0.0

    # Walk outward from center until signal drops below threshold on each side
    left = center
    while left > 0 and abs_signal[left] > threshold:
        left -= 1
    right = center
    while right < len(window) - 1 and abs_signal[right] > threshold:
        right += 1

    width_samples = max(1, right - left)
    return width_samples / sample_rate_hz * 1000.0


class PerceptionAgent:
    def __init__(self, checkpoint_path: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNLSTM().to(self.device)
        if checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()  # dropout disabled at inference, per test_deterministic_output_at_inference

        self._consecutive_abnormal_beats = 0
        self._last_context_vector = None  # (32,) numpy array, for Arm B's adapter

    def get_state(self) -> dict:
        return {"consecutive_abnormal_beats": self._consecutive_abnormal_beats}

    def reset_state(self) -> None:
        """Reset sequence-only clinical state at a recording boundary.

        MIT-BIH records are independent recordings.  Carrying an abnormal-beat
        run from one record into the next would create a synthetic escalation
        label, so corpus-building/evaluation code must reset at that boundary.
        """
        self._consecutive_abnormal_beats = 0

    def get_last_context_vector(self) -> np.ndarray:
        """Returns the 32-dim context vector from the most recent predict()
        call — the exact adapter extraction point (brief Section 4)."""
        if self._last_context_vector is None:
            raise RuntimeError("No prediction has been made yet — call predict() first.")
        return self._last_context_vector

    def predict(self, raw_segment: np.ndarray, rr_interval_ms: float | None = None,
                event_seq: int = 0) -> dict:
        if raw_segment.shape[-1] != WINDOW_LEN:
            raise ValueError(
                f"Expected a {WINDOW_LEN}-sample window, got shape {raw_segment.shape}."
            )
        raw_segment = np.asarray(raw_segment, dtype=np.float32).reshape(-1)

        sqi = compute_sqi(raw_segment)

        # z-score normalize (matches data_prep.py's own normalization —
        # inference-time input must match training-time preprocessing)
        mean, std = raw_segment.mean(), raw_segment.std()
        normalized = (raw_segment - mean) / std if std > 1e-8 else raw_segment.copy()

        x = torch.from_numpy(normalized).float().reshape(1, 1, WINDOW_LEN).to(self.device)
        with torch.no_grad():
            logits, context_vector = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        self._last_context_vector = context_vector.squeeze(0).cpu().numpy()

        if sqi < SQI_LOW_THRESHOLD:
            label = "Q"
            description = "Unclassifiable — low signal quality"
            top_3_labels = ["Q", "N", "S"]  # placeholder ranking, Q forced regardless of model output
            confidence = 1.0 - sqi  # low confidence in the (forced) Q label
            top_3 = [
                {"label": top_3_labels[0], "confidence": round(confidence, 6)},
                {"label": top_3_labels[1], "confidence": round((1 - confidence) * 0.6, 6)},
                {"label": top_3_labels[2], "confidence": round((1 - confidence) * 0.4, 6)},
            ]
            flag_reason = "low_signal_quality"
            requires_urgent = True  # cannot classify automatically -> needs human review
            self._consecutive_abnormal_beats = 0  # Q doesn't count as a tracked abnormal run
        else:
            top_idx = np.argsort(probs)[::-1][:3]
            label = AAMI_CLASSES[top_idx[0]]
            description = _label_description(label)

            # Renormalize the top-3 subset to its own distribution (sums to 1.0) —
            # the raw 5-way softmax slice does NOT sum to 1.0 on its own, since the
            # other 2 classes hold nonzero probability mass too.
            raw_top3 = probs[top_idx].astype(np.float64)  # float64 for rounding safety
            renorm_top3 = raw_top3 / raw_top3.sum()
            confidence = float(renorm_top3[0])
            top_3 = [
                {"label": AAMI_CLASSES[top_idx[i]], "confidence": round(float(renorm_top3[i]), 6)}
                for i in range(3)
            ]
            # 6-decimal rounding keeps drift well under the schema's 1e-5 tolerance
            # without needing a fragile last-entry adjustment that could break ordering.

            if label == "N":
                self._consecutive_abnormal_beats = 0
                requires_urgent = False
                flag_reason = None
            else:
                self._consecutive_abnormal_beats += 1
                requires_urgent = (
                    (label in ("V", "F") and confidence > 0.85)
                    or self._consecutive_abnormal_beats >= 3
                )
                if requires_urgent:
                    if self._consecutive_abnormal_beats >= 3:
                        flag_reason = (
                            f"{self._consecutive_abnormal_beats} consecutive abnormal "
                            f"beats ({label}) detected"
                        )
                    else:
                        flag_reason = f"{label}-class beat detected with confidence {confidence:.2f}"
                else:
                    flag_reason = None

        rr = rr_interval_ms if rr_interval_ms is not None else DEFAULT_RR_INTERVAL_MS
        qrs_ms = estimate_qrs_duration_ms(raw_segment)
        heart_rate_bpm = 60000.0 / rr if rr > 0 else 0.0
        beat_morphology = "wide_complex" if qrs_ms >= QRS_WIDE_THRESHOLD_MS else "narrow_complex"
        if label == "N" and beat_morphology == "narrow_complex":
            beat_morphology = "normal"

        event = {
            "event_id": f"evt_{int(time.time())}_{event_seq:04d}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "classification": {
                "label": label,
                "description": description,
                "confidence": round(confidence, 6),
                "top_3": top_3,
            },
            "signal_features": {
                "rr_interval_ms": round(rr, 2),
                "qrs_duration_ms": round(qrs_ms, 2),
                "heart_rate_bpm": round(heart_rate_bpm, 2),
                "beat_morphology": beat_morphology,
            },
            "segment_metadata": {
                "window_samples": WINDOW_LEN,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "lead": "MLII",
                "signal_quality_index": round(sqi, 4),
            },
            "clinical_flags": {
                "requires_urgent_review": bool(requires_urgent),
                "flag_reason": flag_reason,
                "consecutive_abnormal_beats": self._consecutive_abnormal_beats,
            },
        }

        # Validate before returning — fail loudly here, not downstream in the Reasoning Agent
        HealthEventJSON(**event)
        return event


def _label_description(label: str) -> str:
    return {
        "N": "Normal sinus beat",
        "S": "Supraventricular ectopic beat",
        "V": "Ventricular ectopic beat",
        "F": "Fusion beat",
        "Q": "Unclassifiable beat",
    }[label]
