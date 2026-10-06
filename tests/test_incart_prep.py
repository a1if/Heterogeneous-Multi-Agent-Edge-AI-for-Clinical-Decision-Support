"""INCART conversion (Deviation 21): resampling, annotation mapping, RR, inclusion rules (CPU, synthetic)."""
import numpy as np

import incart_prep as ip


def spikes(n_src, positions, height=5.0):
    sig = np.random.default_rng(0).normal(0, 0.05, n_src)
    for p in positions:
        sig[p] += height
    return sig


def test_index_mapping_and_window_centre():
    pos = [400, 700, 1000]
    w, labels, rr, skipped = ip.convert_record(spikes(1500, pos), pos, ["N", "V", "N"])
    assert w.shape == (3, 360) and labels == ["N", "V", "N"]
    for k in range(3):  # the resampled spike stays at the window centre (±1 sample)
        assert abs(int(np.argmax(w[k])) - 180) <= 1
    assert np.allclose(w.mean(1), 0, atol=1e-5) and np.allclose(w.std(1), 1, atol=1e-4)


def test_rr_from_original_rate_and_first_beat_none():
    pos = [400, 657, 914]  # 257 samples apart = exactly 1000 ms at 257 Hz
    _, _, rr, _ = ip.convert_record(spikes(1500, pos), pos, ["N", "N", "N"])
    assert rr[0] is None and np.allclose(rr[1:], [1000.0, 1000.0])


def test_inclusion_rules():
    pos = [10, 400, 600, 800, 1490]  # first and last windows fall outside the recording
    w, labels, rr, skipped = ip.convert_record(spikes(1500, pos), pos, ["N", "+", "A", "F", "N"])
    assert labels == ["S", "F"]  # '+' (rhythm change) is not a beat symbol; A -> S
    assert skipped == {"window_outside_recording": 2, "symbol_not_mapped": 1}
    assert rr[0] is None and np.isclose(rr[1], (800 - 600) / 257 * 1000)


def test_target_index():
    assert ip.to_target_index(257) == 360 and ip.to_target_index(0) == 0
