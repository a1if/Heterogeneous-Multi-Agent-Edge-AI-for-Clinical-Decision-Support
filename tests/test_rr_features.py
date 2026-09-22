import numpy as np
import pytest

from perception.rr_features import LOCAL_WINDOW, compute_rr_features


def test_post_rr_is_next_beat_and_stops_at_record_boundary():
    rr = np.array([800, 600, 1000, 900, 700], dtype=np.float32)
    rec = np.array([1, 1, 1, 2, 2])
    f = compute_rr_features(rr, rec)
    np.testing.assert_allclose(f[:, 0], [0.8, 0.6, 1.0, 0.9, 0.7], rtol=1e-6)
    # post-RR: next beat within the record; last beat of each record falls back to its pre-RR
    np.testing.assert_allclose(f[:, 1], [0.6, 1.0, 1.0, 0.7, 0.7], rtol=1e-6)


def test_local_average_is_causal_and_resets_per_record():
    rr = np.array([800, 400, 1200, 1000, 500], dtype=np.float32)
    rec = np.array([1, 1, 1, 2, 2])
    f = compute_rr_features(rr, rec)
    # first beat of a record has no history -> its own pre-RR
    assert f[0, 2] == pytest.approx(0.8)
    assert f[3, 2] == pytest.approx(1.0)
    # later beats: mean of previous beats in the same record only, excluding the current one
    assert f[1, 2] == pytest.approx(0.8)
    assert f[2, 2] == pytest.approx(0.6)
    assert f[4, 2] == pytest.approx(1.0)
    assert f[2, 3] == pytest.approx(1.2 / 0.6)


def test_local_window_is_bounded():
    rr = np.full(LOCAL_WINDOW + 5, 1000.0, dtype=np.float32)
    rr[0] = 3000.0  # outside the window by the last beat
    f = compute_rr_features(rr, np.zeros(len(rr), dtype=int))
    assert f[-1, 2] == pytest.approx(1.0)


def test_outliers_are_clipped():
    f = compute_rr_features(np.array([100022.0, 50.0]), np.array([1, 1]))
    assert f[0, 0] == pytest.approx(3.0)
    assert f[1, 0] == pytest.approx(0.2)


def test_non_contiguous_records_rejected():
    with pytest.raises(ValueError):
        compute_rr_features(np.ones(3) * 800, np.array([1, 2, 1]))
