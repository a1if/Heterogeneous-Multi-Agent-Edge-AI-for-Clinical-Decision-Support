"""PerceptionAgent with the RR-branch encoder: inference must reproduce the
training-time feature path exactly (perception/rr_features.py batch function),
and the reference encoder must be unaffected."""
import os

import numpy as np
import pytest
import torch

from perception.model_rr import CNNLSTMRR
from perception.perception_agent import PerceptionAgent, replay_selected
from perception.rr_features import compute_rr_features, fit_standardizer, standardize

DS2 = "data/processed/ds2_test.npz"
REFERENCE = "perception/checkpoints/cnn_lstm.pt"
pytestmark = pytest.mark.skipif(not os.path.exists(DS2), reason="needs data_prep.py output")


@pytest.fixture(scope="module")
def ds2_slice():
    """First record's opening beats plus the start of the second, so the slice
    crosses a record boundary (history reset, post-RR stopping at the boundary)."""
    d = np.load(DS2)
    rec = d["record_ids"]
    b = int(np.flatnonzero(np.diff(rec) != 0)[0]) + 1
    idx = np.r_[b - 40:b, b:b + 40]  # last 40 of record 1, first 40 of record 2
    first = np.r_[0:b]  # the replay must start at record 1's first beat
    return d, idx, first


@pytest.fixture(scope="module")
def rr_checkpoint(tmp_path_factory, ds2_slice):
    d, _, _ = ds2_slice
    torch.manual_seed(0)
    model = CNNLSTMRR()
    stats = fit_standardizer(compute_rr_features(d["rr_interval_ms"], d["record_ids"]))
    path = tmp_path_factory.mktemp("ckpt") / "cnn_lstm_rr_test.pt"
    torch.save({"state_dict": model.state_dict(), "rr_standardizer": stats,
                "rr_feature_names": [], "seed": 0}, path)
    return str(path), model.eval(), stats


def test_rr_agent_matches_batch_training_path(ds2_slice, rr_checkpoint):
    d, idx, _ = ds2_slice
    path, model, stats = rr_checkpoint
    agent = PerceptionAgent(checkpoint_path=path, device="cpu")
    assert agent.uses_rr

    out = replay_selected(agent, d["features"], d["rr_interval_ms"], d["record_ids"], list(idx))

    rr = standardize(compute_rr_features(d["rr_interval_ms"], d["record_ids"])[idx], stats)
    x = torch.from_numpy(d["features"][idx]).unsqueeze(1)
    with torch.no_grad():
        _, batch_context = model(x, torch.from_numpy(rr))
    agent_context = np.stack([out[int(i)][1] for i in idx])
    np.testing.assert_allclose(agent_context, batch_context.numpy(), atol=1e-5)


def test_rr_agent_requires_real_rr(rr_checkpoint, ds2_slice):
    d, _, _ = ds2_slice
    agent = PerceptionAgent(checkpoint_path=rr_checkpoint[0], device="cpu")
    with pytest.raises(ValueError, match="rr_interval_ms"):
        agent.predict(d["features"][0])


def test_reset_state_clears_rr_history(rr_checkpoint, ds2_slice):
    d, _, _ = ds2_slice
    agent = PerceptionAgent(checkpoint_path=rr_checkpoint[0], device="cpu")
    agent.predict(d["features"][0], rr_interval_ms=800.0)
    agent.predict(d["features"][1], rr_interval_ms=400.0)
    agent.reset_state()
    agent.predict(d["features"][2], rr_interval_ms=600.0)
    first = agent.get_last_context_vector().copy()

    fresh = PerceptionAgent(checkpoint_path=rr_checkpoint[0], device="cpu")
    fresh.predict(d["features"][2], rr_interval_ms=600.0)
    np.testing.assert_array_equal(first, fresh.get_last_context_vector())


@pytest.mark.skipif(not os.path.exists(REFERENCE), reason="needs the reference checkpoint")
def test_reference_agent_ignores_next_rr(ds2_slice):
    d, _, _ = ds2_slice
    a = PerceptionAgent(checkpoint_path=REFERENCE, device="cpu")
    b = PerceptionAgent(checkpoint_path=REFERENCE, device="cpu")
    assert not a.uses_rr
    a.predict(d["features"][5], rr_interval_ms=812.0)
    b.predict(d["features"][5], rr_interval_ms=812.0, next_rr_interval_ms=400.0)
    np.testing.assert_array_equal(a.get_last_context_vector(), b.get_last_context_vector())
