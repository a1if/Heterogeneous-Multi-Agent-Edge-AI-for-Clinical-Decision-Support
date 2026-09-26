"""p1_item7_train.train_loop: resuming after an interruption must reproduce an
uninterrupted run exactly; validation-driven best-checkpoint and early stopping.
CPU only, with a tiny stand-in adapter and loss."""
import pytest
import torch
import torch.nn as nn

import p1_item7_train as t7


def setup(tmp_path, tag):
    torch.manual_seed(7)
    adapter = nn.Linear(4, 1)
    opt = torch.optim.AdamW(adapter.parameters(), lr=0.05)
    g = torch.Generator().manual_seed(0)
    train = [{"x": torch.randn(4, generator=g), "y": float(i % 3)} for i in range(24)]
    paths = {"resume": tmp_path / f"{tag}.resume.pt", "best": tmp_path / f"{tag}.pt",
             "state": tmp_path / f"{tag}.json"}
    return adapter, opt, train, paths


def run(tmp_path, tag, fail_at=None, calls=None):
    adapter, opt, train, paths = setup(tmp_path, tag)
    calls = calls if calls is not None else {"n": 0}

    def loss_fn(w):
        calls["n"] += 1
        if fail_at is not None and calls["n"] == fail_at:
            raise KeyboardInterrupt("simulated stop")
        return (adapter(w["x"]) - w["y"]).pow(2).sum()

    def validate_fn(val):
        # deterministic, improves then plateaus -> exercises best-checkpoint + patience
        m = -float(sum(p.detach().abs().sum() for p in adapter.parameters()))
        return {"balanced_accuracy": m, "parse_rate": 1.0, "by_n": {}}

    state = t7.train_loop(adapter=adapter, optimizer=opt, train=train, val=[], loss_fn=loss_fn,
                          validate_fn=validate_fn, seed=3, paths=paths, accum=4, eval_every=2,
                          patience=2, max_epochs=3, resume_every=1, log=lambda *_: None)
    return adapter, state, paths


def test_resume_is_bit_identical(tmp_path):
    ref_adapter, ref_state, _ = run(tmp_path, "ref")
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, "cut", fail_at=23)          # mid-epoch 1 (after several saved updates)
    resumed, state, paths = run(tmp_path, "cut")  # rerun resumes from the checkpoint
    for a, b in zip(ref_adapter.state_dict().values(), resumed.state_dict().values()):
        assert torch.equal(a, b)
    assert [h["loss"] for h in ref_state["history"]] == [h["loss"] for h in state["history"]]
    assert ref_state["best_update"] == state["best_update"]
    assert paths["best"].exists()


def run_sched(tmp_path, tag, fail_at=None):
    adapter, opt, train, paths = setup(tmp_path, tag)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, t7.warmup_cosine(3 * len(train), 0.1, 0.1))
    calls = {"n": 0}

    def loss_fn(w):
        calls["n"] += 1
        if fail_at is not None and calls["n"] == fail_at:
            raise KeyboardInterrupt("simulated stop")
        return (adapter(w["x"]) - w["y"]).pow(2).sum()

    state = t7.train_loop(adapter=adapter, optimizer=opt, train=train, val=[], loss_fn=loss_fn,
                          validate_fn=lambda v: {"balanced_accuracy": 0.0, "parse_rate": 1.0, "by_n": {}},
                          seed=3, paths=paths, accum=1, eval_every=5, patience=99, max_epochs=3,
                          resume_every=4, log=lambda *_: None, scheduler=sched)
    return adapter, state


def test_resume_with_scheduler_is_bit_identical(tmp_path):
    ref, ref_state = run_sched(tmp_path, "ref")
    with pytest.raises(KeyboardInterrupt):
        run_sched(tmp_path, "cut", fail_at=30)   # mid-epoch 2, warm-up finished
    res, state = run_sched(tmp_path, "cut")
    for a, b in zip(ref.state_dict().values(), res.state_dict().values()):
        assert torch.equal(a, b)
    assert [h["lr"] for h in ref_state["history"]] == [h["lr"] for h in state["history"]]
    assert ref_state["history"][0]["lr"] < ref_state["history"][10]["lr"]  # warm-up ran
