"""Atomic saves retry a transient Windows lock (PermissionError) instead of failing (CPU)."""
import os

import p1_io


def test_replace_with_retry_survives_transient_lock(tmp_path, monkeypatch):
    src, dst = tmp_path / "a.tmp", tmp_path / "a.json"
    src.write_text("x")
    real, calls = os.replace, {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(5, "Access is denied")
        real(a, b)

    monkeypatch.setattr(p1_io.os, "replace", flaky)
    p1_io.replace_with_retry(src, dst, wait_s=0)
    assert dst.read_text() == "x" and calls["n"] == 3
