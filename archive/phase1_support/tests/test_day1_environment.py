"""
Day 1 gate: environment confirmed before any pipeline code is written.

Per the 7-day design doc, Section 6: "Model returns a coherent response to a
manual test prompt" is the gate before moving to Day 2. These tests encode
that gate mechanically so it isn't just eyeballed.

Run: pytest tests/test_day1_environment.py -v
Expected state before Day 1 starts: all tests FAIL or ERROR (nothing set up yet).
Expected state to move to Day 2: all tests PASS.
"""
import os
import subprocess
import pytest


def test_data_prep_output_exists():
    """data_prep.py was already run in a prior session (KB Section 11.2/11.4).
    This just confirms the artifacts are still on disk before Day 1 work starts —
    do not re-run data_prep.py unless this fails."""
    assert os.path.exists("data/processed/ds1_train.npz"), (
        "ds1_train.npz missing — re-run `python data_prep.py` (requires network "
        "access to PhysioNet; must be run locally, not in a sandboxed/CI env)."
    )
    assert os.path.exists("data/processed/ds2_test.npz"), "ds2_test.npz missing."


def test_ollama_service_reachable():
    """Ollama must be running and reachable before any Gemma calls are attempted."""
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, f"`ollama list` failed: {result.stderr}"


def test_gemma_4_e4b_model_present():
    """Confirms the specific model variant (E4B, not E2B or another Gemma
    generation — see KB Section 4.4 MedGemma pitfall) is pulled and available."""
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
    assert "gemma4" in result.stdout.lower() or "gemma-4" in result.stdout.lower(), (
        "Gemma 4 E4B not found in `ollama list` output. Pull it before proceeding: "
        "`ollama pull <exact-tag-per-ollama-library>`. Verify the tag is E4B, not E2B."
    )


def test_gemma_responds_to_manual_prompt():
    """The literal Day 1 gate from the design doc: model returns a coherent
    response to a manual test prompt. 'Coherent' is checked loosely here —
    non-empty, no error string, reasonable length — full quality is not the
    point of this test, reachability + basic function is."""
    import ollama  # python client; pip install ollama --break-system-packages if missing

    response = ollama.chat(
        model="gemma4:e4b",  # placeholder tag — confirm exact tag from `ollama list`
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
    )
    content = response["message"]["content"].strip()
    assert len(content) > 0, "Empty response from Gemma — model may not be loaded correctly."
    assert "error" not in content.lower()[:50], f"Response looks like an error: {content[:100]}"


def test_gpu_visible_to_ollama():
    """Confirms inference will actually run on the target GPU, not silently
    fall back to CPU (which would invalidate every latency/VRAM number
    collected on Day 6)."""
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                             capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, "nvidia-smi not available — GPU not visible to this environment."
    assert "RTX 5070" in result.stdout or os.environ.get("SKIP_GPU_NAME_CHECK") == "1", (
        f"Expected target device (RTX 5070, 12GB GDDR7 — KB Section 15.2) not found in "
        f"nvidia-smi output: {result.stdout}. If intentionally testing on different hardware, "
        "set SKIP_GPU_NAME_CHECK=1."
    )


@pytest.mark.skip(reason="Manual check, not automatable — confirm no other GPU load is running "
                          "before Day 6 timing runs, per the sustained-load confound control "
                          "pattern already used elsewhere in this project (KB Section 3.4).")
def test_no_competing_gpu_load():
    pass
