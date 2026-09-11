"""Serialized communication bytes, adapter-only compute time, and GPU energy,
kept deliberately separate from prompt_tokens (input sequence length).

WHY THESE ARE DIFFERENT METRICS
--------------------------------
prompt_tokens (already measured in day6_run_comparison.py etc.) counts LLM
input POSITIONS -- a compute-cost proxy (how much work Gemma's forward pass
does). It is not a byte count: text tokenization and raw float serialization
compress/expand by unrelated amounts, so a smaller sequence length does not
imply a smaller wire payload, and vice versa. This script measures the actual
serialized payload that would cross a real Perception-to-Reasoning network or
IPC boundary, which prompt_tokens has never stood in for.

ARCHITECTURAL ASSUMPTION -- stated explicitly, not silently picked:
Arm B's communication boundary is placed at the 32-dim context vector, BEFORE
the adapter, not after it (i.e. NOT the 4x2560 virtual-token tensor). Reasoning:
the adapter is Gemma-specific glue (load_trained_adapter sizes itself from the
loaded Gemma model; VirtualTokenAdapter.for_model reads Gemma's own embedding
width) -- it has no reason to run on a resource-constrained edge Perception
device, and every reason to run wherever the Reasoning Agent (and its matched
adapter checkpoint) already lives. Under this framing, only the 32 floats cross
the boundary; the adapter's projection to 10,240-dim virtual tokens happens
receiver-side, after the "communication" is already complete. This is a
modelling choice, not an empirical finding -- if the adapter instead ran at
the edge, Arm B's communication bytes would be the much larger 4x2560 virtual
token tensor instead, reversing the comparison. Both are reported so neither
is hidden.

Run (from repo root, GPU required):
    python measure_comm_cost.py
"""
import json
import threading
import time

import numpy as np
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

from perception.perception_agent import PerceptionAgent
from reasoning.adapter_arm import load_trained_adapter, run_adapter_arm_timed
from reasoning.baseline_arm import run_baseline_arm_timed
from reasoning.model_loader import load_model
from reasoning.prompt_template import _extract_prompt_fields
from project_config import (
    ADAPTER_CHECKPOINT, DS2_PATH, PERCEPTION_CHECKPOINT, select_events,
)
from perception.perception_agent import replay_selected

N_EVENTS = 80                  # the full headline set (was 10 for the initial dry run)
ADAPTER_TIMING_REPS = 2000     # repeated calls for the pure-timing microbenchmark (CUDA events, not power)
ADAPTER_ENERGY_WINDOW_S = 12.0 # target wall-clock duration for the energy pass -- see note below
POWER_SAMPLE_HZ = 100
RESULTS_PATH = "results/comm_cost_results.json"


# --- Power sampling ----------------------------------------------------------
def sample_power_during(fn, *args, hz=POWER_SAMPLE_HZ, **kwargs):
    """Runs fn under a background NVML power-polling thread.

    Returns (result, energy_joules, mean_power_w, elapsed_s, n_samples).
    Energy is trapezoidal integration of power(t) over the call's wall time --
    an approximation bounded by NVML's own internal sampling window (driver-
    dependent, typically ~10-50ms), not a calorimeter-grade measurement. This
    is stated once here rather than repeated at every call site.
    """
    if not _HAS_NVML:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        return result, None, None, time.perf_counter() - t0, 0

    samples = []   # (timestamp, watts)
    stop = threading.Event()

    def poll():
        interval = 1.0 / hz
        while not stop.is_set():
            samples.append((time.perf_counter(), pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE) / 1000.0))
            time.sleep(interval)

    thread = threading.Thread(target=poll, daemon=True)
    t_start = time.perf_counter()
    thread.start()
    result = fn(*args, **kwargs)
    t_end = time.perf_counter()
    stop.set()
    thread.join(timeout=1.0)

    samples = [(t, w) for t, w in samples if t_start <= t <= t_end]
    if len(samples) < 2:
        return result, None, None, t_end - t_start, len(samples)

    energy = 0.0
    for (t0, w0), (t1, w1) in zip(samples, samples[1:]):
        energy += (w0 + w1) / 2.0 * (t1 - t0)
    mean_power = sum(w for _, w in samples) / len(samples)
    return result, energy, mean_power, t_end - t_start, len(samples)


def idle_baseline(seconds=3.0):
    if not _HAS_NVML:
        return None
    readings = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        readings.append(pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE) / 1000.0)
        time.sleep(0.02)
    return float(np.mean(readings))


# --- Serialized bytes ---------------------------------------------------------
def arm_a_bytes(health_event: dict) -> dict:
    """The actual JSON object embedded in Arm A's prompt (prompt_template.py's
    own field subset) -- the real transmitted message, not the full
    HealthEventJSON (which includes fields deliberately excluded from the
    prompt, e.g. event_id/timestamp)."""
    fields = _extract_prompt_fields(health_event)
    pretty = json.dumps(fields, indent=2)          # as actually embedded in the prompt
    compact = json.dumps(fields, separators=(",", ":"))  # minimum realistic wire format
    return {
        "as_embedded_in_prompt_bytes": len(pretty.encode("utf-8")),
        "minified_json_bytes": len(compact.encode("utf-8")),
    }


def arm_b_bytes(context_vector: np.ndarray, virtual_tokens_shape) -> dict:
    """Two numbers under two different architectural placements of the
    boundary -- see module docstring. float32 is the numpy/PyTorch default;
    float16 is reported because it is the realistic bandwidth-saving choice
    an actual edge deployment would make for a fixed-length numeric vector."""
    n_context = int(context_vector.size)
    n_virtual = int(np.prod(virtual_tokens_shape))
    return {
        "context_vector_float32_bytes": n_context * 4,
        "context_vector_float16_bytes": n_context * 2,
        "virtual_tokens_float32_bytes": n_virtual * 4,
        "virtual_tokens_float16_bytes": n_virtual * 2,
        "context_vector_dims": n_context,
        "virtual_tokens_dims": n_virtual,
    }


def main():
    print("Loading Perception Agent + DS2...")
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)
    data = np.load(DS2_PATH)
    X, y, rr, record_ids = data["features"], data["labels"], data["rr_interval_ms"], data["record_ids"]
    selected = select_events(y, record_ids)[:N_EVENTS]

    replayed = replay_selected(agent, X, rr, record_ids, selected)
    prepared = [{"idx": i, "health_event": replayed[i][0], "context_vector": replayed[i][1]}
                for i in selected]
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Prepared {len(prepared)} real events.")
    print("Loading Gemma 4 E4B + trained adapter...")
    model, processor = load_model()
    adapter = load_trained_adapter(ADAPTER_CHECKPOINT, model)
    device = next(adapter.parameters()).device

    if _HAS_NVML:
        print("Measuring idle GPU power baseline (3s)...")
        idle_w = idle_baseline()
        print(f"  idle baseline: {idle_w:.2f} W")
    else:
        idle_w = None
        print("pynvml unavailable -- energy numbers will be skipped.")

    # --- 1. Serialized bytes, per event -------------------------------------
    print("\n" + "=" * 70)
    print("SERIALIZED COMMUNICATION BYTES (n=%d real events)" % len(prepared))
    print("=" * 70)
    bytes_rows = []
    for item in prepared:
        he = item["health_event"]
        with torch.no_grad():
            vt = adapter(torch.from_numpy(item["context_vector"]).unsqueeze(0).to(device).float())
        a_b = arm_a_bytes(he)
        b_b = arm_b_bytes(item["context_vector"], tuple(vt.shape))
        bytes_rows.append({"idx": item["idx"], "arm_a": a_b, "arm_b": b_b})
        print(f"idx={item['idx']:>6}  ArmA(prompt-embedded)={a_b['as_embedded_in_prompt_bytes']:>4}B "
              f"ArmA(minified)={a_b['minified_json_bytes']:>4}B  |  "
              f"ArmB(context,f32)={b_b['context_vector_float32_bytes']:>4}B "
              f"ArmB(virtual_tokens,f32)={b_b['virtual_tokens_float32_bytes']:>6}B")

    # --- 2. Adapter-only compute time (excludes Gemma entirely) -------------
    print("\n" + "=" * 70)
    print(f"ADAPTER-ONLY COMPUTE TIME ({ADAPTER_TIMING_REPS} reps x {len(prepared)} real vectors)")
    print("=" * 70)
    ctx_batch = torch.from_numpy(np.stack([p["context_vector"] for p in prepared])).to(device).float()

    # warm-up (excluded from timing: first calls pay one-time CUDA kernel/cache costs)
    for _ in range(20):
        with torch.no_grad():
            adapter(ctx_batch)
    torch.cuda.synchronize()

    adapter_times_ms = []
    for _ in range(ADAPTER_TIMING_REPS):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        with torch.no_grad():
            adapter(ctx_batch)
        end_evt.record()
        torch.cuda.synchronize()
        adapter_times_ms.append(start_evt.elapsed_time(end_evt))
    adapter_times_ms = np.array(adapter_times_ms)
    print(f"per-call (batch={len(prepared)}): mean={adapter_times_ms.mean():.4f}ms "
          f"median={np.median(adapter_times_ms):.4f}ms min={adapter_times_ms.min():.4f}ms "
          f"max={adapter_times_ms.max():.4f}ms")

    adapter_energy = None
    if _HAS_NVML:
        # CRITICAL: the GPU's own power sensor only refreshes every ~200-500ms on
        # this hardware (measured directly: 20 distinct readings across a 9.63s
        # window of tight adapter calls -- confirmed empirically, not assumed).
        # ADAPTER_TIMING_REPS alone (a few hundred calls, ~microseconds each) finishes
        # in under 1ms of wall time -- entirely within a SINGLE sensor refresh
        # interval, so integrating power over that window measures sensor noise,
        # not the adapter's energy. An earlier version of this script did exactly
        # that and reported a number with only 4 real power samples behind it.
        # Fixed by calibrating the rep count to a wall-clock DURATION
        # (ADAPTER_ENERGY_WINDOW_S) long enough to span ~20+ real sensor updates,
        # the same order of magnitude as the multi-second full-generation calls
        # below, so both energy numbers rest on comparably many real readings.
        per_call_ms = adapter_times_ms.mean()
        energy_reps = max(ADAPTER_TIMING_REPS, int(ADAPTER_ENERGY_WINDOW_S * 1000 / per_call_ms))
        print(f"(calibrated {energy_reps} reps for a ~{ADAPTER_ENERGY_WINDOW_S:.0f}s energy-sampling window)")

        def run_adapter_reps():
            for _ in range(energy_reps):
                with torch.no_grad():
                    adapter(ctx_batch)
            torch.cuda.synchronize()
        _, energy_j, mean_w, elapsed_s, n_samples = sample_power_during(run_adapter_reps)
        if energy_j is not None:
            net_w = (mean_w - idle_w) if idle_w is not None else None
            adapter_energy = {
                "reps": energy_reps, "total_joules": energy_j, "mean_power_w": mean_w,
                "elapsed_s": elapsed_s, "n_power_samples": n_samples,
                "joules_per_call": energy_j / energy_reps,
                "mj_per_event": energy_j / energy_reps / len(prepared) * 1000,
                "net_of_idle_mean_power_w": net_w,
                "net_of_idle_mj_per_event": (net_w * elapsed_s / energy_reps / len(prepared) * 1000)
                if net_w is not None else None,
            }
            print(f"energy over {energy_reps} calls ({n_samples} power samples, {elapsed_s:.1f}s): "
                  f"{energy_j:.2f} J total, gross {adapter_energy['mj_per_event']:.4f} mJ/event, "
                  f"net-of-idle {adapter_energy['net_of_idle_mj_per_event']:.4f} mJ/event")

    # --- 3. Full-generation energy per arm, per real event ------------------
    print("\n" + "=" * 70)
    print(f"FULL-GENERATION ENERGY, INTERLEAVED (n={len(prepared)} real events)")
    print("=" * 70)
    gen_rows = []
    for item in prepared:
        he = item["health_event"]
        a_out, a_energy, a_power, a_elapsed, a_n = sample_power_during(
            run_baseline_arm_timed, he, perception_agent=None)
        b_out, b_energy, b_power, b_elapsed, b_n = sample_power_during(
            run_adapter_arm_timed, he, item["context_vector"], model, processor, adapter)
        row = {
            "idx": item["idx"],
            "arm_a": {"joules": a_energy, "mean_power_w": a_power, "elapsed_s": a_elapsed,
                      "output_tokens": a_out["output_tokens"], "prompt_tokens": a_out["prompt_tokens"],
                      "generation_duration_ms": a_out["generation_duration_ms"],
                      "time_to_first_token_ms": a_out["time_to_first_token_ms"]},
            "arm_b": {"joules": b_energy, "mean_power_w": b_power, "elapsed_s": b_elapsed,
                      "output_tokens": b_out["output_tokens"], "prompt_tokens": b_out["prompt_tokens"],
                      "generation_duration_ms": b_out["generation_duration_ms"],
                      "time_to_first_token_ms": b_out["time_to_first_token_ms"]},
        }
        gen_rows.append(row)
        if a_energy is not None:
            print(f"idx={item['idx']:>6}  A: {a_energy:6.1f}J ({a_power:5.1f}W x {a_elapsed:5.2f}s, "
                  f"{a_n} samples)   B: {b_energy:6.1f}J ({b_power:5.1f}W x {b_elapsed:5.2f}s, {b_n} samples)")
        else:
            print(f"idx={item['idx']:>6}  (no NVML) A elapsed={a_elapsed:.2f}s  B elapsed={b_elapsed:.2f}s")

    # --- Save ----------------------------------------------------------------
    out = {
        "n_events": len(prepared), "idle_power_w": idle_w,
        "assumption": "Arm B communication boundary = 32-dim context vector, "
                       "pre-adapter (see module docstring)",
        "bytes_per_event": bytes_rows,
        "adapter_only": {
            "reps": ADAPTER_TIMING_REPS, "batch_size": len(prepared),
            "time_ms_mean": float(adapter_times_ms.mean()), "time_ms_median": float(np.median(adapter_times_ms)),
            "time_ms_min": float(adapter_times_ms.min()), "time_ms_max": float(adapter_times_ms.max()),
            "energy": adapter_energy,
        },
        "generation_energy_per_event": gen_rows,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_PATH}")

    # --- Summary ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    a_prompt_bytes = [r["arm_a"]["as_embedded_in_prompt_bytes"] for r in bytes_rows]
    b_ctx_bytes = [r["arm_b"]["context_vector_float32_bytes"] for r in bytes_rows]
    b_vt_bytes = [r["arm_b"]["virtual_tokens_float32_bytes"] for r in bytes_rows]
    print(f"Arm A serialized bytes (as embedded, mean): {np.mean(a_prompt_bytes):.0f} B")
    print(f"Arm B serialized bytes (context vector, mean): {np.mean(b_ctx_bytes):.0f} B  "
          f"[{'%.1f' % (np.mean(a_prompt_bytes)/np.mean(b_ctx_bytes))}x smaller than Arm A]")
    print(f"Arm B serialized bytes (virtual tokens, alt. framing, mean): {np.mean(b_vt_bytes):.0f} B  "
          f"[{'%.2f' % (np.mean(b_vt_bytes)/np.mean(a_prompt_bytes))}x Arm A -- LARGER, if adapter ran edge-side]")
    print(f"Adapter-only compute time: {adapter_times_ms.mean():.4f} ms/call (batch of {len(prepared)})")
    if adapter_energy:
        print(f"Adapter-only energy: {adapter_energy['mj_per_event']:.4f} mJ/event gross, "
              f"{adapter_energy['net_of_idle_mj_per_event']:.4f} mJ/event net-of-idle")
    a_energies = [r["arm_a"]["joules"] for r in gen_rows if r["arm_a"]["joules"] is not None]
    b_energies = [r["arm_b"]["joules"] for r in gen_rows if r["arm_b"]["joules"] is not None]
    if a_energies:
        print(f"Full-generation energy: Arm A mean={np.mean(a_energies):.1f}J  Arm B mean={np.mean(b_energies):.1f}J "
              f"({(1-np.mean(b_energies)/np.mean(a_energies))*100:.1f}% reduction)")

    # --- Prefill / decode / output-tokens / decode-speed, same pass as bytes+energy ---
    # prefill = time_to_first_token_ms (standard proxy: with greedy decoding the first
    # token cannot emerge before the full-prompt forward pass completes).
    # decode  = generation_duration_ms - prefill; decode_speed excludes the first token
    # (its latency is attributed to prefill, not decode).
    print("\n" + "=" * 70)
    print("PREFILL / DECODE / OUTPUT-TOKENS / DECODE-SPEED (same n=%d events)" % len(prepared))
    print("=" * 70)
    timing_summary = {}
    for arm in ["arm_a", "arm_b"]:
        ttft = np.array([r[arm]["time_to_first_token_ms"] for r in gen_rows], float)
        total = np.array([r[arm]["generation_duration_ms"] for r in gen_rows], float)
        out_tok = np.array([r[arm]["output_tokens"] for r in gen_rows], float)
        decode_ms = total - ttft
        speed = (out_tok - 1) / (decode_ms / 1000.0)
        timing_summary[arm] = {
            "prefill_ms_mean": float(ttft.mean()), "prefill_ms_median": float(np.median(ttft)),
            "decode_ms_mean": float(decode_ms.mean()), "decode_ms_median": float(np.median(decode_ms)),
            "output_tokens_mean": float(out_tok.mean()), "output_tokens_median": float(np.median(out_tok)),
            "decode_speed_tok_s_mean": float(speed.mean()), "decode_speed_tok_s_median": float(np.median(speed)),
        }
        s = timing_summary[arm]
        print(f"{arm}: prefill mean={s['prefill_ms_mean']:.1f}ms median={s['prefill_ms_median']:.1f}ms | "
              f"decode mean={s['decode_ms_mean']:.1f}ms median={s['decode_ms_median']:.1f}ms | "
              f"output_tokens mean={s['output_tokens_mean']:.1f} median={s['output_tokens_median']:.1f} | "
              f"decode_speed mean={s['decode_speed_tok_s_mean']:.2f}tok/s median={s['decode_speed_tok_s_median']:.2f}tok/s")

    out["timing_summary"] = timing_summary
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
