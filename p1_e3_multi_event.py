"""E3 (docs/analysis_plan.md, Deviation 5): interface cost when N events share one
prompt, the regime where per-event payload is multiplied.

For N in {1, 5, 10, 20, 50} consecutive beats from a DS2 record window:
  A-full     scaffold + a JSON list of N full payloads (dissertation Arm A fields)
  A-compact  scaffold + a JSON list of N compact payloads
  B-4        adapter scaffold + N x k virtual tokens (the trained single-event k=4
             adapter applied to each event's context vector, concatenated)
The text scaffold uses the last event's class context (A-arm scaffolds are class
specific; its length varies little). Cost only: output length is forced, content is
not scored, and B-4 was never trained on N > 1 (see item 7 of Deviation 5).

Per (window, N, arm): prefill = median of 3 forced 1-token generations; peak
allocated memory, total time and NVML energy for one forced 16-token generation.
Out-of-memory is recorded as a result, not an error. Arm order rotates; results are
checkpointed atomically after every condition and the run resumes.

Run (from repo root):
    python p1_e3_multi_event.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from measure_comm_cost import _HAS_NVML, idle_baseline, sample_power_during
from p1_io import save_json_atomic
from p1_step1_seeded_headline import provenance, sha256
from p1_step4_baseline_family import B4_CHECKPOINT, chat_inputs
from p1_step5_timing import timed_generate
from perception.perception_agent import PerceptionAgent, replay_selected
from project_config import DS2_PATH, PERCEPTION_CHECKPOINT
from reasoning.adapter_arm import load_trained_adapter
from reasoning.fixed_context import get_fixed_context_for_class
from reasoning.model_loader import load_model
from reasoning.prompt_template import OUTPUT_INSTRUCTIONS, SYSTEM_PROMPT, _extract_prompt_fields
from reasoning.prompt_template_family import _payload
from reasoning.virtual_adapter import (_render_prompt_with_placeholder, _tokenize_and_remove_placeholder,
                                       build_adapter_prompt_parts, compose_adapter_inputs)

NS = (1, 5, 10, 20, 50)
ARMS = ("A-full", "A-compact", "B-4")
N_WINDOWS = 20
PREFILL_REPS = 3
DECODE_TOKENS = 16
RESULTS_PATH = Path("results/p1_e3_multi_event.json")


class MultiEventAdapter(nn.Module):
    """Presents the single-event adapter to compose_adapter_inputs as one adapter
    emitting N x k tokens: (1, N*32) -> (1, N*k, E)."""

    def __init__(self, adapter, n_events):
        super().__init__()
        self.adapter, self.n_events = adapter, n_events
        self.num_tokens = adapter.num_tokens * n_events

    def forward(self, context):
        tokens = self.adapter(context.reshape(self.n_events, -1))  # (N, k, E)
        return tokens.reshape(1, self.num_tokens, tokens.shape[-1])


def windows(data, rng):
    """N_WINDOWS windows of max(NS) consecutive beats, each inside one record."""
    rec = data["record_ids"]
    starts = np.flatnonzero(np.r_[True, np.diff(rec) != 0])
    ends = np.r_[starts[1:], len(rec)]
    out = []
    while len(out) < N_WINDOWS:
        r = rng.integers(len(starts))
        if ends[r] - starts[r] < max(NS) + 1:
            continue
        off = int(rng.integers(starts[r], ends[r] - max(NS)))
        out.append(list(range(off, off + max(NS))))
    return out


def text_prompt(events, payload):
    items = [_extract_prompt_fields(e) if payload == "full" else _payload(e, "compact") for e in events]
    label = events[-1]["classification"]["label"]
    return (f"{SYSTEM_PROMPT}\n\n--- Background context for beat class '{label}' ---\n"
            f"{get_fixed_context_for_class(label)}\n\n--- Event data ---\n"
            f"{json.dumps(items, indent=2)}\n\n--- Instructions ---\n{OUTPUT_INSTRUCTIONS}")


def main():
    if not _HAS_NVML:
        raise RuntimeError("pynvml unavailable")
    state = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {
        "analysis_plan": "docs/analysis_plan.md (Deviation 5, E3)", "provenance": provenance(),
        "b4_checkpoint": {"path": str(B4_CHECKPOINT), "sha256": sha256(B4_CHECKPOINT)}, "rows": []}
    done = {(r["window"], r["n"], r["arm"]) for r in state["rows"]}

    with np.load(DS2_PATH) as z:
        data = {k: z[k] for k in ("features", "labels", "rr_interval_ms", "record_ids")}
    wins = windows(data, np.random.default_rng(0))
    state["windows"] = [[w[0], w[-1]] for w in wins]
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)
    replayed = replay_selected(agent, data["features"], data["rr_interval_ms"], data["record_ids"],
                               sorted({i for w in wins for i in w}))
    del agent

    model, processor = load_model()
    adapter = load_trained_adapter(str(B4_CHECKPOINT), model)
    prefix, suffix = build_adapter_prompt_parts({})
    rendered, s0, s1 = _render_prompt_with_placeholder(processor, prefix, suffix)
    prefix_ids, suffix_ids = _tokenize_and_remove_placeholder(processor, rendered, s0, s1)
    state.setdefault("idle_w", {}).setdefault("start", idle_baseline())

    def inputs_for(arm, idx):
        events = [replayed[i][0] for i in idx]
        if arm == "B-4":
            ctx = torch.from_numpy(np.stack([replayed[i][1] for i in idx]).reshape(1, -1))
            ai = compose_adapter_inputs(model, MultiEventAdapter(adapter, len(idx)), ctx, prefix_ids, suffix_ids)
            return dict(inputs_embeds=ai.inputs_embeds, attention_mask=ai.attention_mask,
                        per_layer_inputs=ai.per_layer_inputs), ai.sequence_length
        ci = chat_inputs(processor, model, text_prompt(events, "full" if arm == "A-full" else "compact"))
        return dict(input_ids=ci["input_ids"], attention_mask=ci["attention_mask"]), ci["input_ids"].shape[1]

    kw, _ = inputs_for("B-4", wins[0][:1])
    timed_generate(model, processor, 4, **kw)  # untimed warm-up
    t0, j = time.time(), 0
    for w, idx_all in enumerate(wins):
        for n in NS:
            order = ARMS[j % len(ARMS):] + ARMS[:j % len(ARMS)]
            j += 1
            for arm in order:
                if (w, n, arm) in done:
                    continue
                row = {"window": w, "n": n, "arm": arm}
                try:
                    kw, prompt_tokens = inputs_for(arm, idx_all[:n])
                    prefill = [timed_generate(model, processor, 1, **kw)["total_ms"] for _ in range(PREFILL_REPS)]
                    torch.cuda.reset_peak_memory_stats()
                    timing, joules, mean_w, _, n_samples = sample_power_during(
                        timed_generate, model, processor, DECODE_TOKENS, **kw)
                    row.update({"prompt_tokens": prompt_tokens, "prefill_ms": float(np.median(prefill)),
                                "prefill_ms_reps": prefill, "total_ms_16": timing["total_ms"],
                                "energy_j_16": joules, "power_samples": n_samples,
                                "peak_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20, "oom": False})
                except torch.cuda.OutOfMemoryError:
                    row.update({"oom": True})
                    torch.cuda.empty_cache()
                state["rows"].append(row)
                save_json_atomic(RESULTS_PATH, state)
        print(f"[window {w+1}/{len(wins)}] elapsed={time.time() - t0:.0f}s", flush=True)

    state["idle_w"]["end"] = idle_baseline()
    state["summary"] = summarize(state)
    save_json_atomic(RESULTS_PATH, state)
    print(json.dumps(state["summary"], indent=2))


def summarize(state):
    get = {(r["window"], r["n"], r["arm"]): r for r in state["rows"]}
    wins = sorted({r["window"] for r in state["rows"]})
    out = {"by_n": {}, "gate": {}}
    for n in NS:
        res = {}
        for arm in ARMS:
            rs = [get[(w, n, arm)] for w in wins if (w, n, arm) in get]
            ok = [r for r in rs if not r["oom"]]
            med = lambda k: float(np.median([r[k] for r in ok])) if ok else None
            res[arm] = {"n_oom": len(rs) - len(ok), "prompt_tokens": med("prompt_tokens"),
                        "prefill_ms_median": med("prefill_ms"), "total_ms_16_median": med("total_ms_16"),
                        "energy_j_16_median": med("energy_j_16"), "peak_mem_mb_median": med("peak_mem_mb")}
        out["by_n"][str(n)] = res
        # Gate (Deviation 5): B-4 vs A-compact prefill, paired over windows.
        pairs = [(get[(w, n, "B-4")], get[(w, n, "A-compact")]) for w in wins
                 if (w, n, "B-4") in get and (w, n, "A-compact") in get]
        pairs = [(b, a) for b, a in pairs if not b["oom"] and not a["oom"]]
        if pairs:
            rel = np.array([100 * (b["prefill_ms"] - a["prefill_ms"]) / a["prefill_ms"] for b, a in pairs])
            boot = np.random.default_rng(0).integers(0, len(rel), (20000, len(rel)))
            ci = np.percentile(rel[boot].mean(1), [2.5, 97.5])
            out["gate"][str(n)] = {"b4_vs_acompact_prefill_rel_pct": float(rel.mean()),
                                   "ci95": [float(ci[0]), float(ci[1])], "n_windows": len(rel),
                                   "passes": bool(ci[1] < 0 and rel.mean() <= -10)}
    return out


if __name__ == "__main__":
    main()
