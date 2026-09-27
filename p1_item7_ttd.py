"""Item 7, Deviation 14: time to the urgency decision (TTD), A-compact vs MEA.

Batch 1, greedy, schema-constrained decoding (reasoning/constrained_json.py): the tier
is always generated token 7, so generation stops right after it. A StoppingCriteria
records CUDA-synchronised wall-clock times from the generate() call to the first
generated token (TTFT) and to the tier token (TTD). DS2 test windows at N in
{1, 5, 10, 20, 50}: the first 7 per tier per N from the Deviation 10 / 13 sets (21 per
N); arm order alternates per window; 3 warm-up calls first. Atomic save per window;
resumable.

Run (from repo root):
    python p1_item7_ttd.py
    python p1_item7_ttd.py --summary
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from p1_io import save_json_atomic

RESULTS = Path("results/p1_item7_ttd.json")
MEA_CKPT = "reasoning/checkpoints/p1_item7_mea_r3_seed101.pt"
NS = (1, 5, 10, 20, 50)
PER_TIER = 7
TIERS = ("routine", "priority", "urgent")
TIER_TOKEN_INDEX = 7  # 6 forced opening tokens, then the tier


class Clock(StoppingCriteria):
    """Records the time each new token is appended; stops after the tier token."""

    def __init__(self, t0):
        self.t0, self.start, self.times = t0, None, []

    def __call__(self, input_ids, scores, **kw):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self.start is None:
            self.start = input_ids.shape[1] - 1  # length before the first generated token
        self.times.append(now - self.t0)
        done = input_ids.shape[1] - self.start >= TIER_TOKEN_INDEX
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


def select(windows):
    out = []
    for n in NS:
        for t in TIERS:
            out += [w for w in windows if w["set"] == "stratified" and w["n"] == n and w["reference"] == t][:PER_TIER]
    return out


def run():
    from p1_item7_common import replay_split
    from p1_item7_eval import load_windows
    from p1_pilot_multi_event import scaffold_parts
    from reasoning.constrained_json import SchemaJsonProcessor, TokenTable
    from reasoning.model_loader import load_model
    from reasoning.multi_event_adapter import MultiEventVirtualAdapter, compose_multi_event_inputs
    from reasoning.prompt_template_family import _payload
    from reasoning.virtual_adapter import _render_prompt_with_placeholder, _tokenize_and_remove_placeholder

    r = replay_split("ds2")
    windows = select(load_windows("ds2", r) + load_windows("ds2_n50", r))
    state = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {
        "analysis_plan": "docs/analysis_plan.md (Deviation 14)", "design": __doc__, "rows": []}
    done = {(x["arm"], x["n"], x["start"]) for x in state["rows"]}
    model, processor = load_model()
    tok = processor.tokenizer
    device = model.get_input_embeddings().weight.device
    table = TokenTable(tok, device=device)
    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    ck = torch.load(MEA_CKPT, map_location="cpu", weights_only=False)["adapter_state_dict"]
    adapter = MultiEventVirtualAdapter.for_model(model, max_events=ck["position"].shape[0]).to(device)
    adapter.load_state_dict(ck)
    adapter.eval()
    scaffold = {}

    def inputs(arm, w):
        n, idx = w["n"], range(w["start"], w["start"] + w["n"])
        pre, suf = scaffold_parts(n)
        if arm == "A-compact":
            payload = json.dumps([_payload(r["events"][i], "compact") for i in idx], indent=2)
            ids = processor.apply_chat_template([{"role": "user", "content": pre + payload + suf}],
                                                add_generation_prompt=True, tokenize=True, return_dict=True,
                                                return_tensors="pt")
            return {"input_ids": ids["input_ids"].to(device), "attention_mask": ids["attention_mask"].to(device)}, \
                int(ids["input_ids"].shape[1])
        if n not in scaffold:
            rendered, s0, s1 = _render_prompt_with_placeholder(processor, pre, suf)
            scaffold[n] = _tokenize_and_remove_placeholder(processor, rendered, s0, s1)
        ai = compose_multi_event_inputs(model, adapter, r["vectors"][w["start"]:w["start"] + n], *scaffold[n])
        return {"inputs_embeds": ai.inputs_embeds, "attention_mask": ai.attention_mask,
                "per_layer_inputs": ai.per_layer_inputs}, int(ai.inputs_embeds.shape[1])

    def timed(arm, w):
        with torch.no_grad():
            kw, n_prompt = inputs(arm, w)
            proc = SchemaJsonProcessor(table, eos, 40)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            clock = Clock(t0)
            out = model.generate(**kw, do_sample=False, max_new_tokens=TIER_TOKEN_INDEX, logits_processor=[proc],
                                 stopping_criteria=StoppingCriteriaList([clock]))
        # The processor only sees a token on the next step, and generation stops right after the
        # tier token, so the tier is read from the generated tokens themselves.
        gen = tok.decode(out[0][-TIER_TOKEN_INDEX:], skip_special_tokens=True)
        tier = next((t for t in TIERS if gen.endswith(t)), None)
        return {"ttft_ms": clock.times[0] * 1000, "ttd_ms": clock.times[-1] * 1000, "n_tokens": len(clock.times),
                "prompt_tokens": n_prompt, "tier": tier, "generated": gen}

    for w in windows[:3]:  # warm-up
        timed("A-compact", w)
        timed("MEA", w)
    t_start = time.time()
    for k, w in enumerate(windows):
        order = ("A-compact", "MEA") if k % 2 == 0 else ("MEA", "A-compact")
        for arm in order:
            if (arm, w["n"], w["start"]) in done:
                continue
            state["rows"].append({"arm": arm, "n": w["n"], "start": w["start"], "reference": w["reference"],
                                  "order": order.index(arm), **timed(arm, w)})
        save_json_atomic(RESULTS, state)
        if (k + 1) % 21 == 0:
            print(f"[{k + 1}/{len(windows)}] elapsed {time.time() - t_start:.0f}s", flush=True)
    state["summary"] = summarize(state["rows"])
    save_json_atomic(RESULTS, state)
    print(json.dumps(state["summary"], indent=1))


def summarize(rows, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for n in NS:
        pair = {}
        for x in rows:
            if x["n"] == n:
                pair.setdefault(x["start"], {})[x["arm"]] = x
        pair = [v for v in pair.values() if len(v) == 2]
        if not pair:
            continue
        res = {"n_windows": len(pair)}
        for m in ("ttft_ms", "ttd_ms"):
            a = np.array([p["A-compact"][m] for p in pair])
            b = np.array([p["MEA"][m] for p in pair])
            rel = b / a - 1
            boots = [np.median(rel[rng.integers(0, len(rel), len(rel))]) for _ in range(n_boot)]
            res[m] = {"a_compact_median": float(np.median(a)), "mea_median": float(np.median(b)),
                      "median_rel_diff": float(np.median(rel)), "ci95": [float(np.percentile(boots, 2.5)),
                                                                        float(np.percentile(boots, 97.5))]}
        res["prompt_tokens_median"] = {arm: float(np.median([p[arm]["prompt_tokens"] for p in pair]))
                                       for arm in ("A-compact", "MEA")}
        res["tier_match_reference"] = {arm: float(np.mean([p[arm]["tier"] == p[arm]["reference"] for p in pair]))
                                       for arm in ("A-compact", "MEA")}
        out[str(n)] = res
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    if ap.parse_args().summary:
        print(json.dumps(summarize(json.loads(RESULTS.read_text(encoding="utf-8"))["rows"]), indent=1))
    else:
        run()
