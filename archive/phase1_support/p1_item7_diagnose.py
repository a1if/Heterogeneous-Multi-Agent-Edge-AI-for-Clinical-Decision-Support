"""Item 7 diagnostics: why seed 202 fails where seed 101 passes (Gate A).

For each checkpoint (seed 101 best = update 80, seed 202 best = update 100, seed 202
final = update 111) and two window sets built exactly as in training:
  val    the 60 validation windows (held-out DS1 records {109, 205, 223})
  train  60 training windows, the first 5 of every (N, tier) cell
it records, per window:
  - the teacher-forced loss on the canonical target (no gradient), and
  - the greedy generation (96 new tokens, as validation), parsed and categorised:
    ok / wrong tier (under- or over-escalated) / unparsed (no JSON, truncated JSON,
    invalid fields), plus whether the justification names the right beat.
Train vs val separates "cannot fit the training data" (optimisation) from "fits
training data but does not transfer" (generalisation). Atomic save per row;
resumable. CPU-free analysis: `python p1_item7_diagnose.py --summary`.

Run (from repo root):
    python p1_item7_diagnose.py
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from p1_io import save_json_atomic
from p1_item7_train import TIER_WEIGHT, TIERS, VAL_MAX_NEW_TOKENS, build_windows

RESULTS = Path("results/p1_item7_diagnose.json")
CHECKPOINTS = {
    "seed101_best": ("reasoning/checkpoints/p1_item7_mea_seed101.pt", "adapter_state_dict"),
    "seed202_best": ("reasoning/checkpoints/p1_item7_mea_seed202.pt", "adapter_state_dict"),
    "seed202_final": ("reasoning/checkpoints/p1_item7_mea_seed202.resume.pt", "adapter_state_dict"),
}
PER_CELL_TRAIN = 5
RANK = {t: i for i, t in enumerate(TIERS)}


def diag_windows(r):
    train, val = build_windows(r)
    sub, seen = [], {}
    for w in train:  # train is ordered by (n, tier) cell
        k = (w["n"], w["tier"])
        if seen.get(k, 0) < PER_CELL_TRAIN:
            sub.append(w)
            seen[k] = seen.get(k, 0) + 1
    return {"val": val, "train": sub}


def categorise(text, ref, n, target_beat):
    """-> dict(category, answer, beat_ok)."""
    from reasoning.baseline_arm import _extract_last_json_object
    from reasoning.output_schema import ReasoningOutput
    try:
        out = ReasoningOutput(**_extract_last_json_object(text))
    except (ValueError, TypeError):
        if "{" not in text:
            cat = "unparsed_no_json"
        elif text.count("{") > text.count("}"):
            cat = "unparsed_truncated"
        else:
            cat = "unparsed_invalid"
        return {"category": cat, "answer": None, "beat_ok": None}
    ans = out.urgency_tier
    m = re.search(r"Beat (\d+) of (\d+)", out.justification or "")
    beat_ok = bool(m and int(m.group(1)) == target_beat + 1 and int(m.group(2)) == n) if ref != "routine" else None
    if ans == ref:
        cat = "ok"
    else:
        cat = "wrong_over" if RANK.get(ans, -1) > RANK[ref] else "wrong_under"
    return {"category": cat, "answer": ans, "beat_ok": beat_ok}


def run():
    from p1_item7_common import replay_split
    from p1_pilot_multi_event import scaffold_parts
    from reasoning.adapter_training import (_append_target_for_teacher_forcing, _labels_for_target,
                                            _target_ids_and_weights, _weighted_teacher_forcing_loss)
    from reasoning.model_loader import load_model
    from reasoning.multi_event_adapter import MultiEventVirtualAdapter, compose_multi_event_inputs
    from reasoning.training_targets import canonical_window_target, most_urgent_index
    from reasoning.virtual_adapter import _render_prompt_with_placeholder, _tokenize_and_remove_placeholder

    r = replay_split("ds1")
    sets = diag_windows(r)
    state = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {"design": __doc__, "rows": []}
    done = {(x["ckpt"], x["set"], x["n"], x["start"]) for x in state["rows"]}

    model, processor = load_model()
    tok = processor.tokenizer
    device = model.get_input_embeddings().weight.device
    scaffold = {}

    def parts(n):
        if n not in scaffold:
            pre, suf = scaffold_parts(n)
            rendered, s0, s1 = _render_prompt_with_placeholder(processor, pre, suf)
            scaffold[n] = _tokenize_and_remove_placeholder(processor, rendered, s0, s1)
        return scaffold[n]

    t0 = time.time()
    for name, (path, key) in CHECKPOINTS.items():
        adapter = MultiEventVirtualAdapter.for_model(model).to(device)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        adapter.load_state_dict(ck[key])
        adapter.eval()
        state.setdefault("checkpoints", {})[name] = {"path": path, "update": ck.get("update", ck.get("state", {}).get("updates")),
                                                      "scale": float(adapter.scale),
                                                      "projection_weight_norm": float(adapter.projection.weight.norm()),
                                                      "position_norm": float(adapter.position.norm())}
        for set_name, windows in sets.items():
            for w in windows:
                if (name, set_name, w["n"], w["start"]) in done:
                    continue
                events = r["events"][w["start"]:w["start"] + w["n"]]
                ai = compose_multi_event_inputs(model, adapter, r["vectors"][w["start"]:w["start"] + w["n"]], *parts(w["n"]))
                with torch.no_grad():
                    tid, tw = _target_ids_and_weights(processor, canonical_window_target(events), w["tier"], TIER_WEIGHT)
                    e, m, p = _append_target_for_teacher_forcing(model, ai, tid)
                    lab, lw = _labels_for_target(ai.sequence_length, tid.to(e.device), tw.to(e.device))
                    keep = tid.shape[1] + 1
                    loss = float(_weighted_teacher_forcing_loss(
                        model(inputs_embeds=e, attention_mask=m, per_layer_inputs=p, use_cache=False,
                              logits_to_keep=keep).logits, lab[:, -keep:], lw[:, -keep:]))
                    out = model.generate(inputs_embeds=ai.inputs_embeds, attention_mask=ai.attention_mask,
                                         per_layer_inputs=ai.per_layer_inputs, do_sample=False,
                                         max_new_tokens=VAL_MAX_NEW_TOKENS)
                text = tok.decode(out[0], skip_special_tokens=True)
                state["rows"].append({"ckpt": name, "set": set_name, "n": w["n"], "start": w["start"], "tier": w["tier"],
                                      "loss": loss, "text": text,
                                      **categorise(text, w["tier"], w["n"], most_urgent_index(events))})
                save_json_atomic(RESULTS, state)
            print(f"[{name} {set_name}] done, elapsed {time.time() - t0:.0f}s", flush=True)
        del adapter
        torch.cuda.empty_cache()
    state["summary"] = summarize(state["rows"])
    save_json_atomic(RESULTS, state)
    print(json.dumps(state["summary"], indent=1))


def summarize(rows):
    out = {}
    for ck in sorted({x["ckpt"] for x in rows}):
        for s in ("train", "val"):
            a = [x for x in rows if x["ckpt"] == ck and x["set"] == s]
            if not a:
                continue
            rec = {t: float(np.mean([x["answer"] == t for x in a if x["tier"] == t])) for t in TIERS}
            cats = {}
            for x in a:
                cats[x["category"]] = cats.get(x["category"], 0) + 1
            beat = [x["beat_ok"] for x in a if x["beat_ok"] is not None]
            out[f"{ck}|{s}"] = {
                "n_windows": len(a), "mean_loss": float(np.mean([x["loss"] for x in a])),
                "loss_by_tier": {t: float(np.mean([x["loss"] for x in a if x["tier"] == t])) for t in TIERS},
                "balanced_accuracy": float(np.mean(list(rec.values()))), "recall": rec,
                "parse_rate": float(np.mean([x["answer"] is not None for x in a])), "categories": cats,
                "beat_named_correctly": float(np.mean(beat)) if beat else None,
                "by_n": {str(n): float(np.mean([np.mean([x["answer"] == t for x in a if x["n"] == n and x["tier"] == t])
                                                for t in TIERS])) for n in sorted({x["n"] for x in a})}}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    if ap.parse_args().summary:
        print(json.dumps(summarize(json.loads(RESULTS.read_text(encoding="utf-8"))["rows"]), indent=1))
    else:
        run()
