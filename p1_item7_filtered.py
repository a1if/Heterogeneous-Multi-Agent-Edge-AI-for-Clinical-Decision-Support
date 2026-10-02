"""Item 7, Deviation 20: filtered/summarised text baseline (A-filtered).

A-filtered lists only the non-normal beats of a window (position + A-compact's fields) and
summarises the rest as a count of normal beats. Everything else (system prompt, neutral
context, instructions, constrained decoder) is A-compact's. Generation on the DS2 test set
runs through p1_item7_eval.py (arm "A-filtered"); this script holds the prompt builder and:

  tokens    prompt tokens of A-compact, A-filtered and MEA for every DS2 v2 window (CPU)
  collect   tier logits for A-filtered on the 147 recipe-r3 validation windows and the DS2 v2
            set, for the Deviation 17 calibration rule (GPU, resumable)
  timing    TTFT and time to decision, A-compact vs A-filtered vs MEA r4 seed 101, on the
            Deviation 14 windows (GPU, resumable)
  analyse   primary comparison, calibration, cost and cost growth (CPU)

Run (from repo root):
    python p1_item7_eval.py --split ds2v2 --suffix r4 --arms A-filtered
    python p1_item7_filtered.py tokens|collect|timing|analyse
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from p1_io import save_json_atomic

TIERS = ("routine", "priority", "urgent")
EVAL = Path("results/p1_item7_eval_ds2v2_r4.json")
import os
_TAG = os.environ.get("P1_ENCODER_TAG", "")  # Deviation 22: per-sender artifacts
TOKENS = Path(f"results/p1_item7_filtered_tokens{_TAG}.json")
LOGITS = Path(f"results/p1_item7_filtered_logits{_TAG}.json")
TIMING = Path(f"results/p1_item7_filtered_timing{_TAG}.json")
OUT = Path("results/p1_item7_filtered.json")
R4 = ("MEA:r4_seed101", "MEA:r4_seed202", "MEA:r4_seed303")
MEA_TIMING_CKPT = f"reasoning/checkpoints/p1_item7_mea_r4{os.environ.get('P1_ENCODER_TAG', '')}_seed101.pt"


def filtered_note(n):
    return (f"The event data below summarises {n} consecutive beats from one recording. Normal (N) beats are not "
            "listed individually: their number is given as normal_beats_not_listed. Every non-normal beat is listed "
            "with its position in the sequence (1 = first beat). Apply the rule to each listed beat and report the "
            f"single most urgent tier among all {n} beats; if no beat is listed, every beat is normal.\n")


def filtered_content(events, start, n):
    """Full user-message text of an A-filtered prompt for events[start:start + n]."""
    from p1_pilot_multi_event import NEUTRAL_CONTEXT, OUTPUT_INSTRUCTIONS, SYSTEM_PROMPT
    from reasoning.prompt_template_family import _payload
    listed = [{"position": k + 1, **_payload(events[start + k], "compact")}
              for k in range(n) if events[start + k]["classification"]["label"] != "N"]
    payload = json.dumps({"total_beats": n, "normal_beats_not_listed": n - len(listed), "non_normal_beats": listed},
                         indent=2)
    return (f"{SYSTEM_PROMPT}\n\n--- Background context ---\n{NEUTRAL_CONTEXT}\n{filtered_note(n)}--- Event data ---\n"
            f"{payload}\n\n--- Instructions ---\n{OUTPUT_INSTRUCTIONS}")


def compact_content(events, start, n):
    from p1_pilot_multi_event import scaffold_parts
    from reasoning.prompt_template_family import _payload
    pre, suf = scaffold_parts(n)
    return pre + json.dumps([_payload(events[i], "compact") for i in range(start, start + n)], indent=2) + suf


def text_ids(processor, content):
    return processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                         tokenize=True, return_dict=True, return_tensors="pt")


def windows(split="ds2v2"):
    path = f"results/p1_item7_testset_v2{_TAG}.json" if split == "ds2v2" else "results/p1_item7_testset_incart.json"
    v2 = json.loads(Path(path).read_text(encoding="utf-8"))["windows"]
    return [{k: w[k] for k in ("set", "n", "start", "reference", "record")} for w in v2]


# ---------- tokens (CPU) ----------
def tokens():
    from transformers import AutoProcessor
    from p1_item7_common import replay_split
    from p1_pilot_multi_event import scaffold_parts
    from reasoning.model_loader import MODEL_ID
    from reasoning.virtual_adapter import _render_prompt_with_placeholder, _tokenize_and_remove_placeholder
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    ev = replay_split("ds2")["events"]
    scaffold = {}
    rows = []
    for w in windows():
        n, s = w["n"], w["start"]
        if n not in scaffold:
            rendered, s0, s1 = _render_prompt_with_placeholder(processor, *scaffold_parts(n))
            pre, suf = _tokenize_and_remove_placeholder(processor, rendered, s0, s1)
            scaffold[n] = int(pre.shape[1] + suf.shape[1])
        rows.append({**w, "abnormal": sum(ev[i]["classification"]["label"] != "N" for i in range(s, s + n)),
                     "A-compact": int(text_ids(processor, compact_content(ev, s, n))["input_ids"].shape[1]),
                     "A-filtered": int(text_ids(processor, filtered_content(ev, s, n))["input_ids"].shape[1]),
                     "MEA": scaffold[n] + 4 * n})
    save_json_atomic(TOKENS, {"design": __doc__, "rows": rows})
    for n in sorted({x["n"] for x in rows}):
        xs = [x for x in rows if x["n"] == n]
        print(n, {a: float(np.median([x[a] for x in xs])) for a in ("A-compact", "A-filtered", "MEA")})


# ---------- calibration logits (GPU) ----------
def collect(text_arm="A-filtered", split="ds2v2"):
    import torch
    from p1_item7_common import replay_split
    from p1_item7_train import RECIPES, build_windows
    from reasoning.constrained_json import TokenTable
    from reasoning.model_loader import load_model
    if split == "incart":  # Deviation 21: confirmatory set only; calibration biases stay frozen from DS1 validation
        sets = {"incart": (replay_split("incart"), windows("incart"))}
        logits_path = Path(f"results/p1_incart_{'filtered' if text_arm == 'A-filtered' else 'compact'}_logits.json")
    else:
        r1, r2 = replay_split("ds1"), replay_split("ds2")
        rc = RECIPES["r3"]
        _, v3 = build_windows(r1, rc["val_per_cell"], rc["val_max_per_record"], rc["ns"])
        sets = {"val": (r1, [{"set": "val", "n": w["n"], "start": w["start"], "reference": w["tier"]} for w in v3]),
                "ds2v2": (r2, windows())}
        logits_path = LOGITS if text_arm == "A-filtered" else Path(f"results/p1_item7_compact_logits{_TAG}.json")
    build = filtered_content if text_arm == "A-filtered" else compact_content
    state = json.loads(logits_path.read_text(encoding="utf-8")) if logits_path.exists() else {"design": __doc__, "arm": text_arm, "rows": []}
    done = {(x["split"], x["set"], x["n"], x["start"]) for x in state["rows"]}
    model, processor = load_model()
    device = model.get_input_embeddings().weight.device
    table = TokenTable(processor.tokenizer, device=device)
    prefix = torch.tensor([table.canon['{"urgency_tier":"']], device=device)
    tier_ids = [table.tiers[t][0] for t in TIERS]
    t0 = time.time()
    for split, (r, ws) in sets.items():
        for k, w in enumerate(x for x in ws if (split, x["set"], x["n"], x["start"]) not in done):
            ids = text_ids(processor, build(r["events"], w["start"], w["n"]))["input_ids"].to(device)
            with torch.no_grad():
                lg = model(input_ids=torch.cat([ids, prefix], 1), use_cache=False, logits_to_keep=1).logits
            state["rows"].append({"split": split, **{k2: w[k2] for k2 in ("set", "n", "start", "reference")},
                                  "logits": lg[0, -1, tier_ids].float().tolist()})
            if (k + 1) % 50 == 0:
                save_json_atomic(logits_path, state)
                print(f"[{split} {k + 1}] {time.time() - t0:.0f}s", flush=True)
        save_json_atomic(logits_path, state)
    print("collect done", len(state["rows"]), flush=True)


# ---------- timing (GPU) ----------
def timing(split="ds2v2"):
    import torch
    from transformers import StoppingCriteriaList
    from p1_item7_common import replay_split, window_vectors
    from p1_item7_eval import load_windows
    from p1_item7_ttd import TIER_TOKEN_INDEX, Clock, select
    from p1_pilot_multi_event import scaffold_parts
    from reasoning.constrained_json import SchemaJsonProcessor, TokenTable
    from reasoning.model_loader import load_model
    from reasoning.multi_event_adapter import MultiEventVirtualAdapter, compose_multi_event_inputs
    from reasoning.virtual_adapter import _render_prompt_with_placeholder, _tokenize_and_remove_placeholder
    timing_path = TIMING if split == "ds2v2" else Path("results/p1_incart_timing.json")
    if split == "incart":  # Deviation 21: first 7 stratified windows per tier per N of the INCART set
        r = replay_split("incart")
        ws = select(windows("incart"))
    else:
        r = replay_split("ds2")
        ws = select(load_windows("ds2", r) + load_windows("ds2_n50", r))
    state = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {"design": __doc__, "rows": []}
    done = {(x["arm"], x["n"], x["start"]) for x in state["rows"]}
    model, processor = load_model()
    tok = processor.tokenizer
    device = model.get_input_embeddings().weight.device
    table = TokenTable(tok, device=device)
    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    ck = torch.load(MEA_TIMING_CKPT, map_location="cpu", weights_only=False)["adapter_state_dict"]
    adapter = MultiEventVirtualAdapter.for_model(model, max_events=ck["position"].shape[0],
                                                 input_dim=ck["projection.weight"].shape[1]).to(device)
    adapter.load_state_dict(ck)
    adapter.eval()
    scaffold = {}

    def inputs(arm, w):
        n, s = w["n"], w["start"]
        if arm in ("A-compact", "A-filtered"):
            content = (compact_content if arm == "A-compact" else filtered_content)(r["events"], s, n)
            ids = text_ids(processor, content)
            return {"input_ids": ids["input_ids"].to(device), "attention_mask": ids["attention_mask"].to(device)}, \
                int(ids["input_ids"].shape[1])
        if n not in scaffold:
            rendered, s0, s1 = _render_prompt_with_placeholder(processor, *scaffold_parts(n))
            scaffold[n] = _tokenize_and_remove_placeholder(processor, rendered, s0, s1)
        ai = compose_multi_event_inputs(model, adapter, window_vectors(r, s, n, adapter.input_dim), *scaffold[n])
        return {"inputs_embeds": ai.inputs_embeds, "attention_mask": ai.attention_mask,
                "per_layer_inputs": ai.per_layer_inputs}, int(ai.inputs_embeds.shape[1])

    def timed(arm, w):
        with torch.no_grad():
            kw, n_prompt = inputs(arm, w)
            proc = SchemaJsonProcessor(table, eos, 40)
            torch.cuda.synchronize()
            clock = Clock(time.perf_counter())
            out = model.generate(**kw, do_sample=False, max_new_tokens=TIER_TOKEN_INDEX, logits_processor=[proc],
                                 stopping_criteria=StoppingCriteriaList([clock]))
        gen = tok.decode(out[0][-TIER_TOKEN_INDEX:], skip_special_tokens=True)
        return {"ttft_ms": clock.times[0] * 1000, "ttd_ms": clock.times[-1] * 1000, "prompt_tokens": n_prompt,
                "tier": next((t for t in TIERS if gen.endswith(t)), None)}

    arms = ("A-compact", "A-filtered", "MEA")
    for w in ws[:3]:
        for a in arms:
            timed(a, w)  # warm-up
    for k, w in enumerate(ws):
        order = arms[k % 3:] + arms[:k % 3]  # rotate arm order per window
        for a in order:
            if (a, w["n"], w["start"]) not in done:
                state["rows"].append({"arm": a, "n": w["n"], "start": w["start"], "reference": w["reference"],
                                      "order": order.index(a), **timed(a, w)})
        save_json_atomic(timing_path, state)
    print("timing done", len(state["rows"]), flush=True)


# ---------- analysis (CPU) ----------
def analyse():
    import p1_item7_tier1 as t1
    from p1_item7_calib import GRID, answer
    from p1_item7_common import replay_split
    rec = np.asarray(replay_split("ds2")["record_ids"])
    rows = json.loads(EVAL.read_text(encoding="utf-8"))["rows"]
    out = {"design": __doc__}

    def compare(rows_, base):
        res = {}
        for n, wins in sorted(t1.windows_by_n(rows_).items()):
            starts = [s for s, w in wins.items() if all(a in w["arms"] for a in R4 + (base,))]
            ba = {a: t1.bal([(wins[s]["reference"], wins[s]["arms"][a]) for s in starts]) for a in R4 + (base,)}
            d = {"base": ba[base], "r4_mean": float(np.mean([ba[m] for m in R4])), "n_windows": len(starts)}
            if n >= 5:
                d["test"] = t1.cluster_ci({s: {"reference": wins[s]["reference"],
                                               "arms": {a: wins[s]["arms"][a] for a in R4 + (base,)}} for s in starts},
                                          list(R4), {s: int(rec[s]) for s in starts}, base=base)
            res[str(n)] = d
        return res

    out["vs_filtered_default"] = compare(rows, "A-filtered")
    out["filtered_vs_compact"] = {n: {"A-filtered": t1.bal([(w["reference"], w["arms"]["A-filtered"]) for w in wins.values()
                                                            if "A-filtered" in w["arms"]]),
                                      "A-compact": t1.bal([(w["reference"], w["arms"]["A-compact"]) for w in wins.values()
                                                           if "A-compact" in w["arms"]])}
                                  for n, wins in ((str(k), v) for k, v in sorted(t1.windows_by_n(rows).items()))}
    # calibration (Deviation 17 primary rule: max validation balanced accuracy, ties -> largest delta)
    lg = json.loads(LOGITS.read_text(encoding="utf-8"))["rows"]
    val = [x for x in lg if x["split"] == "val"]
    curve = [(float(d), t1.bal([(x["reference"], answer(x["logits"], d)) for x in val])) for d in GRID]
    delta = max(curve, key=lambda c: (round(c[1], 6), c[0]))[0]
    saved = {(x["set"], x["n"], x["start"]): x["tier"] for x in rows if x["arm"] == "A-filtered"}
    ds2 = {(x["set"], x["n"], x["start"]): x["logits"] for x in lg if x["split"] == "ds2v2"}
    agree = [answer(v, 0) == saved.get(k) for k, v in ds2.items() if k in saved]
    out["calibration"] = {"delta": delta, "val_at_0": dict(curve)[0.0], "val_at_delta": dict(curve)[delta],
                          "delta0_matches_greedy": {"n": len(agree), "agree": int(sum(agree))}}
    cal_rows = []
    for x in rows:
        if x["arm"] == "A-filtered":
            t = answer(ds2[(x["set"], x["n"], x["start"])], delta)
            x = {**x, "arm": "A-filtered-cal", "tier": t, "correct": t == x["reference"]}
        cal_rows.append(x)
    out["vs_filtered_calibrated"] = compare(cal_rows, "A-filtered-cal")
    # false alarms
    out["false_alarm"] = {a: {s: float(np.mean([x["tier"] != "routine" for x in rows_ if x["arm"] == a and x["reference"] == "routine"
                                                and (s == "all" or x["set"] == s)]))
                              for s in ("all", "natural")}
                          for a, rows_ in (("A-filtered", rows), ("A-filtered-cal", cal_rows), ("A-compact", rows),
                                           *((m, rows) for m in R4))}
    # cost
    tk = json.loads(TOKENS.read_text(encoding="utf-8"))["rows"]
    out["prompt_tokens"] = {str(n): {a: {"median": float(np.median([x[a] for x in tk if x["n"] == n])),
                                         "p90": float(np.percentile([x[a] for x in tk if x["n"] == n], 90)),
                                         "max": int(max(x[a] for x in tk if x["n"] == n))}
                                     for a in ("A-compact", "A-filtered", "MEA")} for n in sorted({x["n"] for x in tk})}
    ab = np.array([x["abnormal"] for x in tk])
    base = np.array([x["A-filtered"] for x in tk])
    X = np.c_[np.ones(len(tk)), ab, [x["n"] for x in tk]]
    coef = np.linalg.lstsq(X, base, rcond=None)[0]
    mea50 = next(x["MEA"] for x in tk if x["n"] == 50)
    out["cost_growth"] = {"tokens_per_abnormal_beat": float(coef[1]), "tokens_per_listed_n": float(coef[2]),
                          "intercept": float(coef[0]),
                          "abnormal_beats_where_filtered_exceeds_mea_at_n50": float((mea50 - coef[0] - coef[2] * 50) / coef[1]),
                          "share_of_windows_above_mea": {str(n): float(np.mean([x["A-filtered"] > x["MEA"] for x in tk if x["n"] == n]))
                                                         for n in sorted({x["n"] for x in tk})}}
    if TIMING.exists():
        tr = json.loads(TIMING.read_text(encoding="utf-8"))["rows"]
        rng = np.random.default_rng(0)
        tim = {}
        for n in sorted({x["n"] for x in tr}):
            by = {}
            for x in tr:
                if x["n"] == n:
                    by.setdefault(x["start"], {})[x["arm"]] = x
            trip = [v for v in by.values() if len(v) == 3]
            d = {"n_windows": len(trip)}
            for m in ("ttft_ms", "ttd_ms", "prompt_tokens"):
                d[m] = {a: float(np.median([v[a][m] for v in trip])) for a in ("A-compact", "A-filtered", "MEA")}
            for m in ("ttft_ms", "ttd_ms"):
                rel = np.array([v["MEA"][m] / v["A-filtered"][m] - 1 for v in trip])
                boots = [np.median(rel[rng.integers(0, len(rel), len(rel))]) for _ in range(20000)]
                d[m]["mea_vs_filtered_median_rel"] = float(np.median(rel))
                d[m]["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            tim[str(n)] = d
        out["timing"] = tim
    # pre-registered interpretation per N
    verdict = {}
    for n in ("5", "10", "20", "50"):
        t = out["vs_filtered_default"][n]["test"]
        fil_ni = t["cluster_ci95"][1] < 0.05  # A-filtered non-inferior to r4 <=> r4 - filtered upper bound < +0.05
        shorter = out["prompt_tokens"][n]["A-filtered"]["median"] <= out["prompt_tokens"][n]["MEA"]["median"]
        verdict[n] = {"filtered_non_inferior_to_r4": bool(fil_ni), "filtered_prompt_not_longer": bool(shorter),
                      "r4_superior_to_filtered": bool(t["superior"]),
                      "advantage_over_text_holds_against_filtering": not (fil_ni and shorter)}
    out["verdict"] = verdict
    save_json_atomic(OUT, out)
    show(out)


def show(out):
    f = lambda d: f"{d['difference']:+.3f} [{d['cluster_ci95'][0]:+.3f}, {d['cluster_ci95'][1]:+.3f}]"
    print("calibration:", out["calibration"])
    for n, d in out["vs_filtered_default"].items():
        c = out["vs_filtered_calibrated"][n]
        line = f"N={n:>2} filtered {d['base']:.3f} (cal {c['base']:.3f}) compact {out['filtered_vs_compact'][n]['A-compact']:.3f} | r4 {d['r4_mean']:.3f}"
        if "test" in d:
            line += f" || r4-filtered {f(d['test'])} || r4-filtered(cal) {f(c['test'])}"
        print(line)
    print("prompt tokens (median):", {n: {a: v["median"] for a, v in d.items()} for n, d in out["prompt_tokens"].items()})
    print("cost growth:", out["cost_growth"])
    print("false alarms:", {a: {k: round(v, 3) for k, v in d.items()} for a, d in out["false_alarm"].items()})
    if "timing" in out:
        for n, d in out["timing"].items():
            print(f"N={n} TTFT {d['ttft_ms']} TTD {d['ttd_ms']}")
    print("verdict:", json.dumps(out["verdict"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("tokens", "collect", "timing", "analyse"))
    ap.add_argument("--text-arm", choices=("A-filtered", "A-compact"), default="A-filtered", help="collect only")
    ap.add_argument("--split", choices=("ds2v2", "incart"), default="ds2v2", help="collect / timing (Deviation 21)")
    a = ap.parse_args()
    if a.mode == "collect":
        collect(a.text_arm, a.split)
    elif a.mode == "timing":
        timing(a.split)
    else:
        {"tokens": tokens, "analyse": analyse}[a.mode]()
