"""Item 7, Deviation 17: false-alarm reduction by tier-threshold calibration.

Under schema-constrained decoding the tier is the argmax over the three tier tokens at
output step 7 (after the 6 forced opening tokens). One forward pass per window over
prompt + those 6 tokens gives the three tier logits; the calibrated answer adds a bias
delta to the routine logit. delta is chosen per arm on the DS1 validation windows (the
147 recipe-r3 validation windows) and applied unchanged to the Deviation 16 DS2 set.

  collect   logits for A-compact and MEA r3 seeds 101/202/303 on val + ds2v2 (GPU,
            resumable, atomic save per batch) -> results/p1_item7_calib_logits.json
  analyse   delta = 0 check against the saved greedy tiers; delta selection on val;
            calibrated DS2 results -> results/p1_item7_calib.json (CPU)

Run (from repo root):
    python p1_item7_calib.py collect
    python p1_item7_calib.py analyse
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from p1_io import save_json_atomic

LOGITS = Path("results/p1_item7_calib_logits.json")
OUT = Path("results/p1_item7_calib.json")
ARMS = ("MEA:r3_seed101", "MEA:r3_seed202", "MEA:r3_seed303", "A-compact")
TIERS = ("routine", "priority", "urgent")
GRID = np.arange(-4.0, 8.0001, 0.25)
BATCH = 8


def collect():
    from p1_item7_common import replay_split
    from p1_item7_eval import load_windows
    from p1_pilot_multi_event import scaffold_parts
    from reasoning.adapter_training import _append_target_for_teacher_forcing
    from reasoning.constrained_json import TokenTable
    from reasoning.model_loader import load_model
    from reasoning.multi_event_adapter import MultiEventVirtualAdapter, compose_multi_event_inputs
    from reasoning.prompt_template_family import _payload
    from reasoning.virtual_adapter import _render_prompt_with_placeholder, _tokenize_and_remove_placeholder

    splits = {"val": (replay_split("ds1"), load_windows("val", replay_split("ds1"))),
              "ds2v2": (replay_split("ds2"), load_windows("ds2v2", replay_split("ds2")))}
    # validation windows for recipe r3 (147), not the r2 set load_windows("val") returns
    from p1_item7_train import RECIPES, build_windows
    rc = RECIPES["r3"]
    _, v3 = build_windows(splits["val"][0], rc["val_per_cell"], rc["val_max_per_record"], rc["ns"])
    splits["val"] = (splits["val"][0], [{"set": "val", "n": w["n"], "start": w["start"], "reference": w["tier"]} for w in v3])

    state = json.loads(LOGITS.read_text(encoding="utf-8")) if LOGITS.exists() else {"design": __doc__, "rows": []}
    done = {(x["split"], x["arm"], x["set"], x["n"], x["start"]) for x in state["rows"]}
    model, processor = load_model()
    tok = processor.tokenizer
    device = model.get_input_embeddings().weight.device
    table = TokenTable(tok, device=device)
    prefix = torch.tensor([table.canon['{"urgency_tier":"']], device=device)
    tier_ids = [table.tiers[t][0] for t in TIERS]
    scaffold = {}

    def parts(n):
        if n not in scaffold:
            pre, suf = scaffold_parts(n)
            rendered, s0, s1 = _render_prompt_with_placeholder(processor, pre, suf)
            scaffold[n] = _tokenize_and_remove_placeholder(processor, rendered, s0, s1)
        return scaffold[n]

    t0 = time.time()
    for split, (r, windows) in splits.items():
        for arm in ARMS:
            todo = [w for w in windows if (split, arm, w["set"], w["n"], w["start"]) not in done]
            if not todo:
                continue
            if arm == "A-compact":
                for w in todo:
                    pre, suf = scaffold_parts(w["n"])
                    payload = json.dumps([_payload(r["events"][i], "compact") for i in range(w["start"], w["start"] + w["n"])],
                                         indent=2)
                    ids = processor.apply_chat_template([{"role": "user", "content": pre + payload + suf}],
                                                        add_generation_prompt=True, tokenize=True, return_dict=True,
                                                        return_tensors="pt")["input_ids"].to(device)
                    with torch.no_grad():
                        lg = model(input_ids=torch.cat([ids, prefix], 1), use_cache=False, logits_to_keep=1).logits
                    state["rows"].append({"split": split, "arm": arm, **{k: w[k] for k in ("set", "n", "start", "reference")},
                                          "logits": lg[0, -1, tier_ids].float().tolist()})
                    if len(state["rows"]) % 50 == 0:
                        save_json_atomic(LOGITS, state)
            else:
                ck = torch.load(f"reasoning/checkpoints/p1_item7_mea_{arm.split(':')[1]}.pt", map_location="cpu",
                                weights_only=False)["adapter_state_dict"]
                adapter = MultiEventVirtualAdapter.for_model(model, max_events=ck["position"].shape[0]).to(device)
                adapter.load_state_dict(ck)
                adapter.eval()
                groups = [[w for w in todo if w["n"] == n] for n in sorted({w["n"] for w in todo})]
                for batch in [g[i:i + BATCH] for g in groups for i in range(0, len(g), BATCH)]:
                    with torch.no_grad():
                        es, ms, ps = [], [], []
                        for w in batch:
                            ai = compose_multi_event_inputs(model, adapter, r["vectors"][w["start"]:w["start"] + w["n"]],
                                                            *parts(w["n"]))
                            e, m, p = _append_target_for_teacher_forcing(model, ai, prefix)
                            es.append(e), ms.append(m), ps.append(p)
                        lg = model(inputs_embeds=torch.cat(es), attention_mask=torch.cat(ms),
                                   per_layer_inputs=torch.cat(ps), use_cache=False, logits_to_keep=1).logits
                    for w, row in zip(batch, lg[:, -1, :][:, tier_ids].float().tolist()):
                        state["rows"].append({"split": split, "arm": arm,
                                              **{k: w[k] for k in ("set", "n", "start", "reference")}, "logits": row})
                    save_json_atomic(LOGITS, state)
                del adapter
                torch.cuda.empty_cache()
            save_json_atomic(LOGITS, state)
            print(f"[{split} {arm}] done, elapsed {time.time() - t0:.0f}s", flush=True)
    save_json_atomic(LOGITS, state)


def answer(logits, delta):
    z = np.asarray(logits, dtype=float).copy()
    z[0] += delta
    return TIERS[int(np.argmax(z))]


def analyse():
    import p1_item7_tier1 as t1
    from p1_item7_common import replay_split
    rows = json.loads(LOGITS.read_text(encoding="utf-8"))["rows"]
    rec = np.asarray(replay_split("ds2")["record_ids"])
    out = {"design": __doc__}

    # delta = 0 must reproduce the saved greedy tiers
    saved = {(x["arm"], x["set"], x["n"], x["start"]): x["tier"]
             for x in json.loads(Path("results/p1_item7_eval_ds2v2.json").read_text(encoding="utf-8"))["rows"]}
    ds2 = [x for x in rows if x["split"] == "ds2v2"]
    agree = [answer(x["logits"], 0) == saved.get((x["arm"], x["set"], x["n"], x["start"])) for x in ds2]
    out["delta0_matches_greedy"] = {"n": len(agree), "agree": int(sum(agree))}

    def metrics(xs, delta):
        ans = [(x["reference"], answer(x["logits"], delta)) for x in xs]
        rout = [a for r, a in ans if r == "routine"]
        return t1.bal(ans), (float(np.mean([a != "routine" for a in rout])) if rout else float("nan"))

    val = [x for x in rows if x["split"] == "val"]
    deltas = {}
    for arm_group in (("MEA:r3_seed101", "MEA:r3_seed202", "MEA:r3_seed303"), ("A-compact",)):
        xs = [x for x in val if x["arm"] in arm_group]
        curve = [(float(d), *metrics(xs, d)) for d in GRID]
        best = max(curve, key=lambda c: (round(c[1], 6), c[0]))  # max bal-acc, ties -> largest delta
        b0 = metrics(xs, 0.0)[0]
        ok = [c for c in curve if c[0] >= 0 and c[2] <= 0.05 and c[1] >= b0 - 0.02]
        constrained = min(ok, key=lambda c: c[0]) if ok else None
        name = "MEA" if len(arm_group) > 1 else "A-compact"
        deltas[name] = {"primary_rule": best[0], "constrained_rule": constrained[0] if constrained else None,
                        "val_at_0": {"bal_acc": b0, "false_alarm": metrics(xs, 0.0)[1]},
                        "val_at_primary": {"bal_acc": best[1], "false_alarm": best[2]},
                        "val_curve": curve}
    out["deltas"] = deltas

    for rule in ("primary_rule", "constrained_rule"):
        dm, da = deltas["MEA"][rule], deltas["A-compact"][rule]
        if dm is None or da is None:
            out[rule] = "no delta satisfies the rule on validation"
            continue
        calibrated = []
        for x in ds2:
            d = dm if x["arm"].startswith("MEA") else da
            calibrated.append({**x, "tier": answer(x["logits"], d), "correct": answer(x["logits"], d) == x["reference"]})
        wbn = t1.windows_by_n(calibrated)
        mea = [a for a in ARMS if a.startswith("MEA")]
        res = {"delta_MEA": dm, "delta_A_compact": da, "by_n": {}}
        for n, wins in sorted(wbn.items()):
            starts = [s for s, w in wins.items() if all(a in w["arms"] for a in ARMS)]
            ba = {a: t1.bal([(wins[s]["reference"], wins[s]["arms"][a]) for s in starts]) for a in ARMS}
            d = {"A-compact": ba["A-compact"], "MEA_mean": float(np.mean([ba[m] for m in mea]))}
            if n >= 5:
                d["primary"] = t1.cluster_ci(wins, mea, {s: int(rec[s]) for s in starts})
            res["by_n"][str(n)] = d
        nat = {}
        for n in sorted({x["n"] for x in calibrated if x["set"] == "natural"}):
            nat[str(n)] = {}
            for a in ARMS:
                xs = [x for x in calibrated if x["set"] == "natural" and x["n"] == n and x["arm"] == a]
                routine = [x for x in xs if x["reference"] == "routine"]
                nat[str(n)][a] = {"accuracy": float(np.mean([x["correct"] for x in xs])),
                                  "false_alarm": float(np.mean([x["tier"] != "routine" for x in routine]))}
        res["natural"] = nat
        allr = [x for x in calibrated if x["reference"] == "routine"]
        res["false_alarm_all_routine"] = {a: float(np.mean([x["tier"] != "routine" for x in allr if x["arm"] == a])) for a in ARMS}
        out[rule] = res
    save_json_atomic(OUT, out)
    show(out)


def show(out):
    print("delta=0 matches greedy:", out["delta0_matches_greedy"])
    for k, v in out["deltas"].items():
        print(f"{k}: primary delta {v['primary_rule']} constrained {v['constrained_rule']} | val at 0 {v['val_at_0']} | val at primary {v['val_at_primary']}")
    for rule in ("primary_rule", "constrained_rule"):
        res = out[rule]
        if isinstance(res, str):
            print(rule, res)
            continue
        print(f"== {rule}: delta MEA {res['delta_MEA']} A-compact {res['delta_A_compact']}")
        for n, d in res["by_n"].items():
            line = f"  N={n:>2} text {d['A-compact']:.3f} MEA {d['MEA_mean']:.3f}"
            if "primary" in d:
                p = d["primary"]
                line += f" diff {p['difference']:+.3f} [{p['cluster_ci95'][0]:+.3f}, {p['cluster_ci95'][1]:+.3f}] NI={p['non_inferior']} sup={p['superior']}"
            print(line)
        print("  false alarms, all routine windows:", {a: round(v, 3) for a, v in res["false_alarm_all_routine"].items()})
        print("  natural:", {n: {a.replace('MEA:r3_', ''): (round(v['accuracy'], 2), round(v['false_alarm'], 3)) for a, v in d.items()} for n, d in res["natural"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("collect", "analyse"))
    collect() if ap.parse_args().mode == "collect" else analyse()
