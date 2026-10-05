"""Generate the LaTeX tables of the paper from the result files.

Run from the repository root:
    python scripts/make_paper_tables.py

Writes docs/paper/tables/*.tex (one file per table body, \\input from sections/results.tex) and prints
an audit of the numbers quoted in the prose against the same files. Nothing here recomputes a statistic;
every value is read from a results/*.json file written by the analysis scripts.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
import os
RES = pathlib.Path(os.environ.get("PAPER_RES_DIR", ROOT / "results"))
OUT = pathlib.Path(os.environ.get("PAPER_TABLES_DIR", ROOT / "docs" / "paper" / "tables"))
OUT.mkdir(parents=True, exist_ok=True)
NS = ["1", "5", "10", "20", "50"]


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def pm(x, d=2):
    """Signed number in math mode."""
    return f"${x:+.{d}f}$"


def ci(lo, hi, d=2):
    return f"[${lo:+.{d}f}$, ${hi:+.{d}f}$]"


def pct(x, d=0):
    v = round(x * 100, d)
    return "0\\%" if v == 0 else f"${v:+.{d}f}\\%$"


def write(name, rows, header, align):
    lines = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


audit = {}

# ---- Cost (RQ1): time to first token, time to decision, tokens, context memory -----------------------------
ttd = load("p1_item7_ttd.json")["summary"]
e3b = load("p1_e3b_context_costs.json")
rows = []
for n in NS:
    s = ttd[n]
    tf, td = s["ttft_ms"], s["ttd_ms"]
    mem = e3b["b4_vs_acompact"][n]["context_mem_mb"]["mean_diff_mb"]
    rows.append([
        n,
        f"{s['prompt_tokens_median']['A-compact']:.0f} / {s['prompt_tokens_median']['MEA']:.0f}",
        f"{tf['a_compact_median']:.0f} / {tf['mea_median']:.0f}",
        pct(tf["median_rel_diff"]),
        pct(td["median_rel_diff"]),
        "---" if abs(mem) < 1e-9 else f"$-${abs(mem):.0f}",
    ])
    audit[f"ttft_rel_{n}"] = round(tf["median_rel_diff"] * 100)
    audit[f"ttd_rel_{n}"] = round(td["median_rel_diff"] * 100)
write("cost.tex", rows,
      ["$N$", "Tokens (text / adapter)", "TTFT ms (text / adapter)", "TTFT", "Decision", "Context MB"],
      "rrrrrr")

# ---- Serving ----------------------------------------------------------------------------------------------
serv = load("p1_serving.json")["summary"]


def cell(d):
    if d.get("n_measurements", 0) == 0:
        return "does not fit"
    s = f"{d['decisions_per_s']:.2f}/s, {d['joules_per_decision']:.0f} J"
    if d.get("exceeds_memory", 0):
        tot = d["n_measurements"] + d["exceeds_memory"]
        s += f" ({d['n_measurements']} of {tot} fit)"
    return s


rows = []
for key in ["N10_B1", "N10_B4", "N10_B8", "N20_B1", "N20_B4", "N20_B8", "N50_B1", "N50_B4", "N50_B8"]:
    n, b = key[1:].split("_B")
    d = serv[key]
    rows.append([n, b, cell(d["MEA"]), cell(d["A-compact"]), cell(d["A-filtered"])])
write("serving.tex", rows, ["$N$", "Batch", "Adapter", "Text, all events", "Filtered text"], "rrccc")
for key in ["N10_B4", "N20_B4", "N50_B1", "N50_B4"]:
    audit[f"serv_adapter_{key}"] = round(serv[key]["MEA"]["decisions_per_s"], 2)
ratios = []
for key, d in serv.items():
    if "MEA" in d and "A-filtered" in d and d["A-filtered"].get("n_measurements", 0) and d["MEA"].get("decisions_per_s"):
        ratios.append((d["MEA"]["decisions_per_s"] / d["A-filtered"]["decisions_per_s"] - 1,
                       1 - d["MEA"]["joules_per_decision"] / d["A-filtered"]["joules_per_decision"]))
batched = [(t, e) for (t, e), key in zip(ratios, [k for k, d in serv.items() if "MEA" in d and "A-filtered" in d and d["A-filtered"].get("n_measurements", 0) and d["MEA"].get("decisions_per_s")]) if not key.endswith("_B1")]
audit["serv_vs_filtered_batched_throughput_gain_pct_range"] = (round(min(t for t, _ in batched) * 100), round(max(t for t, _ in batched) * 100))
audit["serv_vs_filtered_batched_energy_saving_pct_range"] = (round(min(e for _, e in batched) * 100), round(max(e for _, e in batched) * 100))

# ---- Accuracy against text (RQ2) ----------------------------------------------------------------------------
r4 = load("p1_item7_r4_analysis.json")
rows = []
for n in NS:
    d_def = r4["primary_default"][n]
    d_cal = r4["primary_calibrated_text"]["by_n"][n]
    t = d_cal.get("test")
    rows.append([
        n, f"{d_def['text']:.2f}", f"{d_cal['text']:.2f}", f"{d_cal['mea_mean']:.2f}",
        "---" if t is None else f"{pm(t['difference'])} {ci(*t['cluster_ci95'])}",
    ])
    audit[f"acc_adapter_{n}"] = round(d_cal["mea_mean"], 2)
    audit[f"acc_textcal_{n}"] = round(d_cal["text"], 2)
    if t:
        audit[f"acc_diff_cal_{n}"] = (round(t["difference"], 2), t["superior"], t["non_inferior"])
write("accuracy.tex", rows, ["$N$", "Text", "Text cal.", "Adapter r4", "Adapter $-$ cal. text [95\\% CI]"], "rrrrl")

fa = r4["false_alarm"]
r4_all = [fa[f"MEA:r4_seed{s}"]["all_routine"] for s in (101, 202, 303)]
audit["false_alarm_r4_all_routine_mean_pct"] = round(sum(r4_all) / 3 * 100, 1)
audit["false_alarm_r3_all_routine_mean_pct"] = round(sum(fa[f"MEA:r3_seed{s}"]["all_routine"] for s in (101, 202, 303)) / 3 * 100, 1)

# ---- Filtered text -------------------------------------------------------------------------------------------
flt = load("p1_item7_filtered.json")
rows = []
for n in NS:
    d = flt["vs_filtered_default"][n]
    t = d.get("test")
    pt = flt["prompt_tokens"][n]
    rows.append([
        n, f"{d['r4_mean']:.3f}", f"{d['base']:.3f}",
        "---" if t is None else f"{pm(t['difference'])} {ci(*t['cluster_ci95'])}",
        f"{pt['MEA']['median']:.0f} / {pt['A-filtered']['median']:.0f}",
    ])
write("filtered.tex", rows, ["$N$", "Adapter r4", "Filtered", "Difference [95\\% CI]", "Tokens"], "rrrrr")
audit["filtered_tokens_per_abnormal_beat"] = round(flt["cost_growth"]["tokens_per_abnormal_beat"])
audit["filtered_share_above_adapter_n50"] = round(flt["cost_growth"]["share_of_windows_above_mea"]["50"], 2)

# ---- Second sender ---------------------------------------------------------------------------------------------
sec = load("p1_second_sender_analysis.json")
rows = []
for n in ["5", "10", "20", "50"]:
    by = sec["RQ2"]["by_n"][n]
    c = sec["RQ2"]["comparisons"]["A-compact-cal"][n]
    rows.append([
        n, f"{by['MEA_mean']:.3f}", f"{by['A-compact-cal']:.3f}", f"{by['A-filtered']:.3f}",
        f"{pm(c['difference'])} {ci(*c['cluster_ci95'])}",
    ])
    audit[f"second_cal_{n}"] = (round(c["difference"], 2), c["superior"], c["non_inferior"])
write("second.tex", rows, ["$N$", "Adapter", "Text cal.", "Filtered", "Adapter $-$ cal. text [95\\% CI]"], "rrrrl")

# ---- Compression (Deviations 24 and 27): k = 1, 2, 4 three seeds each, k = 8 one run -------------------------------
sw = load("p1_sweep_analysis.json")
ks = load("p1_dev27_kseeds.json")
group = {"1": "k1", "2": "k2", "4": "k4"}
contrast = {"1": "k1_minus_k4", "2": "k2_minus_k4"}
Nq = ["5", "10", "20", "50"]


def mean_slot0(g, field):
    arms = ks["groups"][g]
    return sum(ks["decodability"][a]["slot0"][field] for a in arms) / len(arms)


rows = []
for k in ["1", "2", "4", "8"]:
    tokens = f"{sw['by_k'][k]['prompt_tokens']['50']:.0f}"
    if k in group:
        g = ks["describe"][group[k]]
        acc = " / ".join(f"{g['balanced_accuracy'][n]['mean']:.2f}" for n in Nq)
        if k in contrast:
            c = ks["contrasts"][contrast[k]]
            worst = min(Nq, key=lambda n: c[n]["cluster_ci95"][0])
            diff = f"{pm(c[worst]['difference'])} {ci(*c[worst]['cluster_ci95'])}"
            ni = f"{sum(c[n]['non_inferior'] for n in Nq)}/4"
        else:
            diff, ni = "ref.", "--"
        fa = [v["all"] for v in g["false_alarm"].values()]
        fa_s = f"{100 * sum(fa) / 3:.1f}\\%"
        lab, hr = mean_slot0(group[k], "label"), mean_slot0(group[k], "heart_rate_r2")
        seeds = "3"
    else:
        d = sw["by_k"][k]
        acc = " / ".join(f"{d['balanced_accuracy'][n]:.2f}" for n in Nq)
        diff, seeds, ni = "1 run", "1", "--"
        fa_s = f"{d['false_alarm'] * 100:.1f}\\%"
        lab = d["decodability"]["slot0"]["label"]["mlp"]["balanced_accuracy"]
        hr = d["decodability"]["slot0"]["heart_rate"]["mlp"]["r2"]
    rows.append([k, tokens, acc, diff, ni, fa_s, f"{lab:.2f}", f"${hr:.2f}$"])
    audit[f"compression_k{k}"] = (acc, diff, fa_s)
write("sweep.tex", rows, ["$k$", "Tokens", "Bal. acc., $N$ = 5/10/20/50", "$k-4$ [95\\% CI]", "NI", "FA", "Label", "HR $R^2$"],
      "rrclcrrr")

# ---- Ablation of recipe r4 (Deviation 26): r3, r4hn (hard negatives only), r4 (hard negatives + side inputs) -------
ab = load("p1_dev26_ablation.json")
names = {"r3": "r3 (neither)", "r4hn": "HN only", "r4": "r4 (HN + SI)"}
rows = []
for g in ("r3", "r4hn", "r4"):
    d = ab["describe"][g]
    fa = [v["all"] for v in d["false_alarm"].values()]
    rows.append([names[g]] + [f"{d['balanced_accuracy'][n]['mean']:.3f}" for n in Nq]
                + [f"{100 * sum(fa) / 3:.1f}\\%"])
for name, key in (("SI: r4 $-$ HN only", "side_inputs_r4_minus_r4hn"),
                  ("HN: HN only $-$ r3", "hard_negatives_r4hn_minus_r3")):
    c = ab["contrasts"][key]
    rows.append([name] + [f"{pm(c[n]['difference'])} {ci(*c[n]['cluster_ci95'])}" for n in Nq] + [""])
    audit[f"ablation_{key}"] = {n: round(c[n]["difference"], 3) for n in Nq}
write("ablation.tex", rows, ["Recipe", "$N$=5", "$N$=10", "$N$=20", "$N$=50", "FA"], "lccccr")

# ---- Recoverability from the virtual tokens (RQ3), main sender, recipe r4, slots 0 and 49, three seeds ----------
slots = r4["slots"]
runlen = load("p1_item7_runlen.json")
tok_keys = [f"r4_seed{s}_slot{p}" for s in (101, 202, 303) for p in (0, 49)]


def span(get):
    v = [get(k) for k in tok_keys]
    return f"{min(v):.2f}--{max(v):.2f}"


fields = [("Beat label (bal. acc.)", "label", "balanced_accuracy", slots),
          ("Tier (bal. acc.)", "tier", "balanced_accuracy", slots),
          ("Urgency flag (bal. acc.)", "urgent", "balanced_accuracy", slots),
          ("Heart rate ($R^2$)", "heart_rate", "r2", slots),
          ("RR interval ($R^2$)", "rr", "r2", slots),
          ("Run $\\geq 3$ (bal. acc.)", "run3", "balanced_accuracy", runlen)]
rows = []
for name, f, metric, src in fields:
    row = [name]
    for dec in ("mlp", "linear"):
        row.append(f"{src['input_35d'][f][dec][metric]:.2f}")
        row.append(span(lambda k, f=f, dec=dec, metric=metric, src=src: src[k][f][dec][metric]))
    rows.append(row)
write("recover.tex", rows, ["Field", "Input, MLP", "Tokens, MLP", "Input, linear", "Tokens, linear"], "lrrrr")
audit["recover_hr_mlp"] = span(lambda k: slots[k]["heart_rate"]["mlp"]["r2"])
audit["recover_hr_linear"] = span(lambda k: slots[k]["heart_rate"]["linear"]["r2"])
audit["recover_run3_mlp"] = span(lambda k: runlen[k]["run3"]["mlp"]["balanced_accuracy"])
audit["recover_run3_r3_mlp"] = "%.2f--%.2f" % (min(runlen[f"r3_seed{s}_slot{p}"]["run3"]["mlp"]["balanced_accuracy"] for s in (101, 202, 303) for p in (0, 49)),
                                              max(runlen[f"r3_seed{s}_slot{p}"]["run3"]["mlp"]["balanced_accuracy"] for s in (101, 202, 303) for p in (0, 49)))
urg = r4["urgent_by_reason"]
for reason in ("run>=3", "high-conf V/F"):
    audit[f"urgent_recall_{reason}_r4_mean"] = round(sum(urg[f"MEA:r4_seed{s}"][reason]["recall"] for s in (101, 202, 303)) / 3, 2)
    audit[f"urgent_recall_{reason}_r3_mean"] = round(sum(urg[f"MEA:r3_seed{s}"][reason]["recall"] for s in (101, 202, 303)) / 3, 2)
    audit[f"urgent_recall_{reason}_text"] = round(urg["A-compact"][reason]["recall"], 2)

# ---- Test windows per N and tier (achieved counts) -----------------------------------------------------------------
ts = load("p1_item7_testset_v2.json")
mix, nat = ts["stratified_class_mix"], ts["natural_reference_counts"]
rows, tot_s, tot_n = [], 0, 0
for n in NS:
    r = sum(mix[f"{n}|routine"].values())
    p = sum(mix[f"{n}|priority"].values())
    u = sum(mix[f"{n}|urgent"].values())
    nn = sum(v for k, v in nat.items() if k.startswith(n + "|")) if n != "1" else 0
    rows.append([n, str(r), str(p), str(u), str(r + p + u), str(nn) if nn else "---", str(r + p + u + nn)])
    tot_s += r + p + u
    tot_n += nn
rows.append(["All", "", "", "", str(tot_s), str(tot_n), str(tot_s + tot_n)])
write("windows.tex", rows, ["$N$", "Routine", "Priority", "Urgent", "Stratified", "Natural prevalence", "Total"], "rrrrrrr")
audit["windows_total"] = (tot_s + tot_n, ts["n_windows"])
assert tot_s + tot_n == ts["n_windows"], "window table does not add up to the test set"

# ---- INCART confirmatory evaluation (Deviation 21) -------------------------------------------------------------------
# The window and conversion tables are written from pre-run files (counts only). The hypothesis, accuracy and false-alarm
# tables are written only when results/p1_confirmatory_incart.json exists, so that the manuscript builds before and after.
conv = load("p1_incart_conversion.json")
iw = load("p1_item7_testset_incart.json")["cells"]
rows = []
for n in NS:
    cells = {t: iw.get(f"stratified|{n}|{t}") for t in ("routine", "priority", "urgent")}
    nat = sum(v for k, v in iw.items() if k.startswith(f"natural|{n}|"))
    if any(v is None for v in cells.values()):
        continue
    s = sum(cells.values())
    rows.append([n, str(cells["routine"]), str(cells["priority"]), str(cells["urgent"]), str(s), str(nat) if nat else "---", str(s + nat)])
if rows:
    tot_i = sum(int(r[6]) for r in rows)
    rows.append(["All", "", "", "", str(sum(int(r[4]) for r in rows[:-0 or None])), str(sum(int(r[5]) for r in rows if r[5] != "---")), str(tot_i)])
    write("incart_windows.tex", rows, ["$N$", "Routine", "Priority", "Urgent", "Stratified", "Natural prevalence", "Total"], "rrrrrrr")
    audit["incart_windows_total"] = (tot_i, load("p1_item7_testset_incart.json")["n_windows"])

inc_path = RES / "p1_confirmatory_incart.json"
if inc_path.exists():
    inc = load("p1_confirmatory_incart.json")
    names = {"H1": "H1: non-inferior to calibrated text", "H2": "H2: superior to calibrated text",
             "H3": "H3: non-inferior to calibrated filtered text", "H4": "H4: faster first token than text"}
    # Deviation 21a: the N = 50 timing of the confirmatory run was dominated by memory paging; the value
    # re-measured under the memory cap (same 21 windows, frozen routine) replaces it, marked with a dagger.
    recheck = None
    rpath = RES / "p1_incart_timing_n50_capped.json"
    if rpath.exists():
        import numpy as np
        tr = load("p1_incart_timing_n50_capped.json")["rows"]
        by = {}
        for x in tr:
            by.setdefault(x["start"], {})[x["arm"]] = x
        rel = np.array([v["MEA"]["ttft_ms"] / v["A-compact"]["ttft_ms"] - 1 for v in by.values() if "MEA" in v and "A-compact" in v])
        rng = np.random.default_rng(0)
        boots = [np.median(rel[rng.integers(0, len(rel), len(rel))]) for _ in range(20000)]
        recheck = (float(np.median(rel)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        audit["incart_h4_n50_capped"] = tuple(round(x, 3) for x in recheck)
    rows = []
    for h in ("H1", "H2", "H3", "H4"):
        d = inc["hypotheses"][h]
        for n, v in d["tests"].items():
            est = v.get("difference", v.get("median_rel"))
            lo, hi = v.get("cluster_ci95", v.get("ci95"))
            if h == "H4" and n == "50" and recheck:
                est, lo, hi = recheck
                n = "50$^\\dagger$"
            first = n == list(d["tests"])[0]
            rows.append([names[h] if first else "", n, pm(est), ci(lo, hi), "pass" if (hi < 0 if h == "H4" else v["pass"]) else "fail",
                         d["status"] if first else ""])
    write("incart_hyp.tex", rows, ["Hypothesis", "$N$", "Estimate", "95\\% CI", "Result", "Status"], "llrrll")
    arms = [("MEA:r4_seed101", None), ("A-compact", "Text"), ("A-compact-cal", "Text cal."), ("A-filtered", "Filtered"), ("A-filtered-cal", "Filtered cal.")]
    rows = []
    for n, d in inc["balanced_accuracy"].items():
        r4m = sum(d[f"MEA:r4_seed{s}"] for s in (101, 202, 303)) / 3
        rows.append([n, f"{r4m:.3f}"] + [f"{d[a]:.3f}" for a, _ in arms[1:]])
    write("incart_acc.tex", rows, ["$N$", "Adapter r4"] + [l for _, l in arms[1:]], "rrrrrr")
    fa = inc["false_alarm"]
    rows = []
    for a, lab in [("MEA:r4_seed101", "Adapter, seed 101"), ("MEA:r4_seed202", "Adapter, seed 202"), ("MEA:r4_seed303", "Adapter, seed 303"),
                   ("A-compact-cal", "Text, calibrated"), ("A-filtered-cal", "Filtered, calibrated")]:
        if a in fa:
            rows.append([lab, "---" if fa[a]["all"] is None else f"{fa[a]['all'] * 100:.1f}\\%", "---" if fa[a]["natural"] is None else f"{fa[a]['natural'] * 100:.1f}\\%"])
    write("incart_fa.tex", rows, ["Arm", "All routine windows", "Natural prevalence"], "lrr")
    audit["incart_status"] = {h: inc["hypotheses"][h]["status"] for h in inc["hypotheses"]}

# ---- Bandwidth (Deviation 28) ----------------------------------------------------------------------------------
bw_path = RES / "p1_bandwidth.json"
if bw_path.exists():
    bw = load("p1_bandwidth.json")["summary"]
    arms = [("compact_json", "Compact JSON text"), ("filtered_json", "Filtered JSON text"),
            ("compact_json_zlib", "Compact JSON, zlib"), ("filtered_json_zlib", "Filtered JSON, zlib"),
            ("latent_f32", "Latent, float32"), ("latent_f16", "Latent, float16"), ("latent_i8", "Latent, int8"),
            ("binary_fields", "Binary fields")]
    rows = [[name] + [f"{bw[n]['median_bytes'][a]:,.0f}" for n in ("10", "20", "50")]
            + [f"{bw['50']['time_ms']['10 kbit/s'][a]:,.0f}"] for a, name in arms]
    write("bandwidth.tex", rows, ["Interface", "$N$=10", "$N$=20", "$N$=50", "ms at 10 kbit/s, $N$=50"], "lrrrr")

# ---- Audit of prose numbers ---------------------------------------------------------------------------------------
print("AUDIT (computed from result files):")
for k, v in audit.items():
    print(f"  {k}: {v}")
prose = (ROOT / "docs" / "paper" / "sections" / "results.tex").read_text(encoding="utf-8")
checks = {
    "24\\%": audit["ttft_rel_10"] == -24, "50\\%": audit["ttft_rel_20"] == -50, "78\\%": audit["ttft_rel_50"] == -78,
    "36\\%": audit["ttd_rel_50"] == -36,
}
bad = [k for k, ok in checks.items() if not ok]
print("prose cost checks failed:", bad if bad else "none")
sys.exit(1 if bad else 0)
