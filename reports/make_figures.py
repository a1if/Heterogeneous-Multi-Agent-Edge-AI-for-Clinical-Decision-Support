"""Figures for the Phase 1 technical report, all read from committed results files.

Run (from repo root):
    python reports/make_figures.py
"""
import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))  # repo root
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # adapter, text, calibrated text (validated palette)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False})
load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
NS = [1, 5, 10, 20, 50]


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_pipeline():
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 3.8))
    rows = [("Dissertation (single event)", ["1 ECG beat", "CNN-LSTM\n(32-d vector)", "Adapter\n1 event → 4 tokens", "Gemma 4 E4B\n(frozen)", "Tier for\nthat beat"]),
            ("Phase 1 (multi-event)", ["N beats\n(1–50)", "CNN-LSTM +\nRR branch", "Multi-event adapter\nN events → N×4 tokens", "Gemma 4 E4B\n(frozen, constrained\ndecoding)", "Most urgent tier\n+ which beat"])]
    for ax, (title, boxes) in zip(axes, rows):
        ax.set_xlim(0, 10); ax.set_ylim(0, 1.6); ax.axis("off")
        ax.text(0, 1.45, title, fontsize=10, fontweight="bold", color=INK)
        for i, label in enumerate(boxes):
            x = 0.05 + i * 2.0
            face = "#eaf2fc" if i in (2,) else "#f4f3ef"
            edge = BLUE if i == 2 else INK2
            ax.add_patch(FancyBboxPatch((x, 0.15), 1.7, 1.0, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        facecolor=face, edgecolor=edge, linewidth=1.2))
            ax.text(x + 0.85, 0.65, label, ha="center", va="center", fontsize=8, color=INK)
            if i < len(boxes) - 1:
                ax.annotate("", xy=(x + 1.95, 0.65), xytext=(x + 1.72, 0.65),
                            arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    fig.tight_layout()
    save(fig, "fig1_pipeline.png")


def fig_latency():
    S = load("results/p1_item7_ttd.json")["summary"]
    ns = [int(n) for n in S]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=False)
    for ax, key, title in ((axes[0], "ttft_ms", "Time to first token (prefill)"), (axes[1], "ttd_ms", "Time to the urgency decision")):
        a = [S[str(n)][key]["a_compact_median"] for n in ns]
        b = [S[str(n)][key]["mea_median"] for n in ns]
        ax.plot(ns, a, color=ORANGE, lw=2, marker="o", ms=6, label="Text (A-compact)")
        ax.plot(ns, b, color=BLUE, lw=2, marker="o", ms=6, label="Latent adapter (MEA)")
        rel = S["50"][key]["median_rel_diff"] * 100
        ax.text(0.97, 0.45, f"adapter {rel:+.0f}%\nat N = 50", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
        ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns)
        ax.set_xlabel("Events per prompt (N)"); ax.set_ylabel("Median ms (batch 1)"); ax.set_title(title, fontsize=9.5, color=INK)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "fig2_latency.png")


R4 = "results/p1_item7_r4_analysis.json"
R4_ARMS = ("MEA:r4_seed101", "MEA:r4_seed202", "MEA:r4_seed303")


def fig_accuracy():
    a = load(R4)
    d, c, r3 = a["primary_default"], a["primary_calibrated_text"]["by_n"], a["r3_default"]
    text = [d[str(n)]["text"] for n in NS]
    ctext = [c[str(n)]["text"] for n in NS]
    mean = [d[str(n)]["mea_mean"] for n in NS]
    lo = [min(d[str(n)]["per_seed"].values()) for n in NS]
    hi = [max(d[str(n)]["per_seed"].values()) for n in NS]
    old = [r3[str(n)]["mea_mean"] for n in NS]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.fill_between(NS, lo, hi, color=BLUE, alpha=0.14, linewidth=0)
    ax.plot(NS, mean, color=BLUE, lw=2, marker="o", ms=6, label="Latent adapter, recipe r4 (final), 3-seed mean; band = seed range")
    ax.plot(NS, old, color=INK2, lw=1.3, ls=":", marker="o", ms=4, mfc="white", label="Latent adapter, recipe r3 (previous iteration)")
    ax.plot(NS, text, color=ORANGE, lw=2, marker="o", ms=6, label="Text (A-compact), default decoding")
    ax.plot(NS, ctext, color=AQUA, lw=2, ls="--", marker="s", ms=6, label="Text, calibrated threshold (Deviation 17)")
    for n, v in zip(NS, text):
        ax.text(n, v - 0.045, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    for n, v in zip(NS, mean):
        ax.text(n, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS); ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("Events per prompt (N)"); ax.set_ylabel("Balanced accuracy (DS2 test)")
    ax.legend(loc="upper center", fontsize=7.5, ncol=2, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    save(fig, "fig3_accuracy_vs_n.png")


def fig_forest():
    a = load(R4)
    r3cal = load("results/p1_item7_calib.json")["primary_rule"]["by_n"]
    ns = [5, 10, 20, 50]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    for k, n in enumerate(ns):
        series = ((a["primary_default"][str(n)]["test"], ORANGE, "r4 vs default text", True),
                  (a["primary_calibrated_text"]["by_n"][str(n)]["test"], AQUA, "r4 vs calibrated text", True),
                  (r3cal[str(n)]["primary"], INK2, "r3 vs calibrated text (previous iteration)", False))
        for off, (src, col, lab, filled) in zip((-0.24, 0.0, 0.24), series):
            y = k + off
            d, (l, h) = src["difference"], src["cluster_ci95"]
            ax.plot([l, h], [y, y], color=col, lw=2.2 if filled else 1.4, solid_capstyle="round")
            ax.plot(d, y, "o", color=col if filled else "white", ms=6 if filled else 5, markeredgecolor=col if not filled else "white",
                    markeredgewidth=1.2, label=lab if k == 0 else None)
            ax.text(h + 0.01, y, f"{d:+.2f} [{l:+.2f}, {h:+.2f}]", va="center", fontsize=7, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(-0.05, color="#d03b3b", lw=1, ls=":")
    ax.text(-0.055, -0.75, "NI margin −0.05", color="#d03b3b", fontsize=7.5, ha="right", va="center")
    ax.set_yticks(range(len(ns))); ax.set_yticklabels([f"N = {n}" for n in ns]); ax.set_ylim(len(ns) - 0.5, -1.0)
    ax.set_xlim(-0.2, 0.7); ax.set_xlabel("Adapter minus text, balanced accuracy (record-level 95% CI)")
    ax.legend(loc="lower center", fontsize=7.5, ncol=3, bbox_to_anchor=(0.5, -0.38))
    fig.tight_layout()
    save(fig, "fig4_forest.png")


def fig_why_text_fails():
    from p1_item7_common import replay_split
    from reasoning.training_targets import most_urgent_index
    ev = replay_split("ds2")["events"]
    rows = [x for x in load("results/p1_item7_eval_ds2v2_r4.json")["rows"] if x["arm"] in ("A-compact",) + R4_ARMS]
    pos, why = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for x in rows:
        if x["set"] != "stratified" or x["n"] < 10 or x["reference"] == "routine":
            continue
        w = ev[x["start"]:x["start"] + x["n"]]
        i = most_urgent_index(w)
        third = "First third" if i < x["n"] / 3 else ("Middle third" if i < 2 * x["n"] / 3 else "Last third")
        arm = "text" if x["arm"] == "A-compact" else "adapter"
        pos[arm][third].append(x["correct"])
        if x["reference"] == "urgent":
            reason = "Run of ≥3 abnormal" if w[i]["clinical_flags"]["consecutive_abnormal_beats"] >= 3 else "High-confidence V/F (>0.85)"
            why[arm][reason].append(x["correct"])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    for ax, data, cats, title in ((axes[0], why, ["High-confidence V/F (>0.85)", "Run of ≥3 abnormal"], "Urgent recall by reason (N ≥ 10)"),
                                  (axes[1], pos, ["First third", "Middle third", "Last third"], "Accuracy by position of the key beat (N ≥ 10)")):
        x = np.arange(len(cats))
        for off, arm, col, lab in ((-0.2, "text", ORANGE, "Text (A-compact)"), (0.2, "adapter", BLUE, "Latent adapter")):
            vals = [np.mean(data[arm][c]) for c in cats]
            ax.bar(x + off, vals, width=0.38, color=col, label=lab, edgecolor="white", linewidth=2)
            for xi, v in zip(x + off, vals):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8); ax.set_ylim(0, 1); ax.set_title(title, fontsize=9.5, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Proportion correct"); axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig5_why_text_fails.png")


def fig_confusion():
    rows = [x for x in load("results/p1_item7_eval_ds2v2_r4.json")["rows"] if x["set"] == "stratified" and x["n"] == 50]
    T = ["routine", "priority", "urgent"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, arms, title in ((axes[0], ("A-compact",), "Text (A-compact)"), (axes[1], R4_ARMS, "Latent adapter r4 (3 seeds pooled)")):
        m = np.zeros((3, 3))
        for x in rows:
            if x["arm"] in arms:
                m[T.index(x["reference"]), T.index(x["tier"])] += 1
        frac = m / m.sum(1, keepdims=True)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{frac[i, j]:.2f}\n({int(m[i, j])})", ha="center", va="center", fontsize=8,
                        color="white" if frac[i, j] > 0.55 else INK)
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(T); ax.set_yticklabels(T)
        ax.set_xlabel("Answered"); ax.set_ylabel("Reference tier"); ax.set_title(title, fontsize=9.5, color=INK); ax.grid(False)
    fig.suptitle("Confusion at N = 50 (row-normalised)", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, "fig6_confusion_n50.png")


def fig_training():
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for seed, col, ls in ((101, BLUE, "-"), (202, ORANGE, "--"), (303, AQUA, ":")):
        s = load(f"results/p1_item7_train_r4_seed{seed}.json")
        h = [x for x in s["history"] if "val" in x]
        ax.plot([x["update"] for x in h], [x["val"]["balanced_accuracy"] for x in h], color=col, lw=2, ls=ls,
                marker="o", ms=4, label=f"seed {seed} (best step {s['best_update']}, {s['best_metric']:.2f})")
    ax.axvline(894, color=INK2, lw=0.8, ls=":")
    ax.text(902, 0.97, "early stopping allowed from here", fontsize=7.5, color=INK2, va="top", ha="left")
    ax.set_xlabel("Training step (batch 1)"); ax.set_ylabel("Validation balanced accuracy"); ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig7_training_r4.png")


def fig_false_alarms():
    a = load(R4)["false_alarm"]
    groups = [("Text (A-compact)", ["A-compact"], ORANGE), ("Adapter r3", [f"MEA:r3_seed{s}" for s in (101, 202, 303)], INK2),
              ("Adapter r4 (final)", list(R4_ARMS), BLUE)]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x = np.arange(2)
    for k, (lab, arms, col) in enumerate(groups):
        vals = [[a[m][key] * 100 for m in arms] for key in ("all_routine", "natural_routine")]
        mean = [np.mean(v) for v in vals]
        xs = x - 0.27 + k * 0.27
        ax.bar(xs, mean, width=0.25, color=col, label=lab, edgecolor="white", linewidth=2)
        for xi, v, m in zip(xs, vals, mean):
            if len(v) > 1:
                ax.plot([xi, xi], [min(v), max(v)], color=INK, lw=1)
            ax.text(xi, max(v) + 0.4, f"{m:.1f}%", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(["All routine test windows (479)", "Natural-prevalence routine windows (202)"])
    ax.set_ylabel("Routine windows escalated (%)"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 16)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "fig9_false_alarms.png")


def fig_side_info():
    """Left: R^2 for heart rate and RR (Deviation 18). Right: run >= 3 balanced accuracy (Deviation 18b)."""
    s, rl = load(R4)["slots"], load("results/p1_item7_runlen.json")
    toks = lambda d, t, get: [get(d[f"{t}_seed{sd}_slot{sl}"]) for sd in (101, 202, 303) for sl in (0, 49)]
    panels = [
        ("R² on DS2 (≤ 0 shown as 0)", [("heart_rate", "Heart rate"), ("rr", "RR interval")],
         lambda f: ([s["input_35d"][f]["mlp"]["r2"]], toks(s, "r3", lambda d: d[f]["mlp"]["r2"]), toks(s, "r4", lambda d: d[f]["mlp"]["r2"]))),
        ("Balanced accuracy on DS2", [("run3", "Run of ≥ 3 abnormal beats")],
         lambda f: ([rl["input_35d"][f]["mlp"]["balanced_accuracy"]], toks(rl, "r3", lambda d: d[f]["mlp"]["balanced_accuracy"]),
                    toks(rl, "r4", lambda d: d[f]["mlp"]["balanced_accuracy"])))]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.9), gridspec_kw={"width_ratios": [2, 1.2]})
    labels = [("Input (32-d vector + side inputs)", INK2), ("Adapter r3 tokens", "#9ec5f4"), ("Adapter r4 tokens", BLUE)]
    for ax, (ylab, fields, get) in zip(axes, panels):
        x = np.arange(len(fields))
        for k, (lab, col) in enumerate(labels):
            xs = x - 0.27 + k * 0.27
            vals = [get(f)[k] for f, _ in fields]
            mean = [max(0.0, np.mean(v)) for v in vals]
            ax.bar(xs, mean, width=0.25, color=col, label=lab, edgecolor="white", linewidth=2)
            for xi, v, m in zip(xs, vals, mean):
                if len(v) > 1:
                    ax.plot([xi, xi], [max(0, min(v)), max(v)], color=INK, lw=1)
                ax.text(xi, max(max(v), m) + 0.03, f"{np.mean(v):.2f}" if np.mean(v) > 0 else "≤ 0", ha="center", fontsize=7.5, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in fields]); ax.set_ylim(0, 1.12)
        ax.set_ylabel(ylab); ax.grid(axis="x", visible=False)
    axes[1].axhline(0.5, color=INK2, lw=0.8, ls=":"); axes[1].set_xlim(-0.5, 0.5); axes[1].text(1.01, 0.5 / 1.12, "chance", transform=axes[1].transAxes, fontsize=7, color=INK2, va="center", ha="left")
    axes[0].legend(fontsize=7.5, loc="upper center", ncol=3, bbox_to_anchor=(0.8, -0.12))
    fig.tight_layout()
    save(fig, "fig10_side_info.png")


def fig_audit():
    r = load("results/p1_item7_slot_decoder.json")
    fields = ["label", "tier", "urgent"]
    inp = [r["input_vector"][f]["mlp"]["balanced_accuracy"] for f in fields]
    slots = ["0", "9", "19", "49"]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    x = np.arange(len(fields))
    ax.bar(x - 0.3, inp, width=0.15, color=INK2, label="32-d input vector (upper bound)", edgecolor="white", linewidth=2)
    shades = ["#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"]
    for k, (sl, col) in enumerate(zip(slots, shades)):
        vals = [np.mean([ck["slots"][sl][f]["mlp"]["balanced_accuracy"] for ck in r["checkpoints"].values()]) for f in fields]
        ax.bar(x - 0.15 + k * 0.15, vals, width=0.15, color=col, label=f"slot {sl} tokens", edgecolor="white", linewidth=2)
    ax.set_xticks(x); ax.set_xticklabels(["Predicted label", "Urgency tier", "Urgent flag"]); ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Balanced accuracy (DS2)"); ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    save(fig, "fig8_auditability.png")


def fig_two_senders():
    """Deviations 20 and 22: both senders against default, calibrated and filtered text."""
    a, f = load(R4), load("results/p1_item7_filtered.json")
    s = load("results/p1_second_sender_analysis.json")["RQ2"]["by_n"]
    d, c = a["primary_default"], a["primary_calibrated_text"]["by_n"]
    panels = [("Main sender: CNN-LSTM-RR",
               {"adapter": ([d[str(n)]["mea_mean"] for n in NS], [min(d[str(n)]["per_seed"].values()) for n in NS],
                            [max(d[str(n)]["per_seed"].values()) for n in NS]),
                "text": [d[str(n)]["text"] for n in NS], "cal": [c[str(n)]["text"] for n in NS],
                "filtered": [f["vs_filtered_default"][str(n)]["base"] for n in NS]}),
              ("Second sender: ResNet1D-RR (convolutional)",
               {"adapter": ([s[str(n)]["MEA_mean"] for n in NS],
                            [min(s[str(n)][f"MEA:r4_res_seed{k}"] for k in (101, 202, 303)) for n in NS],
                            [max(s[str(n)][f"MEA:r4_res_seed{k}"] for k in (101, 202, 303)) for n in NS]),
                "text": [s[str(n)]["A-compact"] for n in NS], "cal": [s[str(n)]["A-compact-cal"] for n in NS],
                "filtered": [s[str(n)]["A-filtered"] for n in NS]})]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for ax, (title, v) in zip(axes, panels):
        mean, lo, hi = v["adapter"]
        ax.fill_between(NS, lo, hi, color=BLUE, alpha=0.14, linewidth=0)
        ax.plot(NS, mean, color=BLUE, lw=2, marker="o", ms=5, label="Latent adapter (r4 protocol), 3-seed mean; band = seed range")
        ax.plot(NS, v["text"], color=ORANGE, lw=2, marker="o", ms=5, label="Text, all events (A-compact), default")
        ax.plot(NS, v["cal"], color=AQUA, lw=2, ls="--", marker="s", ms=5, label="Text, all events, calibrated threshold")
        ax.plot(NS, v["filtered"], color=INK2, lw=1.6, ls="-.", marker="^", ms=5, label="Filtered text: abnormal beats listed, normal counted")
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS); ax.set_ylim(0.45, 1.02)
        ax.set_xlabel("Events per prompt (N)")
    axes[0].set_ylabel("Balanced accuracy (DS2 test)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig11_two_senders.png")


if __name__ == "__main__":
    fig_pipeline(); fig_latency(); fig_accuracy(); fig_forest(); fig_why_text_fails(); fig_confusion(); fig_training(); fig_audit()
    fig_false_alarms(); fig_side_info(); fig_two_senders()
