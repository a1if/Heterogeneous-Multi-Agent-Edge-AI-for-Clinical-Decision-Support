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


def fig_accuracy():
    s5, cal = load("results/p1_item7_step5.json"), load("results/p1_item7_calib.json")["primary_rule"]["by_n"]
    mea = ["MEA:r3_seed101", "MEA:r3_seed202", "MEA:r3_seed303"]
    text = [s5["by_n"][str(n)]["balanced_accuracy"]["A-compact"] for n in NS]
    mean = [s5["by_n"][str(n)]["balanced_accuracy"]["MEA_mean"] for n in NS]
    lo = [min(s5["by_n"][str(n)]["balanced_accuracy"][m] for m in mea) for n in NS]
    hi = [max(s5["by_n"][str(n)]["balanced_accuracy"][m] for m in mea) for n in NS]
    ctext = [cal[str(n)]["A-compact"] for n in NS]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.fill_between(NS, lo, hi, color=BLUE, alpha=0.14, linewidth=0)
    ax.plot(NS, mean, color=BLUE, lw=2, marker="o", ms=6, label="Latent adapter, 3-seed mean (band = seed range)")
    ax.plot(NS, text, color=ORANGE, lw=2, marker="o", ms=6, label="Text (A-compact), default decoding")
    ax.plot(NS, ctext, color=AQUA, lw=2, ls="--", marker="s", ms=6, label="Text, calibrated threshold (Deviation 17)")
    for n, v in zip(NS, text):
        ax.text(n, v - 0.045, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    for n, v in zip(NS, mean):
        ax.text(n, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS); ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("Events per prompt (N)"); ax.set_ylabel("Balanced accuracy (DS2 test)")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig3_accuracy_vs_n.png")


def fig_forest():
    s5, cal = load("results/p1_item7_step5.json"), load("results/p1_item7_calib.json")["primary_rule"]["by_n"]
    ns = [5, 10, 20, 50]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for k, n in enumerate(ns):
        for off, (src, col, lab) in zip((-0.14, 0.14), ((s5["by_n"][str(n)]["primary"], ORANGE, "vs default text"),
                                                         (cal[str(n)]["primary"], AQUA, "vs calibrated text"))):
            y = k + off
            d, (l, h) = src["difference"], src["cluster_ci95"]
            ax.plot([l, h], [y, y], color=col, lw=2.2, solid_capstyle="round")
            ax.plot(d, y, "o", color=col, ms=6, markeredgecolor="white", markeredgewidth=1.2, label=lab if k == 0 else None)
            ax.text(h + 0.01, y, f"{d:+.2f} [{l:+.2f}, {h:+.2f}]", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(-0.05, color="#d03b3b", lw=1, ls=":")
    ax.text(-0.055, -0.62, "NI margin −0.05", color="#d03b3b", fontsize=7.5, ha="right", va="center")
    ax.set_yticks(range(len(ns))); ax.set_yticklabels([f"N = {n}" for n in ns]); ax.set_ylim(len(ns) - 0.5, -0.9)
    ax.set_xlim(-0.2, 0.66); ax.set_xlabel("Adapter minus text, balanced accuracy (record-level 95% CI)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "fig4_forest.png")


def fig_why_text_fails():
    from p1_item7_common import replay_split
    from reasoning.training_targets import most_urgent_index
    ev = replay_split("ds2")["events"]
    rows = load("results/p1_item7_eval_ds2v2.json")["rows"]
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
    s5 = load("results/p1_item7_step5.json")
    T = ["routine", "priority", "urgent"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, arm, title in ((axes[0], "A-compact", "Text (A-compact)"), (axes[1], None, "Latent adapter (3 seeds pooled)")):
        c = s5["by_n"]["50"]["confusion"]
        if arm:
            m = np.array([[c[arm][r].get(p, 0) for p in T] for r in T], dtype=float)
        else:
            m = sum(np.array([[c[a][r].get(p, 0) for p in T] for r in T], dtype=float)
                    for a in ("MEA:r3_seed101", "MEA:r3_seed202", "MEA:r3_seed303"))
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
        s = load(f"results/p1_item7_train_r3_seed{seed}.json")
        h = [x for x in s["history"] if "val" in x]
        ax.plot([x["update"] for x in h], [x["val"]["balanced_accuracy"] for x in h], color=col, lw=2, ls=ls,
                marker="o", ms=4, label=f"seed {seed} (best step {s['best_update']})")
    ax.axvline(750, color=INK2, lw=0.8, ls=":")
    ax.text(758, 0.95, "early stopping allowed from here", fontsize=7.5, color=INK2, va="top")
    ax.set_xlabel("Training step (batch 1)"); ax.set_ylabel("Validation balanced accuracy"); ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig7_training_r3.png")


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


if __name__ == "__main__":
    fig_pipeline(); fig_latency(); fig_accuracy(); fig_forest(); fig_why_text_fails(); fig_confusion(); fig_training(); fig_audit()
