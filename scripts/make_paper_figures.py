"""Generate the vector figures of the paper from the result files.

Run from the repository root:
    python scripts/make_paper_figures.py

Writes docs/paper/figures/*.pdf. Colours are the validated categorical slots (blue, orange, aqua, violet);
every series also has its own marker and line style, so identity never depends on colour alone.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NS = ["1", "5", "10", "20", "50"]
X = [int(n) for n in NS]

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
STYLE = {  # arm -> (colour, marker, linestyle, label)
    "adapter": ("#2a78d6", "o", "-", "Adapter (latent)"),
    "compact": ("#eb6834", "s", "--", "Text, every event"),
    "compact_cal": ("#1baf7a", "^", "-.", "Text, every event, calibrated"),
    "filtered": ("#4a3aa7", "D", ":", "Filtered text"),
}
plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def line(ax, arm, xs, ys, **kw):
    c, m, ls, lab = STYLE[arm]
    ax.plot(xs, ys, color=c, marker=m, linestyle=ls, linewidth=1.6, markersize=5.5,
            markeredgecolor="#fcfcfb", markeredgewidth=0.8, label=lab, **kw)


# ---- Figure: accuracy against N ---------------------------------------------------------------------------------
r4 = load("p1_item7_r4_analysis.json")
flt = load("p1_item7_filtered.json")
fig, ax = plt.subplots(figsize=(4.6, 3.1))
line(ax, "filtered", X, [flt["vs_filtered_default"][n]["base"] for n in NS])
line(ax, "adapter", X, [r4["primary_calibrated_text"]["by_n"][n]["mea_mean"] for n in NS])
seeds = [[v for v in r4["primary_calibrated_text"]["by_n"][n]["per_seed"].values()] for n in NS]
ax.fill_between(X, [min(s) for s in seeds], [max(s) for s in seeds], color=STYLE["adapter"][0], alpha=0.15, linewidth=0)
line(ax, "compact_cal", X, [r4["primary_calibrated_text"]["by_n"][n]["text"] for n in NS])
line(ax, "compact", X, [r4["primary_default"][n]["text"] for n in NS])
ax.set_xscale("log")
ax.set_xticks(X, [str(x) for x in X])
ax.set_xlabel("Events per prompt, $N$")
ax.set_ylabel("Balanced accuracy")
ax.set_ylim(0.4, 1.02)
ax.legend(loc="lower left", fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "fig_accuracy.pdf")
plt.close(fig)

# ---- Figure: cost against N (time to first token, time to decision, prompt tokens) -------------------------------
t = flt["timing"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
panels = [("ttft_ms", "Time to first token (ms)"), ("ttd_ms", "Time to decision (ms)")]
for ax, (key, lab) in zip(axs[:2], panels):
    line(ax, "compact", X, [t[n][key]["A-compact"] for n in NS])
    line(ax, "filtered", X, [t[n][key]["A-filtered"] for n in NS])
    line(ax, "adapter", X, [t[n][key]["MEA"] for n in NS])
    ax.set_xscale("log")
    ax.set_xticks(X, [str(x) for x in X])
    ax.set_xlabel("Events per prompt, $N$")
    ax.set_ylabel(lab)
    ax.set_ylim(bottom=0)
ax = axs[2]
pt = flt["prompt_tokens"]
line(ax, "compact", X, [pt[n]["A-compact"]["median"] for n in NS])
line(ax, "filtered", X, [pt[n]["A-filtered"]["median"] for n in NS])
ax.fill_between(X, [pt[n]["A-filtered"]["median"] for n in NS], [pt[n]["A-filtered"]["p90"] for n in NS],
                color=STYLE["filtered"][0], alpha=0.15, linewidth=0)
line(ax, "adapter", X, [pt[n]["MEA"]["median"] for n in NS])
ax.set_xscale("log")
ax.set_xticks(X, [str(x) for x in X])
ax.set_xlabel("Events per prompt, $N$")
ax.set_ylabel("Prompt tokens (median)")
ax.set_ylim(bottom=0)
axs[0].legend(loc="upper left", fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "fig_cost.pdf")
plt.close(fig)

# ---- Figure: serving throughput and energy, by batch size ----------------------------------------------------------
sv = load("p1_serving.json")["summary"]
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
batches = [1, 4, 8]
for j, n in enumerate(["10", "20", "50"]):
    for i, (metric, lab) in enumerate([("decisions_per_s", "Decisions per second"), ("joules_per_decision", "Energy per decision (J)")]):
        ax = axs[i][j]
        for arm, key in [("compact", "A-compact"), ("filtered", "A-filtered"), ("adapter", "MEA")]:
            xs, ys, miss = [], [], []
            for b in batches:
                d = sv.get(f"N{n}_B{b}", {}).get(key)
                if d is None:
                    continue
                if d.get("n_measurements", 0) == 0:
                    miss.append(b)
                else:
                    xs.append(b)
                    ys.append(d[metric])
            line(ax, arm, xs, ys)
            if miss and i == 0:
                ax.plot(miss, [0.0] * len(miss), marker="x", color=STYLE[arm][0], linestyle="none", markersize=7, clip_on=False)
        ax.set_xticks(batches)
        if i == 0:
            ax.set_title(f"$N$ = {n}", fontsize=9, color=INK)
        if i == 1:
            ax.set_xlabel("Batch size")
        if j == 0:
            ax.set_ylabel(lab)
        ax.set_ylim(bottom=0)
axs[0][0].legend(loc="upper left", fontsize=7)
fig.text(0.99, 0.005, "$\\times$ = does not fit in the 12 GB memory cap", ha="right", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(OUT / "fig_serving.pdf")
plt.close(fig)

# ---- Figure: urgent recall by reason (mechanisms), adapter seeds against text ------------------------------------
urg = r4["urgent_by_reason"]
reasons = [("run>=3", "Run of 3+\nabnormal beats"), ("high-conf V/F", "Single high-confidence\nV/F beat")]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.0, 1.0]})
w = 0.34
for i, (rk, rl) in enumerate(reasons):
    tx = urg["A-compact"][rk]["recall"]
    ad = [urg[f"MEA:r4_seed{s}"][rk]["recall"] for s in (101, 202, 303)]
    ax.bar(i - w / 2, tx, width=w, color=STYLE["compact"][0], edgecolor="#fcfcfb", linewidth=1.5, label="Text, every event" if i == 0 else None)
    ax.bar(i + w / 2, sum(ad) / 3, width=w, color=STYLE["adapter"][0], edgecolor="#fcfcfb", linewidth=1.5, label="Adapter (mean of 3 seeds)" if i == 0 else None)
    ax.plot([i + w / 2] * 3, ad, linestyle="none", marker="_", color=INK, markersize=9)
    ax.text(i - w / 2, tx + 0.02, f"{tx:.2f}", ha="center", fontsize=8, color=INK)
    ax.text(i + w / 2, max(ad) + 0.02, f"{sum(ad) / 3:.2f}", ha="center", fontsize=8, color=INK)
ax.set_xticks([0, 1], [rl for _, rl in reasons])
ax.set_ylabel("Urgent recall")
ax.set_ylim(0, 1.05)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", fontsize=7.5, ncol=1)
ax.set_title("By reason the window is urgent", fontsize=8.5, color=INK)

# Position of the deciding beat (exploratory; results/p1_position_effect.json from scripts/position_effect.py)
pos = load("p1_position_effect.json")["by_arm"]
thirds = ["First third", "Middle third", "Last third"]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


for off, arm, key in ((-w / 2, "compact", "text"), (w / 2, "adapter", "adapter")):
    for i, th in enumerate(thirds):
        a, n = pos[key][th]["accuracy"], pos[key][th]["n"]
        lo, hi = wilson(round(a * n), n)
        ax2.bar(i + off, a, width=w, color=STYLE[arm][0], edgecolor="#fcfcfb", linewidth=1.5)
        ax2.plot([i + off] * 2, [lo, hi], color=INK, linewidth=1.0)
        ax2.text(i + off, hi + 0.02, f"n={n}", ha="center", fontsize=6.5, color=MUTED)
ax2.set_xticks(range(3), ["First\nthird", "Middle\nthird", "Last\nthird"])
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1.05)
ax2.grid(axis="x", visible=False)
ax2.set_title("By position of the deciding beat", fontsize=8.5, color=INK)
fig.tight_layout()
fig.savefig(OUT / "fig_mechanisms.pdf")
plt.close(fig)

# ---- Figure: compression (Deviations 24 and 27) ------------------------------------------------------------------
sw = load("p1_sweep_analysis.json")
ks = load("p1_dev27_kseeds.json")
fig, ax = plt.subplots(figsize=(4.4, 3.0))
cols = {"1": "#2a78d6", "2": "#eb6834", "4": "#1baf7a", "8": "#4a3aa7"}
marks = {"1": "o", "2": "s", "4": "^", "8": "D"}
Ns = ["5", "10", "20", "50"]
x = [int(n) for n in Ns]
for k, g in (("4", "k4"), ("2", "k2"), ("1", "k1")):
    d = ks["describe"][g]["balanced_accuracy"]
    ax.fill_between(x, [d[n]["range"][0] for n in Ns], [d[n]["range"][1] for n in Ns], color=cols[k], alpha=0.12, linewidth=0)
    ax.plot(x, [d[n]["mean"] for n in Ns], color=cols[k], marker=marks[k], linewidth=1.5, markersize=5,
            markeredgecolor="#fcfcfb", markeredgewidth=0.8, label=f"$k$ = {k} (3 seeds)")
ax.plot(x, [sw["by_k"]["8"]["balanced_accuracy"][n] for n in Ns], color=cols["8"], marker=marks["8"], linewidth=1.2,
        linestyle="--", markersize=4, label="$k$ = 8 (1 run)")
ax.set_xscale("log")
ax.set_xticks([5, 10, 20, 50], ["5", "10", "20", "50"])
ax.set_xlabel("Events per prompt, $N$")
ax.set_ylabel("Balanced accuracy")
ax.set_ylim(0.45, 0.92)
ax.legend(loc="lower center", fontsize=7, ncol=2, columnspacing=1.0, handlelength=1.6)
fig.tight_layout()
fig.savefig(OUT / "fig_sweep.pdf")
plt.close(fig)

# ---- Figure: validation curves over passes through the training set (exploratory) ----------------------------------
lc_path = RES / "p1_learning_curves.json"
if lc_path.exists():
    lc = load("p1_learning_curves.json")
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    first = True
    for k, r in lc["runs"].items():
        if r["recipe"] == "r4":
            continue
        ax.plot([u / lc["epoch_windows"] for u in r["updates"]], r["val"], color="#b8b6b0", linewidth=0.9, zorder=1,
                label="Variants trained on the same windows" if first else None)
        first = False
    shades = {101: ("#9ec5f4", "o"), 202: ("#3987e5", "s"), 303: ("#1c4f8f", "^")}
    for i, s in enumerate((101, 202, 303), 1):
        r = lc["runs"][f"r4_seed{s}"]
        c, m = shades[s]
        ax.plot([u / lc["epoch_windows"] for u in r["updates"]], r["val"], color=c, marker=m, linewidth=1.6, markersize=4,
                markeredgecolor="#fcfcfb", markeredgewidth=0.6, zorder=3, label=f"Final adapter, run {i}")
    for e in (1, 2):
        ax.axvline(e, color=MUTED, linewidth=0.6, linestyle=":")
    ax.set_xlabel("Passes through the 447 training windows")
    ax.set_ylabel("Validation balanced accuracy")
    ax.set_xlim(0, 3.05)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_learning.pdf", bbox_inches="tight")
    plt.close(fig)

print("figures written to", OUT)
