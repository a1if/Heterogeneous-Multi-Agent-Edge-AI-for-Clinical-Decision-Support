"""Generates two Chapter 4 figures from real per-event Day 6 data.

Figure 1 (forest plot): the three headline reductions (tokens/latency/VRAM)
with paired bootstrap 95% CIs. Point estimates are pulled from
results_ledger.json / results_ledger_updated.json so they match cited prose
exactly (24.7%, 13.3%). CIs are computed here, not invented -- no CI for these
reductions existed anywhere in the project before this script.

Figure 2 (paired per-event distributions): Arm A vs Arm B, tokens and
latency, using the ACTUAL 80-event per-arm records in day6_results.json --
not a reconstruction from summary statistics.

IMPORTANT: bootstrap/plotting source is
results/day6_results.json.bak_pre_rerun_20260816, NOT the live
results/day6_results.json. A later session re-ran day6_run_comparison.py to
check reproducibility; token counts and urgency tiers reproduced exactly
(deterministic), but generation_duration_ms drifted (expected -- latency is a
runtime measurement, not deterministic). The backup is the dataset that
actually produced the ledger's cited 13.3%/24.7% figures; the live file would
silently shift the latency point estimate to ~13.8%. Confirmed by direct
comparison before writing this script.

Run (from repo root):
    python make_headline_figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "results/day6_results.json.bak_pre_rerun_20260816"
OUT_DIRS = [r"D:\Dissertation\latex_src\figures", r"D:\Dissertation\latex_build\figures"]
N_BOOT = 20000
SEED = 0

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_paired():
    d = json.load(open(DATA_PATH, encoding="utf-8"))
    by_idx = {}
    for r in d:
        by_idx.setdefault(r["idx"], {})[r["arm"]] = r
    idxs = sorted(by_idx)
    a = {k: np.array([by_idx[i]["A"][k] for i in idxs], float)
         for k in ("prompt_tokens", "generation_duration_ms", "peak_vram_mb")}
    b = {k: np.array([by_idx[i]["B"][k] for i in idxs], float)
         for k in ("prompt_tokens", "generation_duration_ms", "peak_vram_mb")}
    return idxs, a, b


def reduction_pct(a, b):
    return (1.0 - b.mean() / a.mean()) * 100.0


def paired_bootstrap_ci(a, b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(a)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)      # PAIRED resample: same idx for both arms
        boots[i] = reduction_pct(a[idx], b[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return reduction_pct(a, b), lo, hi


def save(fig, name):
    for d in OUT_DIRS:
        fig.savefig(f"{d}/{name}", dpi=200, bbox_inches="tight")
    print(f"saved {name} to {len(OUT_DIRS)} location(s)")


def make_forest_plot(a, b):
    metrics = [
        ("Prompt tokens", a["prompt_tokens"], b["prompt_tokens"]),
        ("Generation latency", a["generation_duration_ms"], b["generation_duration_ms"]),
        ("Peak VRAM", a["peak_vram_mb"], b["peak_vram_mb"]),
    ]
    rows = []
    for name, av, bv in metrics:
        point, lo, hi = paired_bootstrap_ci(av, bv)
        rows.append((name, point, lo, hi))
        print(f"{name:<20} point={point:6.2f}%  95% CI [{lo:6.2f}, {hi:6.2f}]  (n=80, paired bootstrap, {N_BOOT} resamples)")

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ypos = np.arange(len(rows))[::-1]
    for y, (name, point, lo, hi) in zip(ypos, rows):
        ax.plot([lo, hi], [y, y], color="#333333", lw=1.6, zorder=1)
        ax.plot([lo, lo], [y - 0.08, y + 0.08], color="#333333", lw=1.6, zorder=1)
        ax.plot([hi, hi], [y - 0.08, y + 0.08], color="#333333", lw=1.6, zorder=1)
        ax.scatter([point], [y], color="#c0392b", s=55, zorder=2, edgecolor="white", linewidth=0.8)
        dp = 2 if abs(point) < 1 else 1   # VRAM's true effect is sub-percent; 1dp would print "0.3%" three times
        ax.annotate(f"{point:.{dp}f}%  [{lo:.{dp}f}, {hi:.{dp}f}]", xy=(hi, y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color="#333333")

    ax.axvline(0, color="#999999", lw=1.0, ls="--", zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Reduction, Arm A $\\rightarrow$ Arm B (%)")
    ax.set_xlim(-2, 32)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_title("Headline efficiency reductions with 95% bootstrap CIs (n=80, paired)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig4_5_headline_forest.png")
    plt.close(fig)
    return rows


def make_paired_distributions(idxs, a, b):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    specs = [
        (axes[0], a["prompt_tokens"], b["prompt_tokens"], "Prompt tokens", "tok"),
        (axes[1], a["generation_duration_ms"], b["generation_duration_ms"], "Generation latency", "ms"),
    ]
    for ax, av, bv, title, unit in specs:
        xs = [0, 1]
        for i in range(len(av)):
            ax.plot(xs, [av[i], bv[i]], color="#999999", lw=0.6, alpha=0.5, zorder=1)
        jitter_a = np.random.default_rng(1).normal(0, 0.02, len(av))
        jitter_b = np.random.default_rng(2).normal(0, 0.02, len(bv))
        ax.scatter(np.zeros(len(av)) + jitter_a, av, s=18, color="#2c3e50", zorder=2, label="Arm A")
        ax.scatter(np.ones(len(bv)) + jitter_b, bv, s=18, color="#c0392b", zorder=2, label="Arm B")
        bp = ax.boxplot([av, bv], positions=xs, widths=0.35, showfliers=False,
                         patch_artist=True, zorder=3)
        for patch in bp["boxes"]:
            patch.set_facecolor("white")
            patch.set_alpha(0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Arm A", "Arm B"])
        ax.set_ylabel(unit)
        ax.set_title(title, fontsize=11)

    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Paired per-event distributions, Arm A vs Arm B (n=80)", fontsize=11, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_6_paired_distributions.png")
    plt.close(fig)


def main():
    idxs, a, b = load_paired()
    print(f"loaded {len(idxs)} paired events from {DATA_PATH}\n")
    make_forest_plot(a, b)
    print()
    make_paired_distributions(idxs, a, b)


if __name__ == "__main__":
    main()
