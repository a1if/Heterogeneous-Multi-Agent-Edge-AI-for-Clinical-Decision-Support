"""Exploratory (post hoc, Option 0 of the scaling question): learning curves from existing training logs. CPU.

No new training and no test data: every recipe trained on the same 447 r4 training windows logged validation
balanced accuracy (170 DS1 validation windows) every 111 updates, and one update is one training window, so the
logs give validation accuracy against training windows seen (1 epoch = 447 windows). Per run: the best
validation score reached within 1, 2 and 3 epochs (running maximum), the gain from epoch 2 to epoch 3, the epoch
of the best checkpoint, and whether the run was stopped early (no improvement for three checks after 2 epochs).
Main recipe r4; the other 447-window recipes (r4hn, r4k1, r4k2; three seeds each) are shown as a robustness check.

This measures passes over a fixed data set, not more data; it is evidence about saturation within the recipe,
not a data-scaling curve.

Run (from repo root):  python p1_learning_curves.py   -> results/p1_learning_curves.json, reports/figures/fig13_learning_curves.png
"""
import json
from pathlib import Path

import numpy as np

from p1_io import save_json_atomic

EPOCH = 447
RECIPES = {"r4": "r4 (final)", "r4hn": "hard negatives only", "r4k2": "2 tokens per event", "r4k1": "1 token per event"}
OUT = Path("results/p1_learning_curves.json")


def main():
    runs = {}
    for rec in RECIPES:
        for s in (101, 202, 303):
            p = Path(f"results/p1_item7_train_{rec}_seed{s}.json")
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            h = [(x["update"], x["val"]["balanced_accuracy"]) for x in d["history"] if "val" in x]
            u = np.array([a for a, _ in h]); v = np.array([b for _, b in h])
            best_by = {e: float(v[u <= e * EPOCH].max()) if (u <= e * EPOCH).any() else None for e in (1, 2, 3)}
            stopped_early = bool(u[-1] < 3 * EPOCH - EPOCH // 4)  # last check well before the 3-epoch cap
            runs[f"{rec}_seed{s}"] = {
                "recipe": rec, "seed": s, "updates": u.tolist(), "val": v.tolist(),
                "best": float(v.max()), "best_epoch": float(u[v.argmax()] / EPOCH),
                "best_within_epochs": best_by,
                "gain_epoch2_to_3": (best_by[3] - best_by[2]) if best_by[2] is not None else None,
                "stopped_early": stopped_early}
    healthy = {k: r for k, r in runs.items() if r["best"] >= 0.6}  # excludes the failed k = 1 seed 303 run (0.51)
    gains = [r["gain_epoch2_to_3"] for r in healthy.values()]
    summary = {
        "n_runs": len(runs), "n_healthy": len(healthy),
        "peaked_within_2_epochs": sum(r["best_epoch"] <= 2.0 for r in healthy.values()),
        "stopped_early": sum(r["stopped_early"] for r in healthy.values()),
        "gain_epoch2_to_3": {"median": float(np.median(gains)), "max": float(max(gains)),
                             "n_zero": int(sum(g == 0 for g in gains))},
        "r4": {k: {x: runs[k][x] for x in ("best", "best_epoch", "best_within_epochs", "gain_epoch2_to_3", "stopped_early")}
               for k in runs if runs[k]["recipe"] == "r4"},
    }
    save_json_atomic(OUT, {"design": __doc__, "epoch_windows": EPOCH, "summary": summary, "runs": runs})
    print(json.dumps(summary, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for k, r in runs.items():
        main_ = r["recipe"] == "r4"
        shade = {101: "#9ec5f4", 202: "#3987e5", 303: "#1c4f8f"}[r["seed"]]
        ax.plot(np.array(r["updates"]) / EPOCH, r["val"], color=shade if main_ else "#b8b6b0",
                lw=2 if main_ else 1, marker="o" if main_ else None, ms=3, zorder=3 if main_ else 1,
                label=(f"r4 seed {r['seed']}" if main_ else None))
    ax.plot([], [], color="#b8b6b0", lw=1, label="other recipes on the same 447 windows (9 runs)")
    for e in (1, 2):
        ax.axvline(e, color="#52514e", lw=0.6, ls=":")
    ax.set_xlabel("Training windows seen (epochs of 447)")
    ax.set_ylabel("Validation balanced accuracy")
    ax.set_xlim(0, 3.05); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig("reports/figures/fig13_learning_curves.png", dpi=200)
    print("wrote reports/figures/fig13_learning_curves.png")


if __name__ == "__main__":
    main()
