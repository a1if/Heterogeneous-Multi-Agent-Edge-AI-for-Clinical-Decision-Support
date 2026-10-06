"""E3b: context-size costs from the existing E3 measurements (CPU only, no new runs).

E3's prefill advantage partly reflects re-processing the whole prompt; a streaming
text system with a KV cache would pay only each new event's payload at prefill.
What such a system still pays is the accumulated context, on every generated token
and in memory. From E3's rows (per window, N, arm):
  decode_ms_per_token  (total_ms_16 - prefill_ms) / 15: generation cost per token
                       over the accumulated context (attention grows with context)
  context_mem_mb       peak_mem_mb minus the same arm's N = 1 peak in that window:
                       memory attributable to the accumulated context
  energy_j_16          energy of the 16-token generation (includes one prefill)
B-4 vs A-compact is compared per window (paired), with a 95% bootstrap CI of the
mean relative difference. Memory growth per context token is fitted per arm and
used to extrapolate how many events fit in the remaining GPU memory (an
extrapolation, labelled as such).

Run (from repo root):
    python p1_e3b_context_costs.py
"""
import json

import numpy as np

from p1_io import save_json_atomic

SRC, OUT = "results/p1_e3_multi_event.json", "results/p1_e3b_context_costs.json"
GPU_MB = 12227  # RTX 5070 total, nvidia-smi


def boot_ci(x, seed=0):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (20000, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    s = json.load(open(SRC, encoding="utf-8"))
    rows = {(r["window"], r["n"], r["arm"]): r for r in s["rows"] if not r.get("oom")}
    windows = sorted({w for w, _, _ in rows})
    ns = sorted({n for _, n, _ in rows})
    arms = ("A-full", "A-compact", "B-4")

    def derived(w, n, arm):
        r, base = rows.get((w, n, arm)), rows.get((w, 1, arm))
        if not r or not base:
            return None
        return {"tokens": r["prompt_tokens"], "decode_ms_per_token": (r["total_ms_16"] - r["prefill_ms"]) / 15,
                "context_mem_mb": r["peak_mem_mb"] - base["peak_mem_mb"], "energy_j_16": r["energy_j_16"],
                "peak_mem_mb": r["peak_mem_mb"]}

    out = {"source": SRC, "by_n": {}, "b4_vs_acompact": {}, "memory_fit": {}}
    for n in ns:
        out["by_n"][str(n)] = {}
        for arm in arms:
            d = [x for w in windows if (x := derived(w, n, arm))]
            if d:
                out["by_n"][str(n)][arm] = {k: float(np.median([x[k] for x in d])) for k in d[0]} | {"n_windows": len(d)}
        comp = {}
        for metric in ("decode_ms_per_token", "energy_j_16", "context_mem_mb"):
            pairs = [(derived(w, n, "B-4"), derived(w, n, "A-compact")) for w in windows]
            pairs = [(b, a) for b, a in pairs if b and a]
            if metric == "context_mem_mb":
                diff = [b[metric] - a[metric] for b, a in pairs]  # MB, absolute (0 at N = 1)
                comp[metric] = {"mean_diff_mb": float(np.mean(diff)), "ci95_mb": boot_ci(diff)}
            else:
                rel = [100 * (b[metric] - a[metric]) / a[metric] for b, a in pairs if b[metric] and a[metric]]
                comp[metric] = {"mean_rel_pct": float(np.mean(rel)), "ci95_pct": boot_ci(rel)}
        out["b4_vs_acompact"][str(n)] = comp

    # Memory per context token (least squares over all windows and N) and events that fit.
    for arm in arms:
        pts = [(derived(w, n, arm)["tokens"], derived(w, n, arm)["peak_mem_mb"]) for w in windows for n in ns
               if derived(w, n, arm)]
        t, m = np.array(pts, float).T
        slope, intercept = np.polyfit(t, m, 1)
        tokens_per_event = float(np.median([derived(w, 50, arm)["tokens"] - derived(w, 1, arm)["tokens"]
                                            for w in windows if derived(w, 50, arm)]) / 49)
        n1_tokens = float(np.median([derived(w, 1, arm)["tokens"] for w in windows if derived(w, 1, arm)]))
        max_tokens = (GPU_MB - intercept) / slope
        out["memory_fit"][arm] = {"mb_per_context_token": float(slope), "intercept_mb": float(intercept),
                                  "tokens_per_extra_event": tokens_per_event,
                                  "extrapolated_max_events_in_12gb": float((max_tokens - n1_tokens) / tokens_per_event + 1),
                                  "note": "linear extrapolation of allocated memory; ignores fragmentation and "
                                          "Windows memory paging, which A-full already hit near 11.5 GB"}

    save_json_atomic(OUT, out)
    for n, res in out["by_n"].items():
        print(f"N={n:>2} " + " | ".join(f"{a}: {v['tokens']:.0f}tok decode {v['decode_ms_per_token']:.1f}ms/tok "
                                        f"ctx-mem {v['context_mem_mb']:+.0f}MB E16 {v['energy_j_16']:.0f}J"
                                        for a, v in res.items()))
    for n, c in out["b4_vs_acompact"].items():
        print(f"B-4 vs A-compact N={n:>2}: decode/token {c['decode_ms_per_token']['mean_rel_pct']:+.1f}% "
              f"{[round(v, 1) for v in c['decode_ms_per_token']['ci95_pct']]}, energy {c['energy_j_16']['mean_rel_pct']:+.1f}% "
              f"{[round(v, 1) for v in c['energy_j_16']['ci95_pct']]}, context memory {c['context_mem_mb']['mean_diff_mb']:+.0f} MB "
              f"{[round(v) for v in c['context_mem_mb']['ci95_mb']]}")
    for arm, f in out["memory_fit"].items():
        print(f"{arm}: {f['mb_per_context_token']*1000:.1f} KB/context token, {f['tokens_per_extra_event']:.1f} tokens/event, "
              f"~{f['extrapolated_max_events_in_12gb']:.0f} events fit (extrapolated)")


if __name__ == "__main__":
    main()
