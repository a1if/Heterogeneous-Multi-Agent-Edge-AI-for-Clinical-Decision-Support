"""Deviation 28: bytes a sensor would transmit per window, and nominal transmission time. CPU, no model runs.

For every Deviation 16 DS2 v2 window (main sender):
  compact_json     the A-compact event payload, exactly as placed in the prompt (UTF-8, indent=2)
  filtered_json    the A-filtered payload (non-normal beats with position, normal beats counted)
  *_zlib           the same, zlib level 9
  latent_f32/f16/i8  the 35-d per-event input the hub's adapter needs (int8 adds a 4-byte per-window scale)
  binary_fields    the fields compact text carries, packed: label 1 B, confidence float16 2 B, run length 1 B,
                   signal quality 1 B = 5 B per event (the hub renders them as text for the receiver)
Every message gets an 8-byte header. Reported: median bytes per window by N, and time at 1 Mbit/s,
100 kbit/s and 10 kbit/s (bits / rate; protocol overhead and radio energy not modelled).

Run (from repo root):  python p1_bandwidth.py   -> results/p1_bandwidth.json
"""
import json
import zlib
from pathlib import Path

import numpy as np

from p1_io import save_json_atomic

HEADER = 8
RATES = {"1 Mbit/s": 1e6, "100 kbit/s": 1e5, "10 kbit/s": 1e4}
OUT = Path("results/p1_bandwidth.json")


def main():
    from p1_item7_common import replay_split, window_vectors
    from reasoning.prompt_template_family import _payload
    r = replay_split("ds2")
    ev = r["events"]
    windows = json.loads(Path("results/p1_item7_testset_v2.json").read_text(encoding="utf-8"))["windows"]

    def compact(s, n):
        return json.dumps([_payload(ev[i], "compact") for i in range(s, s + n)], indent=2).encode("utf-8")

    def filtered(s, n):  # same structure as p1_item7_filtered.filtered_content's payload
        listed = [{"position": k + 1, **_payload(ev[s + k], "compact")}
                  for k in range(n) if ev[s + k]["classification"]["label"] != "N"]
        return json.dumps({"total_beats": n, "normal_beats_not_listed": n - len(listed), "non_normal_beats": listed},
                          indent=2).encode("utf-8")

    rows = []
    for w in windows:
        s, n = w["start"], w["n"]
        c, f = compact(s, n), filtered(s, n)
        v = window_vectors(r, s, n, 35)
        size = {
            "compact_json": len(c), "filtered_json": len(f),
            "compact_json_zlib": len(zlib.compress(c, 9)), "filtered_json_zlib": len(zlib.compress(f, 9)),
            "latent_f32": v.size * 4, "latent_f16": v.size * 2, "latent_i8": v.size + 4,
            "binary_fields": 5 * n,
        }
        rows.append({"set": w["set"], "n": n, "start": s, **{k: x + HEADER for k, x in size.items()}})
    arms = [k for k in rows[0] if k not in ("set", "n", "start")]
    summary = {}
    for n in sorted({x["n"] for x in rows}):
        xs = [x for x in rows if x["n"] == n]
        med = {a: float(np.median([x[a] for x in xs])) for a in arms}
        summary[str(n)] = {"n_windows": len(xs), "median_bytes": med,
                           "time_ms": {rate: {a: 1000 * 8 * med[a] / bps for a in arms} for rate, bps in RATES.items()}}
    save_json_atomic(OUT, {"design": __doc__, "header_bytes": HEADER, "summary": summary, "rows": rows})
    print(f"{'N':>3} " + " ".join(f"{a:>18}" for a in arms))
    for n, d in summary.items():
        print(f"{n:>3} " + " ".join(f"{d['median_bytes'][a]:>18.0f}" for a in arms))
    print("time at 10 kbit/s, N = 50 (ms):", {a: round(t) for a, t in summary["50"]["time_ms"]["10 kbit/s"].items()})


if __name__ == "__main__":
    main()
