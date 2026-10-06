"""Item 7, Deviation 18b: run-length decodability from adapter tokens, with a well-posed metric. CPU.

Primary: balanced accuracy of the binary field run >= 3 (MLP); secondary: R^2 of the bounded
target run_z = log(1 + min(run, 20)) / log(4). Same samples, sources and decoders as the
Deviation 18 slot analysis (p1_item7_r4_analysis.slot_decoding). See docs/analysis_plan.md.

Run (from repo root):
    python p1_item7_runlen.py
"""
import time
from pathlib import Path

import numpy as np
import torch

import p1_item7_slot_decoder as sd
import p1_items6_7_decoder_anomaly as item6
from p1_io import save_json_atomic

OUT = Path("results/p1_item7_runlen.json")


def run_fields(events):
    run = np.asarray([ev["clinical_flags"]["consecutive_abnormal_beats"] for ev in events])
    return {"run3": run >= 3, "run_z": np.log1p(np.minimum(run, 20)) / np.log(4.0)}


def main():
    from p1_item7_common import replay_split, window_vectors
    t0 = time.time()
    r1, r2 = replay_split("ds1"), replay_split("ds2")
    rng = np.random.default_rng(0)
    s1 = item6.stratified(np.asarray(r1["labels"]), item6.PER_CLASS_DS1, rng)
    s2 = item6.stratified(np.asarray(r2["labels"]), item6.PER_CLASS_DS2, rng)
    f1, f2 = run_fields([r1["events"][i] for i in s1]), run_fields([r2["events"][i] for i in s2])
    x1 = np.concatenate([window_vectors(r1, i, 1, 35) for i in s1])
    x2 = np.concatenate([window_vectors(r2, i, 1, 35) for i in s2])
    item6.CLASSIFY, item6.REGRESS = ("run3",), ("run_z",)
    out = {"design": __doc__, "n_train": len(s1), "n_test": len(s2),
           "positives": {"ds1": int(f1["run3"].sum()), "ds2": int(f2["run3"].sum())},
           "input_35d": item6.decode(x1, f1, x2, f2, reduce=False)[0]}
    print(f"input done ({time.time() - t0:.0f}s)", flush=True)
    for tag in ("r3", "r4"):
        for seed in (101, 202, 303):
            state = torch.load(f"reasoning/checkpoints/p1_item7_mea_{tag}_seed{seed}.pt", map_location="cpu",
                               weights_only=False)["adapter_state_dict"]
            dim = state["projection.weight"].shape[1]
            a1, a2 = (x1, x2) if dim == 35 else (x1[:, :32], x2[:, :32])
            for slot in (0, 49):
                out[f"{tag}_seed{seed}_slot{slot}"] = item6.decode(sd.slot_tokens(state, a1, slot), f1,
                                                                   sd.slot_tokens(state, a2, slot), f2, reduce=True)[0]
            save_json_atomic(OUT, out)
            print(f"{tag} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    ba = lambda k: out[k]["run3"]["mlp"]["balanced_accuracy"]
    keys = lambda t: [k for k in out if k.startswith(t + "_")]
    inp, r3, r4 = ba("input_35d"), np.mean([ba(k) for k in keys("r3")]), [ba(k) for k in keys("r4")]
    out["verdict"] = {"input": inp, "r3_mean": float(r3), "r4_mean": float(np.mean(r4)), "r4_min": float(min(r4)),
                      "r4_carries_run": bool(np.mean(r4) >= inp - 0.05 and np.mean(r4) >= r3 + 0.10)}
    out["seconds"] = time.time() - t0
    save_json_atomic(OUT, out)
    for k, v in out.items():
        if isinstance(v, dict) and "run3" in v:
            print(f"{k:22s} run>=3 bal-acc {v['run3']['mlp']['balanced_accuracy']:.3f} (linear {v['run3']['linear']['balanced_accuracy']:.3f}) | run_z R2 {v['run_z']['mlp']['r2']:.3f}")
    print("verdict:", out["verdict"])


if __name__ == "__main__":
    main()
