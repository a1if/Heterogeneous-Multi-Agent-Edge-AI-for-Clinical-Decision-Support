"""Pre-flight check for the Deviation 21 confirmatory run, BEFORE the freeze. No INCART data is touched.

Builds a throw-away sandbox copy of the code, a stand-in "incart_test.npz" made from a few MIT-BIH DS2
records (same arrays and keys as incart_prep.py writes, patient_ids grouping records in pairs), and runs
every step of scripts/run_incart_prep.sh and scripts/run_incart.sh on it with a handful of windows:
replay, window builder, adapter generation (full), text generation (--stop-at-tier), tier logits for both
text arms, timing, and the fixed-sequence confirmatory analysis (with a stand-in freeze manifest).
Real results/ and data/ are never written. Outputs here are about DS2 records and are not inspected.

Run (from repo root):  python scripts/preflight_incart.py
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BOX = Path(os.environ.get("PREFLIGHT_DIR", REPO / "tmp" / "preflight_incart"))
PY = str(REPO / "venv" / "Scripts" / "python.exe")
CKPTS = ["perception/checkpoints/cnn_lstm_rr_seed0.pt"] + [
    f"reasoning/checkpoints/p1_item7_mea_r4_seed{s}.pt" for s in (101, 202, 303)]
RESULTS = ["results/p1_item7_calib.json", "results/p1_item7_filtered.json"]
PER_CELL = 2  # windows kept per (set, N, reference) cell


def build_box():
    if BOX.exists():
        shutil.rmtree(BOX)
    for f in REPO.glob("*.py"):
        (BOX / f.name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, BOX / f.name)
    for pkg in ("perception", "reasoning"):
        for f in (REPO / pkg).glob("*.py"):
            (BOX / pkg).mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, BOX / pkg / f.name)
    for rel in CKPTS + RESULTS:
        (BOX / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, BOX / rel)
    for d in ("logs", "data/processed", "cache"):
        (BOX / d).mkdir(parents=True, exist_ok=True)
    # stand-in INCART file: 6 DS2 records, patients = records in pairs
    with np.load(REPO / "data/processed/ds2_test.npz") as z:
        d = {k: z[k] for k in z.files}
    recs = sorted(set(d["record_ids"].tolist()))[:6]
    keep = np.isin(d["record_ids"], recs)
    out = {k: v[keep] for k, v in d.items()}
    out["patient_ids"] = np.array([recs.index(r) // 2 + 1 for r in out["record_ids"]])
    np.savez(BOX / "data/processed/incart_test.npz", **out)
    print(f"sandbox {BOX}: {keep.sum()} beats from DS2 records {recs}")


def run(name, args):
    print(f"--- {name}", flush=True)
    env = {k: v for k, v in os.environ.items() if k not in ("P1_ENCODER", "P1_ENCODER_TAG")}  # main sender
    p = subprocess.run([PY, "-u"] + args, cwd=BOX, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**env, "PYTHONIOENCODING": "utf-8"})
    (BOX / "logs" / f"{name}.log").write_text(p.stdout + p.stderr, encoding="utf-8")
    if p.returncode != 0:
        print((p.stdout + p.stderr)[-3000:])
        raise SystemExit(f"PREFLIGHT FAILED at {name} (exit {p.returncode})")
    print("    ok")


def shrink_windows():
    path = BOX / "results/p1_item7_testset_incart.json"
    v = json.loads(path.read_text(encoding="utf-8"))
    kept, count = [], {}
    for w in v["windows"]:
        key = (w["set"], w["n"], w["reference"])
        if count.get(key, 0) < PER_CELL:
            kept.append(w)
            count[key] = count.get(key, 0) + 1
    v["windows"] = kept
    path.write_text(json.dumps(v), encoding="utf-8")
    print(f"    windows kept: {len(kept)} ({len(count)} cells, built {v['n_windows']}, patients {v['patients_covered']})")


def fake_manifest():
    calib = json.loads((BOX / "results/p1_item7_calib.json").read_text(encoding="utf-8"))
    filt = json.loads((BOX / "results/p1_item7_filtered.json").read_text(encoding="utf-8"))
    m = {"release": "preflight", "calibration_deltas": {"A-compact": calib["deltas"]["A-compact"]["primary_rule"],
                                                        "A-filtered": filt["calibration"]["delta"]}, "manifest_sha256": "preflight"}
    (BOX / "results/p1_freeze_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    print(f"    stand-in manifest deltas {m['calibration_deltas']} (read the same way as p1_freeze.py)")


def main():
    build_box()
    run("replay", ["-c", "from p1_item7_common import replay_split; r = replay_split('incart'); print(len(r['events']))"])
    run("windows", ["p1_incart_windows.py"])
    shrink_windows()
    run("eval_mea", ["p1_item7_eval.py", "--split", "incart", "--arms", "MEA:r4_seed101", "MEA:r4_seed202", "MEA:r4_seed303"])
    run("eval_text_stop", ["p1_item7_eval.py", "--split", "incart", "--stop-at-tier", "--arms", "A-compact", "A-filtered"])
    run("collect_compact", ["p1_item7_filtered.py", "collect", "--text-arm", "A-compact", "--split", "incart"])
    run("collect_filtered", ["p1_item7_filtered.py", "collect", "--text-arm", "A-filtered", "--split", "incart"])
    run("timing", ["p1_item7_filtered.py", "timing", "--split", "incart"])
    fake_manifest()
    run("confirmatory", ["p1_confirmatory_analysis.py"])
    # structural checks only (values are about DS2 records and are not inspected)
    ev = json.loads((BOX / "results/p1_item7_eval_incart.json").read_text(encoding="utf-8"))["rows"]
    arms = {}
    for x in ev:
        arms.setdefault(x["arm"], []).append(x)
    for a, xs in sorted(arms.items()):
        parsed = sum(x["parsed"] for x in xs)
        stopped = sum(bool(x.get("stopped_at_tier")) for x in xs)
        print(f"    {a:16s} rows {len(xs):3d}  parsed {parsed:3d}  stopped_at_tier {stopped:3d}  field_cap None {sum(x['hit_field_cap'] is None for x in xs)}")
    out = json.loads((BOX / "results/p1_confirmatory_incart.json").read_text(encoding="utf-8"))
    print("    confirmatory output keys:", sorted(out), "| hypotheses:", {h: d["status"] for h, d in out["hypotheses"].items()})
    print("    field_cap_rate reported for:", sorted(out["field_cap_rate"]))
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    sys.exit(main())
