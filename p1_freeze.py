"""Deviation 21: freeze manifest for the release r4-confirmatory (docs/confirmatory_plan.md, section 1). CPU.

Records the SHA-256 of every checkpoint and code file the INCART run depends on, the frozen calibration
biases (chosen on DS1 validation only), package versions and the git commit, then refuses to run if the
working tree has uncommitted changes to any of those files. After it, commit the manifest and tag:
    git add results/p1_freeze_manifest.json && git commit -m "Freeze r4-confirmatory" && git tag r4-confirmatory

Run (from repo root):
    python p1_freeze.py
"""
import hashlib
import json
import subprocess
from pathlib import Path

from p1_io import save_json_atomic

OUT = Path("results/p1_freeze_manifest.json")
CHECKPOINTS = ["perception/checkpoints/cnn_lstm_rr_seed0.pt"] + [
    f"reasoning/checkpoints/p1_item7_mea_r4_seed{s}.pt" for s in (101, 202, 303)]
CODE = ["incart_prep.py", "data_prep.py", "p1_incart_windows.py", "p1_confirmatory_analysis.py", "p1_item7_eval.py",
        "p1_item7_filtered.py", "p1_item7_common.py", "p1_item7_testset_v2.py", "p1_pilot2_stratified.py",
        "p1_pilot_multi_event.py", "p1_item7_tier1.py", "p1_item7_calib.py", "p1_item7_ttd.py", "p1_io.py",
        "perception/perception_agent.py", "perception/model_rr.py", "perception/model.py", "perception/rr_features.py",
        "reasoning/constrained_json.py", "reasoning/multi_event_adapter.py", "reasoning/virtual_adapter.py",
        "reasoning/prompt_template.py", "reasoning/prompt_template_family.py", "reasoning/fixed_context_neutral.py",
        "reasoning/training_targets.py", "reasoning/output_schema.py", "reasoning/model_loader.py",
        "docs/confirmatory_plan.md"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    dirty = subprocess.run(["git", "status", "--porcelain", "--"] + CODE, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"uncommitted changes in frozen files, commit first:\n{dirty}")
    import torch
    import transformers
    calib = json.loads(Path("results/p1_item7_calib.json").read_text(encoding="utf-8"))
    filt = json.loads(Path("results/p1_item7_filtered.json").read_text(encoding="utf-8"))
    manifest = {
        "release": "r4-confirmatory", "analysis_plan": "docs/confirmatory_plan.md (Deviation 21)",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "calibration_deltas": {"A-compact": calib["deltas"]["A-compact"]["primary_rule"],
                               "A-filtered": filt["calibration"]["delta"]},
        "checkpoints": {p: sha(p) for p in CHECKPOINTS},
        "code": {p: sha(p) for p in CODE},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "decoding": {"max_new_tokens": 128, "field_cap": 40, "greedy": True},
    }
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    save_json_atomic(OUT, manifest)
    print(json.dumps({k: manifest[k] for k in ("git_commit", "calibration_deltas", "manifest_sha256")}, indent=1))


if __name__ == "__main__":
    main()
