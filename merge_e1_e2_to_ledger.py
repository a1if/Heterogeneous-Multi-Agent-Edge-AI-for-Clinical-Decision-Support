"""Merge E1 (k ablation) and E2 (seed variance) results into results_ledger.json.

Reads results/e1_ablation_results.json and results/e2_seed_variance_results.json
(both written by e1_ablation.py / e2_seed_variance.py) and writes per-point and
aggregate ledger entries. Programmatic, not hand-typed, per the plan's "every
number traceable to a script" rule -- avoids transcription errors on ~30 values.

Run (from repo root):
    python merge_e1_e2_to_ledger.py
"""
import json

from render_ledger import source_run_date

LEDGER_PATH = "results_ledger.json"
E1_RESULTS = "results/e1_ablation_results.json"
E2_RESULTS = "results/e2_seed_variance_results.json"
DAY6_RESULTS = "results/day6_results.json"

# "run" is dated per-entry from whichever result file that entry's numbers actually
# came from, mirroring the existing per-entry `source` selection below: the
# non-retrained (k=4 / existing-checkpoint) points reuse day6_results.json rather
# than the ablation output, so they carry day6's date, not e1/e2's. Previously a
# single hardcoded RUN_DATE was stamped on all of them, which mis-dated those
# reused points whenever the Day 6 run and the ablation runs happened on
# different days -- as they did.


def e1_entries(e1: dict) -> dict:
    entries = {}
    for k_str, r in e1["by_k"].items():
        k = int(k_str)
        prefix = f"e1.k{k}"
        source = "day6_run_comparison.py Arm B (results/day6_results.json)" if not r["retrained"] \
            else "e1_ablation.py"
        run_date = source_run_date(DAY6_RESULTS if not r["retrained"] else E1_RESULTS)
        note = (f"k=4 point reuses the existing headline checkpoint, not retrained -- "
                f"same data as armB.*.n80") if not r["retrained"] else None
        for field, key_suffix, unit in [
            ("accuracy", "accuracy", "%"), ("tokens_mean", "tokens_mean", "tok"),
            ("gen_ms_mean", "gen_ms_mean", "ms"), ("vram_mean", "vram_mean", "MB"),
        ]:
            value = r[field] * 100 if field == "accuracy" else r[field]
            entry = {"value": round(value, 1), "unit": unit, "n": r["n"], "n_failed": r["n_failed"],
                      "source": source, "run": run_date, "checkpoint": r["checkpoint"]}
            if note:
                entry["note"] = note
            entries[f"{prefix}.{key_suffix}"] = entry
    return entries


def e2_entries(e2: dict) -> dict:
    entries = {}
    for seed_key, r in e2["by_seed"].items():
        prefix = f"e2.seed_{seed_key}"
        source = "day6_run_comparison.py Arm B (results/day6_results.json)" if not r["retrained"] \
            else "e2_seed_variance.py"
        run_date = source_run_date(DAY6_RESULTS if not r["retrained"] else E2_RESULTS)
        note = ("pre-dates explicit seed control (unrecorded init); same data as armB.*.n80"
                if not r["retrained"] else None)
        for field, key_suffix, unit in [
            ("accuracy", "accuracy", "%"), ("tokens_mean", "tokens_mean", "tok"),
            ("gen_ms_mean", "gen_ms_mean", "ms"), ("vram_mean", "vram_mean", "MB"),
        ]:
            value = r[field] * 100 if field == "accuracy" else r[field]
            entry = {"value": round(value, 1), "unit": unit, "n": r["n"], "n_failed": r["n_failed"],
                      "source": source, "run": run_date, "checkpoint": r["checkpoint"]}
            if note:
                entry["note"] = note
            entries[f"{prefix}.{key_suffix}"] = entry

    agg = e2["aggregate"]
    entries["e2.aggregate.accuracy_mean"] = {
        "value": round(agg["accuracy_mean"] * 100, 1), "unit": "%",
        "source": "e2_seed_variance.py", "run": source_run_date(E2_RESULTS), "n_seeds": agg["n_seeds"],
    }
    entries["e2.aggregate.accuracy_std"] = {
        "value": round(agg["accuracy_std"] * 100, 1), "unit": "pp",
        "source": "e2_seed_variance.py", "run": source_run_date(E2_RESULTS), "n_seeds": agg["n_seeds"],
    }
    entries["e2.aggregate.tokens_mean"] = {
        "value": round(agg["tokens_mean"], 1), "unit": "tok",
        "source": "e2_seed_variance.py", "run": source_run_date(E2_RESULTS), "n_seeds": agg["n_seeds"],
    }
    entries["e2.aggregate.tokens_std"] = {
        "value": round(agg["tokens_std"], 1), "unit": "tok",
        "source": "e2_seed_variance.py", "run": source_run_date(E2_RESULTS), "n_seeds": agg["n_seeds"],
    }
    entries["e2.aggregate.gen_ms_mean"] = {
        "value": round(agg["gen_ms_mean"], 1), "unit": "ms",
        "source": "e2_seed_variance.py", "run": source_run_date(E2_RESULTS), "n_seeds": agg["n_seeds"],
    }
    entries["e2.aggregate.gen_ms_std"] = {
        "value": round(agg["gen_ms_std"], 1), "unit": "ms",
        "source": "e2_seed_variance.py", "run": source_run_date(E2_RESULTS), "n_seeds": agg["n_seeds"],
    }
    return entries


def main():
    with open(E1_RESULTS, encoding="utf-8") as f:
        e1 = json.load(f)
    with open(E2_RESULTS, encoding="utf-8") as f:
        e2 = json.load(f)
    with open(LEDGER_PATH, encoding="utf-8") as f:
        ledger = json.load(f)

    new_entries = {**e1_entries(e1), **e2_entries(e2)}
    ledger.update(new_entries)

    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)

    print(f"Merged {len(new_entries)} entries into {LEDGER_PATH}:")
    for key in sorted(new_entries):
        print(f"  {key} = {new_entries[key]['value']}{new_entries[key].get('unit', '')}")


if __name__ == "__main__":
    main()
