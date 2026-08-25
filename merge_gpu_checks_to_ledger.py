"""Merge the reseeded day3_norm_check.py + s_class_expanded_check.py outputs
into results_ledger.json. Refreshes norm_check.* (values shift slightly now
that the untrained-adapter baseline is seeded -- expected, not a bug), adds
the Wilcoxon comparison, and adds the N=63 s_class.* entries.

Run (from repo root):
    python merge_gpu_checks_to_ledger.py
"""
import json

from render_ledger import source_run_date

LEDGER_PATH = "results_ledger.json"
NORM_CHECK_RESULTS = "results/day3_norm_check_results.json"
S_CLASS_RESULTS = "results/s_class_expanded_results.json"

# Dated from the source result files rather than hardcoded, so re-running either
# check refreshes its own ledger entries' provenance and re-running this merge
# alone does not. Both groups below are dated from the file each reads.
NORM_CHECK_RUN_DATE = source_run_date(NORM_CHECK_RESULTS)
S_CLASS_RUN_DATE = source_run_date(S_CLASS_RESULTS)


def update_norm_check(ledger: dict, data: dict) -> None:
    for group in ("real_vocab", "virtual_init", "virtual_trained"):
        g = data[group]
        for stat in ("mean", "std"):
            key = f"norm_check.{group}.{stat}"
            ledger[key]["value"] = round(g[stat], 4)
            ledger[key]["run"] = NORM_CHECK_RUN_DATE
        ledger[f"norm_check.{group}.n"]["value"] = g["n"]
        ledger[f"norm_check.{group}.n"]["run"] = NORM_CHECK_RUN_DATE

    trained_mean = data["virtual_trained"]["mean"]
    init_mean = data["virtual_init"]["mean"]
    vocab_mean = data["real_vocab"]["mean"]
    ledger["norm_check.trained_vs_init.ratio"]["value"] = round(trained_mean / init_mean, 1)
    ledger["norm_check.trained_vs_init.ratio"]["run"] = NORM_CHECK_RUN_DATE
    ledger["norm_check.trained_vs_real_vocab.ratio"]["value"] = round(trained_mean / vocab_mean, 1)
    ledger["norm_check.trained_vs_real_vocab.ratio"]["run"] = NORM_CHECK_RUN_DATE

    w = data["comparison"]["init_vs_trained_wilcoxon"]
    ledger["norm_check.comparison.init_vs_trained_wilcoxon"] = {
        "value": w["p_value"], "format": ".2e",
        "statistic": w["statistic"], "n": w["n"], "interpretation": w["interpretation"],
        "source": "day3_norm_check.py", "run": NORM_CHECK_RUN_DATE,
        "note": ("paired Wilcoxon signed-rank, virtual_init vs virtual_trained, n=320 matched "
                 "positions (80 held-out events x 4 virtual tokens). p_value=0.0 is scipy's exact "
                 "float output at this n given the effect size, not a rounded display value -- "
                 "report as p < 1e-4 in prose rather than the literal 0.0."),
    }


def update_s_class_n63(ledger: dict, data: dict) -> None:
    n = data["n"]
    clf, probe = data["classifier"], data["probe"]
    ledger["s_class.classifier_recall_n63"] = {
        "value": round(clf["recall"], 1), "unit": "%", "n": n, "correct": clf["correct"],
        "source": "s_class_expanded_check.py", "run": S_CLASS_RUN_DATE,
        "note": ("expands the original class-matched N=20 check (s_class.classifier_recall_matched, "
                 "30.0%) with 43 additional S-class DS2 events; kept as a separate key rather than "
                 "overwriting the N=20 entry -- both are cited in the Results prose."),
    }
    ledger["s_class.classifier_recall_n63.ci_wilson95.lower"] = {
        "value": clf["ci_wilson95"][0], "unit": "%", "source": "s_class_expanded_check.py",
        "run": S_CLASS_RUN_DATE, "n": n,
    }
    ledger["s_class.classifier_recall_n63.ci_wilson95.upper"] = {
        "value": clf["ci_wilson95"][1], "unit": "%", "source": "s_class_expanded_check.py",
        "run": S_CLASS_RUN_DATE, "n": n,
    }
    ledger["s_class.probe_recall_n63"] = {
        "value": round(probe["recall"], 1), "unit": "%", "n": n, "correct": probe["correct"],
        "source": "s_class_expanded_check.py", "run": S_CLASS_RUN_DATE,
        "note": ("combines the original 20's existing CV out-of-fold predictions with 43 new "
                 "predictions from a single probe fit once on the original 80-event set -- two "
                 "distinct held-out mechanisms, see script docstring. Expands "
                 "s_class.probe_recall_matched (N=20, 60.0%)."),
    }
    ledger["s_class.probe_recall_n63.ci_wilson95.lower"] = {
        "value": probe["ci_wilson95"][0], "unit": "%", "source": "s_class_expanded_check.py",
        "run": S_CLASS_RUN_DATE, "n": n,
    }
    ledger["s_class.probe_recall_n63.ci_wilson95.upper"] = {
        "value": probe["ci_wilson95"][1], "unit": "%", "source": "s_class_expanded_check.py",
        "run": S_CLASS_RUN_DATE, "n": n,
    }
    ledger["s_class.n63.cis_overlap"] = {
        "value": str(data["cis_overlap"]), "source": "s_class_expanded_check.py", "run": S_CLASS_RUN_DATE,
        "note": "whether the classifier and probe Wilson 95% CIs overlap at N=63 (also true at N=20)",
    }


def main():
    with open(NORM_CHECK_RESULTS, encoding="utf-8") as f:
        norm_data = json.load(f)
    with open(S_CLASS_RESULTS, encoding="utf-8") as f:
        s_class_data = json.load(f)
    with open(LEDGER_PATH, encoding="utf-8") as f:
        ledger = json.load(f)

    update_norm_check(ledger, norm_data)
    update_s_class_n63(ledger, s_class_data)

    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)
    print("Ledger updated.")
    for k in sorted(k for k in ledger if k.startswith("norm_check") or k.startswith("s_class")):
        print(f"  {k} = {ledger[k]['value']}")


if __name__ == "__main__":
    main()
