"""E0: paired significance tests on the existing 80-event Day 6 comparison.

McNemar's exact test (one-sided) on paired accuracy, Wilcoxon signed-rank on
paired per-event token counts and generation durations, and exact/Wilson 95%
CIs on each arm's accuracy. See 7day_dissertation_sprint_plan.md §3 (E0) for
the rationale and expected values.

Run:
    python e0_statistics.py
"""
import json

from scipy.stats import beta as beta_dist
from scipy.stats import binomtest, wilcoxon
from statsmodels.stats.proportion import proportion_confint

from render_ledger import source_run_date
from project_config import DAY6_RESULTS

RESULTS_PATH = DAY6_RESULTS
LEDGER_PATH = "results_ledger.json"


def load_paired(results):
    by_idx = {}
    for r in results:
        by_idx.setdefault(r["idx"], {})[r["arm"]] = r
    idxs = sorted(by_idx)
    a = [by_idx[i]["A"] for i in idxs]
    b = [by_idx[i]["B"] for i in idxs]
    return a, b


def mcnemar_one_sided(a, b):
    """b_type: A correct, B incorrect. c_type: A incorrect, B correct."""
    b_type = sum(1 for x, y in zip(a, b) if x["correct"] and not y["correct"])
    c_type = sum(1 for x, y in zip(a, b) if not x["correct"] and y["correct"])
    n = b_type + c_type
    if n == 0:
        # No discordant pairs: the arms agreed on every event, so McNemar has no
        # information to work with and p is 1.0 by definition. Guarded because
        # scipy's binomtest(0, 0, ...) raises ValueError rather than returning
        # this -- an all-agreement run is a perfectly plausible outcome here
        # (both arms already agree on 76/80), so this must not crash the script.
        return b_type, c_type, 1.0
    # One-sided: P(c_type or fewer c-type discordant pairs | n, p=0.5)
    p = binomtest(c_type, n, 0.5, alternative="less").pvalue
    return b_type, c_type, p


def main():
    # encoding="utf-8" everywhere below: this system's platform-default text encoding
    # is NOT UTF-8 and silently corrupts non-ASCII characters (e.g. "§" in the ledger's
    # provenance notes) on a plain read-modify-write round trip -- confirmed in practice.
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    a, b = load_paired(results)
    n = len(a)

    b_type, c_type, mcnemar_p = mcnemar_one_sided(a, b)
    print(f"McNemar discordant pairs: b(A-correct/B-wrong)={b_type}, "
          f"c(A-wrong/B-correct)={c_type}")
    print(f"McNemar one-sided exact p = {mcnemar_p:.4f}")

    a_tokens = [x["prompt_tokens"] for x in a]
    b_tokens = [x["prompt_tokens"] for x in b]
    tokens_p = wilcoxon(a_tokens, b_tokens).pvalue
    print(f"Wilcoxon (prompt tokens) p = {tokens_p:.6g}")

    a_ms = [x["generation_duration_ms"] for x in a]
    b_ms = [x["generation_duration_ms"] for x in b]
    gen_ms_p = wilcoxon(a_ms, b_ms).pvalue
    print(f"Wilcoxon (generation duration) p = {gen_ms_p:.6g}")

    a_correct = sum(x["correct"] for x in a)
    b_correct = sum(x["correct"] for x in b)
    # Arm A sits at the 100% ceiling: a two-sided exact CI degenerates (upper
    # bound is trivially 1.0), so only a one-sided 95% lower confidence bound
    # is informative. Clopper-Pearson one-sided lower bound: beta.ppf(alpha, x, n-x+1).
    a_lo = beta_dist.ppf(0.05, a_correct, n - a_correct + 1)
    a_hi = 1.0
    b_lo, b_hi = proportion_confint(b_correct, n, alpha=0.05, method="wilson")
    print(f"Arm A accuracy {a_correct}/{n} exact one-sided 95% lower bound: {a_lo*100:.1f}%")
    print(f"Arm B accuracy {b_correct}/{n} Wilson 95% CI: [{b_lo*100:.1f}%, {b_hi*100:.1f}%]")

    with open(LEDGER_PATH, encoding="utf-8") as f:
        ledger = json.load(f)

    ledger["e0.mcnemar.p_onesided"]["value"] = round(mcnemar_p, 4)
    ledger["e0.wilcoxon.tokens.p"]["value"] = tokens_p
    ledger["e0.wilcoxon.gen_ms.p"]["value"] = gen_ms_p
    ledger["e0.armA.accuracy.ci_exact95.lower"]["value"] = round(a_lo * 100, 1)
    ledger["e0.armA.accuracy.ci_exact95.upper"]["value"] = round(a_hi * 100, 1)
    ledger["e0.armB.accuracy.ci_wilson95.lower"]["value"] = round(b_lo * 100, 1)
    ledger["e0.armB.accuracy.ci_wilson95.upper"]["value"] = round(b_hi * 100, 1)
    for key in ["e0.mcnemar.p_onesided", "e0.wilcoxon.tokens.p", "e0.wilcoxon.gen_ms.p",
                "e0.armA.accuracy.ci_exact95.lower", "e0.armA.accuracy.ci_exact95.upper",
                "e0.armB.accuracy.ci_wilson95.lower", "e0.armB.accuracy.ci_wilson95.upper"]:
        # Dated from day6_results.json, not date.today(): every value here is
        # recomputed deterministically from that file, so an E0 re-run on unchanged
        # inputs must not re-date the entries and imply a fresher experiment. The
        # date was hardcoded "2026-08-09" and had already gone stale against a later
        # Day 6 re-run.
        ledger[key]["run"] = source_run_date(RESULTS_PATH)
        ledger[key]["n"] = n

    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)
    print(f"\nLedger updated: {LEDGER_PATH}")


if __name__ == "__main__":
    main()
