"""Save every Phase 1 report table and figure in results_ledger.json (keys p1.table.NN, p1.figure.NN). CPU.

The tables and figures are captured exactly as the report builder creates them, so the ledger matches the
report: run the builder with LEDGER_OUT set, then this script.

    LEDGER_OUT=<export.json> node reports/make_report.js        (NODE_PATH pointing at the docx package)
    python p1_export_ledger.py <export.json>

Each table entry: report caption, section, column headers, rows (values as printed in the report, i.e.
rounded; full precision is in the source files), source results files, type (results / qualitative /
reference), and "run" = newest modification date of its source files (the date of the experimental run,
as in render_ledger.source_run_date). Each figure entry: image file, caption, section, the function in
reports/make_figures.py that draws it, and its source files. Existing ledger keys are never modified;
p1.table.* and p1.figure.* keys are replaced on every run. A backup is written first.
"""
import datetime
import json
import re
import shutil
import sys
from pathlib import Path

LEDGER = Path("results_ledger.json")
REPORT = "reports/Phase1_Technical_Report.docx"

TABLE_SOURCES = {  # report table number -> (type, source files)
    1: ("reference", ["results_ledger.json (dissertation keys armA.* / armB.*, n80)"]),
    2: ("qualitative", ["docs/analysis_plan.md"]),
    3: ("results", ["results/p1_rr_seeds.json"]),
    4: ("results", ["results/p1_e3_multi_event.json", "results/p1_e3b_context_costs.json", "results/p1_item7_ttd.json"]),
    5: ("results", ["results/p1_serving.json"]),
    6: ("results", ["results/p1_item7_r4_analysis.json", "results/p1_item7_calib.json"]),
    7: ("results", ["results/p1_item7_r4_analysis.json"]),
    8: ("results", ["results/p1_item7_r4_analysis.json"]),
    9: ("results", ["results/p1_item7_filtered.json"]),
    10: ("results", ["results/p1_second_sender_analysis.json"]),
    11: ("results", ["results/p1_dev26_ablation.json"]),
    12: ("results", ["results/p1_dev27_kseeds.json", "results/p1_sweep_analysis.json"]),
    13: ("qualitative", ["docs/analysis_plan.md"]),
    14: ("reference", ["docs/analysis_plan.md"]),
}
FIGURE_SOURCES = {  # image file -> (make_figures.py function, source files)
    "fig1_pipeline.png": ("fig_pipeline", []),
    "fig2_latency.png": ("fig_latency", ["results/p1_item7_ttd.json", "results/p1_item7_r4_analysis.json"]),
    "fig3_accuracy_vs_n.png": ("fig_accuracy", ["results/p1_item7_r4_analysis.json"]),
    "fig4_forest.png": ("fig_forest", ["results/p1_item7_r4_analysis.json", "results/p1_item7_calib.json"]),
    "fig5_why_text_fails.png": ("fig_why_text_fails", ["results/p1_item7_eval_ds2v2_r4.json"]),
    "fig6_confusion_n50.png": ("fig_confusion", ["results/p1_item7_eval_ds2v2_r4.json"]),
    "fig7_training_r4.png": ("fig_training", [f"results/p1_item7_train_r4_seed{s}.json" for s in (101, 202, 303)]),
    "fig8_auditability.png": ("fig_audit", ["results/p1_item7_slot_decoder.json"]),
    "fig9_false_alarms.png": ("fig_false_alarms", ["results/p1_item7_r4_analysis.json"]),
    "fig10_side_info.png": ("fig_side_info", ["results/p1_item7_r4_analysis.json", "results/p1_item7_runlen.json"]),
    "fig11_two_senders.png": ("fig_two_senders", ["results/p1_item7_r4_analysis.json", "results/p1_item7_filtered.json",
                                                  "results/p1_second_sender_analysis.json"]),
    "fig12_compression.png": ("fig_compression", ["results/p1_item7_eval_ds2v2_r4.json", "results/p1_item7_eval_ds2v2_sweep.json",
                                                  "results/p1_sweep_analysis.json"]),
}


def run_date(paths):
    times = [Path(p).stat().st_mtime for p in paths if Path(p).exists()]
    return datetime.date.fromtimestamp(max(times)).isoformat() if times else None


def number(caption, kind):
    m = re.match(rf"{kind} (\d+)\.", caption or "")
    return int(m.group(1)) if m else None


def main(export_path):
    export = json.loads(Path(export_path).read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    backup = LEDGER.with_name(LEDGER.name + ".bak_pre_p1_tables")
    if not backup.exists():
        shutil.copy2(LEDGER, backup)
    ledger = {k: v for k, v in ledger.items() if not k.startswith(("p1.table.", "p1.figure."))}
    note = "values as printed in the report (rounded); full precision in the source files"
    for t in export["tables"]:
        n = number(t["caption"], "Table")
        if n is None:
            raise SystemExit(f"table without a numbered caption in section {t['section']}")
        kind, sources = TABLE_SOURCES[n]
        ledger[f"p1.table.{n:02d}"] = {
            "caption": t["caption"], "section": t["section"], "subsection": t["subsection"], "type": kind,
            "columns": t["rows"][0], "rows": t["rows"][1:], "source": sources,
            "run": run_date([s for s in sources if s.startswith("results/")]), "report": REPORT, "note": note}
    for f in export["figures"]:
        n = number(f["caption"], "Figure")
        name = Path(f["file"]).name
        func, sources = FIGURE_SOURCES[name]
        ledger[f"p1.figure.{n:02d}"] = {
            "file": f["file"], "caption": f["caption"], "section": f["section"], "subsection": f["subsection"],
            "script": f"reports/make_figures.py:{func}", "source": sources, "run": run_date(sources), "report": REPORT}
    LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n_t = sum(k.startswith("p1.table.") for k in ledger)
    n_f = sum(k.startswith("p1.figure.") for k in ledger)
    print(f"results_ledger.json: {n_t} tables, {n_f} figures (p1.*); {len(ledger)} keys in all; backup {backup}")


if __name__ == "__main__":
    main(sys.argv[1])
