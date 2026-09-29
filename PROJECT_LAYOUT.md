# Project layout

## Packages and data (unchanged)
- `perception/` — CNN-LSTM Perception Agent (model, checkpoint, schema).
- `reasoning/` — Arm A/B reasoning pipeline (baseline arm, adapter arm, prompts, checkpoints).
- `data/` — MIT-BIH raw records (`mitdb/`) and processed DS1/DS2 splits (`processed/`).
- `tests/` — pytest suite (`conftest.py` at root adds the repo root to `sys.path` for this).

## Scripts at the repository root

Phase 1 (journal extension) scripts: `p1_*.py`. Shared infrastructure that current
code still imports: `project_config.py`, `p1_io.py`, `ablation_common.py`,
`day7_auditability_probe.py`, `measure_comm_cost.py`, `data_prep.py`,
`train_perception_agent.py`, `train_perception_agent_rr.py`, `perception_eval.py`,
`render_ledger.py`.

These use bare `from perception.X import Y` / `from reasoning.X import Y` imports, which
resolve because Python puts a directly invoked script's own directory on `sys.path[0]`.
**Run them from the repository root** (`python <name>.py`).

`day7_auditability_probe.py` and `ablation_common.py` date from the dissertation. They
stay at the root because current code and the archived scripts both import them.

## `archive/dissertation/`

Scripts that produced the submitted dissertation's results, moved here unchanged on
2026-09-29: the day3–day7 checks, E0–E4b, S-class checks, the quantisation pilot, the
aux-objective run, the ledger merges and the headline figures. Run them from the root
with the root on the import path (`PYTHONPATH=. python archive/dissertation/<name>.py`).
Sibling imports between archived scripts resolve through `sys.path[0]`. See
`archive/dissertation/README.md` for the list and for what was deleted.

## `project_config.py` — shared constants and helpers
Single home for the literals and helpers that used to be re-typed in every script:
`DS1_PATH`/`DS2_PATH`, `PERCEPTION_CHECKPOINT`, `ADAPTER_CHECKPOINT`, `DAY6_RESULTS`,
`LEDGER_PATH`, `PER_CLASS`/`MAX_PER_RECORD`, and the `select_events`, `measure_vram`
and `wilson_ci` helpers. Before this, `data/processed/ds2_test.npz` appeared in 19
files, `perception/checkpoints/cnn_lstm.pt` in 17, and `AAMI_CLASSES` was redeclared
in 6 despite already existing in `perception/model.py`.

The important one was `select_events`. It had TWO independent definitions —
`day6_run_comparison.py` and `day7_auditability_probe.py` — with six other scripts
importing the day7 copy while day6 used its own. Every one of those scripts claims to
score "the same 80 events", but nothing enforced it: editing either copy would have
broken that guarantee silently. Both were verified to return byte-identical output
before being merged, so the consolidation preserves behaviour exactly and removes the
drift risk. `day7_auditability_probe.select_events` is now a re-export, so the existing
`from day7_auditability_probe import select_events` in six scripts still resolves.

**Scope is deliberately limited to root and archived scripts.** The
`perception/`/`reasoning/` packages do NOT import from it — a package depending on a
repo-root module inverts the dependency direction and would break if those packages
were imported from outside this checkout. `AAMI_CLASSES` is likewise NOT re-exported
from `project_config`; it stays in `perception.model` and is imported from there, so
this module removes a duplicate rather than becoming a seventh copy (and importing it
here would drag `torch` into CPU-only scripts).

Deliberate local overrides survive: `archive/dissertation/check_adapter_collapse.py` keeps its
own `PER_CLASS = 3`, because the migration only replaced constants whose value matched
the canonical one exactly.

## `results_ledger.json`, `render_ledger.py`, `methodology_footnote_e0.md`
Also stay at root — this is the dissertation "single source of numeric truth"
infrastructure (see `7day_dissertation_sprint_plan.md` §2.1), referenced by
its literal root-relative path from every script and worth keeping undisturbed.

## `results/`
Raw JSON outputs from the scripts above: `day6_results.json`,
`day7_auditability_results_v2.json`, `perception_eval_results.json`,
`s_class_matched_check_results.json`, `arm_a_disagreement_diagnosis.json`,
`e1_ablation_results.json`, `e2_seed_variance_results.json`,
`day3_norm_check_results.json`, `s_class_expanded_results.json`,
`s_class_per_record_results.json`, `class_heading_ablation_results.json`, and
`quantization_coupling_pilot_results.json`. Producing scripts' `RESULTS_PATH` constants
point here; nothing else reads these files directly except `results_ledger.json`'s
`source` field (documentation only).

The last three above were found loose at repo root during this pass (their
`RESULTS_PATH` constants were plain filenames with no `results/` prefix) and were
moved here, updating `day3_norm_check.py`, `s_class_expanded_check.py`,
`check_s_class_per_record.py`, and the two constants in `merge_gpu_checks_to_ledger.py`
that read them (`NORM_CHECK_RESULTS`, `S_CLASS_RESULTS`) to match. These are plain
string paths resolved relative to cwd (repo root), not `__file__`-relative, so this
move carries none of the sys.path[0] risk the scripts themselves do.

## `cache/`
Expensive-to-recompute intermediate `.npy` arrays, read/written by the archived
`s_class_expanded_check.py`, `check_s_class_per_record.py`, and
`find_additional_s_events.py`: `s_class_per_event_original20.npy`,
`s_class_per_event_new43.npy`, `s_class_adapter_vectors_80_cache.npy`,
`s_class_adapter_vectors_new43_cache.npy`, `additional_s_class_indices.npy`.
Also found loose at root and moved here, with the corresponding path constants
in those three scripts updated to match.

## `logs/`
Captured stdout/stderr from manual runs: `e1_e2_run.log`, `e1_seeded_run.log`,
`gpu_checks_run.log`, `s_class_rerun.log`. Confirmed (grep) that nothing reads
these programmatically — pure human-readable run history, moved here from root
with no code changes needed.

## `docs/`
`7day_prototype_design_doc.docx` — pure documentation, no code depends on it.

## A bug found and fixed while organizing this
Several scripts opened files without an explicit `encoding="utf-8"`. On this
system the platform-default text encoding is not UTF-8, so any non-ASCII
character (em dashes, `α`, `§` — all present in real dissertation prose)
gets silently corrupted on a read-modify-write round trip. This wasn't
theoretical: it corrupted `§` in `results_ledger.json`'s `armA/B.*.n60`
provenance notes the moment `e0_statistics.py` re-wrote the ledger. Repaired,
and `encoding="utf-8"` added to every `open()` across the active scripts and
`render_ledger.py` (the one that matters most, since it processes real
chapter prose before submission).
