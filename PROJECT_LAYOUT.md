# Project layout

## Packages and data (unchanged)
- `perception/` — CNN-LSTM Perception Agent (model, checkpoint, schema).
- `reasoning/` — Arm A/B reasoning pipeline (baseline arm, adapter arm, prompts, checkpoints).
- `data/` — MIT-BIH raw records (`mitdb/`) and processed DS1/DS2 splits (`processed/`).
- `tests/` — pytest suite (`conftest.py` at root adds the repo root to `sys.path` for this).

## Active pipeline scripts — deliberately flat at repo root
`data_prep.py`, `train_perception_agent.py`, `day6_run_comparison.py`,
`day7_auditability_probe.py`, `day3_norm_check.py`, `e0_statistics.py`,
`perception_eval.py`, `s_class_matched_check.py`, `render_ledger.py`,
plus (added after this doc was first written, same constraint applies):
`ablation_common.py`, `e1_ablation.py`, `e2_seed_variance.py`,
`merge_e1_e2_to_ledger.py`, `merge_gpu_checks_to_ledger.py`,
`run_gpu_checks_combined.py`, `s_class_expanded_check.py`,
`check_s_class_per_record.py`, `check_s_class_headroom.py`,
`find_additional_s_events.py`, `export_norm_check_raw.py`.

These all do bare `from perception.X import Y` / `from reasoning.X import Y` imports,
which only resolve because Python adds a directly-invoked script's own directory to
`sys.path[0]` — i.e. they work *because* they sit next to the `perception/`/`reasoning/`
packages. Moving them into a subfolder would break every one of these imports project-wide
(verified while organizing this folder). Not worth the risk this close to the Day 3+ GPU
runs, so they stay put. **Always run them from the repo root** (`python <name>.py`), same
as their docstrings say.

`day7_auditability_probe.py` is bare-imported (`from day7_auditability_probe import
select_events`) by SIX other root scripts — `ablation_common.py`, `day3_norm_check.py`,
`find_additional_s_events.py`, `export_norm_check_raw.py`, `s_class_expanded_check.py`,
`s_class_matched_check.py` — so it can never move without updating all six.
`ablation_common.py` is likewise bare-imported by `e1_ablation.py` and
`e2_seed_variance.py` (and itself imports from `day7_auditability_probe.py`, chaining
the same constraint). `run_gpu_checks_combined.py` goes one step further and does
`import s_class_expanded_check` / `import day3_norm_check` as bare module imports
(to load Gemma once instead of twice across both checks) — another reason those two
can't move either.

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
`s_class_per_record_results.json`. Producing scripts' `RESULTS_PATH` constants
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
Expensive-to-recompute intermediate `.npy` arrays, read/written by
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

## `diagnostics/`
One-off manual verification scripts, never imported by anything else and not
part of pytest's collection (`smoke_test_*.py` doesn't match pytest's default
`test_*.py` pattern by design — these are meant to be read by a human, not run
in CI): the four `smoke_test_*.py` files, `diagnose_arm_a_disagreements.py`,
`check_adapter_collapse.py`.

Each got a `sys.path.insert(0, ...)` shim added at the top (same pattern
`conftest.py` already used for pytest) so their `perception.*`/`reasoning.*`
imports still resolve from the new location. Verified by static import check
(no GPU needed — only `__main__`-guarded code needs a GPU/model).
**Run from the repo root**, e.g. `python diagnostics/check_adapter_collapse.py`.

RESOLVED (was: "known pre-existing issue"): `diagnose_structured_output_overhead.py`
imported `MODEL_TAG` from `reasoning.baseline_arm`, which doesn't define it. Confirmed
dead code from the pre-HuggingFace Ollama era — the whole script targets the removed
Ollama path, and its `GENERATION_CONFIG` is now HF-shaped so it would be silently wrong
even with a valid tag. Moved to `archive/` with a header explaining both breakages.

## `docs/`
`7day_prototype_design_doc.docx` — pure documentation, no code depends on it.

## `archive/`
`day7_auditability_probe_pre-v2.1_draft.py` — confirmed-stale earlier draft of
`day7_auditability_probe.py` (diffed; superseded by the v2.1 scaling/StandardScaler/
Wilcoxon fixes). Kept for reference rather than deleted outright.

`diagnose_structured_output_overhead.py` — Ollama-era diagnostic, dead since the
HuggingFace rewrite of `reasoning/baseline_arm.py`. Does not run (ImportError on
`MODEL_TAG`); the latency question it measured is moot because the HF path uses
retry-parse rather than grammar-constrained structured output. See its own header.

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
