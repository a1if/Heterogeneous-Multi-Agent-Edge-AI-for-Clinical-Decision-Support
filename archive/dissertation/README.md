# Dissertation scripts (archived unchanged)

These scripts produced the results in the submitted MSc dissertation. They were
moved here from the repository root on 2026-09-29, when Phase 1 (the journal
extension) became the active work. The move only changed their location: the
code is byte-identical (`git log --follow <file>` shows the full history), and
their outputs in `results/` were not touched.

`results_ledger.json` and the README's "Where each reported result comes from"
table still name these scripts. The ledger uses bare file names, so read
`e1_ablation.py` there as `archive/dissertation/e1_ablation.py`.

## Running one

Run from the repository root, with the root on the import path. The scripts
import `perception.*`, `reasoning.*`, `project_config`, `ablation_common` and
`day7_auditability_probe` from the root. They import each other (for example,
`s_class_expanded_check` imports `day3_norm_check`) from this folder, which
Python adds to the path automatically.

```bash
PYTHONPATH=. python archive/dissertation/e1_ablation.py
```

In PowerShell: `$env:PYTHONPATH="."; python archive/dissertation/e1_ablation.py`.

A static check on 2026-09-29 confirmed that every import in these files still
resolves this way.

## Contents

| Group | Scripts |
|---|---|
| Headline comparison and statistics | `day6_run_comparison.py`, `e0_statistics.py`, `make_headline_figures.py`, `make_intro_figure.py` |
| Ablations and sweeps | `e1_ablation.py`, `e2_seed_variance.py`, `e3_training_compute_ladder.py`, `e4_training_set_size.py`, `e4b_per_class_27.py`, `day6_class_heading_ablation.py` |
| Auditability probes | `day7_auditability_probe_aux.py`, `day7_probe_target_comparison.py`, `run_aux_experiment.py`, `train_perception_agent_aux.py` |
| Embedding-norm checks | `day3_norm_check.py`, `export_norm_check_raw.py`, `check_adapter_collapse.py`, `run_gpu_checks_combined.py` (runs `day3_norm_check` and `s_class_expanded_check` with one model load) |
| Arm A diagnosis | `diagnose_arm_a_disagreements.py` |
| S-class checks | `find_additional_s_events.py` (selects the 43 extra S events), `s_class_expanded_check.py`, `s_class_matched_check.py`, `s_class_headroom_bias_check.py`, `check_s_class_per_record.py` |
| Quantisation pilot | `quantization_coupling_pilot.py` |
| Ledger merges | `merge_e1_e2_to_ledger.py`, `merge_gpu_checks_to_ledger.py` |

## What stayed at the root, and why

Some dissertation-era modules are still imported by current code, so they stay
at the root:

- `day7_auditability_probe.py`, `ablation_common.py`, `measure_comm_cost.py`, `project_config.py`
- `data_prep.py`, `train_perception_agent.py`, `perception_eval.py`, `render_ledger.py`
- the `perception/` and `reasoning/` packages, including the modules only the
  archived scripts use (`model_loader_8bit`, the class-heading ablation prompt
  and arm, `perception/reconstruction_decoder`, and `reasoning/audit_adapter_embeddings`),
  because they are package members

## Deleted in the same cleanup

These files produced no reported result or cached input; git history keeps them:

- the four `diagnostics/smoke_test_*.py` files
- two stale drafts previously in `archive/`: `day7_auditability_probe_pre-v2.1_draft.py`
  and the Ollama-era `diagnose_structured_output_overhead.py`, which no longer ran
- `check_s_class_headroom.py`, superseded by `s_class_headroom_bias_check.py`
- `demo_one_event.py`

Restore any of them with `git show b66e982:<path>`.
