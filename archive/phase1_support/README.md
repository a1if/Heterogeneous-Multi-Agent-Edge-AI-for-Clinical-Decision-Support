# Phase 1 supporting files (archived)

Moved here on 2026-10-06 to keep the top level to the code that produces the paper.
Nothing in this folder is needed to reproduce the paper's tables and figures.

Contents keep their original relative paths:
- one-off diagnostics and pilots (`p1_*_diag`, `p1_item7_parsecheck.py`, `p1_task5_tier_balanced.py`, ...);
- GPU run queues and shutdown scripts (`scripts/*.sh`) and the INCART pre-flight sandbox;
- the supervisor report (`reports/`) and its ledger (`results_ledger.json`, `render_ledger.py`, `p1_export_ledger.py`);
- results not read by the paper (dissertation-era runs, smoke tests, earlier adapters' training logs).

Scripts here still use paths relative to the repository root (`results/...`); to run one, copy it back to its
original location first. The frozen confirmatory state is also preserved unchanged at git tag `r4-confirmatory`.
