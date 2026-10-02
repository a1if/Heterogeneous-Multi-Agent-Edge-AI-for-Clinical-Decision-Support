#!/usr/bin/env bash
# Deviation 21, step 15 of docs/run_order.md: the single confirmatory run of r4-confirmatory on INCART (GPU).
# Idempotent and resumable (technical retries only). Launch detached (see TASKS.md). Log: logs/incart_queue.log
set -u
cd "$(dirname "$0")/.."
PY=venv/Scripts/python.exe
log() { echo "$(date '+%m-%d %H:%M') $*"; }
step() { log "start $1"; shift; "$@"; c=$?; log "exit=$c"; [ $c -eq 0 ] || { log "STOP"; exit $c; }; }
[ -f results/p1_item7_testset_incart.json ] || { log "STOP: run scripts/run_incart_prep.sh first"; exit 2; }
step "generation (r4 x3, full)" sh -c "$PY -u p1_item7_eval.py --split incart --arms MEA:r4_seed101 MEA:r4_seed202 MEA:r4_seed303 >> logs/p1_incart_eval.log 2>&1"
# Deviation 25 (decided on DS2 before the freeze: 2,390/2,390 tiers identical): text arms stop at the tier token
step "generation (A-compact, A-filtered; stop at tier)" sh -c "$PY -u p1_item7_eval.py --split incart --stop-at-tier --arms A-compact A-filtered >> logs/p1_incart_eval.log 2>&1"
step "tier logits A-compact" sh -c "$PY -u p1_item7_filtered.py collect --text-arm A-compact --split incart >> logs/p1_incart_collect.log 2>&1"
step "tier logits A-filtered" sh -c "$PY -u p1_item7_filtered.py collect --text-arm A-filtered --split incart >> logs/p1_incart_collect.log 2>&1"
step "timing" sh -c "$PY -u p1_item7_filtered.py timing --split incart >> logs/p1_incart_timing.log 2>&1"
step "confirmatory analysis" sh -c "$PY -u p1_confirmatory_analysis.py > logs/p1_confirmatory.log 2>&1"
log "INCART QUEUE DONE"
