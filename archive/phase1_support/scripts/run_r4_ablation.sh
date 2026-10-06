#!/usr/bin/env bash
# Deviation 26: r4 ablation (hard negatives only, no side inputs), seeds 101/202/303, then DS2 v2 evaluation.
# Idempotent and resumable. Launch detached. Log: logs/ablation_queue.log
set -u
cd "$(dirname "$0")/.."
unset P1_ENCODER P1_ENCODER_TAG
PY=venv/Scripts/python.exe
log() { echo "$(date '+%m-%d %H:%M') $*"; }
step() { log "start $1"; shift; "$@"; c=$?; log "exit=$c"; [ $c -eq 0 ] || { log "STOP"; exit $c; }; }
for S in 101 202 303; do
  step "train r4hn seed $S" sh -c "$PY -u p1_item7_train.py --recipe r4hn --seed $S >> logs/p1_item7_r4hn_seed$S.log 2>&1"
done
step "DS2 eval (r4hn x3)" sh -c "$PY -u p1_item7_eval.py --split ds2v2 --suffix ablation --arms MEA:r4hn_seed101 MEA:r4hn_seed202 MEA:r4hn_seed303 >> logs/p1_item7_eval_ablation.log 2>&1"
log "QUEUE DONE"
