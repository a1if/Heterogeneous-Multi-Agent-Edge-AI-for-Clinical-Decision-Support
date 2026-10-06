#!/usr/bin/env bash
# Deviation 27: seeds 202 and 303 for k = 1 and k = 2, then DS2 v2 evaluation (appends to the sweep file).
# Idempotent and resumable. Launch detached. Log: logs/kseeds_queue.log
set -u
cd "$(dirname "$0")/.."
unset P1_ENCODER P1_ENCODER_TAG
PY=venv/Scripts/python.exe
log() { echo "$(date '+%m-%d %H:%M') $*"; }
step() { log "start $1"; shift; "$@"; c=$?; log "exit=$c"; [ $c -eq 0 ] || { log "STOP"; exit $c; }; }
for K in 1 2; do for S in 202 303; do
  step "train r4k$K seed $S" sh -c "$PY -u p1_item7_train.py --recipe r4k$K --seed $S >> logs/p1_item7_r4k${K}_seed$S.log 2>&1"
done; done
step "DS2 eval (k = 1, 2; seeds 202, 303)" sh -c "$PY -u p1_item7_eval.py --split ds2v2 --suffix sweep --arms MEA:r4k1_seed202 MEA:r4k1_seed303 MEA:r4k2_seed202 MEA:r4k2_seed303 >> logs/p1_item7_eval_sweep.log 2>&1"
log "QUEUE DONE"
