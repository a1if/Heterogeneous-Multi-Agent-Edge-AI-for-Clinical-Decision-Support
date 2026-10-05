#!/usr/bin/env bash
# Deviation 29: training-data scaling (r4d25, r4d50, r4dmax; seeds 101/202/303), then DS2 v2 evaluation.
# Idempotent and resumable. Launch detached. Log: logs/scaling_queue.log
set -u
cd "$(dirname "$0")/.."
unset P1_ENCODER P1_ENCODER_TAG
PY=venv/Scripts/python.exe
log() { echo "$(date '+%m-%d %H:%M') $*"; }
step() { log "start $1"; shift; "$@"; c=$?; log "exit=$c"; [ $c -eq 0 ] || { log "STOP"; exit $c; }; }
for R in r4d25 r4d50 r4dmax; do for S in 101 202 303; do
  step "train $R seed $S" sh -c "$PY -u p1_item7_train.py --recipe $R --seed $S >> logs/p1_item7_${R}_seed$S.log 2>&1"
done; done
step "DS2 eval (scaling x9)" sh -c "$PY -u p1_item7_eval.py --split ds2v2 --suffix scaling --arms MEA:r4d25_seed101 MEA:r4d25_seed202 MEA:r4d25_seed303 MEA:r4d50_seed101 MEA:r4d50_seed202 MEA:r4d50_seed303 MEA:r4dmax_seed101 MEA:r4dmax_seed202 MEA:r4dmax_seed303 >> logs/p1_item7_eval_scaling.log 2>&1"
log "QUEUE DONE"
