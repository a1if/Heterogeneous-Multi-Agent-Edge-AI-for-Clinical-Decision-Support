#!/usr/bin/env bash
# Deviation 24, step 9-10 of docs/run_order.md. Idempotent and resumable (training resumes from
# .resume.pt; evaluation skips finished windows). Run from the repo root:  bash scripts/run_compression_sweep.sh
set -u
cd "$(dirname "$0")/.."
PY=venv/Scripts/python.exe
log() { echo "$(date +%H:%M) $*"; }
step() { log "start $1"; shift; "$@"; c=$?; log "exit=$c"; [ $c -eq 0 ] || { log "STOP"; exit $c; }; }
if [ ! -f results/p1_item7_smoke_r4k8.json ]; then
  step "smoke r4k8 (memory)" sh -c "$PY -u p1_item7_train.py --smoke --recipe r4k8 > logs/p1_smoke_r4k8.log 2>&1"
fi
$PY -c "import json,sys;s=json.load(open('results/p1_item7_smoke_r4k8.json'));print('memory_ok',s['memory_ok']);sys.exit(0 if s['memory_ok'] else 1)" || { log "STOP: memory gate"; exit 4; }
for K in 1 2 8; do
  step "train r4k$K seed 101" sh -c "$PY -u p1_item7_train.py --recipe r4k$K --seed 101 >> logs/p1_item7_r4k${K}_seed101.log 2>&1"
done
step "DS2 eval (k = 1, 2, 8)" sh -c "$PY -u p1_item7_eval.py --split ds2v2 --suffix sweep --arms MEA:r4k1_seed101 MEA:r4k2_seed101 MEA:r4k8_seed101 >> logs/p1_item7_eval_sweep.log 2>&1"
log "QUEUE DONE"
