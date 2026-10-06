#!/usr/bin/env bash
# Deviation 22, steps 1-5 of docs/run_order.md. Idempotent: re-running skips finished steps,
# and adapter training resumes from its .resume.pt (saved every 25 updates).
# Run from the repo root:  bash scripts/run_second_sender.sh
set -u
cd "$(dirname "$0")/.."
PY=venv/Scripts/python.exe
export P1_ENCODER=perception/checkpoints/resnet1d_rr_seed0.pt P1_ENCODER_TAG=_res
log() { echo "$(date +%H:%M) $*"; }
step() { log "start $1"; shift; "$@"; c=$?; log "exit=$c"; [ $c -eq 0 ] || { log "STOP"; exit $c; }; }

# 1. sender (skip if trained) + gate on DS1 validation accuracy at the selected (best val-loss) epoch
if [ ! -f results/p1_rr_encoder_results_resnet1d_rr_seed0.json ]; then
  step "1 train ResNet1D-RR sender" sh -c "$PY -u train_perception_agent_rr.py --arch resnet1d_rr --seed 0 > logs/p1_res_sender.log 2>&1"
fi
gate=$($PY -c "import json;h=json.load(open('results/p1_rr_encoder_results_resnet1d_rr_seed0.json'))['history'];b=min(h,key=lambda e:e['val_loss']);print(round(b['val_accuracy'],4))")
log "gate: DS1 validation accuracy at selected epoch = $gate (need >= 0.85)"
$PY -c "import sys;sys.exit(0 if $gate >= 0.85 else 1)" || { log "STOP: sender failed the pre-registered gate"; exit 3; }

# 2. replay (cached on disk by checkpoint hash; instant on re-run)
step "2 replay DS1 + DS2" sh -c "$PY -u -c \"from p1_item7_common import replay_split; replay_split('ds1'); replay_split('ds2')\" > logs/p1_res_replay.log 2>&1"

# 3. test windows
[ -f results/p1_item7_testset_v2_res.json ] || step "3 build DS2 test windows" sh -c "$PY -u p1_item7_testset_v2.py > logs/p1_res_testset.log 2>&1"

# 4. smoke + memory gate
if [ ! -f results/p1_item7_smoke_r4_res.json ]; then
  step "4 r4 smoke" sh -c "$PY -u p1_item7_train.py --smoke --recipe r4 > logs/p1_res_smoke.log 2>&1"
fi
$PY -c "import json,sys;s=json.load(open('results/p1_item7_smoke_r4_res.json'));print('memory_ok',s['memory_ok']);sys.exit(0 if s['memory_ok'] else 1)" || { log "STOP: memory gate"; exit 4; }

# 5. adapter seeds (each resumes; a finished seed returns at once)
for S in 101 202 303; do
  step "5 adapter seed $S" sh -c "$PY -u p1_item7_train.py --recipe r4 --seed $S >> logs/p1_item7_r4_res_seed$S.log 2>&1"
done
log "QUEUE DONE"
