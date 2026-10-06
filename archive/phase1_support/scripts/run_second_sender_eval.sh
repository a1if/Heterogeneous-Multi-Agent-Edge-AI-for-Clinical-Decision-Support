#!/usr/bin/env bash
# Deviation 22, step 7-8 of docs/run_order.md: second-sender DS2 evaluation. Idempotent and resumable:
# every script saves per generation / per batch and skips finished windows on re-run.
# Run from the repo root:  bash scripts/run_second_sender_eval.sh
set -u
cd "$(dirname "$0")/.."
PY=venv/Scripts/python.exe
export P1_ENCODER=perception/checkpoints/resnet1d_rr_seed0.pt P1_ENCODER_TAG=_res
log() { echo "$(date +%H:%M) $*"; }
step() { log "start $1"; shift; "$@"; c=$?; log "exit=$c"; [ $c -eq 0 ] || { log "STOP"; exit $c; }; }
step "7 DS2 generation (MEA x3, A-compact, A-filtered)" sh -c "$PY -u p1_item7_eval.py --split ds2v2 --suffix r4_res --arms MEA:r4_res_seed101 MEA:r4_res_seed202 MEA:r4_res_seed303 A-compact A-filtered >> logs/p1_item7_eval_ds2v2_r4_res.log 2>&1"
step "8a calibration logits A-compact" sh -c "$PY -u p1_item7_filtered.py collect --text-arm A-compact >> logs/p1_res_collect_compact.log 2>&1"
step "8b calibration logits A-filtered" sh -c "$PY -u p1_item7_filtered.py collect --text-arm A-filtered >> logs/p1_res_collect_filtered.log 2>&1"
step "8c prompt tokens" sh -c "$PY -u p1_item7_filtered.py tokens >> logs/p1_res_tokens.log 2>&1"
step "8d timing" sh -c "$PY -u p1_item7_filtered.py timing >> logs/p1_res_timing.log 2>&1"
log "QUEUE DONE"
