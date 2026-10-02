#!/usr/bin/env bash
# Serving benchmark under the memory rule (Deviation 23a), then the Deviation 25 stop-at-tier check.
# Both resumable. Launch detached. Log: logs/serving_queue.log
cd "$(dirname "$0")/.."
PY=venv/Scripts/python.exe
log() { echo "$(date '+%m-%d %H:%M') $*"; }
unset P1_ENCODER P1_ENCODER_TAG
log "start serving benchmark (memory cap 90%)"
$PY -u p1_serving.py >> logs/p1_serving.log 2>&1
c=$?; log "serving exit=$c"
log "start stop-at-tier generation (A-compact, A-filtered; DS2 v2)"
$PY -u p1_item7_eval.py --split ds2v2 --suffix stoptier --stop-at-tier --arms A-compact A-filtered >> logs/p1_stoptier_eval.log 2>&1
c=$?; log "stop-at-tier generation exit=$c"; [ $c -eq 0 ] || exit $c
$PY -u p1_stop_at_tier_check.py
log "QUEUE DONE"
