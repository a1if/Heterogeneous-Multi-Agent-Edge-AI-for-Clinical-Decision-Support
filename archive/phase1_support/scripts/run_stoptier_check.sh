#!/usr/bin/env bash
# Deviation 25: stop-at-tier check on DS2 (main sender, text arms). Waits for the Friday queue to finish
# (or stop), then generates and compares. Resumable. Launch detached. Log: logs/stoptier_check.log
cd "$(dirname "$0")/.."
PY=venv/Scripts/python.exe
log() { echo "$(date '+%m-%d %H:%M') $*"; }
until grep -qE "FRIDAY QUEUE DONE|STOPPED" logs/friday_queue.log 2>/dev/null; do sleep 60; done
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'p1_serving.py|p1_item7_' }) { exit 0 } else { exit 1 }"; do sleep 60; done
log "start stop-at-tier generation (A-compact, A-filtered; DS2 v2)"
unset P1_ENCODER P1_ENCODER_TAG
$PY -u p1_item7_eval.py --split ds2v2 --suffix stoptier --stop-at-tier --arms A-compact A-filtered >> logs/p1_stoptier_eval.log 2>&1
c=$?; log "generation exit=$c"; [ $c -eq 0 ] || exit $c
$PY -u p1_stop_at_tier_check.py
log "STOPTIER CHECK DONE"
