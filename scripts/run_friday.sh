#!/usr/bin/env bash
# Friday GPU queue, detached from the Claude session (survives its background time limit).
# Waits for any running p1_item7_eval.py, then re-runs the idempotent second-sender evaluation queue
# (finished windows are skipped) and the serving benchmark. Log: logs/friday_queue.log
cd "$(dirname "$0")/.."
log() { echo "$(date '+%m-%d %H:%M') $*"; }
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'p1_item7_eval.py' }) { exit 0 } else { exit 1 }"; do sleep 60; done
log "no evaluation process running; continuing the second-sender queue"
"$BASH" scripts/run_second_sender_eval.sh || { log "second-sender queue STOPPED"; exit 1; }
log "start serving benchmark"
venv/Scripts/python.exe -u p1_serving.py >> logs/p1_serving.log 2>&1
log "serving exit=$?"
log "FRIDAY QUEUE DONE"
