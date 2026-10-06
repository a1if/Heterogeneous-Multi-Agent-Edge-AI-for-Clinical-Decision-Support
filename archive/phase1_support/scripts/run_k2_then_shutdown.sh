#!/usr/bin/env bash
# Resume only the compression-sweep k=2 training (Deviation 24), then shut the PC down (user request 2026-10-03).
# Log: logs/sweep_queue.log
cd "$(dirname "$0")/.."
unset P1_ENCODER P1_ENCODER_TAG
log() { echo "$(date '+%m-%d %H:%M') $*"; }
log "resume train r4k2 seed 101 (then shutdown)"
venv/Scripts/python.exe -u p1_item7_train.py --recipe r4k2 --seed 101 >> logs/p1_item7_r4k2_seed101.log 2>&1
log "r4k2 exit=$?; shutting down in 120 s"
/c/Windows/System32/shutdown.exe //s //t 120 //c "Compression sweep k=2 finished; shutting down"
