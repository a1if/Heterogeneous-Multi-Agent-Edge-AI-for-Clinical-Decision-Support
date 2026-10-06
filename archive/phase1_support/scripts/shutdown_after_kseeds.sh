#!/usr/bin/env bash
# Shut the PC down 2 min after the Deviation 27 queue ends (done or stopped). User request 2026-10-03.
cd "$(dirname "$0")/.."
until grep -qE "QUEUE DONE|STOP" logs/kseeds_queue.log; do sleep 60; done
echo "$(date '+%m-%d %H:%M') kseeds queue ended; shutting down in 120 s" >> logs/kseeds_queue.log
/c/Windows/System32/shutdown.exe //s //t 120 //c "Deviation 27 evaluation finished; shutting down"
