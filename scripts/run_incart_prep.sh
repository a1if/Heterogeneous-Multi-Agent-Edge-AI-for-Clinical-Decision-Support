#!/usr/bin/env bash
# Deviation 21, step 13-14 of docs/run_order.md (CPU, run AFTER the freeze/tag). Conversion, replay through
# the frozen perception agent, confirmatory windows. Inspect only the conversion checks and window counts.
set -u
cd "$(dirname "$0")/.."
PY=venv/Scripts/python.exe
[ -f results/p1_freeze_manifest.json ] || { echo "STOP: run p1_freeze.py and tag r4-confirmatory first"; exit 2; }
$PY -u incart_prep.py > logs/p1_incart_prep.log 2>&1 || { echo "STOP: conversion"; exit 1; }
$PY -u -c "from p1_item7_common import replay_split; replay_split('incart')" > logs/p1_incart_replay.log 2>&1 || { echo "STOP: replay"; exit 1; }
$PY -u p1_incart_windows.py > logs/p1_incart_windows.log 2>&1 || { echo "STOP: windows"; exit 1; }
echo "INCART PREP DONE"; tail -3 logs/p1_incart_prep.log; head -1 logs/p1_incart_windows.log
