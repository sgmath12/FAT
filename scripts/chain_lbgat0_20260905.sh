#!/usr/bin/env bash
# Run LBGAT0 CIFAR-10 after the 50-epoch ladder rebuild (2026-09-05).
# Chained on a PID rather than launched beside the queue: main.py's lock is a 60 s poll, so a waiting
# process loses every handoff to a freshly started one.
set -u
cd "$(dirname "$0")/.."
PREV_PID="${1:?usage: chain_lbgat0_20260905.sh <pid to wait on>}"
echo "=== $(date '+%m-%d %H:%M') lbgat0 chain armed, waiting on pid $PREV_PID ==="
while kill -0 "$PREV_PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') starting LBGAT0 CIFAR-10 ==="
bash scripts/run_lbgat0_c10_20260905.sh 2>&1
