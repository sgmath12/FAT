#!/usr/bin/env bash
# Run the 50-epoch ladder rebuild after the seed repeat (2026-09-05).
#
# Chained on the seed-repeat chain's PID rather than on the queue's, so the order is
# queue -> seed repeat -> this, with one waiter on the lock at a time.  main.py's lock is a 60 s
# poll and a fresh process beats a sleeping waiter, so two chains armed on the same PID would race.
set -u
cd "$(dirname "$0")/.."
PREV_PID="${1:?usage: chain_ladder50_20260905.sh <pid to wait on>}"
echo "=== $(date '+%m-%d %H:%M') ladder-50 chain armed, waiting on pid $PREV_PID ==="
while kill -0 "$PREV_PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') starting 50-epoch ladder rebuild ==="
bash scripts/run_ladder_fh_50ep_20260905.sh 2>&1
