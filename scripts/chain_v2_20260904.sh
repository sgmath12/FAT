#!/usr/bin/env bash
# START MASTER QUEUE V2 ONLY AFTER champ_gnorm1_l2 HAS HAD THE CARD (2026-09-04).
#
# Waits on the gnorm1 driver's PID rather than launching beside it, for the reason recorded in
# master_queue_v2: main.py's lock is a 60 s poll, so an already-waiting process loses every handoff
# to a freshly started one.  Launching the queue now would restart the starvation this is fixing.
set -u
cd "$(dirname "$0")/.."
GNORM_PID="${1:?usage: chain_v2_20260904.sh <pid of run_gnorm1_l2 driver>}"
echo "=== $(date '+%m-%d %H:%M') chain v2 armed, waiting on gnorm1 driver pid $GNORM_PID ==="
while kill -0 "$GNORM_PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') gnorm1 driver exited, master queue v2 starting ==="
bash scripts/master_queue_v2_20260904.sh 2>&1
