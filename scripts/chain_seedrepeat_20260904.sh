#!/usr/bin/env bash
# RUN THE SEED REPEAT AFTER THE QUEUE (2026-09-04 21:1x).
#
# run_seedrepeat_20260904 was added to master_queue_v2 at about 18:00, after that script had been
# running since 17:13.  bash had already parsed the for loop, so the queue skipped it -- the file has
# the entry and the running shell does not.  Editing a live script is the trap this repo has hit
# before; this launches the cell as its own driver instead.
#
# It waits on the queue's PID rather than launching beside it, because main.py's lock is a 60 s poll:
# a process already waiting is asleep when the lock frees while a freshly started one takes it
# immediately, so a waiter loses every handoff to a queue that keeps starting cells.
set -u
cd "$(dirname "$0")/.."
QUEUE_PID="${1:?usage: chain_seedrepeat_20260904.sh <pid of master_queue_v2>}"
echo "=== $(date '+%m-%d %H:%M') seed-repeat chain armed, waiting on queue pid $QUEUE_PID ==="
while kill -0 "$QUEUE_PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') queue exited, seed repeat starting ==="
bash scripts/run_seedrepeat_20260904.sh 2>&1
