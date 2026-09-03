#!/usr/bin/env bash
# KEEP THE CARD BUSY AFTER THE SECTION 3 CONTROLS (2026-09-04).
#
# `run_diffrank_l2_20260904.sh` (champ_p0_l2 -> champ_diffrank_l2) is the priority: it is the hole in
# the paper, not a table row.  Everything else waits behind it.
#
# Why this waits on the PID rather than just launching and letting flock sort it out: flock wake order
# is NOT FIFO, so a second driver launched now could take the card ahead of champ_diffrank_l2 and push
# the result we are actually waiting on hours later.  Waiting for the controls' driver to exit makes
# the order deterministic.
#
# Then, in order:
#   run_epssignal_l2_20260904   champ_diffeps_l2, champ_margineps_l2 -- the other two signal cells,
#                               on the shipped recipe.  Replaces run_epssignal_20260903, whose cells
#                               carry student_norm True + freeze_lr_epoch 0.65 (stale regime).
#   master_queue_resume_20260904  rpat baselines, kdswa, std baselines, lowerps, awpfix -- the queue
#                               that was stopped at 00:40 so the controls could go first.
set -u
cd "$(dirname "$0")/.."
CONTROLS_PID="${1:?usage: chain_after_controls_20260904.sh <pid of run_diffrank_l2 driver>}"
echo "=== $(date '+%m-%d %H:%M') chain armed, waiting on controls driver pid $CONTROLS_PID ==="
while kill -0 "$CONTROLS_PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') controls driver exited, chain starting ==="
for q in run_epssignal_l2_20260904 master_queue_resume_20260904; do
  echo "=== $(date '+%m-%d %H:%M') >>> $q ==="
  bash "scripts/$q.sh" 2>&1
done
echo "=== $(date '+%m-%d %H:%M') CHAIN DONE ==="
