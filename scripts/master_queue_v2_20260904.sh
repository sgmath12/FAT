#!/usr/bin/env bash
# MASTER QUEUE, REPRIORITISED (2026-09-04 14:00).
#
# Two changes from master_queue_resume_20260904.
#
# run_std_baselines_20260902 is GONE.  pgdat / trades / mart on both CIFAR datasets are being run on
# another machine, so six cells and about 13 h come out of this queue entirely.
#
# The order is now by what the paper depends on rather than by what was queued first:
#
#   1  awpfix_rest    corrections to numbers the paper already reports (app:awp says every
#                     shipped-recipe figure is post-correction, so these are not optional)
#      (run_adr_wa_20260904 was queued here and removed 2026-09-04.  The configs stay in the tree
#      -- config/*/adr_wa_200ep.yaml -- so it can be run later if the AT + WA + AWP + ADR row is
#      wanted; the table reports what we actually have, which is AT + AWP + ADR.)
#   2  lbgat_c10      diverged to chance and was skipped by a completeness test that read the key
#                     rather than the value
#   3  lowerps        champ_eps6/7 -- the low end of tab:radius, which is a table in the paper
#   4  kdswa          abl_kdswa_t4/t16 -- one ablation row
#   5  rpat_rest      pgdat_wa, pgdat_wa_awp, consistency -- table-filling baselines, last
#
# champ_gnorm1_l2 is not in this list because it is already running ahead of it, and the reason it
# had to be let in first is worth recording: main.py's lock is a 60-second poll loop, so a process
# that is already waiting is asleep at the moment the lock frees while a freshly started one tries
# immediately and wins.  A waiter therefore loses every handoff to a queue that keeps starting new
# cells -- champ_gnorm1_l2 lost three in a row and had been waiting 5.5 h.  Stopping the queue is
# what lets a waiter in; that is why this script is launched behind it rather than beside it.
set -u
cd "$(dirname "$0")/.."
for q in run_awpfix_rest_20260904 \
         run_lbgat_c10_rerun_20260904 \
         run_lowerps_20260902 \
         run_kdswa_20260903 \
         run_rpat_rest_20260904; do
  echo "=== $(date '+%m-%d %H:%M') >>> $q ==="
  bash "scripts/$q.sh" 2>&1
done
echo "=== $(date '+%m-%d %H:%M') MASTER QUEUE V2 DONE ==="
