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
#   The AWP re-runs were dropped on 2026-09-04.  The shipped recipe is already post-correction on
#   both datasets (62.17 / 28.86 and 84.96 / 51.74, re-run 09-02), and the cells that remained
#   pre-correction are ablation rows whose numbers move by 0.01-0.22 NRR.  tab:ladder's rungs are
#   also a different design from the shipped one -- they train the head, where the shipped recipe
#   freezes it -- so re-running them would not have produced the paper's numbers anyway.  The
#   captions say which tables were measured before the fix.
#      (run_adr_wa_20260904 was queued here and removed 2026-09-04.  The configs stay in the tree
#      -- config/*/adr_wa_200ep.yaml -- so it can be run later if the AT + WA + AWP + ADR row is
#      wanted; the table reports what we actually have, which is AT + AWP + ADR.)
#   Order note (2026-09-04).  champ_gnorm1_l2 is on the card and may move the recipe from an L2
#   gradient norm to the L1 the derivation actually gives.  If it does, every cell that allocates a
#   per-sample radius has to be retrained, so the queue runs the cells that do NOT allocate first:
#   abl_kdswa_t4/t16 have no featdir_angeps_p at all, and neither does ladder_p0_fh.  What is exposed
#   is ladder_angeps_fh, ladder_angeps_wa_fh and the champ_eps6/7 pair.
#
#   kdswa also happens to be the one wanted soonest: it fills tab:main's "logit anchor + our stack"
#   row, the control that separates what the target does from what the schedule does.
#   3  lbgat_c10      diverged to chance and was skipped by a completeness test that read the key
#                     rather than the value
#   4  lowerps        champ_eps6/7 -- the low end of tab:radius
#      (run_rpat_rest_20260904 removed 2026-09-04: pgdat_wa, pgdat_wa_awp and consistency are being
#      run on another machine, along with pgdat/trades/mart and ADR.  Nothing published is trained
#      on this card any more -- what is left is our own ablations.)
#
# champ_gnorm1_l2 is not in this list because it is already running ahead of it, and the reason it
# had to be let in first is worth recording: main.py's lock is a 60-second poll loop, so a process
# that is already waiting is asleep at the moment the lock frees while a freshly started one tries
# immediately and wins.  A waiter therefore loses every handoff to a queue that keeps starting new
# cells -- champ_gnorm1_l2 lost three in a row and had been waiting 5.5 h.  Stopping the queue is
# what lets a waiter in; that is why this script is launched behind it rather than beside it.
set -u
cd "$(dirname "$0")/.."
for q in run_kdswa_20260903 \
         run_seedrepeat_20260904 \
         run_ladder_fh_20260904 \
         run_lbgat_c10_rerun_20260904 \
         run_lowerps_20260902; do
  echo "=== $(date '+%m-%d %H:%M') >>> $q ==="
  bash "scripts/$q.sh" 2>&1
done
echo "=== $(date '+%m-%d %H:%M') MASTER QUEUE V2 DONE ==="
