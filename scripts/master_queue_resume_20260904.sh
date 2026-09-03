#!/usr/bin/env bash
# MASTER QUEUE, RESUMED AFTER THE CONTROL RE-RUN (2026-09-04 00:40).
#
# The 2026-09-04 master queue was stopped mid-`run_rpat_baselines` (CIFAR100/adr_200ep, ~4 min in,
# nothing salvageable) so that the Section 3 per-sample-eps controls could take the card first: the
# controls the paper cites were built on the pre-2026-08-31 directional regime and cannot be quoted
# as they stand, which makes them a hole in the paper rather than a table row.
#
# Everything before run_rpat_baselines had already finished: run_hat_rerun (both datasets) and
# run_lbgat_rerun (both skipped, already present).  This restarts from run_rpat_baselines.
#
# NOTE `run_epssignal_20260903` is deliberately absent -- its two cells (champ_diffeps,
# champ_margineps) carry student_norm True and freeze_lr_epoch 0.65, i.e. the same stale regime, and
# their shipped-recipe replacements (champ_diffeps_l2, champ_margineps_l2) run in
# run_diffrank_l2_20260904 instead.
#
# Logging is NOT tee'd per line here.  The old master script piped each echo through
# `tee -a logs/master_queue.log` while the script as a whole was ALSO wrapped in a tee to the same
# file, so every line landed twice; redirect this one's stdout instead:
#   nohup ./scripts/master_queue_resume_20260904.sh >> logs/master_queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
for q in run_rpat_baselines_20260903 \
         run_kdswa_20260903 \
         run_std_baselines_20260902 \
         run_lowerps_20260902 \
         run_awpfix_rest_20260902; do
  echo "=== $(date '+%m-%d %H:%M') >>> $q ==="
  bash "scripts/$q.sh" 2>&1
done
echo "=== $(date '+%m-%d %H:%M') MASTER QUEUE DONE ==="
