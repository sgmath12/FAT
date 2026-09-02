#!/usr/bin/env bash
# MASTER QUEUE, REORDERED (2026-09-03).
#
# Two changes from 09-02.
#
# 1. The natural-teacher trade-off baselines run FIRST among the new work.  LBGAT and HAT both consume
#    a naturally trained network AND both target clean accuracy, which makes them our most direct
#    competitors -- same asset, same goal.  LBGAT in particular is the only measurement that can
#    support "feature beats logit at the same read point with no temperature on either side", which
#    the paper currently asserts without a number.  ADR follows for the same reason: it is the row the
#    headline comparison is stated against.
#
# 2. The AWP re-runs stay last.  The fix moved NRR by 0.01 on CIFAR-100 and 0.22 on CIFAR-10, so those
#    seven cells are decimal-place updates; nothing in the paper waits on them.
#
# Ordering rule: a cell that can change a SENTENCE outranks a cell that can change a DIGIT.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

# The 09-02 ablation driver is still running its last four cells and does NOT skip completed ones, so
# it is left alone to finish and is NOT in the list below -- re-running it would redo all nine.
# Waiting on main.py alone is not enough: there is a gap between its cells where no training exists,
# and starting here in that gap would put two on one GPU.  Wait on the DRIVER, then on main.py.
#
# That failure has now happened twice.  On 09-02 a scratchpad waiter from the previous day launched a
# second copy of the ablation driver one minute after the master queue launched its own, and every
# ablation cell since 08:54 ran twice on one GPU.  The results were unharmed -- seed 0 is
# deterministic and both copies agreed to the last digit -- but each cell cost twice what it should.
while pgrep -f "[r]un_ablations_20260901.sh" > /dev/null; do sleep 120; done
while pgrep -f "[m]ain.py --config_name" > /dev/null; do sleep 120; done
echo "=== $(date '+%m-%d %H:%M') GPU free, starting ===" | tee -a logs/master_queue.log

for q in run_hat_lbgat_20260903 \
         run_rpat_baselines_20260903 \
         run_kdswa_20260903 \
         run_epssignal_20260903 \
         run_std_baselines_20260902 \
         run_lowerps_20260902 \
         run_awpfix_rest_20260902; do
  echo "=== $(date '+%m-%d %H:%M') >>> $q ===" | tee -a logs/master_queue.log
  bash "scripts/$q.sh" 2>&1 | tee -a logs/master_queue.log
done
echo "=== $(date '+%m-%d %H:%M') MASTER QUEUE DONE ===" | tee -a logs/master_queue.log
