#!/usr/bin/env bash
# SINGLE SEQUENTIAL QUEUE (2026-09-02).
#
# The five run queues were chained with independent `while pgrep ...; do sleep` waiters, and they
# raced: at 00:02 one driver exited and two waiters both saw an idle GPU within the same polling
# window, so two trainings shared the card.  A lock file did not fix it either, because the driver
# already running had been launched before the lock existed and never held it.
#
# One process, one `for` loop, no polling and nothing to race against.  Everything after the
# currently-running AWP-fix queue goes here, in the order the paper needs it.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "run_awpfix_rerun_20260901" > /dev/null; do sleep 120; done
echo "=== $(date '+%m-%d %H:%M') awpfix queue clear ==="
for q in run_ablations_20260901 run_std_baselines_20260902 run_lowerps_20260902; do
  echo "=== $(date '+%m-%d %H:%M') >>> $q ==="
  bash "scripts/$q.sh"
  echo "=== $(date '+%m-%d %H:%M') <<< $q done ==="
done
echo "=== $(date '+%m-%d %H:%M') MASTER QUEUE DONE ==="
