#!/usr/bin/env bash
# MASTER QUEUE (2026-09-04).  This machine: ResNet-18, CIFAR-10/100 only.
# WideResNet and Tiny-ImageNet are the other server's -- see memo.md.
#
# No pgrep waiting anywhere.  main.py holds a per-GPU flock and blocks until the card is free, so the
# queue is just a sequential list and extra launchers are harmless.  The previous version waited on
# drivers by name, which is what let a leftover waiter start a second training four times running.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
for q in run_hat_rerun_20260903 \
         run_lbgat_rerun_20260904 \
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
