#!/bin/bash
# NIGHT QUEUE 2026-07-13->14 (user asleep; goal: story-faithful loss that TIES baseline 41.77
# at 3-step). Order: (1) selfkl flagship (self-metric direction KL; prediction = recovery),
# (2) alpha-undetach {0.25, 1.0} (teacher-confidence routing dose), (3) span_random + lamda100
# combo (stack the two known levers +1.2/+0.6), (4) seed-1 insurance for selfkl & span_random.
# New code paths smoke-gated. GPU free at queue time.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log

smoke_run () {  # $1 config  $2 extra args...  (smoke then full run)
  cfg=$1; shift
  echo "=== $cfg SMOKE START $(date) ===" >> $LOG
  if $PY -u main.py --config_name $cfg.yaml --dataset CIFAR100 --seed 0 --epochs 1 "$@" \
      > results/CIFAR100/${cfg}_smoke_driver.log 2>&1; then
    echo "=== $cfg smoke OK, full START $(date) ===" >> $LOG
    $PY -u main.py --config_name $cfg.yaml --dataset CIFAR100 --seed 0 "$@" \
      > results/CIFAR100/${cfg}_driver.log 2>&1
    echo "=== $cfg DONE $(date) ===" >> $LOG
  else
    echo "!!! $cfg SMOKE FAILED, skipped $(date)" >> $LOG
  fi
}
run () { cfg=$1; shift; echo "=== $cfg $* START $(date) ===" >> $LOG; \
  $PY -u main.py --config_name $cfg.yaml --dataset CIFAR100 "$@" \
    > results/CIFAR100/${cfg}_$(echo $* | tr ' ' '_' | tr -d -- '-')_driver.log 2>&1; \
  echo "=== $cfg $* DONE $(date) ===" >> $LOG; }

smoke_run featdir_selfkl
smoke_run featdir_alpha025
run featdir_alpha10 --seed 0
run featdir_span_random --seed 0 --lamda 100.0
run featdir_selfkl --seed 1
run featdir_span_random --seed 1
echo "NIGHT_SELFMETRIC_DONE $(date)" >> $LOG
