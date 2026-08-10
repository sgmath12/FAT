#!/bin/bash
# Sharper head-target dose on the STORY cell (user's magnitude hypothesis, 2026-07-13 15:05):
# featdir head ||w_c|| stays 1.69 (vs baseline 9.23) because the tau16 target is too soft to
# demand amplification -- tau {8, 4} forces the head to grow ||w|| to match target confidence.
# dir-attack variant isolates the head-geometry effect (attack never sees the head);
# the full attack-feedback version is klattack tau8 (round2 chain). Parse featdir/output.log by tau.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/featdir_round2_chain.log
until grep -q "FEATDIR_LAMDA_SWEEP_DONE" $LOG 2>/dev/null; do sleep 120; done
for t in 8 4; do
  echo "=== featdir tau $t START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir.yaml --dataset CIFAR100 --seed 0 --tau $t \
    > results/CIFAR100/featdir_tau${t}_driver.log 2>&1
  echo "=== featdir tau $t DONE $(date) ===" >> $LOG
done
echo "FEATDIR_TAU_DONE $(date)" >> $LOG
