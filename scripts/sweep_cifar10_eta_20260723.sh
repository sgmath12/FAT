#!/bin/bash
# 2026-07-23: CIFAR10 eta(=k subspace dim) sweep of the champion recipe (featdir k+WA+lamda4,
# tau16/beta1/eps8, mixup clean teacher). Q: does the CIFAR100 k-curve symptom reproduce on
# CIFAR10 -- left cliff at small k (clean crash) / robust decay toward k=512 / interior optimum
# (capacity reservation)? Sequential, single GPU. All runs share the featdir_mixupT_wa output
# folder; per-run logs are timestamped (distinct), checkpoints overwrite (metrics live in logs).
cd /mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
DRIVER=results/CIFAR10/sweep_eta_20260723_driver.log
mkdir -p results/CIFAR10
for eta in 5 25 50 100 200 350 512; do
  echo "=== START eta=${eta} $(date) ===" >> "$DRIVER"
  $PY -u main.py --config_name featdir_mixupT_wa.yaml --dataset CIFAR10 --seed 0 --eta ${eta} --lamda 4.0 >> "$DRIVER" 2>&1
  echo "=== DONE  eta=${eta} rc=$? $(date) ===" >> "$DRIVER"
done
echo "CIFAR10_ETA_SWEEP_DONE $(date)" >> "$DRIVER"
