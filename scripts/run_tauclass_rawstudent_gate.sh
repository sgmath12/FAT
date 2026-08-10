#!/bin/bash
# RAW-STUDENT tauclass DECISIVE GATE (user, 2026-07-09).
# Context: rawstudent tau16 showed g0->g0.1 bump (clean +0.28, pgd20 +0.19, cw +0.28), BUT that
#   +0.19pp pgd20 is INSIDE the +-0.5pp noise band that already produced a FALSE POSITIVE in the
#   norm-student arm (g0.05 looked +0.5pp, died under placebo 32.13==32.05 and seed repetition).
#   Also tau16-g0 is the WEAKEST base point (t8=31.31,t12=31.61,t16=31.24), so the "gain" may be
#   regression-to-pack. Best pgd20 overall is tau12-g0 (31.61) with NO gamma.
# This gate is the CHEAP DECISIVE FILTER before any dense tau x gamma grid: at tau16, does the real
#   gnorm-aligned tau_c (g0.1) beat BOTH (a) its own g0 baseline AND (b) a difficulty-shuffled placebo,
#   ACROSS 3 seeds? Same gate that killed norm-student. If it survives -> dense sweep justified.
#   If not -> tauclass is dead on raw students too; pivot to the FN+free-head paper story.
# Design (9 runs, ~8h, one GPU, one overnight):
#   arm A baseline : gate config,        gamma 0.0, seeds 0/1/2  (shuffle no-op at g0)
#   arm B real     : gate config,        gamma 0.1, seeds 0/1/2  (gnorm-aligned tau_c)
#   arm C placebo  : gate_placebo config, gamma 0.1, seeds 0/1/2  (same dispersion, alignment scrambled)
#   Results -> results/CIFAR100/temp_tauclass_rawstudent_gate{,_placebo}/output.log (parse by gamma+seed).
# Chains behind the rawstudent gamma sweep (waits for RAWSTUDENT_GAMMASWEEP_DONE).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
CHAIN=results/CIFAR100/rawstudent_tauclass_chain.log

until grep -q "RAWSTUDENT_GAMMASWEEP_DONE" "$CHAIN" 2>/dev/null; do
  sleep 120
done

for s in 0 1 2; do
  echo "=== GATE baseline g0.0 seed $s $(date) ===" >> "$CHAIN"
  $PY -u main.py --config_name temp_tauclass_rawstudent_gate.yaml --dataset CIFAR100 --seed $s --gamma 0.0 \
    > results/CIFAR100/gate_g0.0_seed${s}_driver.log 2>&1
done

for s in 0 1 2; do
  echo "=== GATE real g0.1 seed $s $(date) ===" >> "$CHAIN"
  $PY -u main.py --config_name temp_tauclass_rawstudent_gate.yaml --dataset CIFAR100 --seed $s --gamma 0.1 \
    > results/CIFAR100/gate_real_g0.1_seed${s}_driver.log 2>&1
done

for s in 0 1 2; do
  echo "=== GATE placebo g0.1 seed $s $(date) ===" >> "$CHAIN"
  $PY -u main.py --config_name temp_tauclass_rawstudent_gate_placebo.yaml --dataset CIFAR100 --seed $s --gamma 0.1 \
    > results/CIFAR100/gate_placebo_g0.1_seed${s}_driver.log 2>&1
done

echo "RAWSTUDENT_GATE_DONE $(date)" >> "$CHAIN"
