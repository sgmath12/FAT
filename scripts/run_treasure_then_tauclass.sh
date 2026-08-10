#!/bin/bash
# Post-basetau-sweep queue (user, 2026-07-08 night). Priority order ("list insert" -- treasure first):
# (1) TREASURE PAIR: hard-CE AT (madry_at) freehead vs coshead, 3-step, matched to KD cells
#     (AdamW 0.021 cyclic 50ep, clean-init, student_norm, EMA eval via weight_avg True).
#     Tests the HE sign-flip scoping: prediction = under hard CE the coshead penalty (KD: -1.8 H)
#     vanishes (HE's own ablation: FN+WN synergize under CE). If pair ties -> supervision type
#     (soft vs hard) is the switch, proven in OUR codebase.
# (2) placebo: shuffled tau_c (same dispersion, difficulty-alignment destroyed), 10-step g0.05 seed0.
# (3) tauclass 10-step gamma 0.1 seeds 1,2 (second positive cell 3-seed) + per-run ckpt archiving.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT

# wait for the basetau sweep to finish (marker in its chain log)
until grep -q "TAUCLASS_BASETAU_SWEEP_DONE" results/CIFAR100/basetau_sweep_chain.log 2>/dev/null; do
  sleep 120
done

for cfg in at_ce_freehead at_ce_coshead; do
  echo "=== $cfg $(date) ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 \
    > results/CIFAR100/${cfg}_driver.log 2>&1
done

echo "=== placebo g0.05 $(date) ==="
$PY -u main.py --config_name temp_tauclass_fixed_10step_placebo.yaml --dataset CIFAR100 --seed 0 --gamma 0.05 \
  > results/CIFAR100/tauclass_placebo_driver.log 2>&1

for s in 1 2; do
  echo "=== g0.1 seed $s $(date) ==="
  $PY -u main.py --config_name temp_tauclass_fixed_10step.yaml --dataset CIFAR100 --seed $s --gamma 0.1 \
    > results/CIFAR100/tauclass_g01_seed${s}_driver.log 2>&1
  cp CIFAR100/checkpoint/temp_tauclass_fixed_10step/temperature_tauclass_fixed_last.pkl \
     CIFAR100/checkpoint/temp_tauclass_fixed_10step/g01_seed${s}_last.pkl 2>/dev/null
done

echo "TREASURE_TAUCLASS_QUEUE_DONE $(date)"
