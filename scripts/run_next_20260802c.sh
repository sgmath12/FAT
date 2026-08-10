#!/usr/bin/env bash
# Queue rev. C (2026-08-02). The head ladder, in order of what each pair isolates:
#   frozen  vs tau1  -> ADAPTATION alone (same target function: the teacher's raw logits)
#   tau1    vs tau16 -> smoothing
#   tau16   vs CE    -> supervision source (teacher vs true label)
# Theorem 2 (Tsipras model): the student's feature has mean 0.9x and variance 3x the teacher's, so
# a head calibrated on phi_t is miscalibrated on phi_s(x_adv). Measured cos_adv 0.69 (46 degrees).
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

while pgrep -f "config""_name featdir_champ200_tau1" > /dev/null; do sleep 60; done

echo "=== $(date '+%m-%d %H:%M') start freezehead ==="
$PY main.py --config_name featdir_champ200_freezehead.yaml --dataset CIFAR100 --seed 0 \
    > logs/featdir_champ200_freezehead.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done freezehead ==="

for c in plain_tr_sr_kl plain_tr_sr_ce; do
  echo "=== $(date '+%m-%d %H:%M') AA $c ==="
  $PY scripts/eval_aa_generic.py \
      "$c|CIFAR100/checkpoint/fg_$c/feat_direction_last.pkl" 2>&1 | tee -a logs/aa_head_axis.log
done

echo "=== $(date '+%m-%d %H:%M') start fg_cos_th_sh_ce ==="
$PY main.py --config_name fg_cos_th_sh_ce.yaml --dataset CIFAR100 --seed 0 \
    > logs/fg_cos_th_sh_ce.log 2>&1
echo "=== queue complete $(date) ==="
