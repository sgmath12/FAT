#!/usr/bin/env bash
# Queue rev. D (2026-08-02).
#   1. fg_plain_nofeat_kl -- NO feature term, in the fast 50ep grid regime. Compare against
#      fg_plain_th_sh_kl = 62.61 / 29.16 / 26.63 (same regime, feature term present).
#      Is the smoothed teacher logit the whole method?
#   2. AA on plain_tr_sr_{kl,ce} -- identical backbone, CE vs KD head.
#   3. fg_cos_th_sh_ce -- last cell of the pruned grid.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

while pgrep -f "config""_name featdir_champ200_freezehead" > /dev/null; do sleep 60; done

echo "=== $(date '+%m-%d %H:%M') start fg_plain_nofeat_kl ==="
$PY main.py --config_name fg_plain_nofeat_kl.yaml --dataset CIFAR100 --seed 0 \
    > logs/fg_plain_nofeat_kl.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done fg_plain_nofeat_kl ==="

for c in plain_tr_sr_kl plain_tr_sr_ce; do
  echo "=== $(date '+%m-%d %H:%M') AA $c ==="
  $PY scripts/eval_aa_generic.py \
      "$c|CIFAR100/checkpoint/fg_$c/feat_direction_last.pkl" 2>&1 | tee -a logs/aa_head_axis.log
done

echo "=== $(date '+%m-%d %H:%M') start fg_cos_th_sh_ce ==="
$PY main.py --config_name fg_cos_th_sh_ce.yaml --dataset CIFAR100 --seed 0 \
    > logs/fg_cos_th_sh_ce.log 2>&1
echo "=== queue complete $(date) ==="
