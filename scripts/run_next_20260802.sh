#!/usr/bin/env bash
# Pruned queue (2026-08-02). The 16-cell grid collapses to 8 meaningful cells: only BOTH-raw and
# BOTH-hat backbones are interpretable.
#   th_sr (hat teacher, raw student) -- collapses, clean 39.5: the unit-norm target crushes ||phi_s||.
#   tr_sh (raw teacher, hat student) -- redundant: a unit vector cannot close the magnitude gap to a
#                                       norm-11 target, so the loss reduces to direction matching.
#                                       Measured 62.66/29.21/26.77 vs th_sh 62.61/29.16/26.63.
#
# 1. featdir_champ200_rawfeat -- the decisive run. Champion stack, backbone target swapped to raw
#    L2. Bar: 60.74 clean / 34.94 PGD / 30.53 CW / 28.69 AA / NRR 38.97.
# 2. cos_th_sh_{kl,ce}        -- the last two cells of the pruned 2x2x2.
# 3. AA on plain_tr_sr_{kl,ce} -- same backbone, different head. Settles whether the head's +3.2 PGD
#    is real robustness or a soft-target artifact (CW says +0.15..0.40, i.e. nothing).
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

run () {
  echo "=== $(date '+%m-%d %H:%M') start $2 ==="
  $PY main.py --config_name "$1" --dataset CIFAR100 --seed 0 > "logs/$2.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $2 (exit $?) ==="
}

run featdir_champ200_rawfeat.yaml featdir_champ200_rawfeat
run fg_cos_th_sh_kl.yaml fg_cos_th_sh_kl
run fg_cos_th_sh_ce.yaml fg_cos_th_sh_ce

for c in plain_tr_sr_kl plain_tr_sr_ce; do
  echo "=== $(date '+%m-%d %H:%M') AA $c ==="
  $PY scripts/eval_aa_generic.py \
      "$c|CIFAR100/checkpoint/fg_$c/feat_direction_last.pkl" >> logs/aa_head_axis.log 2>&1
done
echo "=== queue complete $(date) ==="
