#!/usr/bin/env bash
# Normalization 2x2, done properly (2026-08-02). "Normalized" applies to that side everywhere at
# once -- direction loss, head input, and the eval-time forward -- rather than only inside the
# direction loss as the earlier fg_* cells did.
#
#   n2_traw_sraw    teacher raw        student raw
#   n2_traw_snorm   teacher raw        student normalized
#   n2_tnorm_sraw   teacher normalized student raw          <- expected to collapse
#   n2_tnorm_snorm  teacher normalized student normalized
#
# Regime: 50ep / 10-step / WA off / AWP off / eps 8 / lamda 0 / k512 / seed 0 / clean_200ep teacher.
# Head: detached KD, unchanged from the champion. Reference: fg_plain_th_sh_kl 62.61/29.16/26.63.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

for c in n2_traw_sraw n2_traw_snorm n2_tnorm_sraw n2_tnorm_snorm; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== n2 grid complete $(date) ==="
