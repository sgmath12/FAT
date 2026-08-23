#!/usr/bin/env bash
# Temperature sensitivity of logit- vs feature-following, BASE regime (50ep, no WA, no AWP,
# eps 8/255).  An ablation belongs here, not on the champion recipe.
#
# A naturally-trained teacher is over-confident, so its logits need a temperature before they can
# serve as a KD target and someone has to choose it.  Its features need no calibration at all --
# ||Phi_hat_s - Phi_hat_t||^2 contains no temperature.  Prediction: the pure-KD row degrades sharply
# as tau falls, the featdir rows barely move, and featdir with the head term removed entirely (no tau
# anywhere) is no worse than featdir with a good tau.
#
# Reference at this regime, AA not previously measured:
#   pure KD tau 16   61.69 / PGD 31.44 / CW 27.05
#   featdir tau 16   62.61 / PGD 29.16 / CW 26.63
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for c in tausens_kd_t1 tausens_kd_t4 tausens_kd_t16 tausens_fd_t1 tausens_fd_nohd; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
