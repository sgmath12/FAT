#!/usr/bin/env bash
# THE LOGIT ANCHOR AT THE FULL RECIPE (2026-09-02), i.e. our KD+SWA analogue.
#
# RPAT's SOTA leaderboard (its Tables 3 and 4) puts KD+SWA (Chen et al., ICLR'21) closest to us
# structurally: distil to smooth the logits, then average weights to smooth the parameters.  Our
# recipe has that shape with a feature target instead of a logit one, so the difference between them
# has to be measured rather than asserted.  Section 2.2 shows the feature anchor beating the best
# temperature in the base regime; these two cells ask whether that survives the full stack, which is
# the regime KD+SWA actually operates in.
#
# Both cells are `l2_bestrecipe_freezehead` with the loss swapped to logit KL -- WA, AWP proxy,
# eps_train 8.8/255, teacher warm start, AdamW 0.021, OneCycle, 100 epochs all unchanged.  tau = 4 is
# the temperature that wins on AA in the base sweep; tau = 16 wins on PGD-20 and loses on AA and CW,
# and is run second because if tau = 4 lands far from 28.86 the second cell is confirmation rather
# than a question.
#
# DO NOT start this while the master queue is running -- one training per GPU.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

for c in abl_kdswa_t4 abl_kdswa_t16; do
  [ -f "config/CIFAR100/${c}.yaml" ] || { echo "skip $c (config 없음)"; continue; }
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/CIFAR100_${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
