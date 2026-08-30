#!/usr/bin/env bash
# ANGEPS LADDER, CIFAR-100 / ResNet-18 (2026-08-30).
#
# Cumulative ablation of the shipped recipe on the raw-L2 feature anchor, at both 50 and 100 epochs,
# so each component can be read as a property of the objective or of the schedule length:
#
#   step 1  anchor + sensitivity-matched eps
#   step 2  + weight averaging
#   step 3  + AWP proxy               = the shipped recipe
#
# train_eps is held at 8.8/255 across the whole ladder so that eps is not a hidden fourth step, and
# freeze_lr is absent throughout (a net loss for the raw target: NRR 38.83 -> 39.29 on CIFAR-100 when
# removed, against +0.30 for the directional design).
#
# The partner ladder WITHOUT angeps already exists at 100 epochs:
#   anchor              `wadec_raw_nowa`            62.40 / AA 24.34
#   + WA                `wadec_raw_wa`              61.58 / AA 26.55
#   + WA + eps8.8       `wadec_raw_wa_eps88`        60.09 / AA 27.00
#   + WA + eps8.8 + AWP `wadec_raw_wa_eps88_awp`    60.21 / AA 28.14
#   + angeps            `l2_bestrecipe_angeps`      62.35 / AA 28.68
#
# `ladder_angeps_waawp_100ep` is byte-identical to `l2_bestrecipe_angeps` and therefore doubles as a
# consistency check: it should reproduce 62.35 / 28.68.
#
# 50-epoch cells first, since they are half the cost and already answer whether the components need
# the longer schedule.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

for c in ladder_angeps_50ep ladder_angeps_wa_50ep ladder_angeps_waawp_50ep \
         ladder_angeps_100ep ladder_angeps_wa_100ep ladder_angeps_waawp_100ep; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
