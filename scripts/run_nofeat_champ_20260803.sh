#!/usr/bin/env bash
# Queued behind featdir_champ200_fullraw (pid passed as $1, default 913846). Answers the reviewer
# question "if you match raw features anyway, why is the feature loss there?" at the champion recipe.
#   nofeat_champ200_raw   method:temperature, student_norm False  <- the cell the user asked for
#   nofeat_champ200_norm  method:temperature, student_norm True   <- champion minus feature loss
# Both are champion schedule verbatim (100ep, WA, proxy AWP, freeze_lr .65, wa_start .2, aa True).
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

WAIT_PID="${1:-913846}"
echo "=== $(date '+%m-%d %H:%M') waiting on pid $WAIT_PID (featdir_champ200_fullraw) ==="
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') pid $WAIT_PID gone, starting queue ==="

for c in nofeat_champ200_raw nofeat_champ200_norm; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== nofeat queue complete $(date) ==="
