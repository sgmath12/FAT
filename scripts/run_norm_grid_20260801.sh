#!/usr/bin/env bash
# Normalization grid, 12 cells (2026-08-01).
#   {teacher feature norm} x {teacher head norm} x {student head: free/cosine/gain}
# All cells share one regime (see any ngrid_*.yaml header): 50ep, 10-step, WA on, k=512, lamda 0,
# no AWP, teacher = clean_200ep. Runs strictly sequentially on one GPU, ~1h/cell => ~12h.
# Waits for any in-flight main.py to finish first so it can be launched immediately.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

while pgrep -f "main.py --config_name featdir_champ200" > /dev/null; do sleep 60; done

CELLS="t00_free t00_cos t01_free t01_cos t10_free t10_cos t11_free t11_cos"
for c in $CELLS; do
  echo "=== $(date +%H:%M) start ngrid_$c ==="
  $PY main.py --config_name "ngrid_${c}.yaml" --dataset CIFAR100 --seed 0 \
      > "logs/ngrid_${c}.log" 2>&1
  echo "=== $(date +%H:%M) done ngrid_$c (exit $?) ==="
done
echo "=== grid complete $(date) ==="
