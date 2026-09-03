#!/usr/bin/env bash
# LBGAT on its own schedule (lr_schedule: lbgat), both datasets.
#
# CIFAR-100 already landed at 56.46 / 26.02, against 57.10 / 25.99 under our flat protocol -- the
# schedule barely matters there, which is consistent with LBGAT having trained fine under both.
# CIFAR-10 is the one that collapsed to 9.99 under the flat protocol and is the reason this exists.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for ds in CIFAR10 CIFAR100; do
  if grep -ql "last_aa_acc" results/$ds/*/lbgat_100ep/*.log 2>/dev/null && \
     [ "$(ls -t results/$ds/*/lbgat_100ep/*.log 2>/dev/null | head -1 | xargs -r stat -c %Y)" -gt 1756900000 ]; then
    echo "=== $(date '+%m-%d %H:%M') skip $ds/lbgat_100ep ==="; continue
  fi
  echo "=== $(date '+%m-%d %H:%M') start $ds/lbgat_100ep ==="
  $PY -u main.py --config_name lbgat_100ep.yaml --dataset "$ds" --seed 0 > "logs/${ds}_lbgat_100ep.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $ds/lbgat_100ep (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
