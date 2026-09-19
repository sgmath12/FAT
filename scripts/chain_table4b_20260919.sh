#!/usr/bin/env bash
# 2026-09-19 09:30.  Replaces chain_table4_20260919.sh: only the cells the paper's tables are missing.
# Four fill Table 4 (own recipe + natural init + stack), three fill the teacher-initialization rows of
# tab:natteacher.  The two *_full cells, which gave baselines our recipe as well, are dropped -- that
# comparison now lives in notes/rebuttal.md and its table is already complete.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=76590
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "=== $(date '+%m-%d %H:%M') adaadigdm_natinit_stack_100ep finished ==="
run_fat() {
  DS="$1"; cfg="$2"
  echo "=== $(date '+%m-%d %H:%M') start $DS/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset $DS --seed 0 > logs/${DS}_${cfg}_s0.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $DS/$cfg (exit $?) ==="
}
for entry in CIFAR100:consistency_natinit_stack_100ep CIFAR100:lbgat_natinit_stack_100ep \
             CIFAR100:adr_natinit_stack_200ep \
             CIFAR100:consistency_natinit_100ep CIFAR100:lbgat_natinit_100ep \
             CIFAR100:adr_natinit_200ep; do
  run_fat "${entry%%:*}" "${entry##*:}"
done
