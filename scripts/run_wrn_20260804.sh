#!/usr/bin/env bash
# WRN-34-10 / CIFAR10 plan:
#   B  wrn_champ_angeps_mixupT  (existing mixup teacher)            ~14 h + AA
#   A1 clean_wrn_200ep          (plain WRN teacher, so A matches R18) ~3.4 h
#   A2 wrn_champ_angeps_plainT                                       ~14 h + AA
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
WAIT_PID="${1:?}"
echo "=== $(date '+%m-%d %H:%M') waiting on beta_b20 pid $WAIT_PID ==="
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done

echo "=== $(date '+%m-%d %H:%M') B start wrn_champ_angeps_mixupT ==="
$PY main.py --config_name wrn_champ_angeps_mixupT.yaml --dataset CIFAR10 --seed 0 > logs/wrn_mixupT.log 2>&1
echo "=== $(date '+%m-%d %H:%M') B done (exit $?) ==="

echo "=== $(date '+%m-%d %H:%M') A1 start clean_wrn_200ep (teacher) ==="
$PY main.py --config_name clean_wrn_200ep.yaml --dataset CIFAR10 --seed 0 > logs/c10_clean_wrn_200ep.log 2>&1
rc=$?; echo "=== $(date '+%m-%d %H:%M') A1 done (exit $rc) ==="
if [ $rc -ne 0 ] || [ ! -f CIFAR10/checkpoint/clean_wrn_200ep/clean_last.pkl ]; then
  echo "plain WRN teacher missing -> skip A2"; exit 1; fi

echo "=== $(date '+%m-%d %H:%M') A2 start wrn_champ_angeps_plainT ==="
$PY main.py --config_name wrn_champ_angeps_plainT.yaml --dataset CIFAR10 --seed 0 > logs/wrn_plainT.log 2>&1
echo "=== $(date '+%m-%d %H:%M') A2 done (exit $?) ==="
echo "=== WRN plan complete $(date) ==="
