#!/usr/bin/env bash
# RESUME after the 22:46 stop (2026-08-31 night).
#
# champ_eps8 had already FINISHED training when it was killed -- main.py writes `_last.pkl` before the
# evaluation block and that file is dated 22:27, one minute after the 100th epoch.  It was killed
# inside AutoAttack.  So it does not need retraining, only re-evaluating from the checkpoint, which
# saves 2.2 hours.  The remaining four cells start from scratch.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

echo "=== $(date '+%m-%d %H:%M') re-evaluating champ_eps8 from its checkpoint ==="
$PY -u scripts/eval_tin_from_ckpt.py --cell champ_eps8 --dataset CIFAR100 --bs 128 --aa_bs 256 \
    > logs/CIFAR100_champ_eps8_evalonly.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done champ_eps8 eval (exit $?) ==="
grep -E "^EVAL|^AA|aa " logs/CIFAR100_champ_eps8_evalonly.log | tail -3

run () {
  echo "=== $(date '+%m-%d %H:%M') start $1/$2 ==="
  $PY -u main.py --config_name "$2.yaml" --dataset "$1" --seed 0 > "logs/$1_$2.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $1/$2 (exit $?) ==="
}
run CIFAR100 champ_eps10
run CIFAR10  champ_eps88
run CIFAR10  champ_eps8
run CIFAR10  champ_eps10
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
