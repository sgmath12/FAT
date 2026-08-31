#!/usr/bin/env bash
# TRAINING-RADIUS SWEEP ON THE CHAMPION RECIPE (2026-08-31 night).
#
# eps_train in {8, 8.8, 10}/255, evaluation fixed at 8/255, everything else identical.  8.8 is a
# non-standard number in the shipped recipe and a reviewer will ask about it; this answers with the
# standard 8/255 and with 10/255 on both CIFAR datasets.  CIFAR-100 already has 8.8
# (`l2_bestrecipe_freezehead`, 62.65 / 28.77 / 39.43), so only two cells are needed there.
#
# CIFAR-10 gets three, because it has never been run on the raw-L2 design at all -- its 84.66 / 51.87
# champion is the directional variant with freeze_lr_epoch 0.65.  These three put CIFAR-10 on the
# objective the method section actually describes, and the 8.8 cell is what compares with 84.66.
#
# CIFAR-100 first: it is the dataset every other table in the paper is built on, and it is cheaper.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

run () {  # run <dataset> <config>
  echo "=== $(date '+%m-%d %H:%M') start $1/$2 ==="
  $PY -u main.py --config_name "$2.yaml" --dataset "$1" --seed 0 > "logs/$1_$2.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $1/$2 (exit $?) ==="
}

run CIFAR100 champ_eps8
run CIFAR100 champ_eps10
run CIFAR10  champ_eps88
run CIFAR10  champ_eps8
run CIFAR10  champ_eps10
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
