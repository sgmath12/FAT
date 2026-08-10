#!/usr/bin/env bash
# Reordered queue (2026-08-02). The 16-cell grid's headline finding is that raw-vs-raw feature
# matching beat the champion's hat-vs-hat direction matching (+1.35 PGD, +0.61 CW, tied clean) in
# the stripped regime. Testing whether that survives the full stack outranks the remaining cosine-
# teacher cells, whose first two results are negative on CW (25.05 vs 27.24).
#
#   1. featdir_champ200_rawfeat  -- champion stack, direction loss swapped to raw L2. Bar: AA 28.69.
#   2. the remaining cos-teacher cells.
#
# cos_th_sr_{ce,kl} are DROPPED: the matching plain cells collapsed to 39.5 clean because a raw
# student chasing a unit-norm teacher target has its feature norm crushed to 1. Degenerate by
# construction, not worth 1.7h.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

run () {
  echo "=== $(date '+%m-%d %H:%M') start $2 ==="
  $PY main.py --config_name "$1" --dataset CIFAR100 --seed 0 > "logs/$2.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $2 (exit $?) ==="
}

# wait out the cell that is running right now
while pgrep -f "config""_name fg_" > /dev/null; do sleep 60; done

run featdir_champ200_rawfeat.yaml featdir_champ200_rawfeat

for c in cos_tr_sh_kl cos_th_sh_ce cos_th_sh_kl; do
  run "fg_${c}.yaml" "fg_${c}"
done
echo "=== queue complete $(date) ==="
