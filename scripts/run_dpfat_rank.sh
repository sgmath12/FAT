#!/bin/bash
# DPFAT_rank sweep over (tau, alpha), recommended values, best-guess first.
#
# Temperature model (methods.py:469, config DPFAT_rank.yaml):
#   T_i = tau * rank_i + alpha,   rank_i in (0,1]  ->  T_i in [alpha, alpha+tau]
#   - rank-normalizing the signal makes it scale-invariant and forces full
#     per-sample dispersion (the whole point of the _rank variant).
#   - alpha = temperature floor (sharpest targets), tau = dispersion width.
#
# Why these values: a well-tuned *constant* temperature ~0.5 already works well
# on CIFAR (DPFAT_adaptive best effective temp ~0.505). So we keep the floor
# alpha near 0.5 and widen tau to add real dispersion, plus a lower-floor
# (alpha=0.25) pair for sharper targets on confident samples.
#
# NOTE: output/ckpt paths derive from config_name only (main.py:37,163), so every
# run writes to results/CIFAR10/DPFAT_rank/output.log (appended; each run is
# delimited by its "Experiment Configuration" header) and overwrites
# CIFAR10/checkpoint/DPFAT_rank/*.pkl. Same behavior as run_fad_fs.sh — read
# per-run results from the appended output.log.
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

# (tau, alpha) pairs, best-guess first.  T range shown for reference.
RUNS=(
  "0.5 0.5"    # T in [0.50, 1.00]  mild dispersion, near known-good constant temp  <-- top pick
  "1.0 0.5"    # T in [0.50, 1.50]
  "2.0 0.5"    # T in [0.50, 2.50]  (config default)
  "0.75 0.25"  # T in [0.25, 1.00]  lower floor -> sharper targets for confident samples
  "1.5 0.25"   # T in [0.25, 1.75]
)

for pair in "${RUNS[@]}"; do
  set -- $pair
  tau=$1; alpha=$2
  echo "############ DPFAT_rank tau=$tau alpha=$alpha START $(date) ############"
  $PY -u main.py --config_name DPFAT_rank.yaml --tau "$tau" --alpha "$alpha"
  echo "############ DPFAT_rank tau=$tau alpha=$alpha DONE  $(date) ############"
done

echo "############ ALL DONE $(date) ############"
