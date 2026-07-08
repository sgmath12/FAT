#!/bin/bash
# Close out the 4-cell {norm,margin}x{raw,rank} temperature ablation.
#
# Context: the margin+rank (tau,alpha) sweep (run_dpfat_rank.sh) is flat
#   (pgd20 53.75-53.87, cw 51.41-51.76 over 5x range) AND below the
#   DPFAT_adaptive baseline (pgd20 54.88 / cw 52.51). So per-sample temperature
#   dispersion via rank does not help here.
#
# Two decisive runs left:
#   1) margin+rank EXTREME -- alpha=0.1, tau=10  => T in [0.1, 10.1], 100x spread.
#      Steelman of the "forced dispersion helps" thesis: low-rank samples get
#      near-hard targets, high-rank get near-uniform. If even max dispersion is
#      flat/worse, the rank story is fully closed. (writes results/CIFAR10/DPFAT_rank/output.log)
#   2) margin+raw -- the only untested cell with REAL, undistorted margin
#      dispersion (signal=margin, rank=False). UNLIKE rank, tau here multiplies
#      the actual margin magnitude (~0-15), so the operating point depends on
#      scale -- a single tau could land saturated/dead and look like baseline
#      unfairly. So sweep tau (alpha=0.5 floor fixed) to give it the same shot
#      the rank cell already got. If the whole tau sweep ~= baseline, per-sample
#      temperature gives nothing over a well-tuned constant temp, regardless of
#      signal. (writes results/CIFAR10/DPFAT_margin_raw/output.log)
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

echo "############ [1] margin+rank EXTREME  alpha=0.1 tau=10  START $(date) ############"
$PY -u main.py --config_name DPFAT_rank.yaml --tau 10 --alpha 0.1
echo "############ [1] margin+rank EXTREME  DONE $(date) ############"

# margin+raw sweep. Teacher margin (top1-top2 logit) measured: mean 7.91, p5 1.47,
# p50 8.65, p95 11.87 (scratchpad/margin_stats.py). So T = tau*margin + alpha.
# Three (alpha=0.5) points walk median temp 0.93 -> 2.23 with dispersion 1.9x -> 3.6x;
# tau=0.4 dropped (median T 3.96 already uniform-hot, predictably degrades like the
# rank tau=10 run). Fourth point lowers alpha to get HIGH dispersion (5.2x) while
# keeping median T ~1 (the good operating temp) -- the real "does margin dispersion
# help at a sane temperature" test.
RAW_RUNS=(
  "0.05 0.5"   # T med 0.93, spread 1.9x  (near known-good constant temp)
  "0.1  0.5"   # T med 1.37, spread 2.6x  (config default)
  "0.2  0.5"   # T med 2.23, spread 3.6x
  "0.1  0.1"   # T med 0.97, spread 5.2x  <-- high dispersion, cool median (steelman)
)
for pair in "${RAW_RUNS[@]}"; do
  set -- $pair
  tau=$1; alpha=$2
  echo "############ margin+raw  tau=$tau alpha=$alpha  START $(date) ############"
  $PY -u main.py --config_name DPFAT_margin_raw.yaml --tau "$tau" --alpha "$alpha"
  echo "############ margin+raw  tau=$tau alpha=$alpha  DONE $(date) ############"
done

echo "############ ALL DONE $(date) ############"
