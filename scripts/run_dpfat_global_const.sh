#!/bin/bash
# Missing 2x2 cell: NO normalization + GLOBAL CONSTANT temperature.
#
# reformation=False (no L2-norm) + tau=0 (kills the per-sample norm term) =>
# T = tau*||Phi|| + alpha = alpha  (one constant temperature for every sample).
# The most stripped-down distillation baseline: no normalization, no adaptivity.
#
# alpha = the constant temperature. Raw feat norm (reform=False): mean 6.46,
# p5 5.28, p50 6.44, p95 7.65. Best adaptive no_normalize run (tau=1.0) sits at
# effective T(med)~6.9, so sweep alpha to bracket that operating point.
#
# Completes the 2x2 (pgd20):
#                 |  const (tau=0)   |  adaptive (tau>0)
#   no-norm (F)   |  THIS sweep      |  54.73 (tau=1.0)
#   norm    (T)   |  54.90           |  54.88
# If this lands ~54.7, all four cells are flat => the entire DPD apparatus
# (normalization + adaptive temperature) reduces to a single well-tuned constant.
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

# tau=0 always; alpha = constant temperature, bracketing effective T(med)~6.9
ALPHAS=(6 7 5 8)

for alpha in "${ALPHAS[@]}"; do
  echo "############ global_const reform=F tau=0 alpha=$alpha START $(date) ############"
  $PY -u main.py --config_name DPFAT_adaptive_no_normalize.yaml --tau 0 --alpha "$alpha"
  echo "############ global_const reform=F tau=0 alpha=$alpha DONE  $(date) ############"
done

echo "############ ALL DONE $(date) ############"
