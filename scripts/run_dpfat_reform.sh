#!/bin/bash
# Normalization isolation experiments.
#
# Hypothesis: the ~0.7pp gap (DPFAT_adaptive 54.88 vs margin/rank ~54.25) is due
# to reformation=True (ResNet18_z, unit-norm features) not per-sample adaptivity.
#
# With reformation=True, feat.norm() ≈ 1 for all samples, so the original
# DPFAT_adaptive's tau*norm+alpha ≈ constant. We test two things:
#
# [1] DPFAT_const_reform  (tau=0.0, reformation=True):
#     Pure constant temperature + normalization, zero adaptivity.
#     If pgd20 ≈ 54.88 → normalization explains everything, adaptivity does nothing.
#
# [2] DPFAT_margin_reform  (signal=margin, reformation=True, tau sweep):
#     reformation=True logit scale ~6x smaller → tau needs 6x rescale.
#     no_normalize best: tau=0.2, T_med≈2.1 → reform target: tau~1.0-1.5, alpha=0.5.
#     If still ≤ 54.88 → margin signal adds nothing even with correct normalization.
#
# Results write to:
#   results/CIFAR10/DPFAT_const_reform/output.log
#   results/CIFAR10/DPFAT_margin_reform/output.log
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

echo "############ [1] DPFAT_const_reform  tau=0.0 alpha=0.5  START $(date) ############"
$PY -u main.py --config_name DPFAT_const_reform.yaml --tau 0.0 --alpha 0.5
echo "############ [1] DPFAT_const_reform  DONE $(date) ############"

# tau sweep for margin+reform: best-guess first
# T = tau*margin + alpha,  reform margin: mean~1.3, p5~0.25, p95~2.0
REFORM_RUNS=(
  "1.0 0.5"   # T_mean~1.8, T_p5~0.75, T_p95~2.5  (top pick: matches no_normalize best op point)
  "1.5 0.5"   # T_mean~2.5, T_p5~0.88, T_p95~3.5
  "0.5 0.5"   # T_mean~1.2, T_p5~0.62, T_p95~1.5  (conservative)
  "2.0 0.5"   # T_mean~3.1, T_p5~1.0,  T_p95~4.5  (hot, likely too warm)
)

for pair in "${REFORM_RUNS[@]}"; do
  set -- $pair
  tau=$1; alpha=$2
  echo "############ [2] DPFAT_margin_reform  tau=$tau alpha=$alpha  START $(date) ############"
  $PY -u main.py --config_name DPFAT_margin_reform.yaml --tau "$tau" --alpha "$alpha"
  echo "############ [2] DPFAT_margin_reform  tau=$tau alpha=$alpha  DONE $(date) ############"
done

echo "############ ALL DONE $(date) ############"
