#!/bin/bash
# CIFAR-100 overnight: train natural teacher -> measure dispersion (DECISION GATE)
# -> isolated 2x2 (self-calibrating alpha from measured raw norm).
# The dispersion number answers: is C100 norm dispersion big enough for the
# softening mechanism to fire (unlike CIFAR10's 1.45x)?
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p results/CIFAR100

run () { # cfg tau alpha
  echo "############ C100 $1  tau=$2 alpha=$3  START $(date) ############"
  $PY -u main.py --config_name "$1" --tau "$2" --alpha "$3" --dataset CIFAR100
  echo "############ C100 $1  tau=$2 alpha=$3  DONE  $(date) ############"
}

# 1) train natural teacher (clean) -> CIFAR100/checkpoint/clean/clean_last.pkl
echo "######## C100 TEACHER TRAIN START $(date) ########"
$PY -u main.py --config_name clean.yaml --dataset CIFAR100
echo "######## C100 TEACHER TRAIN DONE $(date) ########"

# 2) measure dispersion (writes results/CIFAR100/c100_norm_stats.txt + median file)
$PY -u scratchpad/measure_c100_norm.py
R=$(cat results/CIFAR100/c100_rawnorm_median.txt)
echo "######## C100 raw-norm median R=$R ########"

# centered alpha for norm-adaptive: alpha = median_kappa - tau*R  (median_kappa=0.5)
a_iso4_005=$(awk "BEGIN{printf \"%.3f\", 0.5 - 0.05*$R}")
a_iso4_010=$(awk "BEGIN{printf \"%.3f\", 0.5 - 0.10*$R}")
# no-norm const: bracket R
a_lo=$(awk "BEGIN{printf \"%.2f\", 0.7*$R}")
a_hi=$(awk "BEGIN{printf \"%.2f\", 1.3*$R}")

# 3) isolated 2x2 on C100
# norm row (directional logits -> alpha ~0.5 scale, dataset-robust)
run iso3_norm_const.yaml  0     0.3
run iso3_norm_const.yaml  0     0.5
run iso3_norm_const.yaml  0     0.7
run iso4_norm_adapt.yaml  0.05  "$a_iso4_005"
run iso4_norm_adapt.yaml  0.10  "$a_iso4_010"
# no-norm row (raw scale -> alpha from R)
run iso1_nonorm_const.yaml  0    "$a_lo"
run iso1_nonorm_const.yaml  0    "$R"
run iso1_nonorm_const.yaml  0    "$a_hi"
run iso2_nonorm_adapt.yaml  1.0  0.5
run iso2_nonorm_adapt.yaml  1.5  0.5

echo "######## C100 ALL DONE $(date) ########"
