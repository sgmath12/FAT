#!/usr/bin/env bash
# Queue rev. B (2026-08-02). rawfeat came back a robustness TIE (PGD identical to 3 s.f., AA -0.12)
# and clean -0.95, so normalization is a clean-accuracy lever, not a robustness mechanism, and the
# 50ep grid's raw>hat ordering did NOT survive the full stack. Focus shifts to the head axis.
#
# 1. featdir_champ200_tau1      -- champion with a SHARP head target (tau 1). Separates smoothing
#    from teacher dark knowledge. Bar: 60.74 / 34.94 / 30.53 / AA 28.69.
# 2. AA on plain_tr_sr_{kl,ce}  -- identical backbone, CE vs KD head, no training needed. If AA
#    tracks CW (+0.15) rather than PGD (+3.46), the head gain is an attack-loss artifact.
# 3. fg_cos_th_sh_ce            -- last cell of the pruned grid.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

while pgrep -f "config""_name fg_cos_th_sh_kl" > /dev/null; do sleep 60; done

echo "=== $(date '+%m-%d %H:%M') start tau1 ==="
$PY main.py --config_name featdir_champ200_tau1.yaml --dataset CIFAR100 --seed 0 \
    > logs/featdir_champ200_tau1.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done tau1 ==="

for c in plain_tr_sr_kl plain_tr_sr_ce; do
  echo "=== $(date '+%m-%d %H:%M') AA $c ==="
  $PY scripts/eval_aa_generic.py \
      "$c|CIFAR100/checkpoint/fg_$c/feat_direction_last.pkl" 2>&1 | tee -a logs/aa_head_axis.log
done

echo "=== $(date '+%m-%d %H:%M') start fg_cos_th_sh_ce ==="
$PY main.py --config_name fg_cos_th_sh_ce.yaml --dataset CIFAR100 --seed 0 \
    > logs/fg_cos_th_sh_ce.log 2>&1
echo "=== queue complete $(date) ==="
