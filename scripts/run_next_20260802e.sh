#!/usr/bin/env bash
# Queue rev. E (2026-08-02).
#   1. fg_plain_nofeat_kl  -- CIFAR-100, no feature term (KL only) in the 50ep grid regime.
#                             Compare: fg_plain_th_sh_kl = 62.61 / 29.16 / 26.63.
#   2. c10_tr_sr_kl        -- CIFAR-10 replication, raw feature matching.
#   3. c10_th_sh_kl        -- CIFAR-10 replication, direction matching (the ordering partner).
#   4. AA on plain_tr_sr_{kl,ce}
#   5. fg_cos_th_sh_ce     -- last cell of the pruned CIFAR-100 grid.
# CIFAR-10 teacher: CIFAR10/checkpoint/clean/clean_last.pkl, measured 94.20% clean.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

while pgrep -f "config""_name featdir_champ200_freezehead" > /dev/null; do sleep 60; done

run () {  # run <config> <dataset> <logname>
  echo "=== $(date '+%m-%d %H:%M') start $3 ==="
  $PY main.py --config_name "$1" --dataset "$2" --seed 0 > "logs/$3.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $3 (exit $?) ==="
}

run fg_plain_nofeat_kl.yaml CIFAR100 fg_plain_nofeat_kl
run fg_plain_nofeat_kl_raw.yaml CIFAR100 fg_plain_nofeat_kl_raw
run fg_plain_nofeat_kl_tnorm.yaml CIFAR100 fg_plain_nofeat_kl_tnorm
run fg_plain_nofeat_kl_tnorm_raw.yaml CIFAR100 fg_plain_nofeat_kl_tnorm_raw
run c10_tr_sr_kl.yaml       CIFAR10  c10_tr_sr_kl
run c10_th_sh_kl.yaml       CIFAR10  c10_th_sh_kl

for c in plain_tr_sr_kl plain_tr_sr_ce; do
  echo "=== $(date '+%m-%d %H:%M') AA $c ==="
  $PY scripts/eval_aa_generic.py \
      "$c|CIFAR100/checkpoint/fg_$c/feat_direction_last.pkl" 2>&1 | tee -a logs/aa_head_axis.log
done

run fg_cos_th_sh_ce.yaml CIFAR100 fg_cos_th_sh_ce
echo "=== queue complete $(date) ==="
