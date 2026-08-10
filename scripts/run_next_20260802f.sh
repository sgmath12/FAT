#!/usr/bin/env bash
# Queue rev. F (2026-08-02). Restart after the tnorm cell was launched with a stale config: it
# picked up tau 16 at 18:48:05, seconds before the tau 1 edit landed, so its target was effectively
# uniform (max prob 0.0106 vs uniform 0.0100). Killed and its results/checkpoint removed.
#
# Remaining, in order:
#   fg_plain_nofeat_kl_tnorm      teacher phi_t_hat (tau 1, NOT divided) / student normalized
#   fg_plain_nofeat_kl_tnorm_raw  teacher phi_t_hat (tau 1)              / student raw
#   c10_tr_sr_kl / c10_th_sh_kl   CIFAR-10 replication of the backbone axis
#   AA on plain_tr_sr_{kl,ce}     identical backbone, CE vs KD head
#   fg_cos_th_sh_ce               last cell of the pruned CIFAR-100 grid
#
# Done so far (50ep grid, CIFAR-100):  clean / PGD-20 / CW
#   tr_sr_kl        62.68 / 30.51 / 27.24    (raw matching + student norm)
#   nofeat_kl       61.69 / 31.44 / 27.05    (no feature term + student norm)
#   th_sh_kl        62.61 / 29.16 / 26.63    (direction matching + student norm)
#   nofeat_kl_raw   58.43 / 31.98 / 26.70    (no feature term, no norm)
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

run () {  # run <config> <dataset> <logname>
  echo "=== $(date '+%m-%d %H:%M') start $3 ==="
  $PY main.py --config_name "$1" --dataset "$2" --seed 0 > "logs/$3.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $3 (exit $?) ==="
}

run fg_plain_nofeat_kl_tnorm.yaml     CIFAR100 fg_plain_nofeat_kl_tnorm
run fg_plain_nofeat_kl_tnorm_raw.yaml CIFAR100 fg_plain_nofeat_kl_tnorm_raw
run c10_tr_sr_kl.yaml                 CIFAR10  c10_tr_sr_kl
run c10_th_sh_kl.yaml                 CIFAR10  c10_th_sh_kl

for c in plain_tr_sr_kl plain_tr_sr_ce; do
  echo "=== $(date '+%m-%d %H:%M') AA $c ==="
  $PY scripts/eval_aa_generic.py \
      "$c|CIFAR100/checkpoint/fg_$c/feat_direction_last.pkl" 2>&1 | tee -a logs/aa_head_axis.log
done

run fg_cos_th_sh_ce.yaml CIFAR100 fg_cos_th_sh_ce
echo "=== queue complete $(date) ==="
