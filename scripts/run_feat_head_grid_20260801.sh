#!/usr/bin/env bash
# Feature/head grid, 16 cells (2026-08-01). CIFAR100, 50ep, 10-step, lamda 0, WA on, AWP OFF, seed 0.
#
#   teacher   plain = clean_200ep       |  cos = clean_cos200ep (cosine classifier at training)
#   direction tr = raw phi_t            |  th = phi_t_hat
#   direction sr = raw phi_s            |  sh = phi_s_hat
#   head      ce = CE(head(x_adv), y)   |  kl = KL to z_t/tau16
# WA and AWP are both OFF in every cell.
#
# Head INPUT is the normalized student feature in every cell; only the head target varies.
# P cells run first (teacher on disk), then teacher C is trained, then the C cells.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

run () {
  echo "=== $(date '+%m-%d %H:%M') start $2 ==="
  $PY main.py --config_name "$1" --dataset CIFAR100 --seed 0 > "logs/$2.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $2 (exit $?) ==="
}

while pgrep -f "main.py --config_name" > /dev/null; do sleep 60; done

for c in plain_tr_sr_ce plain_tr_sr_kl plain_tr_sh_ce plain_tr_sh_kl plain_th_sr_ce plain_th_sr_kl plain_th_sh_ce plain_th_sh_kl; do
  run "fg_${c}.yaml" "fg_${c}"
done

run clean_cos200ep.yaml clean_cos200ep      # teacher C

for c in cos_tr_sr_ce cos_tr_sr_kl cos_tr_sh_ce cos_tr_sh_kl cos_th_sr_ce cos_th_sr_kl cos_th_sh_ce cos_th_sh_kl; do
  run "fg_${c}.yaml" "fg_${c}"
done
echo "=== feat/head grid complete $(date) ==="
