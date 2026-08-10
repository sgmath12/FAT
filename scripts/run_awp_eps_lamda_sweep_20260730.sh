#!/bin/bash
# 2026-07-30 (user): sweep on the long-schedule AWP recipe. Sequential, single GPU.
# Every cell has `aa: True`, so AutoAttack runs at the end of each run -- no separate AA pass.
# Eval eps is always 8/255; only the TRAINING attack (train_eps) moves.
#
# Ordered by the user's stated priority, so the tail can be killed without losing the front:
#   A. k512 (mechanism OFF) + 100ep + AWP, lamda {0, 1.5}     ~2h10 each  ->  4h20
#   B. 100ep + AWP: train_eps {9,10,12} @lamda4, lamda {2,8} @eps8   ~2h10 each -> 10h50
#   C. 200ep + AWP: train_eps {8,10,12} @lamda4               ~4h20 each  -> 12h45
# Full run is ~28h, i.e. LONGER than a day -- kill from the back if that is too long.
#
# Context: AWP works at 100ep (control loses PGD 1.58 / CW 1.18 / AA 1.13 to the 50ep champion;
# AWP recovers all the PGD/CW and +0.82 AA) but 100ep+AWP is still 0.31 AA short of the 50ep
# champion. So this sweep is looking for a dial that makes the long schedule actually pay.
# Bars: champion (50ep) 62.75/33.96/cw28.41/AA26.29/NRR37.06 · ADR-full AA28.50/NRR38.08.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
CHAIN=results/CIFAR100/awp_sweep_20260730.log
echo "=== sweep start $(date) ===" >> $CHAIN

run_cell () {   # $1 = config basename, $2 = eta
  echo "=== $1 (eta $2) START $(date) ===" >> $CHAIN
  $PY -u main.py --config_name $1.yaml --dataset CIFAR100 --seed 0 --eta $2 \
      > results/CIFAR100/$1_driver.log 2>&1
  rc=$?
  line=$(grep -ah "last_aa_acc\|last_pgd20_acc" results/CIFAR100/ResNet18/$1/*.log 2>/dev/null | tail -2 | tr '\n' ' ')
  echo "=== $1 DONE rc=$rc $(date) | $line" >> $CHAIN
}

# A. mechanism-off + low lamda (user's latest ask)
run_cell featdir_awp_100ep_k512_lamda0  512
run_cell featdir_awp_100ep_k512_lamda15 512
echo "SWEEP_GROUP_A_DONE $(date)" >> $CHAIN

# B. 100ep train_eps / lamda
for c in featdir_awp_100ep_eps9 featdir_awp_100ep_eps10 featdir_awp_100ep_eps12 \
         featdir_awp_100ep_lamda2 featdir_awp_100ep_lamda8; do
  run_cell $c 350
done
echo "SWEEP_GROUP_B_DONE $(date)" >> $CHAIN

# C. 200ep train_eps
for c in featdir_awp_200ep_eps8 featdir_awp_200ep_eps10 featdir_awp_200ep_eps12; do
  run_cell $c 350
done
echo "AWP_EPS_LAMDA_SWEEP_DONE $(date)" >> $CHAIN
