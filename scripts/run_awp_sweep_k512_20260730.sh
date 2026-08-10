#!/bin/bash
# 2026-07-30 21:45 — REPLACES run_awp_eps_lamda_sweep_20260730.sh (its driver was killed; the
# k512_lamda0 cell it had already launched was left to finish untouched). Two changes the user
# asked for: (1) add the k512+lamda4 BRIDGE cell so the dimension effect and the lamda effect can
# be attributed separately, (2) unify EVERY remaining cell to eta 512.
#
# Reference row to beat (k350, lamda4, 100ep+AWP): clean 63.07 / pgd20 34.12 / cw 28.20 / AA 25.98
# Other bars: champion (50ep, k350, lamda4) 62.75/33.96/28.41/AA 26.29 · ADR-full AA 28.50.
# All cells: 100 or 200 ep, AWP proxy gamma 0.005 warmup 10, AdamW 0.021, WA kappa 0.999,
# tau16/beta1, 10-step, eval eps 8/255, `aa: True` so AutoAttack runs at the end of each cell.
#
#   A  k512 lamda 0.0   <- already running when this script was written, NOT relaunched here
#      k512 lamda 1.5                                                    ~2h10
#      k512 lamda 4.0   (bridge)                                         ~2h10
#   B  k512 100ep: train_eps 9 / 10 / 12 (lamda4), lamda 2 / 8 (eps8)    ~2h10 each -> 10h50
#   C  k512 200ep: train_eps 8 / 10 / 12 (lamda4)                        ~4h20 each -> 12h45
# ~28h total from here. Kill from the back if that is too long.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
CHAIN=results/CIFAR100/awp_sweep_20260730.log

# Wait for the in-flight k512_lamda0 cell (launched by the previous driver) to exit.
while pgrep -f "featdir_awp_100ep_k512_lamda0" > /dev/null; do sleep 60; done
line=$(grep -ah "last_aa_acc\|last_pgd20_acc" results/CIFAR100/ResNet18/featdir_awp_100ep_k512_lamda0/*.log 2>/dev/null | tail -2 | tr '\n' ' ')
echo "=== featdir_awp_100ep_k512_lamda0 DONE (prev driver) $(date) | $line" >> $CHAIN
echo "=== requeued sweep start $(date) ===" >> $CHAIN

run_cell () {
  echo "=== $1 (eta 512) START $(date) ===" >> $CHAIN
  $PY -u main.py --config_name $1.yaml --dataset CIFAR100 --seed 0 --eta 512 \
      > results/CIFAR100/$1_driver.log 2>&1
  rc=$?
  line=$(grep -ah "last_aa_acc\|last_pgd20_acc" results/CIFAR100/ResNet18/$1/*.log 2>/dev/null | tail -2 | tr '\n' ' ')
  echo "=== $1 DONE rc=$rc $(date) | $line" >> $CHAIN
}

run_cell featdir_awp_100ep_k512_lamda15
run_cell featdir_awp_100ep_k512_lamda4
echo "SWEEP_GROUP_A_DONE $(date)" >> $CHAIN

for c in featdir_awp_100ep_eps9 featdir_awp_100ep_eps10 featdir_awp_100ep_eps12 \
         featdir_awp_100ep_lamda2 featdir_awp_100ep_lamda8; do
  run_cell $c
done
echo "SWEEP_GROUP_B_DONE $(date)" >> $CHAIN

for c in featdir_awp_200ep_eps8 featdir_awp_200ep_eps10 featdir_awp_200ep_eps12; do
  run_cell $c
done
echo "AWP_K512_SWEEP_DONE $(date)" >> $CHAIN
