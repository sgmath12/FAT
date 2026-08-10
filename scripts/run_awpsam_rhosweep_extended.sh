#!/bin/bash
# Overnight queue (2026-07-19 00:xx, user request): after the currently-running rho=0.005/
# warmup=0 AWP-SAM cell (PID 23383, scripts/run_awpsam_warmup0_overnight.sh) finishes, run
# rho=0.001/warmup=0 as a comparison point, auto-pick the winner by last_cw_acc, then re-run
# the winner at 100 epochs (annealing/cyclic-LR/WA-decay all rescale to the new epoch count, so
# this is a genuinely longer recovery window, not just "more of the same schedule"). Sized to
# ~5h to fit the user's ~6h budget; 150ep intentionally left OUT (would add ~4h more) -- rerun
# manually with --epochs 150 on the winning rho if the 100ep result looks promising.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
AWPLOG=results/CIFAR100/awpsam_chain.log
OUTLOG=results/CIFAR100/featdir_k350wa_awpsam/output.log
CKDIR=CIFAR100/checkpoint/featdir_k350wa_awpsam

until grep -q "AWPSAM_WARMUP0_DONE" $AWPLOG 2>/dev/null; do sleep 60; done
RHO005_RESULT=$(tail -1 $OUTLOG)
echo "=== rho=0.005 warmup=0 (50ep) result: $RHO005_RESULT ===" >> $AWPLOG

echo "=== rho=0.001 warmup=0 smoke epochs:2 START $(date) ===" >> $AWPLOG
if $PY -u main.py --config_name featdir_k350wa_awpsam.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --awp_gamma 0.001 --awp_warmup 0 --epochs 2 \
    > results/CIFAR100/awpsam_rho001_w0_smoke_driver.log 2>&1; then
  echo "=== smoke OK, full50 START $(date) ===" >> $AWPLOG
  $PY -u main.py --config_name featdir_k350wa_awpsam.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --awp_gamma 0.001 --awp_warmup 0 \
    > results/CIFAR100/awpsam_rho001_w0_driver.log 2>&1
  if [ $? -eq 0 ]; then
    cp $CKDIR/feat_direction_last.pkl $CKDIR/awpsam_rho001_w0_last.pkl 2>/dev/null
    echo "=== rho=0.001 warmup=0 (50ep) DONE $(date) ===" >> $AWPLOG
  else
    echo "!!! rho=0.001 warmup=0 FULL RUN FAILED $(date) ===" >> $AWPLOG
  fi
else
  echo "!!! rho=0.001 warmup=0 SMOKE FAILED $(date) ===" >> $AWPLOG
fi
RHO001_RESULT=$(tail -1 $OUTLOG)
echo "=== rho=0.001 warmup=0 (50ep) result: $RHO001_RESULT ===" >> $AWPLOG
echo "AWP_RHO_SWEEP_DONE $(date)" >> $AWPLOG

CW005=$(echo "$RHO005_RESULT" | sed -n "s/.*'last_cw_acc': \([0-9.]*\).*/\1/p")
CW001=$(echo "$RHO001_RESULT" | sed -n "s/.*'last_cw_acc': \([0-9.]*\).*/\1/p")
CW005=${CW005:-0}
CW001=${CW001:-0}
echo "=== cw005=$CW005 cw001=$CW001 ===" >> $AWPLOG

WINNER_RHO=0.005
if awk "BEGIN {exit !($CW001 > $CW005)}"; then
  WINNER_RHO=0.001
fi
echo "=== WINNER rho=$WINNER_RHO -- launching 100ep retest $(date) ===" >> $AWPLOG

echo "=== winner rho=$WINNER_RHO @ 100ep smoke epochs:3 START $(date) ===" >> $AWPLOG
if $PY -u main.py --config_name featdir_k350wa_awpsam.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --awp_gamma $WINNER_RHO --awp_warmup 0 --epochs 3 \
    > results/CIFAR100/awpsam_100ep_smoke_driver.log 2>&1; then
  echo "=== 100ep smoke OK, full START $(date) ===" >> $AWPLOG
  $PY -u main.py --config_name featdir_k350wa_awpsam.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --awp_gamma $WINNER_RHO --awp_warmup 0 --epochs 100 \
    > results/CIFAR100/awpsam_100ep_driver.log 2>&1
  if [ $? -eq 0 ]; then
    cp $CKDIR/feat_direction_last.pkl $CKDIR/awpsam_100ep_last.pkl 2>/dev/null
    echo "=== 100ep (rho=$WINNER_RHO) DONE $(date) ===" >> $AWPLOG
  else
    echo "!!! 100ep FULL RUN FAILED $(date) ===" >> $AWPLOG
  fi
else
  echo "!!! 100ep SMOKE FAILED $(date) ===" >> $AWPLOG
fi
echo "AWP_100EP_DONE $(date)" >> $AWPLOG
echo "AWP_QUEUE_DONE $(date) -- 150ep NOT queued (time budget); if 100ep looks promising rerun manually: --awp_gamma $WINNER_RHO --awp_warmup 0 --epochs 150" >> $AWPLOG
