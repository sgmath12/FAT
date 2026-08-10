#!/bin/bash
# Overnight run (2026-07-18 23:5x, user going to sleep): AWP-SAM with TRUE ADR settings --
# rho=0.005 (ADR's own resnet18_pgd_awp_adr.gin default) AND awp_warmup=0 (ADR applies AWP from
# epoch 0, no warmup -- the earlier awp_warmup:5 in the yaml was inherited from the OLD dropped
# proxy-AWP convention, not from ADR itself; confirmed by reading ADR's actual gin configs, no
# warmup param exists there for AWP). Prior probes (warmup=5, rho=0.005 then rho=0.001) both
# collapsed hard the instant AWP switched on (clean ~55->~2-3%) then SLOWLY partially recovered
# (rho=0.001: 2.81->15.98->23.43 clean by ep12, still far below baseline) -- testing whether
# removing the abrupt on/off discontinuity (warmup=0, AWP live from the very first batch) avoids
# that collapse entirely.
# Bar to beat (kappa 0.999, no AWP): clean62.75/pgd20 33.96/pgd10 34.18/cw28.41, H(pgd)44.07/H(cw)39.11, AA26.29.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
AWPLOG=results/CIFAR100/awpsam_chain.log
CKDIR=CIFAR100/checkpoint/featdir_k350wa_awpsam

echo "=== AWPSAM warmup0 (ADR-exact: rho=0.005, warmup=0) smoke epochs:2 START $(date) ===" >> $AWPLOG
if $PY -u main.py --config_name featdir_k350wa_awpsam.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --awp_gamma 0.005 --awp_warmup 0 --epochs 2 \
    > results/CIFAR100/awpsam_warmup0_smoke_driver.log 2>&1; then
  echo "=== smoke OK, full 50ep START $(date) ===" >> $AWPLOG
  $PY -u main.py --config_name featdir_k350wa_awpsam.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --awp_gamma 0.005 --awp_warmup 0 \
    > results/CIFAR100/awpsam_warmup0_driver.log 2>&1
  if [ $? -eq 0 ]; then
    cp $CKDIR/feat_direction_last.pkl $CKDIR/awpsam_warmup0_last.pkl 2>/dev/null
    echo "=== AWPSAM warmup0 DONE (ckpt archived) $(date) ===" >> $AWPLOG
  else
    echo "!!! AWPSAM warmup0 FULL RUN FAILED $(date) ===" >> $AWPLOG
  fi
else
  echo "!!! AWPSAM warmup0 SMOKE FAILED $(date) ===" >> $AWPLOG
fi
echo "AWPSAM_WARMUP0_DONE $(date)" >> $AWPLOG
