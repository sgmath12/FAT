#!/bin/bash
# AWP-SAM cell (2026-07-18), queued behind KAPPA_EXT3_DONE (i.e. after kappa=0.995 --
# see run_kappa_0.995_now.sh; 0.9999 was cancelled by user after 0.9995 came in worse than
# champion, so the kappa sweep is {0.9, 0.9995, 0.995}, no 0.9999). Tests the ADR-repo SAM-style
# AWP
# port (utils.AdvWeightPerturbSAM, awp_style: sam) on the champion (k350+WA+lamda4, tau16/beta1/
# eps8, eta350), rho=0.005 matching ADR's own gin config verbatim. The proxy-model AWP
# (AdvWeightPerturb) was already dropped permanently on 07-17 (gamma sweep 0.005/0.01/0.02 all
# monotonically worse) -- this is a mechanically different port (no proxy network, perturbs all
# params incl. the head bias, transient look-ahead instead of a persistent add/restore around the
# real optimizer step), so it is not assumed to fail the same way; see methods.py's AWP comment.
# Bar to beat (kappa 0.999, no AWP): clean62.75/pgd20 33.96/pgd10 34.18/cw28.41, H(pgd)44.07/H(cw)39.11, AA26.29.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/kappa_sweep_chain.log
CKDIR=CIFAR100/checkpoint/featdir_k350wa_awpsam
AWPLOG=results/CIFAR100/awpsam_chain.log

until grep -q "KAPPA_EXT3_DONE" $LOG 2>/dev/null; do sleep 180; done

echo "=== AWPSAM SMOKE START $(date) ===" >> $AWPLOG
if $PY -u main.py --config_name featdir_k350wa_awpsam.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --epochs 1 \
    > results/CIFAR100/awpsam_smoke_driver.log 2>&1; then
  echo "=== AWPSAM smoke OK, full START $(date) ===" >> $AWPLOG
  $PY -u main.py --config_name featdir_k350wa_awpsam.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 \
    > results/CIFAR100/awpsam_driver.log 2>&1
  cp $CKDIR/feat_direction_last.pkl $CKDIR/awpsam_last.pkl 2>/dev/null
  echo "=== AWPSAM DONE (ckpt archived) $(date) ===" >> $AWPLOG
else
  echo "!!! AWPSAM SMOKE FAILED, skipped $(date) ===" >> $AWPLOG
fi

echo "AWPSAM_DONE $(date)" >> $AWPLOG
