#!/bin/bash
# CARVE-ONLY (noren) DENSE beta sweep — alpha=1 FIXED, NO global, NO renorm.
# Research question (user): can carve ALONE soften logits to /13 level (entropy~4.58) AND still train?
#   noren: mean(w)~exp(-beta) so beta is the ONLY temperature knob (alpha stays 1).
# Known so far: beta=3.0 -> /13-soft but COLLAPSES (clean 9.7, stuck to step5). beta=2.0 only seen at
#   step0 (11.4/5.38) -> NOT confirmed, must run to completion. beta=1.0 trained healthy early.
# This sweep RUNS EVERY BETA TO COMPLETION (do not kill) so we get real final clean/pgd20 trajectories.
# DESCENDING (soft-first): let the high-beta /13-soft points finish first to answer "soft carve-only viable?".
# steps=10 (2/255), carve PGD-2 (gamma=2), norm student, alpha=1. config fgsm_noren_s10.yaml.
# Results APPEND -> results/CIFAR100/fgsm_noren_s10/output.log (parse by 'tau', take last_clean_acc/last_pgd20_acc).
# Baseline to beat: /13 == L2norm teacher = clean 62.6 / pgd20 31.7 / HARMONIC 42.13.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for b in 2.0 1.75 1.5 1.25 1.0 0.75 0.5 0.25; do
 echo "############ NOREN-DENSE beta=$b alpha=1 START $(date) ############"
 $PY -u main.py --config_name fgsm_noren_s10.yaml --tau "$b" --gamma 2 --alpha 1.0 --dataset CIFAR100
 echo "############ NOREN-DENSE beta=$b DONE $(date) ############"
done
echo "############ C100 NOREN-DENSE ALL DONE $(date) ############"
