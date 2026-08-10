#!/bin/bash
# 2026-07-29 night, third stage of the queue. Runs AFTER the AdamW 100ep+AWP pair.
# User's ask was "SGD at 100/200 epochs, then stack AWP like ADR does". Deliberate deviation:
# these cells change the SCHEDULE (OneCycle -> piecewise, /10 at 50%/75%) rather than only
# stretching the epoch count, because the 50ep SGD sweep did not fail from too few epochs --
# it diverged AT THE OneCycle PEAK (lr0.1: clean 71->50->40->3 by ep15-25; lr0.05 identical
# shape one step later). Stretching OneCycle to 100/200ep reproduces the same peak, slower.
# Two cells isolate warm-start vs scratch (ADR trains from scratch; our divergence looked like
# the large LR destroying the clean warm start). AWP goes on whichever survives -- not stacked
# blind tonight, since AWP on a diverging run tells us nothing.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
CHAIN=results/CIFAR100/sgd_lrsweep_20260729.log
until grep -qE "100EP_AWP_PAIR_DONE|100ep\+AWP pair skipped" $CHAIN 2>/dev/null; do sleep 120; done

for cell in featdir_sgd_pw_100ep featdir_sgd_pw_100ep_scratch; do
  echo "=== $cell START $(date) ===" >> $CHAIN
  $PY -u main.py --config_name ${cell}.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 \
      > results/CIFAR100/${cell}_driver.log 2>&1
  echo "=== $cell DONE $(date) ===" >> $CHAIN
done
echo "SGD_PIECEWISE_NIGHT_DONE $(date)" >> $CHAIN
