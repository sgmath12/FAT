#!/bin/bash
# featdir round 2 (2026-07-13 afternoon): beta sweep KILLED (AdamW scale-invariance -> provably
# +empirically flat; featdir betas 0.25-4 all 64.53/28.8-28.9). Replacement knobs that move:
#   (1) featdir_klattack       -- attack isolation; recovery => story-faithful method == baseline
#   (2) klattack tau {8, 32}   -- head-target sharpness dose on the story-method candidate
#   (3) featdir_gainhead_nodecay -- close the user's no-decay question in the featdir pipeline
# Bars: baseline 41.77 (63.04/31.23), featdir dir-attack 39.93 (64.53/28.91).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/featdir_round2_chain.log

run () {
  echo "=== $1 (tau ${2:-16}) START $(date) ===" >> $LOG
  $PY -u main.py --config_name $1.yaml --dataset CIFAR100 --seed 0 ${2:+--tau $2} \
    > results/CIFAR100/$1_tau${2:-16}_driver.log 2>&1
  echo "=== $1 (tau ${2:-16}) DONE $(date) ===" >> $LOG
}
run featdir_klattack
run featdir_klattack 8
run featdir_klattack 32
run featdir_gainhead_nodecay
echo "FEATDIR_ROUND2_DONE $(date)" >> $LOG
