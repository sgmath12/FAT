#!/bin/bash
# Lamda (self-consistency) sweep on the STORY method (user's plan, 2026-07-13 afternoon):
# featdir (dir-attack, the pure cell: better cw 26.12 + teacher head deleted everywhere) with
# lamda {0.3, 1, 3} -- trade the +1.5 clean headroom back into robustness. Consistency term is
# teacher-free (student(x_adv) vs student(x)) so story purity is intact. PLUS the reviewer-defense
# control: baseline KL pipeline with the same lamda=1 (its config had lamda 0.0).
# klattack verdict (14:51): attack hypothesis REJECTED (65.67/28.54 H 39.79) -> frontier shift is
# intrinsic to the feature-direction loss; lamda is the real lever.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/featdir_round2_chain.log
until grep -q "FEATDIR_ROUND2_DONE" $LOG 2>/dev/null; do sleep 120; done

for l in 0.3 1.0 3.0; do
  echo "=== featdir lamda $l START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir.yaml --dataset CIFAR100 --seed 0 --lamda $l \
    > results/CIFAR100/featdir_lamda${l}_driver.log 2>&1
  echo "=== featdir lamda $l DONE $(date) ===" >> $LOG
done
echo "=== baseline lamda 1.0 control START $(date) ===" >> $LOG
$PY -u main.py --config_name temp_studentNorm_teacherRaw.yaml --dataset CIFAR100 --seed 0 --tau 16 --lamda 1.0 \
  > results/CIFAR100/baseline_lamda1_driver.log 2>&1
echo "FEATDIR_LAMDA_SWEEP_DONE $(date)" >> $LOG
