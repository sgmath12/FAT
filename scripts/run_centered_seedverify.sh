#!/bin/bash
# Paired seed verification (2026-07-06 night): is centered-deltanet's +0.38 (H 40.93 vs 40.55, seed0)
# real, given that delta==0 makes a mechanism-driven gain impossible? Same protocol that exposed the
# swap seed-0 spike (+0.55 -> +0.21+-0.41 over 3 seeds). Runs seeds 1,2 for BOTH configs -> paired
# diffs over 3 seeds. If the gap survives, investigate meta-phase side effects; if it evaporates
# (expected), the centered tie is final and CIFAR stays closed.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
run () {  # $1=config  $2=seed  $3=tag
  $PY main.py --config_name "$1.yaml" --dataset CIFAR100 --seed "$2" \
      > "results/CIFAR100/${3}_seed${2}_driver.log" 2>&1
  echo "${3}_seed${2} DONE $(date)"
}
run temp_baseline_val45k 1 base45k
run temp_deltanet_bilevel_centered 1 centered
run temp_baseline_val45k 2 base45k
run temp_deltanet_bilevel_centered 2 centered
echo "ALL_CENTERED_SEEDVERIFY_DONE $(date)"
