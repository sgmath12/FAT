#!/bin/bash
# DeltaNet targeted sweep (2026-07-06), chained behind pilot -> baseline_val45k.
# Priority order per the pilot-diagnostics plan:
#   1-2. delta_meta_lr {30, 300}: lr=100 was tuned for taunet's SCALAR head; DeltaNet's 100-dim
#        output head has a different meta-gradient scale -- most likely binding knob.
#   3-4. delta_r {0.15, 0.6}: edit-budget cap; smoke showed p95(||delta||)=0.13 << 0.3, so this
#        only binds if structure grows -- swept second.
# All 3-step, seed0, window 0-10, 45000-train (post-valfix). Results append to
# results/CIFAR100/temp_deltanet_bilevel/output.log -- parse by (delta_r, delta_meta_lr) in the
# config dict line of each run.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
until grep -q "BASELINE_VAL45K_DONE" results/CIFAR100/baseline_val45k_chain.log 2>/dev/null; do sleep 60; done
echo "baseline_val45k done, starting deltanet sweep $(date)"

run_cell () {  # $1=delta_meta_lr  $2=delta_r  $3=tag
  $PY main.py --config_name temp_deltanet_bilevel.yaml --dataset CIFAR100 --seed 0 \
      --delta_meta_lr "$1" --delta_r "$2" \
      > "results/CIFAR100/deltanet_sweep_${3}_driver.log" 2>&1
  echo "deltanet ${3} DONE $(date)"
}

run_cell 30  0.3  "mlr30"
run_cell 300 0.3  "mlr300"
run_cell 100 0.15 "r015"
run_cell 100 0.6  "r060"
echo "ALL_DELTANET_SWEEP_DONE $(date)"
