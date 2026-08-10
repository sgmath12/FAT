#!/bin/bash
# AWP track re-examination (2026-07-16, queued to run unattended): default gamma=5e-3 gave
# near-zero robust gain (and net-negative when stacked with lamda4) on 2026-07-15 night's runs.
# Hypothesis: our L2-feature-normalization partially absorbs AWP's norm-relative weight
# perturbation (diff scaled by ||w_old||/||diff||), muting the intended effect -- consistent with
# the project's recurring "normalization absorbs perturbations" theme. Plan:
#   (1) AA-verify the existing AWP survivors + the lamda-ceiling cells (eval-only, cheap)
#   (2) gamma sweep {0.01, 0.02} (2x, 4x default) on BOTH pipelines' lamda0 isolation cells,
#       each immediately AA-evaluated (separate config folders -> no ckpt overwrite)
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
EVAL=scripts/eval_aa_generic.py

until grep -q "LAMDA_HIGHDOSE_DONE" $LOG 2>/dev/null; do sleep 120; done

echo "=== AWP track2: AA verification START $(date) ===" >> $LOG
$PY -u $EVAL \
  "baseline+WA+AWP+lamda10|CIFAR100/checkpoint/temp_baseline_10step_wa_awp/temperature_last.pkl" \
  "k350+WA+AWP+lamda4|CIFAR100/checkpoint/featdir_k350wa_awp/feat_direction_last.pkl" \
  "baseline+WA lamda300 (ceiling)|CIFAR100/checkpoint/temp_baseline_10step_wa/temperature_last.pkl" \
  "k350+WA lamda800 (scaled-high)|CIFAR100/checkpoint/featdir_span_random_10step_wa/feat_direction_last.pkl" \
  > results/CIFAR100/aa_verify_20260716.log 2>&1
echo "=== AWP track2: AA verification DONE $(date) ===" >> $LOG

run_and_aa () {
  cfg=$1; label=$2; ckdir=$3; ckfile=$4
  echo "=== $cfg START $(date) ===" >> $LOG
  $PY -u main.py --config_name $cfg.yaml --dataset CIFAR100 --seed 0 --eta 350 \
    > results/CIFAR100/${cfg}_driver.log 2>&1
  echo "=== $cfg DONE $(date) ===" >> $LOG
  $PY -u $EVAL "${label}|CIFAR100/checkpoint/${ckdir}/${ckfile}" >> results/CIFAR100/aa_verify_20260716.log 2>&1
}

run_and_aa "temp_baseline_10step_wa_awp_g010" "baseline+WA+AWP gamma0.01" "temp_baseline_10step_wa_awp_g010" "temperature_last.pkl"
run_and_aa "featdir_k350wa_awp_g010"          "k350+WA+AWP gamma0.01"     "featdir_k350wa_awp_g010"          "feat_direction_last.pkl"
run_and_aa "temp_baseline_10step_wa_awp_g020" "baseline+WA+AWP gamma0.02" "temp_baseline_10step_wa_awp_g020" "temperature_last.pkl"
run_and_aa "featdir_k350wa_awp_g020"          "k350+WA+AWP gamma0.02"     "featdir_k350wa_awp_g020"          "feat_direction_last.pkl"

echo "AWP_GAMMA_DONE $(date)" >> $LOG
