#!/usr/bin/env bash
# 2026-09-17 16:10.  Replaces chain_nostack4_20260916.sh: the CIFAR-100 Consistency-AT + RPAT run went
# NaN at epoch 1 (the same unclamped KL target, this time in RPAT_Benchmarks/training/__init__.py) and
# is requeued after the CIFAR-10 one, which started after the fix.
# was 2026-09-16 18:20.  Replaces chain_nostack3_20260916.sh; Consistency-AT + RPAT moves ahead of the
# CIFAR-10 RPAT++ cells., which replaced chain_rpatawp2_20260916.sh (killed while the natural-init CIFAR-100
# RPAT++ run, pid 9830, was training; that python keeps running and this waits for it).
# Next: CIFAR-10 CFA with the stack removed at both radii, then ADR with its stack removed on both
# datasets, then the RPAT++ CIFAR-10 cells, then the queue that was already pending.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=38141
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "=== $(date '+%m-%d %H:%M') Consistency-AT + RPAT cifar10 finished ==="
run_fat() {
  DS="$1"; cfg="$2"
  echo "=== $(date '+%m-%d %H:%M') start $DS/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset $DS --seed 0 > logs/${DS}_${cfg}_s0.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $DS/$cfg (exit $?) ==="
}
run_bench() {
  DS="$1"
  echo "=== $(date '+%m-%d %H:%M') start Consistency-AT + RPAT $DS ==="
  DATASET=$DS bash /mnt/d/research/RPAT/run_benchmarks.sh > logs/rpat_bench_${DS}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done Consistency-AT + RPAT $DS (exit $?) ==="
}
run_rpat() {
  NC="$1"; DD="$2"; FN="$3"; NI="${4:-}"
  echo "=== $(date '+%m-%d %H:%M') start RPAT++ + AWP $FN ==="
  BACKGROUND=0 NEW_LOG=1 NUM_CLASSES=$NC DATA_DIR=$DD FNAME=$FN AWP_GAMMA=0.005 AWP_WARMUP=20 \
    NATURAL_INIT="$NI" bash /mnt/d/research/RPAT/run.sh > logs/rpat_${FN}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done RPAT++ + AWP $FN (exit $?) ==="
}
run_bench cifar100
run_rpat 10 /mnt/d/research/FAT/data/CIFAR10 rpatpp_awp_c10_resnet18
run_rpat 10 /mnt/d/research/FAT/data/CIFAR10 rpatpp_awp_natinit_c10_resnet18 \
  /mnt/d/research/FAT/CIFAR10/checkpoint/clean_200ep/clean_last.pkl
for entry in CIFAR100:adaadigdm_natinit_stack_100ep CIFAR100:consistency_natinit_stack_100ep \
             CIFAR100:lbgat_natinit_stack_100ep \
             CIFAR100:consistency_natinit_100ep CIFAR100:lbgat_natinit_100ep \
             CIFAR100:adr_natinit_200ep CIFAR100:adr_natinit_stack_200ep \
             CIFAR100:consistency_full_100ep CIFAR100:adr_full_100ep; do
  run_fat "${entry%%:*}" "${entry##*:}"
done
