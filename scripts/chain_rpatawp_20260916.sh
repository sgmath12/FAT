#!/usr/bin/env bash
# 2026-09-16 07:55.  Relaunch after the 07:48 reboot; the 30-epoch WA cells are dropped at the user request.
# ResNet-18, 200 epochs, its own WA) + AWP gamma 0.005 warmup 20 on both datasets, then the rest.
# RPAT++ runs outside main.py and so takes no GPU lock; running it inside this serial chain is what
# keeps it from overlapping another job.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=0
run_fat() {
  DS="$1"; cfg="$2"
  echo "=== $(date '+%m-%d %H:%M') start $DS/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset $DS --seed 0 > logs/${DS}_${cfg}_s0.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $DS/$cfg (exit $?) ==="
}
run_rpat() {
  NC="$1"; DD="$2"; FN="$3"; NI="${4:-}"
  echo "=== $(date '+%m-%d %H:%M') start RPAT++ + AWP $FN ==="
  BACKGROUND=0 NEW_LOG=1 NUM_CLASSES=$NC DATA_DIR=$DD FNAME=$FN AWP_GAMMA=0.005 AWP_WARMUP=20 \
    NATURAL_INIT="$NI" bash /mnt/d/research/RPAT/run.sh > logs/rpat_${FN}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done RPAT++ + AWP $FN (exit $?) ==="
}
run_rpat 100 /mnt/d/research/FAT/data/CIFAR100 rpatpp_awp_natinit_c100_resnet18 \
  /mnt/d/research/FAT/CIFAR100/checkpoint/clean_200ep/clean_last.pkl
run_rpat 10  /mnt/d/research/FAT/data/CIFAR10  rpatpp_awp_c10_resnet18
run_rpat 10  /mnt/d/research/FAT/data/CIFAR10  rpatpp_awp_natinit_c10_resnet18 \
  /mnt/d/research/FAT/CIFAR10/checkpoint/clean_200ep/clean_last.pkl
for entry in CIFAR100:adaadigdm_natinit_stack_100ep CIFAR100:consistency_natinit_stack_100ep \
             CIFAR100:lbgat_natinit_stack_100ep \
             CIFAR100:consistency_natinit_100ep CIFAR100:lbgat_natinit_100ep \
             CIFAR100:adr_natinit_200ep CIFAR100:adr_natinit_stack_200ep \
             CIFAR100:consistency_full_100ep CIFAR100:adr_full_100ep; do
  run_fat "${entry%%:*}" "${entry##*:}"
done
