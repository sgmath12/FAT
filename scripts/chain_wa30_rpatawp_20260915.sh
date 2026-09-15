#!/usr/bin/env bash
# 2026-09-15 22:00.  Replaces chain_wa30_20260915.sh (killed while CIFAR-100 cfa_wa_eps88_50ep ran; that
# python keeps running).  CFA sample-wise eps + WA at 30 epochs (4 cells), then RPAT++ (official repo,
# ResNet-18, 200 epochs, its own WA) + AWP gamma 0.005 warmup 20 on both datasets, then the rest.
# RPAT++ runs outside main.py and so takes no GPU lock; running it inside this serial chain is what
# keeps it from overlapping another job.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=25664
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "=== $(date '+%m-%d %H:%M') CIFAR-100 cfa_wa_eps88_50ep finished ==="
run_fat() {
  DS="$1"; cfg="$2"
  echo "=== $(date '+%m-%d %H:%M') start $DS/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset $DS --seed 0 > logs/${DS}_${cfg}_s0.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $DS/$cfg (exit $?) ==="
}
run_rpat() {
  NC="$1"; DD="$2"; FN="$3"
  echo "=== $(date '+%m-%d %H:%M') start RPAT++ + AWP $FN ==="
  BACKGROUND=0 NEW_LOG=1 NUM_CLASSES=$NC DATA_DIR=$DD FNAME=$FN AWP_GAMMA=0.005 AWP_WARMUP=20 \
    bash /mnt/d/research/RPAT/run.sh > logs/rpat_${FN}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done RPAT++ + AWP $FN (exit $?) ==="
}
for entry in CIFAR10:cfa_wa_eps8_30ep CIFAR100:cfa_wa_eps8_30ep CIFAR10:cfa_wa_eps88_30ep CIFAR100:cfa_wa_eps88_30ep; do
  run_fat "${entry%%:*}" "${entry##*:}"
done
run_rpat 100 /mnt/d/research/FAT/data/CIFAR100 rpatpp_awp_c100_resnet18
run_rpat 10  /mnt/d/research/FAT/data/CIFAR10  rpatpp_awp_c10_resnet18
for entry in CIFAR100:adaadigdm_natinit_stack_100ep CIFAR100:consistency_natinit_stack_100ep \
             CIFAR100:lbgat_natinit_stack_100ep \
             CIFAR100:consistency_natinit_100ep CIFAR100:lbgat_natinit_100ep \
             CIFAR100:adr_natinit_200ep CIFAR100:adr_natinit_stack_200ep \
             CIFAR100:consistency_full_100ep CIFAR100:adr_full_100ep; do
  run_fat "${entry%%:*}" "${entry##*:}"
done
