#!/usr/bin/env bash
# FILL THE Tiny-ImageNet AND WideResNet TABLES (2026-09-02).
#
#   usage:  bash scripts/run_tin_wrn_fill.sh TIN        # Tiny-ImageNet, ResNet-18
#           bash scripts/run_tin_wrn_fill.sh WRN10      # CIFAR-10, WideResNet-34-10
#           bash scripts/run_tin_wrn_fill.sh WRN100     # CIFAR-100, WideResNet-34-10
#           bash scripts/run_tin_wrn_fill.sh TIN core   # the four rows the argument needs, only
#
# `core` runs PGD-AT, AdaAD, PGD-AT-at-teacher-init and skips the rest.  Those four carry every claim
# the tables are used for: the reference point, the strongest distillation objective on a natural
# teacher, the "do not distil at all" bound the four fall below, and ours.  The full set adds TRADES,
# MART, ARD, RSLAD and IGDM for completeness.
#
# DEPENDENCIES, and they are not optional:
#   TIN     needs TinyImageNet/checkpoint/clean_200ep -- NOT on this machine.  Either train
#           config/TinyImageNet/clean_200ep.yaml first, or copy that checkpoint from the other server.
#           (clean_80ep is here, but the reported 55.16 / 20.54 uses the 200-epoch teacher.)
#   WRN10   needs CIFAR10/checkpoint/clean_wrn_200ep  -- on the other server.
#   WRN100  needs CIFAR100/checkpoint/clean_wrn_200ep -- train config/CIFAR100/clean_wrn_200ep.yaml.
#
# Per-cell cost relative to CIFAR/ResNet-18: Tiny-ImageNet about x4 (64x64, 100k images),
# WideResNet-34-10 about x4.5 (48.3M parameters against 11.2M).
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
WHICH="${1:?TIN | WRN10 | WRN100}"
MODE="${2:-full}"
case "$WHICH" in
  TIN)    DS=TinyImageNet; SFX="";     TEACHER=TinyImageNet/checkpoint/clean_200ep/clean_last.pkl ;;
  WRN10)  DS=CIFAR10;      SFX="_wrn"; TEACHER=CIFAR10/checkpoint/clean_wrn_200ep/clean_last.pkl ;;
  WRN100) DS=CIFAR100;     SFX="_wrn"; TEACHER=CIFAR100/checkpoint/clean_wrn_200ep/clean_last.pkl ;;
  *) echo "unknown: $WHICH"; exit 1 ;;
esac
[ -f "$TEACHER" ] || { echo "missing teacher $TEACHER -- see the dependency note in this file"; exit 1; }

CORE="pgdat_100ep adaad_nat100ep at_teacherinit_matched"
FULL="$CORE trades_100ep mart_100ep ard_nat100ep rslad_nat100ep adaadigdm_nat100ep"
LIST="$CORE"; [ "$MODE" = full ] && LIST="$FULL"

for c in $LIST; do
  cfg="${c}${SFX}"
  echo "=== $(date '+%m-%d %H:%M') start $DS/$cfg ==="
  $PY -u main.py --config_name "${cfg}.yaml" --dataset "$DS" --seed 0 > "logs/${DS}_${cfg}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $DS/$cfg (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ($WHICH $MODE) ==="
