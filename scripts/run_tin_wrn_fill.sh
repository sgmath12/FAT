#!/usr/bin/env bash
# FILL THE Tiny-ImageNet AND WideResNet TABLES (2026-09-02).
#
#   usage:  bash scripts/run_tin_wrn_fill.sh TIN        # Tiny-ImageNet, ResNet-18
#           bash scripts/run_tin_wrn_fill.sh WRN10      # CIFAR-10, WideResNet-34-10
#           bash scripts/run_tin_wrn_fill.sh WRN100     # CIFAR-100, WideResNet-34-10
#           bash scripts/run_tin_wrn_fill.sh TIN core   # the four rows the argument needs, only
#
# `core` runs OURS, PGD-AT, AdaAD and PGD-AT-at-teacher-init, and skips the rest.  Those four carry
# every claim the tables are used for: ours, the reference point, the strongest distillation
# objective on a natural teacher, and the "do not distil at all" bound that all four distillation
# rows fall below.  The full set adds TRADES, MART, ARD, RSLAD and IGDM for completeness.
#
# OURS RUNS FIRST in both modes.  It is the row the tables exist for, and if the machine is taken
# back or the teacher turns out to be wrong, the cell that must not be missing is that one.
#
# The champion config is byte-identical across all three tables apart from dataset, arch and the
# checkpoint paths -- one recipe, no per-dataset tuning, which is a claim the paper makes.  Do not
# tune it here.  Full specification, and the two commits this checkout must contain, in
# `writting_docs/paper/TODO.md` section 4.
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
  # OURS is named per table because the champion config is not the same file everywhere.  In
  # particular Tiny-ImageNet's is `featdir_tin_champ`, NOT `featdir_tin_100ep`: the latter points at
  # the 80-epoch teacher and gives 57.08 / 18.96, while the paper's row is the 200-epoch teacher at
  # 55.16 / 20.54.  The epoch in that older name is the student's, and the difference was invisible.
  TIN)    DS=TinyImageNet; SFX="";     OURS=featdir_tin_champ
          TEACHER=TinyImageNet/checkpoint/clean_200ep/clean_last.pkl ;;
  WRN10)  DS=CIFAR10;      SFX="_wrn"; OURS=wrn_champ_freezehead
          TEACHER=CIFAR10/checkpoint/clean_wrn_200ep/clean_last.pkl ;;
  WRN100) DS=CIFAR100;     SFX="_wrn"; OURS=wrn_champ_freezehead
          TEACHER=CIFAR100/checkpoint/clean_wrn_200ep/clean_last.pkl ;;
  *) echo "unknown: $WHICH"; exit 1 ;;
esac
[ -f "$TEACHER" ] || { echo "missing teacher $TEACHER -- see the dependency note in this file"; exit 1; }

# A CIFAR-100 WideResNet built before commit c5d4458 is an 11.2M ResNet-18 that logs itself as a
# 48.3M WideResNet-34-10, so every cell here would be silently wrong.  Refuse to start.
if ! $PY scripts/check_arch.py > /dev/null 2>&1; then
  echo "scripts/check_arch.py FAILED -- this checkout predates c5d4458; git pull before running"
  $PY scripts/check_arch.py
  exit 1
fi

# HAT is in CORE on Tiny-ImageNet specifically.  The row we currently quote there -- 52.60 / 18.14 --
# is PreActResNet-18: ADR lists it under ResNet-18 but the numbers come from HAT's own Table 9, whose
# caption says PreAct, and HAT's code asserts `'preact-resnet' in name` for this dataset and refuses
# anything else.  It is also the highest published clean accuracy on Tiny-ImageNet, so it is the row
# our claim is measured against and the one worth having as our own measurement rather than a quote.
CORE="pgdat_100ep adaad_nat100ep at_teacherinit_matched hat_50ep"
FULL="$CORE trades_100ep mart_100ep ard_nat100ep rslad_nat100ep adaadigdm_nat100ep \
      lbgat_100ep consistency_100ep adr_200ep"
LIST="$CORE"; [ "$MODE" = full ] && LIST="$FULL"

echo "=== $(date '+%m-%d %H:%M') start $DS/$OURS (ours) ==="
$PY -u main.py --config_name "${OURS}.yaml" --dataset "$DS" --seed 0 > "logs/${DS}_${OURS}.log" 2>&1
echo "=== $(date '+%m-%d %H:%M') done $DS/$OURS (exit $?) ==="

for c in $LIST; do
  cfg="${c}${SFX}"
  echo "=== $(date '+%m-%d %H:%M') start $DS/$cfg ==="
  $PY -u main.py --config_name "${cfg}.yaml" --dataset "$DS" --seed 0 > "logs/${DS}_${cfg}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $DS/$cfg (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ($WHICH $MODE) ==="
