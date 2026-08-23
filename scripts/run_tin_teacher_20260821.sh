#!/usr/bin/env bash
# Tiny-ImageNet-200 CLEAN TEACHER (2026-08-21).  Everything else for this dataset has been ready
# since 2026-08-07 -- data prepared and val/ rebuilt into ImageFolder layout, dataset.TinyImageNet,
# the utils.get_model branch, the architecture policy -- but no checkpoint has ever been trained,
# and our method warm-starts from a clean teacher, so nothing else on this dataset can run until
# this does.
#
# 80 epochs, matching ADR, which also uses 80 for this dataset alone (200 elsewhere) -- presumably
# for the same cost reason: the smoke test measured 7.4 min/epoch, so 80ep is ~10h against ~25h for
# 200ep.  Architecture policy is ADR's: keep the CIFAR backbone (3x3 stem, stride 1, no maxpool) and
# absorb the larger input with adaptive final pooling only.  Swapping in an ImageNet-style 7x7
# stride-2 stem would break the comparison.
#
# Evaluation split: the labelled official val/ is our test set, per the AT literature's convention
# and ADR's; the shipped test/ has no labels.
#
# The bar: ADR ResNet-18 clean 48.27 / AA 20.10 (WRN-34-10 51.44 / 23.35).
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
echo "=== $(date '+%m-%d %H:%M') start TinyImageNet clean_80ep ==="
$PY -u main.py --config_name clean_80ep.yaml --dataset TinyImageNet --seed 0 \
    > logs/tin_clean_80ep.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
