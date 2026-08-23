#!/usr/bin/env bash
# PURE-KD DESIGN AXIS with the raw cell made fair (2026-08-21).
#
# `nofeat_champ200_{norm,raw}` are the design axis in its simplest setting: one KL term, no feature
# loss, differing only in student_norm.
#   direction (student_norm True)   58.92 / AA 28.71 / NRR 38.61
#   L2        (student_norm False)  45.84 / AA 22.66 / NRR 30.33   <- diverged at step 0, unusable
#
# The divergence is the finding, not a nuisance: the student warm-starts from the natural teacher,
# so its logits ARE z_t at step 0 while the target is z_t/16 -- 16x off.  The normalized student
# starts at z_t/||Phi_t|| ~ z_t/11.2, within 1.43x of the same target.  With a raw student, target
# softness and student logit scale are coupled; normalization decouples them.
#
# These cells price the coupling by giving the raw student a tau that matches its own scale:
#   tau 1.43 -> same 1.43x initial mismatch the normalized cell has
#   tau 4    -> intermediate
# Read against nofeat_champ200_norm: if raw@1.43 catches up, normalization is convenience; if not,
# smoothing itself is worth something and only the normalized design can have both.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for c in nofeat_raw_tau143 nofeat_raw_tau4; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
