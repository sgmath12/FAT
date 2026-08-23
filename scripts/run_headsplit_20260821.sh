#!/usr/bin/env bash
# HEAD-SEPARATION ablations (2026-08-21).  The split currently has a theorem (T.2) and no isolated
# measurement.  The +1.82 clean it is credited with (champion 60.74/28.69 vs pure KD 58.92/28.71)
# conflates the detach with the presence of a feature loss at all.
#   featdir_champ200_freezehead  -- head frozen at the teacher's solution.  Directly tests Thm 2
#     ("the head cannot be inherited").  A previous attempt stopped at epoch 80 with 60.62 / PGD
#     31.00 against the champion's 60.74 / 34.94, i.e. PGD -3.94 and still unfinished; no AA.
#   featdir_alpha1_champion      -- featdir_alpha 1.0, so the head KD is NOT detached and trains the
#     backbone too.  This is the cell that isolates the detach itself; never run.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for c in featdir_champ200_freezehead featdir_alpha1_champion; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
