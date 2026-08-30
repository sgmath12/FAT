#!/usr/bin/env bash
# 2026-08-31 QUEUE.  Two jobs, in this order.
#
# 1. `l2_bestrecipe_freezehead` -- the shipped CIFAR-100 recipe with featdir_freeze_head: True.
#    The reported cell trains the classifier by a DETACHED head-KD term (beta 1.0, tau 16) that the
#    method section does not describe; the base-regime ablation says freezing is better anyway
#    (NRR 36.64 vs 36.35).  If this reproduces or beats 62.35 / 28.68, tau and beta leave the method
#    and "no temperature, no loss weight" becomes literally true rather than scoped.
#
# 2. The four published robust-distillation objectives -- ARD, RSLAD, AdaAD, AdaAD+IGDM -- ported
#    into methods.py from the official IGDM release and run on OUR NATURAL TEACHER instead of the
#    robust WRN-28-10 their own scripts load.  Their published recipe, not ours: SGD 0.1, step LR
#    x0.1 at epochs 70/90, random init, eps 8/255, no WA, no AWP, no teacher warm start.  100 epochs
#    to match the ladder's budget.  The comparison cell is `ladder_p0_100ep` (61.21 / AA 25.24).
#
#    Proposition 1 predicts the ORDER and can be killed by it: ARD and RSLAD query the teacher at x
#    only, AdaAD queries it at x_adv, AdaAD+IGDM additionally at x - delta.  Off-manifold queries to
#    a naturally trained teacher should hurt, so AdaAD should degrade most and IGDM should not
#    rescue it.  If AdaAD wins here, section 3 is wrong.
#
# Ordered cheapest-first among the baselines (measured: 3.0 / 3.5 / 5.7 / 6.7 h per 100ep).
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

for c in l2_bestrecipe_freezehead ard_nat100ep rslad_nat100ep adaad_nat100ep adaadigdm_nat100ep; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
