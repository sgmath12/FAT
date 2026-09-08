#!/usr/bin/env bash
# 2026-09-09.  Ordered from writting_docs/정리된생각.md.  P0 first: complete the teacher ladder into
# the weak-teacher region, separate the warm start from the teacher effect, and give the three
# representative rungs more than one seed.  Then the shipped-recipe logit cells that tab:main needs,
# then the queue that was already standing (table-5 remainder, epoch and lr sensitivity).
#
# Entries may carry a seed as cfg@seed; the default is 0.  Checkpoints go to per-seed directories
# (main.py:142), so a repeat no longer overwrites the run it repeats.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1708996
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
for entry in c10_t50_anchor c10_t50_logitmse c10_t50_kdsym c10_t50_kdasym \
             clean_5ep clean_10ep clean_20ep clean_40ep \
             tladder_clean_5ep tladder_clean_10ep tladder_clean_20ep tladder_clean_40ep \
             tladder_randinit_10ep tladder_randinit_100ep tladder_randinit_300ep \
             tladder_clean_10ep@1 tladder_clean_10ep@2 \
             tladder_clean@1 tladder_clean@2 \
             tladder_clean_300ep@1 tladder_clean_300ep@2 \
             kdsym_stack_t16 logitmse_stack_100ep logitmse_100ep kdsym_t16_100ep \
             adaadigdm_ourrecipe_100ep trades_ourrecipe_100ep mart_ourrecipe_100ep \
             trades_100ep mart_100ep trades_natinit_100ep mart_natinit_100ep hat_natinit_50ep \
             ship_ep10 ship_ep25 ship_lr0p007 ship_lr0p014 ship_lr0p03 ship_lr0p042 ship_ep200; do
  cfg="${entry%@*}"; seed="${entry#*@}"; [ "$seed" = "$entry" ] && seed=0
  DS=CIFAR100; case "$cfg" in c10_*) DS=CIFAR10;; esac
  echo "=== $(date '+%m-%d %H:%M') start $DS/$cfg seed $seed ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset $DS --seed $seed \
      > logs/${DS}_${cfg}_s${seed}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg seed $seed (exit $?) ==="
done
