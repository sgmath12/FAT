#!/bin/bash
# High-dose lamda ceiling check (user, 2026-07-15 night): does robustness keep climbing
# indefinitely with lamda, or does it truly saturate? (1) baseline+WA lamda300 -- push past 100
# (still climbing there: pgd 33.43->34.12, cw 28.18->28.92) to find the real ceiling/collapse.
# (2) k350+WA at PROPERLY SCALED high dose: raw lamda 400/800 (grad-ratio 4.2x -> baseline-
# equivalent ~95/190) -- resolves whether k350's apparent "flat at 10-100" was real saturation
# or just undersampling below its true high-dose regime (matching baseline's still-climbing 100).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
until grep -q "AWP_CELLS_DONE" $LOG 2>/dev/null; do sleep 180; done

echo "=== basewa lamda 300 (ceiling) START $(date) ===" >> $LOG
$PY -u main.py --config_name temp_baseline_10step_wa.yaml --dataset CIFAR100 --seed 0 --lamda 300.0 \
  > results/CIFAR100/basewa_lamda300_driver.log 2>&1
echo "=== basewa lamda 300 (ceiling) DONE $(date) ===" >> $LOG

for l in 400.0 800.0; do
  echo "=== k350wa lamda $l (scaled-high) START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda $l \
    > results/CIFAR100/k350wa_lamda${l}_driver.log 2>&1
  echo "=== k350wa lamda $l (scaled-high) DONE $(date) ===" >> $LOG
done
echo "LAMDA_HIGHDOSE_DONE $(date)" >> $LOG
