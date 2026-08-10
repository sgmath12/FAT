#!/bin/bash
# 2026-07-25: WideResNet-34-10 CIFAR10 featdir 2x2x2 sweep -- eta(=k subspace dim) {480,320} x
# lamda {0,4} x WA {on,off}, all epochs=50. WRN-34-10 feature dim is 640 (not 512), so k=480
# leaves 160 free dims (~= the ResNet18 champion's 162-dim reservation at k=350/512) and k=320
# is the half-graded point. Teacher = 200ep WideResNet mixup clean net
# (CIFAR10/checkpoint/clean_mixup_200ep/clean_mixup_last.pkl, ~97.1 clean); student = WideResNet_z.
# WA is a config field (no CLI override), so it picks the config:
#   WA off -> featdir_wrn_mixupT.yaml    (weight_avg: False)
#   WA on  -> featdir_wrn_mixupT_wa.yaml (weight_avg: True)
# eta and lamda are CLI overrides. Sequential, single GPU. Per-run logs are timestamped and land
# under results/CIFAR10/WideResNet/<config>/ ; checkpoints overwrite, metrics live in the logs.
cd /mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
DRIVER=results/CIFAR10/WideResNet/sweep_wrn_2x2x2_20260725_driver.log
mkdir -p results/CIFAR10/WideResNet
EPOCHS=50
for wa in off on; do
  if [ "$wa" = "on" ]; then CFG=featdir_wrn_mixupT_wa.yaml; else CFG=featdir_wrn_mixupT.yaml; fi
  for eta in 480 320; do
    for lamda in 0 4; do
      echo "=== START wa=${wa} eta=${eta} lamda=${lamda} epochs=${EPOCHS} $(date) ===" >> "$DRIVER"
      $PY -u main.py --config_name ${CFG} --dataset CIFAR10 --seed 0 \
          --eta ${eta} --lamda ${lamda} --epochs ${EPOCHS} >> "$DRIVER" 2>&1
      echo "=== DONE  wa=${wa} eta=${eta} lamda=${lamda} rc=$? $(date) ===" >> "$DRIVER"
    done
  done
done
echo "CIFAR10_WRN_2x2x2_SWEEP_DONE $(date)" >> "$DRIVER"
