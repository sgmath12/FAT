#!/bin/bash
# Wait for the ADR cifar100.zip checkpoint download (PID 77979) to finish, extract it, locate
# the ResNet18 ADR+WA+AWP checkpoint, and verify its published number (AA 28.52% / clean 57.36%)
# using FAT's own eval code (scripts/eval_adr_checkpoint.py).
cd /mnt/d/research/ADR/pretrained
LOG=/mnt/d/research/FAT/results/CIFAR100/adr_verify.log

DL_PID=5504
EXPECTED_SIZE=1714695315
echo "=== waiting for cifar100.zip download (PID $DL_PID) $(date) ===" >> $LOG
while kill -0 $DL_PID 2>/dev/null; do
  sleep 15
done
sleep 5

SIZE=$(stat -c%s cifar100.zip 2>/dev/null || echo 0)
echo "=== download finished, size=$SIZE bytes (expected $EXPECTED_SIZE) $(date) ===" >> $LOG
if [ "$SIZE" -ne "$EXPECTED_SIZE" ]; then
  echo "!!! SIZE MISMATCH -- download incomplete/truncated, aborting $(date) ===" >> $LOG
  exit 1
fi

echo "=== extracting $(date) ===" >> $LOG
mkdir -p extracted
unzip -q -o cifar100.zip -d extracted >> $LOG 2>&1
if [ $? -ne 0 ]; then
  echo "!!! unzip FAILED $(date) ===" >> $LOG
  exit 1
fi
echo "=== extraction done $(date) ===" >> $LOG

echo "=== searching for resnet18 ADR+WA+AWP checkpoint ===" >> $LOG
find extracted -iname "*.pt" | tee -a $LOG
CKPT=$(find extracted -iname "*.pt" | grep -i "resnet18" | grep -i "awp" | grep -i "adr" | head -1)
if [ -z "$CKPT" ]; then
  echo "!!! no exact match, falling back to broader search ===" >> $LOG
  CKPT=$(find extracted -iname "*.pt" | grep -i "resnet18" | grep -i "awp" | head -1)
fi
echo "=== using checkpoint: $CKPT $(date) ===" >> $LOG

if [ -z "$CKPT" ]; then
  echo "!!! COULD NOT LOCATE CHECKPOINT, aborting eval $(date) ===" >> $LOG
  exit 1
fi

cd /mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
echo "=== running FAT eval (WA/ema weights, matches ADR+WA+AWP row) $(date) ===" >> $LOG
$PY -u scripts/eval_adr_checkpoint.py "/mnt/d/research/ADR/pretrained/$CKPT" --ema >> $LOG 2>&1
echo "=== ADR_VERIFY_DONE $(date) ===" >> $LOG
