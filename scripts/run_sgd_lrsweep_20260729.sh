#!/bin/bash
# 2026-07-29 overnight-ish chain (user away ~2.5h):
#   1. wait for the running k=512 control to finish
#   2. pick the subspace dim: 512 if it TIES the champion (pgd20 within 0.5 AND clean within 1.0
#      of 62.75/33.96), else fall back to the champion's 350
#   3. SGD switch, lr sweep {0.1, 0.05}, everything else = champion recipe
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
K512_LOG=results/CIFAR100/ResNet18/featdir_k512wa_lamda4/2607291929.log
CHAIN=results/CIFAR100/sgd_lrsweep_20260729.log

echo "=== chain start $(date) ===" >> $CHAIN
# 1. wait for k512's final eval line
until grep -q "last_pgd20_acc" $K512_LOG 2>/dev/null; do
  pgrep -f "featdir_k512wa_lamda4" > /dev/null || { sleep 30; grep -q "last_pgd20_acc" $K512_LOG 2>/dev/null || { echo "K512 DIED without final eval $(date)" >> $CHAIN; break; }; }
  sleep 60
done

# 2. decide eta
ETA=$($PY - <<'EOF'
import re
try:
    t = open("results/CIFAR100/ResNet18/featdir_k512wa_lamda4/2607291929.log").read()
    m = re.findall(r"'last_clean_acc': ([\d.]+).*?'last_pgd20_acc': ([\d.]+)", t)
    c, p = float(m[-1][0]), float(m[-1][1])
    print(512 if (p >= 33.96 - 0.5 and c >= 62.75 - 1.0) else 350)
except Exception:
    print(350)
EOF
)
echo "ETA_CHOSEN $ETA $(date)" >> $CHAIN

# 3. SGD lr sweep
for cell in lr01:0.1 lr005:0.05; do
  name=${cell%%:*}; lr=${cell##*:}
  echo "=== sgd $name (lr $lr, eta $ETA) START $(date) ===" >> $CHAIN
  $PY -u main.py --config_name featdir_wa_sgd_${name}.yaml --dataset CIFAR100 --seed 0 \
      --eta $ETA --lamda 4.0 \
      > results/CIFAR100/sgd_${name}_driver.log 2>&1
  echo "=== sgd $name DONE $(date) ===" >> $CHAIN
done
echo "SGD_LRSWEEP_DONE $(date)" >> $CHAIN
