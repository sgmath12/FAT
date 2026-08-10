#!/bin/bash
# 2026-07-29 ~23:00. Rewritten to put the DECISIVE cheap check first (user's q: "is k350 even
# useful now?"). k=512 makes Q a full-rank orthonormal 512x512 matrix, i.e. the projection is an
# isometry and the subspace mechanism is a NO-OP -- so the k512 cell is the clean ablation of the
# whole mechanism. It came in at clean +0.48 / pgd20 -1.11 / cw +0.02 (a TIE on CW). Since AA
# tracks CW more closely than PGD, AA on the k512 checkpoint decides whether k350 buys anything
# on the metric the paper is judged by. 25 min, and it changes what is worth running after it.
# Then: AdamW 100ep control (no AWP, MATCHED baseline) + 100ep with proxy AWP (g0.005, warmup10).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
CHAIN=results/CIFAR100/sgd_lrsweep_20260729.log
LOG001=$(ls -t results/CIFAR100/ResNet18/featdir_wa_sgd_lr001/*.log 2>/dev/null | head -1)

until grep -q "sgd lr001 DONE" $CHAIN 2>/dev/null; do sleep 60; done

# 0. DECISIVE: AA on the k512 (= mechanism-off) checkpoint, vs champion k350 AA 26.29
echo "=== AA k512 START $(date) ===" >> $CHAIN
$PY -u scripts/eval_aa_generic.py \
    "k512 (mechanism-off ablation)|CIFAR100/checkpoint/featdir_k512wa_lamda4/feat_direction_last.pkl" \
    > results/CIFAR100/aa_k512_20260729.log 2>&1
grep -a "AA_RESULT" results/CIFAR100/aa_k512_20260729.log >> $CHAIN
echo "=== AA k512 DONE $(date) ===" >> $CHAIN

FAILED=$($PY - "$LOG001" <<'EOF'
import re, sys
try:
    t = open(sys.argv[1]).read()
    p = float(re.findall(r"'last_pgd20_acc': ([\d.]+)", t)[-1])
    print("yes" if p < 33.0 else "no")
except Exception:
    print("yes")     # no final line at all => it died => proceed
EOF
)
echo "SGD_LR001_FAILED=$FAILED $(date)" >> $CHAIN
[ "$FAILED" = "no" ] && { echo "lr0.01 SUCCEEDED -- 100ep+AWP pair skipped $(date)" >> $CHAIN
                          echo "100EP_AWP_PAIR_DONE (skipped) $(date)" >> $CHAIN; exit 0; }

for cell in featdir_k350wa_100ep featdir_k350wa_100ep_awp; do
  echo "=== $cell START $(date) ===" >> $CHAIN
  $PY -u main.py --config_name ${cell}.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 \
      > results/CIFAR100/${cell}_driver.log 2>&1
  echo "=== $cell DONE $(date) ===" >> $CHAIN
done
echo "100EP_AWP_PAIR_DONE $(date)" >> $CHAIN
