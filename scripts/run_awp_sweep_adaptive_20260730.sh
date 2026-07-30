#!/bin/bash
# 2026-07-30 22:5x — replaces run_awp_sweep_k512_20260730.sh (driver killed; its in-flight
# lamda15 cell was left running and is picked up below). Change the user asked for: group B's
# train_eps cells must NOT be hard-coded to lamda 4 — they should inherit whichever lamda group A
# proves best. k512+lamda0 already beat k512-adjacent lamda4 (CW 28.67 vs 28.20, AA 26.06 vs
# 25.98), so sweeping eps on top of lamda4 would be sweeping on top of a losing setting.
#
# Flow:
#   A  k512 lamda 0.0   DONE  63.75 / pgd 33.56 / cw 28.67 / AA 26.06
#      k512 lamda 1.5   in flight when this script was written
#      k512 lamda 4.0   (bridge: isolates the dimension effect from the lamda effect)
#   -> pick best lamda by AA across the three, REWRITE the eps configs with it
#   B  train_eps 10, 9, 12  (best-first by the 50ep evidence: eps10 was the only dial that ever
#      raised AA, 26.29 -> 26.80; eps12 overshot, 54.21 clean / AA 26.50) then lamda 8, lamda 2
#      (lamda2 is largely redundant once A gives 0/1.5/4 — it is last on purpose, cut it freely)
#   C  200ep eps 8/10/12 — still queued LAST and still questionable: 100ep did not beat 50ep on
#      AA (25.98/26.06 vs 26.29), so 12h45 on the epoch axis is hard to justify. Kill from the
#      back. A 3-seed replication of the winner is probably worth more (every call today rests on
#      single-seed gaps of 0.2-0.3 AA, which is this project's own observed noise scale).
# All cells: eta 512, AWP proxy g0.005 warmup10, eval eps 8/255, aa: True.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
CHAIN=results/CIFAR100/awp_sweep_20260730.log

run_cell () {
  echo "=== $1 (eta 512) START $(date) ===" >> $CHAIN
  $PY -u main.py --config_name $1.yaml --dataset CIFAR100 --seed 0 --eta 512 \
      > results/CIFAR100/$1_driver.log 2>&1
  rc=$?
  line=$(grep -ah "last_aa_acc\|last_pgd20_acc" results/CIFAR100/ResNet18/$1/*.log 2>/dev/null | tail -2 | tr '\n' ' ')
  echo "=== $1 DONE rc=$rc $(date) | $line" >> $CHAIN
}

# --- finish group A ---
while pgrep -f "featdir_awp_100ep_k512_lamda15" > /dev/null; do sleep 60; done
line=$(grep -ah "last_aa_acc\|last_pgd20_acc" results/CIFAR100/ResNet18/featdir_awp_100ep_k512_lamda15/*.log 2>/dev/null | tail -2 | tr '\n' ' ')
echo "=== featdir_awp_100ep_k512_lamda15 DONE (prev driver) $(date) | $line" >> $CHAIN
run_cell featdir_awp_100ep_k512_lamda4
echo "SWEEP_GROUP_A_DONE $(date)" >> $CHAIN

# --- pick the winning lamda by AA, stamp it into the eps configs ---
BEST=$($PY - <<'EOF'
import re, glob
best, blam = -1, 0.0
for lam, cell in [(0.0,"lamda0"), (1.5,"lamda15"), (4.0,"lamda4")]:
    aa = -1
    for f in glob.glob(f"results/CIFAR100/ResNet18/featdir_awp_100ep_k512_{cell}/*.log"):
        m = re.findall(r"'last_aa_acc': ([\d.]+)", open(f).read())
        if m: aa = max(aa, float(m[-1]))
    if aa > best: best, blam = aa, lam
print(blam if best > 0 else 4.0)
EOF
)
echo "BEST_LAMDA=$BEST (by AA) $(date)" >> $CHAIN
$PY - "$BEST" <<'EOF'
import sys, re
lam = float(sys.argv[1])
for name in ("featdir_awp_100ep_eps10","featdir_awp_100ep_eps12","featdir_awp_100ep_eps14",
             "featdir_awp_200ep_eps10","featdir_awp_200ep_eps12"):
    p = f"config/CIFAR100/{name}.yaml"
    t = re.sub(r"^lamda : .*$", f"lamda : {lam}", open(p).read(), flags=re.M)
    open(p,"w").write(t)
print("eps configs stamped with lamda", lam)
EOF

# --- group B: eps best-first, then the remaining lamda extension points ---
for c in featdir_awp_100ep_eps10 featdir_awp_100ep_eps12 featdir_awp_100ep_eps14; do
  run_cell $c
done
echo "SWEEP_GROUP_B_DONE $(date)" >> $CHAIN

# --- group C: 200ep. Cut this if the day is better spent on seeds. ---
for c in featdir_awp_200ep_eps10 featdir_awp_200ep_eps12; do
  run_cell $c
done
echo "AWP_ADAPTIVE_SWEEP_DONE $(date)" >> $CHAIN
