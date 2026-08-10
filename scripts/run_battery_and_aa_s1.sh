#!/bin/bash
# Behind BASEWA_LAMDA_DONE (2026-07-15): (1) AA on k350wa_seed1 (AA seed-variance check on the
# -0.45 deficit), (2) mechanism battery (1)(2)(3).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
until grep -q "BASEWA_LAMDA_DONE" $LOG 2>/dev/null; do sleep 120; done
echo "=== AA k350wa_seed1 START $(date) ===" >> $LOG
$PY - << 'PYEOF' > results/CIFAR100/aa_k350wa_s1.log 2>&1
import sys, os
sys.path.insert(0, "/mnt/d/research/FAT"); os.chdir("/mnt/d/research/FAT")
import torch, dataset as dataset_mod
from CIFAR10.models.resnet_z import ResNet18_z
from converter import Converter
from utils import evaluate_final_aa
class A: pass
args = A(); args.eps = 8/255.0; args.batch_size = 512
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
_, _, tl = dataset_mod.CIFAR100(root="./data/CIFAR100", download=False, batch_size=512, val=False)
m = Converter(ResNet18_z(num_classes=100), mean, std)
m.load_state_dict(torch.load("CIFAR100/checkpoint/featdir_span_random_10step_wa/k350wa_seed1_last.pkl", map_location="cpu"), strict=False)
m.cuda().eval()
print("AA_RESULT | k350+WA (s1) | aa_acc %.2f" % evaluate_final_aa(m, tl, args), flush=True)
PYEOF
echo "=== battery START $(date) ===" >> $LOG
$PY -u scripts/diag_mechanism_battery.py > results/CIFAR100/battery_20260715.log 2>&1
echo "BATTERY_AA_DONE $(date)" >> $LOG
