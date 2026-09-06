# FAB-T AND SQUARE ON THE FULL TEST SET (2026-09-06).
#
# The first pass ran the two expensive attacks on 1000 examples, which is the AutoAttack convention
# but is a weaker claim than the rest of the paper, where every AutoAttack number is all 10000.  A
# masking check reported on a tenth of the data invites the doubt it is meant to remove, so both
# attacks are rerun here on the full test set.
#
# Two things are measured:
#   (a) each attack STANDALONE, from the clean test set -- the ordering that decides whether the
#       black-box attack is weaker than the white-box ones;
#   (b) the CASCADE, apgd-ce + apgd-t (what the paper reports) against all four (AutoAttack's own
#       standard version) -- whether adding FAB and Square would move the reported number at all.
import sys, os, time
sys.path.insert(0, "/mnt/d/research/FAT"); os.chdir("/mnt/d/research/FAT")
import torch
import dataset as dataset_mod
from CIFAR10.models.resnet import ResNet18
from converter import Converter
from autoattack import AutoAttack

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
m = Converter(ResNet18(num_classes=100), MEAN, STD)
m.load_state_dict(torch.load("CIFAR100/checkpoint/l2_bestrecipe_freezehead/feat_direction_last.pkl",
                             map_location="cpu"), strict=False)
m = m.cuda().eval()

_, _, test_loader = dataset_mod.CIFAR100(root="./data/CIFAR100", download=False, batch_size=512, val=False)
x = torch.cat([a for a, _ in test_loader], 0)
y = torch.cat([b for _, b in test_loader], 0)
print(f"AAFULL | n {len(x)}", flush=True)

def acc(xa, ya, bs=512):
    c = 0
    with torch.no_grad():
        for i in range(0, len(xa), bs):
            c += (m(xa[i:i+bs].cuda()).argmax(1) == ya[i:i+bs].cuda()).sum().item()
    return 100.0 * c / len(xa)

def run(atks, bs=512):
    a = AutoAttack(m, norm="Linf", eps=8/255, version="standard")
    a.attacks_to_run = atks
    t0 = time.time()
    adv = a.run_standard_evaluation(x, y, bs=bs)
    return acc(adv.cpu(), y), time.time() - t0

for name in ["apgd-ce", "apgd-t", "fab-t", "square"]:
    a, s = run([name])
    print(f"AAFULL | standalone {name:8s} | acc {a:.2f} | {s/60:.1f} min", flush=True)

for label, atks in [("reported (apgd-ce, apgd-t)", ["apgd-ce", "apgd-t"]),
                    ("full standard (4 attacks)",  ["apgd-ce", "apgd-t", "fab-t", "square"])]:
    a, s = run(atks)
    print(f"AAFULL | cascade {label:28s} | acc {a:.2f} | {s/60:.1f} min", flush=True)
