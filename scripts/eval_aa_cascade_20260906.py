# Does the reported AA number change if FAB and Square are added? (2026-09-06)
#
# The per-attack numbers in eval_gradmask are STANDALONE -- each attack starts from the clean set.
# AutoAttack's reported accuracy is a cascade: a point counts robust only if every attack fails on it,
# so a standalone-weaker attack can still take points the others missed.  This runs both cascades on
# the same 1000 examples and prints the difference.
import sys, os
sys.path.insert(0, "/mnt/d/research/FAT"); os.chdir("/mnt/d/research/FAT")
import torch
import dataset as dataset_mod
from CIFAR10.models.resnet import ResNet18
from converter import Converter
from autoattack import AutoAttack

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
m = Converter(ResNet18(num_classes=100), MEAN, STD)
m.load_state_dict(torch.load("CIFAR100/checkpoint/l2_bestrecipe_freezehead_seed1/feat_direction_last.pkl",
                             map_location="cpu"), strict=False)
m = m.cuda().eval()

_, _, test_loader = dataset_mod.CIFAR100(root="./data/CIFAR100", download=False, batch_size=512, val=False)
x = torch.cat([a for a, _ in test_loader], 0)[:1000]
y = torch.cat([b for _, b in test_loader], 0)[:1000]

def acc(model, xa, ya, bs=256):
    c = 0
    with torch.no_grad():
        for i in range(0, len(xa), bs):
            c += (model(xa[i:i+bs].cuda()).argmax(1) == ya[i:i+bs].cuda()).sum().item()
    return 100.0 * c / len(xa)

for name, atks in [("reported (apgd-ce, apgd-t)", ['apgd-ce', 'apgd-t']),
                   ("full (+ fab-t, square)",     ['apgd-ce', 'apgd-t', 'fab-t', 'square'])]:
    a = AutoAttack(m, norm="Linf", eps=8/255, version="standard")
    a.attacks_to_run = atks
    adv = a.run_standard_evaluation(x, y, bs=256)
    print(f"CASCADE | {name:28s} | acc {acc(m, adv.cpu(), y):.2f}", flush=True)
