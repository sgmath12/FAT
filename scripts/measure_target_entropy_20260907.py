# TARGET ENTROPY AND CONFIDENCE BY TEMPERATURE (2026-09-07).
#
# The analysis table reports the mean maximum probability of the softened teacher target per
# temperature.  Entropy is the other half of the same description and is what B-MTARD's knowledge
# scale (log C - H) is written in, so it is recorded here for the same grid the sweep runs on.
# Teacher-only measurement: one forward pass over the clean test set, no training.
import sys, os, math
sys.path.insert(0, "/mnt/d/research/FAT"); os.chdir("/mnt/d/research/FAT")
import torch, torch.nn.functional as F
import dataset as dataset_mod
from CIFAR10.models.resnet import ResNet18
from converter import Converter

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
CKPTS = [("natural teacher",             "CIFAR100/checkpoint/clean_200ep/clean_last.pkl"),
         ("PGD-AT, teacher init, stack",  "CIFAR100/checkpoint/at_teacherinit_matched/madry_at_last.pkl"),
         ("label CE, base regime",        "CIFAR100/checkpoint/abl_ce_nostack/madry_at_last.pkl"),
         ("anchored student (seed 1)",    "CIFAR100/checkpoint/l2_bestrecipe_freezehead_seed1/feat_direction_last.pkl")]
TAUS = [1, 2, 4, 8, 16]
C = 100

_, _, test_loader = dataset_mod.CIFAR100(root="./data/CIFAR100", download=False, batch_size=512, val=False)
x = torch.cat([a for a, _ in test_loader], 0)
y = torch.cat([b for _, b in test_loader], 0)

for name, path in CKPTS:
    if not os.path.exists(path):
        print(f"ENT | {name:28s} | checkpoint missing: {path}", flush=True); continue
    m = Converter(ResNet18(num_classes=C), MEAN, STD)
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    m = m.cuda().eval()
    logits, correct = [], 0
    with torch.no_grad():
        for i in range(0, len(x), 512):
            z = m(x[i:i+512].cuda())
            correct += (z.argmax(1) == y[i:i+512].cuda()).sum().item()
            logits.append(z.cpu())
    z = torch.cat(logits, 0)
    print(f"ENT | {name} | clean acc {100.0*correct/len(x):.2f} | ln C = {math.log(C):.4f}", flush=True)
    for t in TAUS:
        p = F.softmax(z / t, dim=1)
        H = -(p * p.clamp_min(1e-12).log()).sum(1)
        print(f"ENT |   tau {t:2d} | max prob {p.max(1).values.mean():.4f} "
              f"| entropy {H.mean():.4f} nats | normalized {H.mean()/math.log(C):.4f} "
              f"| knowledge scale logC-H {math.log(C)-H.mean():.4f}", flush=True)
