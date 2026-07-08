"""3-way golden-logit diagnostic (teacher-only, no training). Compare teacher-target softening:
  (1) TEMPERATURE   : softmax(z(x) / T)
  (2) TEMPERED      : tempered_softmax(z(x), t)   (Tsallis exp_t; t<1 = polynomial/heavy-ish tail)
  (3) EPS-BALL SMOOTH: softmax( mean_δ z(x+δ) / T ),  δ ~ Uniform[-eps,eps]  (randomized-smoothing target)
Golden-logit conditions -> metrics per method:
  - entropy            : softness (want soft but NOT uniform; 100-class uniform = 4.605)
  - top5 mass          : structure / dark-knowledge preserved (higher = keeps class geometry)
  - LOCAL KL (KEY)     : E_x,δ'[ KL(target(x) || target(x+δ')) ] for a FRESH perturbation δ'.
                         robustness == target FUNCTION locally flat -> LOWER is better (robust-shaped).
Only (3) averages over the ball, so (3) should give the lowest LOCAL KL at comparable softness.
"""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, "/mnt/d/research/FAT")
from CIFAR10.models.resnet import ResNet18
from converter import Converter
import torchvision, torchvision.transforms as T

mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
EPS = 8/255
N, BS = 1000, 100
K_SMOOTH = 8       # samples to build the eps-ball smoothed logits
K_PROBE  = 4       # fresh perturbations to measure LOCAL KL

teacher = Converter(ResNet18(num_classes=100), mean, std).cuda()
teacher.load_state_dict(torch.load("/mnt/d/research/FAT/CIFAR100/checkpoint/clean/clean_last.pkl"))
teacher.eval()
ds = torchvision.datasets.CIFAR100("/mnt/d/research/FAT/data/CIFAR100", train=False, download=False, transform=T.ToTensor())
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(N)), batch_size=BS)

@torch.no_grad()
def z_of(x):                      # teacher logits
    return teacher(x)

@torch.no_grad()
def z_smooth(x, k):               # eps-ball averaged logits (uniform noise in the linf ball)
    acc = 0
    for _ in range(k):
        d = (torch.rand_like(x) * 2 - 1) * EPS
        acc = acc + z_of((x + d).clamp(0, 1))
    return acc / k

def tempered_softmax(z, t):       # Tsallis; t<1 safe (base>=0, exponent>0). t=1 == softmax.
    if abs(t - 1.0) < 1e-6:
        return F.softmax(z, dim=1)
    zz = z - z.max(dim=1, keepdim=True).values
    num = torch.clamp(1 + (1 - t) * zz, min=0.0) ** (1.0 / (1 - t))
    return num / num.sum(dim=1, keepdim=True)

def ent(p):   return (-(p * (p + 1e-12).log()).sum(1)).mean().item()
def top5(p):  return p.topk(5, dim=1).values.sum(1).mean().item()

# target builders: x -> probability target
def make_temp(Tv):   return lambda x: F.softmax(z_of(x) / Tv, dim=1)
def make_temp_t(tv): return lambda x: tempered_softmax(z_of(x), tv)
def make_eb(Tv, k):  return lambda x: F.softmax(z_smooth(x, k) / Tv, dim=1)

methods = [
    ("temperature T=16",   make_temp(16.0)),
    ("temperature T=4",    make_temp(4.0)),
    ("tempered t=0.5",     make_temp_t(0.5)),
    ("tempered t=0.8",     make_temp_t(0.8)),
    ("eps-ball T=16",      make_eb(16.0, K_SMOOTH)),
    ("eps-ball T=4",       make_eb(4.0,  K_SMOOTH)),
]

agg = {name: {"ent": [], "t5": [], "lkl": []} for name, _ in methods}
for x, _ in loader:
    x = x.cuda()
    for name, f in methods:
        p = f(x)
        agg[name]["ent"].append(ent(p)); agg[name]["t5"].append(top5(p))
        # LOCAL KL: fresh random perturbations, KL(target(x) || target(x+d'))
        lk = 0.0
        for _ in range(K_PROBE):
            d = (torch.rand_like(x) * 2 - 1) * EPS
            p2 = f((x + d).clamp(0, 1))
            lk += (p * ((p + 1e-12).log() - (p2 + 1e-12).log())).sum(1).mean().item()
        agg[name]["lkl"].append(lk / K_PROBE)

import numpy as np
print(f"{'method':18s} | entropy | top5 mass | LOCAL KL (lower=flatter=robust-shaped)")
for name, _ in methods:
    e = np.mean(agg[name]["ent"]); t5 = np.mean(agg[name]["t5"]); lk = np.mean(agg[name]["lkl"])
    print(f"{name:18s} |  {e:5.3f}  |   {t5:5.3f}   |  {lk:.4f}")
print("(100-class uniform entropy = 4.605; top5 mass high = structure kept)")
