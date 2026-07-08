"""Diagnostic: is there an input-dependent carve signal ORTHOGONAL to class info?

Only uses the teacher checkpoint (results-folder philosophy; no past runs).
For the raw CIFAR100 teacher, per (sample i, dim c):
    fragility_{i,c}  = |Phi_t(x)_c - Phi_t(x_adv)_c|         (carve recipe: PGD-2, CE, eps=8/255)
    class_need_{i,c} = |W[y_i, c] * Phi_t(x)_c|              (contribution to TRUE-class logit)
corr(fragility, class_need) high+ => fragile dims ARE class-relevant => carving erodes class info
(input-dependent carve doomed). Low corr => vulnerable-but-redundant dims exist => exploitable.
Also: teacher accuracy / softness of the carved target as tau grows (is carve just softening?).
"""
import sys, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "/mnt/d/research/FAT")
from CIFAR10.models.resnet import ResNet18
from converter import Converter
import torchvision, torchvision.transforms as T

mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
EPS, CSTEPS = 8/255, 2
CSTEP = EPS / CSTEPS
N_SAMPLES, BS = 2000, 100

teacher = Converter(ResNet18(num_classes=100), mean, std).cuda()
teacher.load_state_dict(torch.load("/mnt/d/research/FAT/CIFAR100/checkpoint/clean/clean_last.pkl"))
teacher.eval()
W = teacher.encoder.linear.weight.detach()          # [100, 512]

ds = torchvision.datasets.CIFAR100("/mnt/d/research/FAT/data/CIFAR100", train=False, download=False,
                                   transform=T.ToTensor())
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(N_SAMPLES)), batch_size=BS)

frag_all, need_all, feat_all, y_all = [], [], [], []
for x, y in loader:
    x, y = x.cuda(), y.cuda()
    with torch.no_grad():
        feat_clean, _ = teacher(x, feat=True)
    x_adv = x.clone().detach()
    for _ in range(CSTEPS):
        x_adv.requires_grad_(True)
        _, logits = teacher(x_adv, feat=True)
        g = torch.autograd.grad(F.cross_entropy(logits, y), x_adv)[0]
        x_adv = (x_adv.detach() + CSTEP * g.sign())
        x_adv = torch.min(torch.max(x_adv, x - EPS), x + EPS).clamp(0, 1)
    with torch.no_grad():
        feat_adv, _ = teacher(x_adv, feat=True)
    frag = (feat_clean - feat_adv).abs()             # [B,512]
    need = (W[y] * feat_clean).abs()                 # [B,512]
    frag_all.append(frag.cpu()); need_all.append(need.cpu())
    feat_all.append(feat_clean.cpu()); y_all.append(y.cpu())

frag = torch.cat(frag_all); need = torch.cat(need_all)
feat = torch.cat(feat_all); y = torch.cat(y_all)

def pearson(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()
def spearman(a, b):
    ra = a.flatten().argsort().argsort().float()
    rb = b.flatten().argsort().argsort().float()
    return pearson(ra, rb)

print("=== corr(fragility, class_need) ===")
print(f"pooled per-(sample,dim)  Pearson {pearson(frag, need):+.3f}   Spearman {spearman(frag, need):+.3f}")
fd, nd = frag.mean(0), need.mean(0)                   # [512] aggregated over samples
print(f"per-dim aggregated       Pearson {pearson(fd, nd):+.3f}   Spearman {spearman(fd, nd):+.3f}")
per_sample = [pearson(frag[i], need[i]) for i in range(min(500, frag.shape[0]))]
print(f"per-sample (avg of {len(per_sample)})   Pearson {np.mean(per_sample):+.3f}  (std {np.std(per_sample):.3f})")

# fraction of fragility mass on class-relevant dims (top/bottom half of class_need, per sample)
med = need.median(dim=1, keepdim=True).values
frac_hi = (frag * (need >= med)).sum() / frag.sum()
print(f"\nfragility mass on class-relevant (need>=median) dims: {frac_hi.item()*100:.1f}%  (50% = neutral)")

print("\n=== carved teacher target vs tau (is carve just softening?) ===")
yb = y.cuda(); featb = feat.cuda()
clean_logits = teacher.encoder.linear(featb)
base_acc = (clean_logits.argmax(1) == yb).float().mean().item()
print(f"tau=0 (no carve): acc {base_acc*100:5.2f}  maxprob {F.softmax(clean_logits,1).max(1).values.mean():.3f}")
for tau in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
    w = torch.exp(-tau * frag.cuda())
    cl = teacher.encoder.linear(featb * w)
    acc = (cl.argmax(1) == yb).float().mean().item()
    p = F.softmax(cl, 1)
    maxp = p.max(1).values.mean().item()
    ent = (-(p * (p + 1e-12).log()).sum(1)).mean().item()
    print(f"tau={tau:<5} acc {acc*100:5.2f}  maxprob {maxp:.3f}  entropy {ent:.3f}")
