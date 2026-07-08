"""Per-sample diagnostic: entropy / margin / vulnerability as candidate signals for an
input-dependent (dynamic) global temperature. Teacher checkpoint only.

Per sample i (raw CIFAR100 teacher):
    entropy_i       = H(softmax(z_i))                 (clean confidence; low=confident)
    margin_i        = top1(z_i) - top2(z_i)           (clean confidence)
    vulnerability_i = p_clean[y_i] - p_adv[y_i]       (true-class prob drop under PGD-10; robustness need)
Reports: (1) dispersion of each (p5/p50/p95, p95/p5) -- does it vary enough to matter?
         (2) pairwise corr among the three -- is vulnerability independent of clean confidence?
The bet: a signal only beats a global temperature if it has real dispersion AND is the RIGHT
axis. If entropy~=margin and both weakly track vulnerability, clean-confidence temp is a re-tread
and vulnerability is the only orthogonal per-sample handle.
"""
import sys, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, "/mnt/d/research/FAT")
from CIFAR10.models.resnet import ResNet18
from converter import Converter
import torchvision, torchvision.transforms as T

mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
EPS, STEPS = 8/255, 10
STEP = 2/255
N_SAMPLES, BS = 2000, 100

teacher = Converter(ResNet18(num_classes=100), mean, std).cuda()
teacher.load_state_dict(torch.load("/mnt/d/research/FAT/CIFAR100/checkpoint/clean/clean_last.pkl"))
teacher.eval()

ds = torchvision.datasets.CIFAR100("/mnt/d/research/FAT/data/CIFAR100", train=False, download=False,
                                   transform=T.ToTensor())
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(N_SAMPLES)), batch_size=BS)

ent_all, mar_all, vul_all = [], [], []
for x, y in loader:
    x, y = x.cuda(), y.cuda()
    with torch.no_grad():
        z = teacher(x)
        p = F.softmax(z, 1)
        ent = -(p * (p + 1e-12).log()).sum(1)
        top2 = z.topk(2, 1).values
        margin = top2[:, 0] - top2[:, 1]
        p_clean_y = p[torch.arange(len(y)), y]
    # PGD-10 for vulnerability
    x_adv = x.clone().detach() + 0.001 * torch.randn_like(x)
    for _ in range(STEPS):
        x_adv.requires_grad_(True)
        z_adv = teacher(x_adv)
        g = torch.autograd.grad(F.cross_entropy(z_adv, y), x_adv)[0]
        x_adv = x_adv.detach() + STEP * g.sign()
        x_adv = torch.min(torch.max(x_adv, x - EPS), x + EPS).clamp(0, 1)
    with torch.no_grad():
        p_adv_y = F.softmax(teacher(x_adv), 1)[torch.arange(len(y)), y]
    vul = (p_clean_y - p_adv_y)
    ent_all.append(ent.cpu()); mar_all.append(margin.cpu()); vul_all.append(vul.cpu())

ent = torch.cat(ent_all); mar = torch.cat(mar_all); vul = torch.cat(vul_all)

def disp(a, name):
    p5, p50, p95 = np.percentile(a.numpy(), [5, 50, 95])
    r = p95 / (p5 if abs(p5) > 1e-6 else 1e-6)
    print(f"  {name:13s} mean {a.mean():7.3f}  p5 {p5:7.3f}  p50 {p50:7.3f}  p95 {p95:7.3f}  p95/p5 {r:6.2f}x")
def pearson(a, b):
    a, b = a.float(), b.float(); a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()
def spearman(a, b):
    return pearson(a.argsort().argsort().float(), b.argsort().argsort().float())

print("=== dispersion across samples (does it vary enough?) ===")
disp(ent, "entropy"); disp(mar, "margin"); disp(vul, "vulnerability")
print("\n=== pairwise corr (Pearson / Spearman) ===")
for (a, na), (b, nb) in [((ent, "entropy"), (mar, "margin")),
                          ((ent, "entropy"), (vul, "vulnerability")),
                          ((mar, "margin"), (vul, "vulnerability"))]:
    print(f"  {na:8s} vs {nb:13s}  {pearson(a, b):+.3f} / {spearman(a, b):+.3f}")
print("\nnote: margin high = confident; entropy high = UNconfident -> expect margin/entropy strongly NEGATIVE.")
