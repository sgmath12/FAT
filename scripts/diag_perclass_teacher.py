# Per-CLASS teacher statistics (user's question, 2026-07-07): before/while learning tau_c (per-class
# temperature, train_temperature_tauclass_bilevel), check whether the frozen clean teacher's logit
# magnitude / input-gradient magnitude etc. actually DIFFER across classes -- if they don't, tau_c
# has nothing to key on (same "adaptivity doesn't fire" failure as the per-sample norm temperature).
#
# Two groupings, both on the CIFAR100 TEST set (10000 imgs, no aug):
#   A. by TRUE class y==c (100 samples/class): mean ||z||_2, margin(top1-top2), entropy,
#      input-grad norm ||dCE/dx||, teacher top-1 accuracy.
#   B. by logit COORDINATE c (what tau_c actually divides): mean/std of z_c over ALL samples,
#      and mean z_c restricted to own-class samples (y==c).
# Prints dispersion summaries + top/bot-5 classes; dumps raw arrays to
# results/CIFAR100/diag_perclass_teacher.npz for later correlation with the learned tau_c vector.
import os, sys, yaml
sys.path.insert(0, "/mnt/d/research/FAT")
os.chdir("/mnt/d/research/FAT")
import torch
import torch.nn.functional as F
import numpy as np
import dataset as dataset_mod
from utils import get_model

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

with open("config/CIFAR100/temp_tauclass_bilevel.yaml") as f:
    config = dotdict(yaml.safe_load(f))
config.config_name = "diag_perclass_teacher"

teacher, _ = get_model(config)
teacher.eval()

_, _, test_loader = dataset_mod.CIFAR100(root="./data/CIFAR100", download=False, batch_size=256, val=False)

C = 100
dev = "cuda"
cnt        = torch.zeros(C, device=dev)
sum_norm   = torch.zeros(C, device=dev)
sum_margin = torch.zeros(C, device=dev)
sum_ent    = torch.zeros(C, device=dev)
sum_gnorm  = torch.zeros(C, device=dev)
sum_corr   = torch.zeros(C, device=dev)
sum_zc     = torch.zeros(C, device=dev)   # coordinate mean over ALL samples
sumsq_zc   = torch.zeros(C, device=dev)
sum_zc_own = torch.zeros(C, device=dev)   # z_c on own-class samples only
n_total = 0

for x, y in test_loader:
    x, y = x.cuda(), y.cuda()
    x.requires_grad_(True)
    _, z = teacher(x, feat=True)
    ce = F.cross_entropy(z, y, reduction="sum")   # sum -> per-sample input grads (inputs independent)
    g = torch.autograd.grad(ce, x)[0]
    with torch.no_grad():
        gnorm = g.flatten(1).norm(dim=1)
        znorm = z.norm(dim=1)
        top2 = z.topk(2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        p = F.softmax(z, dim=1)
        ent = -(p * p.clamp_min(1e-12).log()).sum(dim=1)
        corr = (z.argmax(dim=1) == y).float()
        cnt.index_add_(0, y, torch.ones_like(gnorm))
        sum_norm.index_add_(0, y, znorm)
        sum_margin.index_add_(0, y, margin)
        sum_ent.index_add_(0, y, ent)
        sum_gnorm.index_add_(0, y, gnorm)
        sum_corr.index_add_(0, y, corr)
        sum_zc += z.sum(dim=0)
        sumsq_zc += (z ** 2).sum(dim=0)
        sum_zc_own.index_add_(0, y, z.gather(1, y.unsqueeze(1)).squeeze(1))
        n_total += x.shape[0]

per = {
    "znorm":  (sum_norm / cnt).cpu().numpy(),
    "margin": (sum_margin / cnt).cpu().numpy(),
    "entropy": (sum_ent / cnt).cpu().numpy(),
    "gnorm":  (sum_gnorm / cnt).cpu().numpy(),
    "acc":    (sum_corr / cnt).cpu().numpy(),
    "zc_mean_all": (sum_zc / n_total).cpu().numpy(),
    "zc_std_all":  torch.sqrt(sumsq_zc / n_total - (sum_zc / n_total) ** 2).cpu().numpy(),
    "zc_own":      (sum_zc_own / cnt).cpu().numpy(),
}
np.savez("results/CIFAR100/diag_perclass_teacher.npz", **per)

def show(name, v, hi_is="high"):
    idx = np.argsort(v)
    print(f"{name:12s} mean {v.mean():8.3f}  std {v.std():7.3f}  min {v.min():8.3f}  max {v.max():8.3f}  "
          f"max/min {v.max()/max(v.min(),1e-9):6.2f}x | bot5 {idx[:5].tolist()} top5 {idx[-5:].tolist()}")

print(f"n={n_total} test images, teacher = clean_last.pkl (frozen)\n")
print("A. grouped by TRUE class (100 samples each):")
show("||z||_2", per["znorm"])
show("margin", per["margin"])
show("entropy", per["entropy"])
show("gradnorm", per["gnorm"])
show("teacher_acc", per["acc"])
print("\nB. per logit COORDINATE c (what tau_c divides):")
show("z_c all", per["zc_mean_all"])
show("z_c std", per["zc_std_all"])
show("z_c own", per["zc_own"])
print("\ncorrelations across classes (Pearson):")
for a, b in [("gnorm", "acc"), ("gnorm", "entropy"), ("znorm", "acc"),
             ("zc_own", "acc"), ("entropy", "acc"), ("zc_own", "znorm")]:
    r = np.corrcoef(per[a], per[b])[0, 1]
    print(f"  {a:8s} vs {b:8s}: r = {r:+.3f}")
