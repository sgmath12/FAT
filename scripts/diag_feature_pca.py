# Feature-PCA bases for the "important subspace" cells (user, 2026-07-14 morning):
# user's hypothesis = the subspace worth following is the ROBUST-important one, which a natural
# teacher doesn't know ("중요 피처는 robust하면서 바뀐다"). Build top-k projectors from
#   (a) NATURAL teacher (clean_last)  vs  (b) ROBUST teacher (at_teacher/madry_at_last, plain AT)
# on TRAIN-set normalized features (no test leakage; aug on -- fine for a basis). At k < r
# (r ~ 56-76 by participation/90%-energy on the natural teacher), selection content can matter:
# pca_robust > random -> user's claim wins; pca_natural == random -> importance is not natural-
# teacher-knowable; all equal -> only dimension count matters even below r.
# Saves V (512x512, columns = PCA directions) for both into results/CIFAR100/feature_pca_bases.npz.
import sys, os
sys.path.insert(0, "/mnt/d/research/FAT")
os.chdir("/mnt/d/research/FAT")
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import dataset as dataset_mod
from utils import get_model

class dd(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

cfg = dd(yaml.safe_load(open("config/CIFAR100/featdir.yaml")))
cfg.config_name = "diag_feature_pca"
natural, _ = get_model(cfg)     # Converter(ResNet18) teacher loaded from clean_last
natural.eval()

cfg_r = dd(yaml.safe_load(open("config/CIFAR100/featdir.yaml")))
cfg_r.config_name = "diag_feature_pca_r"
cfg_r.checkpoint = "CIFAR100/checkpoint/at_teacher/madry_at_last.pkl"
cfg_r.finetune = False
robust, _ = get_model(cfg_r)    # same plain-ResNet18 teacher arch, robust weights
robust.eval()

train_loader, _, _ = dataset_mod.CIFAR100(root="./data/CIFAR100", download=False, batch_size=512, val=False)

def pca_V(model):
    feats = []
    with torch.no_grad():
        for x, y in train_loader:
            f, _ = model(x.cuda(), feat=True)
            feats.append(F.normalize(f, dim=1).cpu())
    Phi = torch.cat(feats).double()
    Phi = Phi - Phi.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(Phi, full_matrices=False)
    e = (S**2 / (S**2).sum()).cumsum(0)
    r90 = int((e < 0.90).sum().item()) + 1
    pr = float((S**2).sum()**2 / (S**4).sum())
    print(f"  r90 {r90}  participation {pr:.1f}")
    return Vh.T.float().numpy()          # 512 x 512, columns ordered by variance

print("natural teacher:")
V_nat = pca_V(natural)
print("robust teacher:")
V_rob = pca_V(robust)
np.savez("results/CIFAR100/feature_pca_bases.npz", natural=V_nat, robust=V_rob)
print("saved results/CIFAR100/feature_pca_bases.npz")
