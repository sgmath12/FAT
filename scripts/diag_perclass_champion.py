"""Per-class clean / PGD-20 / CW-20 accuracy for the CIFAR100 champion
(feat_direction k350 + WA + lamda4)."""
import sys, os
sys.path.insert(0, "/mnt/d/research/FAT"); os.chdir("/mnt/d/research/FAT")
import torch, numpy as np, torchattacks
import dataset as dataset_mod
from CIFAR10.models.resnet_z import ResNet18_z
from converter import Converter
from utils import cw_Linf_attack

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "CIFAR100/checkpoint/featdir_span_random_10step_wa/k350wa_lamda4_last.pkl"
EPS = 8/255.0
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

_, _, tl = dataset_mod.CIFAR100(root="./data/CIFAR100", download=False, batch_size=256, val=False)
classes = tl.dataset.classes

m = Converter(ResNet18_z(num_classes=100), mean, std)
m.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
m.cuda().eval()

pgd = torchattacks.PGD(m, eps=EPS, steps=20, alpha=2/255, random_start=True)

n = np.zeros(100); c_ok = np.zeros(100); p_ok = np.zeros(100); w_ok = np.zeros(100)
for x, y in tl:
    x, y = x.cuda(), y.cuda()
    x_pgd = pgd(x, y)
    x_cw = cw_Linf_attack(m, x, y, EPS, 2/255, 20, 1)
    with torch.no_grad():
        pc = m(x).argmax(1); pp = m(x_pgd).argmax(1); pw = m(x_cw).argmax(1)
    for cls in y.unique().tolist():
        msk = y == cls
        n[cls] += msk.sum().item()
        c_ok[cls] += (pc[msk] == cls).sum().item()
        p_ok[cls] += (pp[msk] == cls).sum().item()
        w_ok[cls] += (pw[msk] == cls).sum().item()

clean, pgd_a, cw_a = 100*c_ok/n, 100*p_ok/n, 100*w_ok/n
print("ckpt: %s" % CKPT)
print("OVERALL | clean %.2f | pgd20 %.2f | cw %.2f" % (clean.mean(), pgd_a.mean(), cw_a.mean()))
print("\nidx  class                 clean   pgd20    cw")
for i in np.argsort(pgd_a):
    print("%3d  %-20s %6.1f %7.1f %5.1f" % (i, classes[i], clean[i], pgd_a[i], cw_a[i]))
np.save("scratchpad/perclass_champion.npy", np.stack([clean, pgd_a, cw_a]))
