# CLASS GEOMETRY OF THE TEMPERATURE SWEEP'S STUDENTS (2026-09-08).
#
# tab:targetcomparison reports what each target buys in accuracy.  It does not say why softening
# stops short, which is what makes it readable as a bare sweep.  The mechanism section measures class
# separation between the anchor and label cross-entropy; this applies the same measurement to the
# five temperature students, so the table can be read as "the target's softness moves robustness, and
# the separation the student ends up with is what does not move".
#
# Sw/Sb is the within-class over between-class scatter of unit features (diag_teacher_collapse), so
# lower is better separated.  Measured on clean test inputs and under a true-label PGD-20.
import sys, os, torch, torchattacks
sys.path.insert(0, "/mnt/d/research/FAT"); sys.path.insert(0, "/mnt/d/research/FAT/scripts")
os.chdir("/mnt/d/research/FAT")
import dataset
from converter import Converter
from CIFAR10.models.resnet import ResNet18
from diag_teacher_collapse import collapse_stats

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
CELLS = [("logits, tau=1",  "tausens_kd_t1/temperature_last.pkl"),
         ("logits, tau=2",  "tausens_kd_t2/temperature_last.pkl"),
         ("logits, tau=4",  "tausens_kd_t4/temperature_last.pkl"),
         ("logits, tau=8",  "tausens_kd_t8/temperature_last.pkl"),
         ("logits, tau=16", "tausens_kd_t16/temperature_last.pkl"),
         ("feature anchor", "tausens_fd_nohd/feat_direction_last.pkl"),
         ("natural teacher","clean_200ep/clean_last.pkl")]
N_ATTACK = 2000

_, _, loader = dataset.CIFAR100(root="./data/CIFAR100", download=False, batch_size=250)
for label, rel in CELLS:
    path = f"CIFAR100/checkpoint/{rel}"
    if not os.path.exists(path):
        print(f"GEO | {label:16s} | missing {path}", flush=True); continue
    m = Converter(ResNet18(num_classes=100), MEAN, STD).cuda()
    m.load_state_dict(torch.load(path, map_location="cuda"), strict=False)
    m.eval()
    atk = torchattacks.PGD(m, eps=8/255, alpha=2/255, steps=20, random_start=True)
    fc, fa, ys, ok, n = [], [], [], 0, 0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        xa = atk(x, y) if n < N_ATTACK else None
        with torch.no_grad():
            f, z = m.extract_feature(x)
            fc.append(f.cpu()); ys.append(y.cpu()); ok += (z.argmax(1) == y).sum().item()
            if xa is not None:
                fa.append(m.extract_feature(xa)[0].cpu())
        n += len(y)
    fc, ys = torch.cat(fc).cuda(), torch.cat(ys).cuda()
    fa = torch.cat(fa).cuda()
    sw_clean = collapse_stats(fc, ys, 100)
    sw_adv = collapse_stats(fa, ys[:len(fa)], 100)
    print(f"GEO | {label:16s} | clean acc {100.0*ok/n:5.2f} | Sw/Sb clean {sw_clean:6.3f} "
          f"| Sw/Sb adv {sw_adv:6.3f}", flush=True)
