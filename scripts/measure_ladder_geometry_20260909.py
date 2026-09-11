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
CELLS = [("teacher 5ep",    "clean_5ep/clean_last.pkl"),
         ("teacher 10ep",   "clean_10ep/clean_last.pkl"),
         ("teacher 20ep",   "clean_20ep/clean_last.pkl"),
         ("teacher 40ep",   "clean_40ep/clean_last.pkl"),
         ("teacher 50ep",   "clean/clean_last.pkl"),
         ("teacher 100ep",  "clean_100ep/clean_last.pkl"),
         ("teacher 150ep",  "clean_150ep/clean_last.pkl"),
         ("teacher 200ep",  "clean_200ep/clean_last.pkl"),
         ("teacher 300ep",  "clean_300ep/clean_last.pkl"),
         ("student t5",     "tladder_clean_5ep/feat_direction_last.pkl"),
         ("student t10",    "tladder_clean_10ep/feat_direction_last.pkl"),
         ("student t20",    "tladder_clean_20ep/feat_direction_last.pkl"),
         ("student t50",    "tladder_clean/feat_direction_last.pkl"),
         ("student t100",   "tladder_clean_100ep/feat_direction_last.pkl"),
         ("student t150",   "tladder_clean_150ep/feat_direction_last.pkl"),
         ("student t200",   "tausens_fd_nohd/feat_direction_last.pkl"),
         ("student t300",   "tladder_clean_300ep/feat_direction_last.pkl")]
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
