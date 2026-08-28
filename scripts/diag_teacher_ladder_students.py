"""Closes the loop for METHOD.md section 8: teacher Sw/Sb -> student oscillation -> student AA.

The teacher ladder (diag_teacher_collapse) shows the teacher's within-class scatter falls
monotonically with training length. The student runs (tladder_*) show clean falling and AA
rising along that same ladder. The interpretation is that Sw is both the transferable signal
(F) and the obstruction to being locally constant (O), so the missing measurement is the
student's own oscillation: it should RISE with the teacher's Sw/Sb.

    python scripts/diag_teacher_ladder_students.py
"""
import sys, torch, torchattacks
sys.path.insert(0, "/mnt/d/research/FAT")
sys.path.insert(0, "/mnt/d/research/FAT/scripts")
import dataset
from converter import Converter
from CIFAR10.models.resnet import ResNet18
from CIFAR10.models.resnet_z import ResNet18_z
from diag_teacher_collapse import collapse_stats, eff_rank


def angular_margin(f, y, k=100):
    """Angle (deg) from each sample to its own class mean, and the angle between the two
    nearest class means. The attack has to rotate a sample past roughly half the gap between
    its class mean and the nearest other one, so the meaningful scale for the measured attack
    rotation is this margin, not the rotation on its own."""
    fn = torch.nn.functional.normalize(f, dim=1)
    mu = torch.nn.functional.normalize(
        torch.stack([fn[y == c].mean(0) for c in range(k)]), dim=1)
    own = (fn * mu[y]).sum(1).clamp(-1, 1).arccos().rad2deg().mean().item()
    g = (mu @ mu.T).clamp(-1, 1)
    g.fill_diagonal_(-1.0)
    nearest = g.max(1).values.arccos().rad2deg()          # angle to the closest other class
    return own, nearest.mean().item()

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
CK = "/mnt/d/research/FAT/CIFAR100/checkpoint"
# (label, teacher Sw/Sb, teacher dir, student run dir)
CELLS = [("50ep", 1.100, "clean", "tladder_clean"),
         ("100ep", 0.982, "clean_100ep", "tladder_clean_100ep"),
         ("150ep", 0.887, "clean_150ep", "tladder_clean_150ep"),
         ("200ep", 0.808, "clean_200ep", "tausens_fd_nohd"),
         ("300ep", 0.712, "clean_300ep", "tladder_clean_300ep")]


def load(path, z):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    if isinstance(ck, dict) and "state_dict" in ck:
        ck = ck["state_dict"]
    net = ResNet18_z(num_classes=100, scale=1.0) if z else ResNet18(num_classes=100)
    m = Converter(net, MEAN, STD).cuda()
    m.load_state_dict(ck)
    m.eval()
    return m


def feats(m, z, loader, attack=True, n_max=10000):
    atk = torchattacks.PGD(m, eps=8/255, alpha=2/255, steps=10, random_start=True) if attack else None
    fc, fa, ys, ok, n = [], [], [], 0, 0
    for x, y in loader:
        if n >= n_max:
            break
        x, y = x.cuda(), y.cuda()
        xa = atk(x, y) if attack else None
        with torch.no_grad():
            c = m(x, feat=True) if z else m.extract_feature(x)
            fc.append(c[0].cpu()); ys.append(y.cpu())
            ok += (c[1].argmax(1) == y).sum().item(); n += len(y)
            if attack:
                fa.append((m(xa, feat=True) if z else m.extract_feature(xa))[0].cpu())
    return (torch.cat(fc).cuda(), torch.cat(fa).cuda() if attack else None,
            torch.cat(ys).cuda(), 100 * ok / n)


def main():
    _, _, loader = dataset.CIFAR100(root="/mnt/d/research/FAT/data/CIFAR100",
                                    download=False, batch_size=250)
    print(f"{'teacher':8}{'T.Sw/Sb':>9}{'S.clean':>9}{'S.angle':>9}{'S.reldisp':>11}"
          f"{'cos(S,T)':>10}{'S.Sw/Sb':>9}{'to own mu':>10}{'mu-mu gap':>11}{'ang/margin':>12}")
    for name, sw, tdir, sdir in CELLS:
        t = load(f"{CK}/{tdir}/clean_last.pkl", z=False)
        ft, _, _, tacc = feats(t, False, loader, attack=False)
        del t; torch.cuda.empty_cache()
        s = load(f"{CK}/{sdir}/feat_direction_last.pkl", z=True)
        fc, fa, y, sacc = feats(s, True, loader)
        del s; torch.cuda.empty_cache()
        cos_adv = torch.nn.functional.cosine_similarity(fc, fa, dim=1).clamp(-1, 1)
        ang = cos_adv.arccos().rad2deg().mean().item()
        rel = ((fa - fc).norm(dim=1) / fc.norm(dim=1).clamp_min(1e-8)).mean().item()
        cst = torch.nn.functional.cosine_similarity(fc, ft, dim=1).mean().item()
        own, gap = angular_margin(fc, y)
        print(f"{name:8}{sw:9.3f}{sacc:9.2f}{ang:9.1f}{rel:11.3f}{cst:10.4f}"
              f"{collapse_stats(fc, y, 100):9.3f}{own:10.1f}{gap:11.1f}{ang/(gap/2):12.3f}")
        del fc, fa, ft; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
