"""Does a natural teacher's feature geometry keep changing after its accuracy saturates?

Motivated by Tiny-ImageNet 2026-08-26: two natural teachers of nearly equal clean accuracy
(65.97 vs 66.29) sent the student to operating points 1.9 clean / 1.6 AA apart. If accuracy is
not what differs, geometry is. CIFAR-100 has clean teachers at 50/100/150/200/300 epochs on
disk, so the question is answerable without training.

Predicted, as epochs grow at saturated accuracy: feature norm up, within-class variance down
relative to between-class (neural collapse), effective rank down. Those are the properties the
anchor copies, so they should predict which way a student anchored on that teacher moves.

    python scripts/diag_teacher_collapse.py [--n_attack 2000]
"""
import argparse
import sys

import torch
import torchattacks

sys.path.insert(0, "/mnt/d/research/FAT")

import dataset
from converter import Converter
from CIFAR10.models.resnet import ResNet18

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
CK = "/mnt/d/research/FAT/CIFAR100/checkpoint"
TEACHERS = [("50ep", "clean"), ("100ep", "clean_100ep"), ("150ep", "clean_150ep"),
            ("200ep", "clean_200ep"), ("300ep", "clean_300ep")]


def eff_rank(f):
    ev = torch.linalg.eigvalsh(torch.cov((f - f.mean(0, keepdim=True)).T.double())).clamp_min(0)
    return (ev.sum() ** 2 / ev.pow(2).sum()).item()


def collapse_stats(f, y, k=100):
    """NC1-style: within-class scatter against between-class scatter, on unit features."""
    f = torch.nn.functional.normalize(f, dim=1).double()
    g = f.mean(0, keepdim=True)
    mu = torch.stack([f[y == c].mean(0) for c in range(k)])
    sw = sum(((f[y == c] - mu[c]) ** 2).sum() for c in range(k)) / len(f)
    sb = ((mu - g) ** 2).sum() / k
    return (sw / sb).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_attack", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=250)
    args = ap.parse_args()

    _, _, loader = dataset.CIFAR100(root="/mnt/d/research/FAT/data/CIFAR100",
                                    download=False, batch_size=args.batch_size)
    print(f"{'teacher':9}{'clean':>7}{'||Phi||':>9}{'norm CV':>9}{'eff.rank':>10}"
          f"{'Sw/Sb':>9}{'angle':>8}{'|a|/|c|':>9}{'PGD':>7}")
    for name, d in TEACHERS:
        m = Converter(ResNet18(num_classes=100), MEAN, STD).cuda()
        m.load_state_dict(torch.load(f"{CK}/{d}/clean_last.pkl", map_location="cuda",
                                     weights_only=False))
        m.eval()
        atk = torchattacks.PGD(m, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True)
        fs, ys, ang, nr, ok, ok_a, n, na = [], [], [], [], 0, 0, 0, 0
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                f_c, z_c = m.extract_feature(x)
            fs.append(f_c.cpu()); ys.append(y.cpu())
            ok += (z_c.argmax(1) == y).sum().item(); n += len(y)
            if na < args.n_attack:
                xa = atk(x, y)
                with torch.no_grad():
                    f_a, z_a = m.extract_feature(xa)
                cos = torch.nn.functional.cosine_similarity(f_c, f_a, dim=1).clamp(-1, 1)
                ang.append(cos.arccos().rad2deg().cpu())
                nr.append((f_a.norm(dim=1) / f_c.norm(dim=1).clamp_min(1e-8)).cpu())
                ok_a += (z_a.argmax(1) == y).sum().item(); na += len(y)
        f = torch.cat(fs).cuda(); y = torch.cat(ys).cuda()
        print(f"{name:9}{100*ok/n:7.2f}{f.norm(dim=1).mean():9.2f}"
              f"{(f.norm(dim=1).std()/f.norm(dim=1).mean()):9.3f}"
              f"{eff_rank(torch.nn.functional.normalize(f, dim=1)):10.1f}"
              f"{collapse_stats(f, y):9.3f}{torch.cat(ang).mean():8.1f}"
              f"{torch.cat(nr).mean():9.3f}{100*ok_a/na:7.2f}")
        del m, atk, f
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
