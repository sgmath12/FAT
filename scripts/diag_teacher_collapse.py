"""Does a natural teacher's feature geometry keep changing after its accuracy saturates?

Motivated by Tiny-ImageNet 2026-08-26: two natural teachers of nearly equal clean accuracy
(65.97 vs 66.29) sent the student to operating points 1.9 clean / 1.6 AA apart. If accuracy is
not what differs, geometry is. CIFAR-100 has clean teachers at 50/100/150/200/300 epochs on
disk, so the question is answerable without training.

Predicted, as epochs grow at saturated accuracy: feature norm up, within-class variance down
relative to between-class (neural collapse), effective rank down. Those are the properties the
anchor copies, so they should predict which way a student anchored on that teacher moves.

    python scripts/diag_teacher_collapse.py                       # CIFAR-100, the 5-teacher ladder
    python scripts/diag_teacher_collapse.py --dataset TinyImageNet \
           --teachers 100ep=clean_100ep 200ep=clean_200ep          # other server

Sw/Sb needs enough samples per class: CIFAR-100 gives 100/class, Tiny-ImageNet 50/class, so
the TIN estimate is noisier. Compare TIN teachers against each other, not against the CIFAR
numbers.
"""
import argparse
import sys

import torch
import torchattacks

sys.path.insert(0, "/mnt/d/research/FAT")

import dataset
from converter import Converter
from CIFAR10.models.resnet import ResNet18

ROOT = "/mnt/d/research/FAT"
# mean/std and class counts mirror utils.py's per-dataset branch exactly.
DATASETS = {
    "CIFAR100": dict(
        mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
        num_classes=100,
        default=[("50ep", "clean"), ("100ep", "clean_100ep"), ("150ep", "clean_150ep"),
                 ("200ep", "clean_200ep"), ("300ep", "clean_300ep")],
    ),
    "TinyImageNet": dict(
        mean=(0.4802, 0.4481, 0.3975),
        std=(0.2302, 0.2265, 0.2262),
        num_classes=200,
        default=[("80ep", "clean_80ep"), ("200ep", "clean_200ep")],
    ),
}


def eff_rank(f):
    ev = torch.linalg.eigvalsh(torch.cov((f - f.mean(0, keepdim=True)).T.double())).clamp_min(0)
    return (ev.sum() ** 2 / ev.pow(2).sum()).item()


def separation_stats(f, y, k):
    """Split Sw/Sb into its two halves: how far a sample sits from its own class mean, and how
    far the class means sit from each other. Both in degrees on the unit sphere, so a falling
    Sw/Sb can be read as either the numerator shrinking or the denominator growing."""
    fn = torch.nn.functional.normalize(f, dim=1)
    mu = torch.nn.functional.normalize(
        torch.stack([fn[y == c].mean(0) for c in range(k)]), dim=1)
    own = (fn * mu[y]).sum(1).clamp(-1, 1).arccos().rad2deg().mean().item()
    g = (mu @ mu.T).clamp(-1, 1)
    g.fill_diagonal_(-1.0)
    return own, g.max(1).values.arccos().rad2deg().mean().item()


def collapse_stats(f, y, k):
    """NC1-style: within-class scatter against between-class scatter, on unit features.
    Lower = more collapsed = the teacher has kept sharpening after its accuracy saturated."""
    f = torch.nn.functional.normalize(f, dim=1).double()
    g = f.mean(0, keepdim=True)
    mu = torch.stack([f[y == c].mean(0) for c in range(k)])
    sw = sum(((f[y == c] - mu[c]) ** 2).sum() for c in range(k)) / len(f)
    sb = ((mu - g) ** 2).sum() / k
    return (sw / sb).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CIFAR100", choices=sorted(DATASETS))
    ap.add_argument("--teachers", nargs="*", default=None,
                    help="LABEL=CKPT_DIRNAME pairs, e.g. 100ep=clean_100ep 200ep=clean_200ep")
    ap.add_argument("--n_attack", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=250)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    mean, std, k = cfg["mean"], cfg["std"], cfg["num_classes"]
    ck = f"{ROOT}/{args.dataset}/checkpoint"
    teachers = ([tuple(s.split("=", 1)) for s in args.teachers] if args.teachers
                else cfg["default"])
    root = f"{ROOT}/data/{args.dataset}" if args.dataset == "CIFAR100" else f"{ROOT}/data"
    _, _, loader = getattr(dataset, args.dataset)(root=root, download=False,
                                                  batch_size=args.batch_size)
    print(f"dataset={args.dataset}  classes={k}  teachers={[n for n, _ in teachers]}")
    print(f"{'teacher':9}{'clean':>7}{'||Phi||':>9}{'norm CV':>9}{'eff.rank':>10}"
          f"{'Sw/Sb':>9}{'to own mu':>10}{'mu-mu gap':>11}{'angle':>8}{'|a|/|c|':>9}{'PGD':>7}")
    for name, d in teachers:
        m = Converter(ResNet18(num_classes=k), mean, std).cuda()
        m.load_state_dict(torch.load(f"{ck}/{d}/clean_last.pkl", map_location="cuda",
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
              f"{collapse_stats(f, y, k):9.3f}"
              f"{separation_stats(f, y, k)[0]:10.1f}{separation_stats(f, y, k)[1]:11.1f}"
              f"{torch.cat(ang).mean():8.1f}"
              f"{torch.cat(nr).mean():9.3f}{100*ok_a/na:7.2f}")
        del m, atk, f
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
