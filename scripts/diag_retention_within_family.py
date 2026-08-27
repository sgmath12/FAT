"""METHOD.md 260825 section 8: retention vs oscillation, WITHIN our own run family.

Why this and not a comparison against ADR: every cell below starts from the *same* teacher
checkpoint and runs the *same* recipe, differing in one term. So the warm-start confound that
invalidates a cross-method retention comparison is removed by construction, and every cell shares
the teacher's coordinate basis, which makes the direct cos(Phi_s, Phi_t) of T.3 meaningful.

The controlled pair that matters:
    featdir_champ200_100ep   feature anchor, p=0        clean 60.74  AA 28.69
    nofeat_champ200_norm     anchor deleted, logit KD   clean 58.92  AA 28.71
Same init, same recipe, AA identical, clean differs by 1.82. Section 8.6 predicts the difference
is RETENTION of the teacher's clean geometry at matched oscillation -- if instead the two cells
retain equally, the clean gap is not a retention effect and 8.6 is wrong.

    python scripts/diag_retention_within_family.py [--n 10000]
"""
import argparse
import sys

import torch
import torchattacks

sys.path.insert(0, "/mnt/d/research/FAT")

import dataset
from converter import Converter
from CIFAR10.models.resnet import ResNet18
from CIFAR10.models.resnet_z import ResNet18_z

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
CK = "/mnt/d/research/FAT/CIFAR100/checkpoint"

CELLS = [
    ("teacher",   f"{CK}/clean_200ep/clean_last.pkl",                       "plain"),
    ("champ p=1", f"{CK}/featdir_champ200_angeps/feat_direction_last.pkl",  "z"),
    ("champ p=0", f"{CK}/featdir_champ200_100ep/feat_direction_last.pkl",   "z"),
    ("rawfeat",   f"{CK}/featdir_champ200_rawfeat/feat_direction_last.pkl", "z"),
    ("alpha1",    f"{CK}/champ_alpha1/feat_direction_last.pkl",             "z"),
    ("nofeat",    f"{CK}/nofeat_champ200_norm/temperature_last.pkl",        "z"),
    # 8.4-3: label-CE adversarial training initialized from the SAME clean teacher, so it shares
    # the teacher's coordinate basis and its cos is not confounded by warm start (METHOD.md 5).
    ("AT clean-init", f"{CK}/at_ce_freehead/madry_at_last.pkl",              "z"),
]


def build(path, kind):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    if isinstance(ck, dict) and "state_dict" in ck:
        ck = ck["state_dict"]
    net = ResNet18(num_classes=100) if kind == "plain" else ResNet18_z(num_classes=100, scale=1.0)
    m = Converter(net, MEAN, STD).cuda()
    m.load_state_dict(ck)
    m.eval()
    return m, kind


def raw_feature(model, kind, x):
    if kind == "plain":
        return model.extract_feature(x)
    return model(x, feat=True)


def label_probe(x_tr, y_tr, x_te, y_te, k=100, lam=1e-2):
    x_tr = torch.cat([x_tr, torch.ones(len(x_tr), 1, device=x_tr.device)], 1)
    x_te = torch.cat([x_te, torch.ones(len(x_te), 1, device=x_te.device)], 1)
    y1 = torch.zeros(len(y_tr), k, device=x_tr.device)
    y1[torch.arange(len(y_tr)), y_tr] = 1.0
    a = x_tr.T @ x_tr + lam * torch.eye(x_tr.shape[1], device=x_tr.device) * x_tr.shape[0]
    w = torch.linalg.solve(a, x_tr.T @ y1)
    return 100.0 * ((x_te @ w).argmax(1) == y_te).float().mean().item()


def eff_rank(f):
    ev = torch.linalg.eigvalsh(torch.cov((f - f.mean(0, keepdim=True)).T.double())).clamp_min(0)
    return (ev.sum() ** 2 / ev.pow(2).sum()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=250)
    args = ap.parse_args()

    _, _, loader = dataset.CIFAR100(root="/mnt/d/research/FAT/data/CIFAR100",
                                    download=False, batch_size=args.batch_size)
    rows, feats, labels = {}, {}, None
    for name, path, kind in CELLS:
        model, kind = build(path, kind)
        atk = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=args.steps, random_start=True)
        fc, lab, ang, rel, ok_c, ok_a, n = [], [], [], [], 0, 0, 0
        for x, y in loader:
            if n >= args.n:
                break
            x, y = x.cuda(), y.cuda()
            xa = atk(x, y)
            with torch.no_grad():
                f_c, z_c = raw_feature(model, kind, x)
                f_a, z_a = raw_feature(model, kind, xa)
                cos = torch.nn.functional.cosine_similarity(f_c, f_a, dim=1).clamp(-1, 1)
                ang.append(cos.arccos().rad2deg().cpu())
                rel.append(((f_a - f_c).norm(dim=1) / f_c.norm(dim=1).clamp_min(1e-8)).cpu())
                fc.append(f_c.cpu()); lab.append(y.cpu())
                ok_c += (z_c.argmax(1) == y).sum().item()
                ok_a += (z_a.argmax(1) == y).sum().item()
                n += x.size(0)
        feats[name] = torch.cat(fc)
        labels = torch.cat(lab)
        rows[name] = dict(clean=100 * ok_c / n, pgd=100 * ok_a / n,
                          angle=torch.cat(ang).mean().item(), rel=torch.cat(rel).mean().item())
        print(f"  {name:10s} clean={rows[name]['clean']:6.2f} pgd10={rows[name]['pgd']:6.2f}")
        del model, atk
        torch.cuda.empty_cache()

    labels = labels.cuda()
    feats = {k: v.cuda() for k, v in feats.items()}
    t = feats["teacher"]
    h = len(t) // 2
    print(f"\n{'cell':14}{'clean':>7}{'PGD-10':>8}{'angle':>8}{'reldisp':>9}"
          f"{'cos vs T':>10}{'lin.probe':>11}{'eff.rank':>10}")
    for name, _, _ in CELLS:
        r = rows[name]
        f = feats[name]
        c = torch.nn.functional.cosine_similarity(f, t, dim=1).mean().item()
        print(f"{name:14}{r['clean']:7.2f}{r['pgd']:8.2f}{r['angle']:8.1f}{r['rel']:9.3f}"
              f"{c:10.4f}{label_probe(f[:h], labels[:h], f[h:], labels[h:]):11.2f}{eff_rank(f):10.1f}")


if __name__ == "__main__":
    main()
