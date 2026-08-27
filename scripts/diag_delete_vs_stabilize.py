"""METHOD.md 260825 section 8.8-2: deletion versus stabilization.

The claim under test (section 8.6): AT reaches robustness by DELETING the attack-volatile
directions -- which is what costs it clean accuracy -- while the feature anchor is forbidden
that route by the fidelity term F and must instead STABILIZE, keeping the teacher's clean
geometry while shrinking its oscillation.

Prediction:
    teacher   high retention (=1 by definition), high oscillation
    ADR (AT)  LOW retention, low oscillation      <- reached robustness by deleting
    ours      HIGH retention, low oscillation     <- reached robustness by stabilizing

Retention must be basis-free: ADR is a separately-trained network, so a raw cosine against the
teacher's coordinates is meaningless. Two basis-free observables are used --
  * linear CKA(Phi_model(x), Phi_teacher(x)) on clean inputs
  * ridge probe R^2: how much of the teacher's clean feature is linearly recoverable from the
    model's own clean feature (fit on half the test set, scored on the other half)
Oscillation is measured inside each model's own feature space and is therefore comparable:
  * angle between Phi(x) and Phi(x_adv), and the relative displacement.

    python scripts/diag_delete_vs_stabilize.py [--n 10000] [--steps 10]
"""
import argparse
import sys

import torch
import torchattacks

sys.path.insert(0, "/mnt/d/research/FAT")
sys.path.insert(0, "/mnt/d/research/ADR/src")

import dataset
from converter import Converter
from CIFAR10.models.resnet import ResNet18
from CIFAR10.models.resnet_z import ResNet18_z

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
TEACHER = "/mnt/d/research/FAT/CIFAR100/checkpoint/clean_200ep/clean_last.pkl"
OURS = "/mnt/d/research/FAT/CIFAR100/checkpoint/featdir_champ200_angeps/feat_direction_last.pkl"
ADR = "/mnt/d/research/ADR/pretrained/cifar100/resnet18_cifar100_adr_wa_awp.pt"


def penultimate_hook(module):
    """Capture the input of the final linear layer's *raw* predecessor via a forward hook."""
    store = {}

    def hook(_m, inp, _out):
        store["feat"] = inp[0].detach()

    module.register_forward_hook(hook)
    return store


def build_model(which):
    models = {}

    if which != "teacher":
        pass
    ck = torch.load(TEACHER, map_location="cuda", weights_only=False)
    t = Converter(ResNet18(num_classes=100), MEAN, STD).cuda()
    t.load_state_dict(ck)
    t.eval()
    if which == "teacher":
        return t, None

    ck = torch.load(OURS, map_location="cuda", weights_only=False)
    if isinstance(ck, dict) and "state_dict" in ck:
        ck = ck["state_dict"]
    s = Converter(ResNet18_z(num_classes=100, scale=1.0), MEAN, STD).cuda()
    s.load_state_dict(ck)
    s.eval()
    if which == "ours":
        return s, None

    from model.resnet import ResNet as ADRResNet

    ck = torch.load(ADR, map_location="cuda", weights_only=False)
    key = "model_ema" if "model_ema" in ck else "model"
    a = ADRResNet((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), num_classes=100,
                  depth=18, activation_fn=torch.nn.ReLU, adaptive_pooling=False).cuda()
    a.load_state_dict(ck[key])
    a.eval()
    print(f"  ADR checkpoint: loaded '{key}' (epoch {ck.get('epoch')})")
    return a, penultimate_hook(a.linear)


def raw_feature(name, model, store, x):
    """Raw penultimate feature (pre head-normalization) for any of the three models."""
    if name == "ours":
        return model(x, feat=True)          # resnet_z returns (raw feat, logits)
    if name == "teacher":
        return model.extract_feature(x)     # plain resnet returns (raw feat, logits)
    logits = model(x)                       # ADR: captured by the forward hook
    return store["feat"], logits


def linear_cka(a, b):
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    num = (a.T @ b).pow(2).sum()
    den = (a.T @ a).norm() * (b.T @ b).norm()
    return (num / den.clamp_min(1e-12)).item()


def ridge_r2(x_tr, y_tr, x_te, y_te, lam=1e-2):
    x_tr = torch.cat([x_tr, torch.ones(len(x_tr), 1, device=x_tr.device)], 1)
    x_te = torch.cat([x_te, torch.ones(len(x_te), 1, device=x_te.device)], 1)
    d = x_tr.shape[1]
    a = x_tr.T @ x_tr + lam * torch.eye(d, device=x_tr.device) * x_tr.shape[0]
    w = torch.linalg.solve(a, x_tr.T @ y_tr)
    resid = (x_te @ w - y_te).pow(2).sum()
    total = (y_te - y_te.mean(0, keepdim=True)).pow(2).sum()
    return (1 - resid / total).item()


def label_probe(x_tr, y_tr, x_te, y_te, k=100, lam=1e-2):
    """Linear probe on frozen clean features -> labels. Ridge on one-hot, argmax accuracy.
    Measures how much clean-label information the backbone still carries, head removed."""
    x_tr = torch.cat([x_tr, torch.ones(len(x_tr), 1, device=x_tr.device)], 1)
    x_te = torch.cat([x_te, torch.ones(len(x_te), 1, device=x_te.device)], 1)
    y1 = torch.zeros(len(y_tr), k, device=x_tr.device)
    y1[torch.arange(len(y_tr)), y_tr] = 1.0
    d = x_tr.shape[1]
    a = x_tr.T @ x_tr + lam * torch.eye(d, device=x_tr.device) * x_tr.shape[0]
    w = torch.linalg.solve(a, x_tr.T @ y1)
    return 100.0 * ((x_te @ w).argmax(1) == y_te).float().mean().item()


def eff_rank(f):
    c = torch.cov((f - f.mean(0, keepdim=True)).T.double())
    ev = torch.linalg.eigvalsh(c).clamp_min(0)
    return (ev.sum() ** 2 / ev.pow(2).sum()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=250)
    args = ap.parse_args()

    _, _, test_loader = dataset.CIFAR100(root="/mnt/d/research/FAT/data/CIFAR100",
                                         download=False, batch_size=args.batch_size)

    feats_clean, stats, labels = {}, {}, None
    for name in ("teacher", "ours", "ADR"):
        print(f"loading {name}")
        model, store = build_model(name)
        attack = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=args.steps,
                                  random_start=True)
        fc, lab, ang, rel, nr, ok_c, ok_a, n = [], [], [], [], [], 0, 0, 0
        for x, y in test_loader:
            if n >= args.n:
                break
            x, y = x.cuda(), y.cuda()
            x_adv = attack(x, y)
            with torch.no_grad():
                f_c, z_c = raw_feature(name, model, store, x)
                f_a, z_a = raw_feature(name, model, store, x_adv)
                cos = torch.nn.functional.cosine_similarity(f_c, f_a, dim=1).clamp(-1, 1)
                ang.append(cos.arccos().rad2deg().cpu())
                rel.append(((f_a - f_c).norm(dim=1) / f_c.norm(dim=1).clamp_min(1e-8)).cpu())
                nr.append((f_a.norm(dim=1) / f_c.norm(dim=1).clamp_min(1e-8)).cpu())
                fc.append(f_c.cpu()); lab.append(y.cpu())
                ok_c += (z_c.argmax(1) == y).sum().item()
                ok_a += (z_a.argmax(1) == y).sum().item()
                n += x.size(0)
        feats_clean[name] = torch.cat(fc)
        labels = torch.cat(lab)
        del model, store, attack
        torch.cuda.empty_cache()
        stats[name] = dict(clean=100 * ok_c / n, pgd=100 * ok_a / n,
                           angle=torch.cat(ang).mean().item(),
                           rel=torch.cat(rel).mean().item(),
                           normratio=torch.cat(nr).mean().item(), n=n)
        print(f"  {name:8s} done  n={n}  clean={stats[name]['clean']:.2f}  pgd10={stats[name]['pgd']:.2f}")

    labels = labels.cuda()
    feats_clean = {k: v.cuda() for k, v in feats_clean.items()}
    t = feats_clean["teacher"]
    h = len(t) // 2
    print(f"\n{'':10}{'clean':>7}{'PGD-10':>8}{'angle':>8}{'relat.':>8}{'|a|/|c|':>9}"
          f"{'CKA vs T':>10}{'probe R2':>10}{'lin.probe':>11}{'eff.rank':>10}")
    for name in ("teacher", "ADR", "ours"):
        s = stats[name]
        f = feats_clean[name]
        cka = linear_cka(f, t)
        r2 = ridge_r2(f[:h], t[:h], f[h:], t[h:])
        lp = label_probe(f[:h], labels[:h], f[h:], labels[h:])
        print(f"{name:10}{s['clean']:7.2f}{s['pgd']:8.2f}{s['angle']:8.1f}{s['rel']:8.3f}"
              f"{s['normratio']:9.3f}{cka:10.3f}{r2:10.3f}{lp:11.2f}{eff_rank(f):10.1f}")

    same = torch.nn.functional.cosine_similarity(feats_clean["ours"], t, dim=1).mean().item()
    print(f"\nsame-basis cos(ours, teacher) on clean inputs = {same:.4f}   "
          f"(T.3 reports 0.8345 for the champion family)")


if __name__ == "__main__":
    main()
