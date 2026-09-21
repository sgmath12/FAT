"""Class geometry and displacement on the FINAL models (2026-09-21).

The appendix diagnostics were measured on earlier base-regime checkpoints whose classifier is trained by
a detached KD term, which is not the frozen-head model the paper reports.  Three reviewers asked for the
same measurement on the shipped configuration, so this recomputes it for any pair of checkpoints under
one protocol: test-set batches, true-label PGD at the stated radius, unit-normalized features.

  S_w / S_b   within-class over between-class scatter
  own         mean angle from a sample to its own class centroid
  gap         mean angle from a class centroid to the nearest other centroid
  disp        mean ||Phi(x_adv) - Phi(x)|| / ||Phi(x)||, the relative displacement under attack
"""
import argparse, json, os, sys
import torch, torch.nn.functional as F
import torchvision, torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def build(ckpt, num_classes):
    from CIFAR10.models.resnet import ResNet18
    from converter import Converter
    net = ResNet18(num_classes=num_classes)
    sd = torch.load(ckpt, map_location='cpu')
    sd = {k[len('encoder.'):] if k.startswith('encoder.') else k: v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    return Converter(net, MEAN, STD).cuda().eval()


def feats(m, x):
    return m(x, feat=True)[0]


def pgd(m, x, y, eps, steps=10, alpha=2 / 255):
    xa = x + torch.empty_like(x).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(m(xa), y)
        g = torch.autograd.grad(loss, [xa])[0]
        xa = (xa.detach() + alpha * g.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return xa.detach()


def angles_and_scatter(Fn, Y):
    mu = Fn.mean(0)
    cents = torch.stack([Fn[Y == c].mean(0) for c in Y.unique()])
    sw = sum(((Fn[Y == c] - cents[i]) ** 2).sum().item() for i, c in enumerate(Y.unique()))
    sb = sum((Y == c).sum().item() * ((cents[i] - mu) ** 2).sum().item() for i, c in enumerate(Y.unique()))
    cu = F.normalize(cents, dim=1)
    own = torch.rad2deg(torch.acos(
        (F.normalize(Fn, dim=1) * cu[Y]).sum(1).clamp(-1, 1))).mean().item()
    cc = cu @ cu.t()
    cc.fill_diagonal_(-2)
    gap = torch.rad2deg(torch.acos(cc.max(1).values.clamp(-1, 1))).mean().item()
    return sw / sb, own, gap


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', nargs='+', required=True, help='name=path/to.pkl pairs')
    ap.add_argument('--eps', type=float, default=8 / 255)
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--out', default='writting_docs/paper/notes/geometry_final.json')
    a = ap.parse_args()
    ds = torchvision.datasets.CIFAR100('./data/CIFAR100', train=False, transform=T.ToTensor())
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(a.n)), batch_size=250)
    rows = {}
    for spec in a.checkpoints:
        name, path = spec.split('=')
        m = build(path, 100)
        Fc, Fa, Ys, disp = [], [], [], []
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            xa = pgd(m, x, y, a.eps)
            with torch.no_grad():
                f0, f1 = feats(m, x), feats(m, xa)
            disp.append(((f1 - f0).norm(dim=1) / f0.norm(dim=1)).cpu())
            Fc.append(f0.cpu()); Fa.append(f1.cpu()); Ys.append(y.cpu())
        Fc, Fa, Ys = torch.cat(Fc), torch.cat(Fa), torch.cat(Ys)
        c = angles_and_scatter(Fc, Ys); adv = angles_and_scatter(Fa, Ys)
        rows[name] = dict(clean_sw_sb=c[0], clean_own=c[1], clean_gap=c[2],
                          adv_sw_sb=adv[0], adv_own=adv[1], adv_gap=adv[2],
                          disp=torch.cat(disp).mean().item())
        print(name, json.dumps({k: round(v, 4) for k, v in rows[name].items()}), flush=True)
    json.dump(rows, open(a.out, 'w'), indent=2)
    print('wrote', a.out)
