r"""Why does the sensitivity-matched radius buy clean accuracy and not robustness? (2026-09-24)

The allocation gives a smaller radius to examples whose anchor loss responds strongly to the input and a
larger one to the rest, at a fixed batch mean.  The first-order argument says nothing about which axis
should move, and the measured effect is one-sided: +2.19 clean, +0.03 AA over three paired seeds.  This
script tests the obvious candidate explanation, that the two arms differ in OPPOSITE directions on the
two ends of the sensitivity distribution and the robust effects cancel while the clean ones do not.

Examples are binned by a quantity that belongs to neither arm: the TEACHER's per-example feature
sensitivity ||Phi_t(x+d) - Phi_t(x)|| / ||d||, averaged over several uniform d at the training radius.
The allocation's own signal, the gradient of ||Phi_s - Phi_t||^2, is zero at initialization (student and
teacher coincide, which is why the method floors it) and arm-dependent later, so it cannot define a
shared split.  The teacher's sensitivity is the fixed property whose ranking that signal inherits early.

Reported per bin, for each arm: the radius multiplier the allocation assigns, the clean anchor error
||Phi_s(x) - Phi_t(x)||, the adversarial anchor error under an anchor-loss PGD, and clean and PGD-20
accuracy.  PGD-20 stands in for AutoAttack because the comparison is within one pair of models across
bins rather than a headline robustness number.

What this can and cannot settle.  If the high-sensitivity bins are the ones that receive smaller radii
AND the ones whose clean anchor error falls, while robust accuracy moves in opposite directions at the
two ends and cancels, then "a uniform radius over-constrains the sensitive examples" has evidence behind
it.  It would still be an association across bins of one pair of models, not a mechanism, and the
first-order argument of the method predicts none of it -- that argument motivates equalizing the
perturbation effect and says nothing about which accuracy axis should move.
"""
import argparse, json, os, sys
import torch, torch.nn.functional as F
import torchvision, torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def build(ckpt, num_classes=100):
    from CIFAR10.models.resnet import ResNet18
    from converter import Converter
    net = ResNet18(num_classes=num_classes)
    sd = torch.load(ckpt, map_location='cpu')
    sd = {k[len('encoder.'):] if k.startswith('encoder.') else k: v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    return Converter(net, MEAN, STD).cuda().eval()


def anchor_pgd(m, x, phi_t, eps, steps=10, alpha=2 / 255):
    """PGD on the anchor loss itself, which is the attack the method trains against."""
    xa = x.detach() + 0.001 * torch.randn_like(x)
    for _ in range(steps):
        xa.requires_grad_()
        with torch.enable_grad():
            loss = (m(xa, feat=True)[0] - phi_t).pow(2).sum(dim=1).mean()
        g = torch.autograd.grad(loss, [xa])[0]
        xa = (xa.detach() + alpha * g.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return xa.detach()


def alloc_weights(m, x, phi_t, lo=0.5, hi=1.5):
    """The multiplier the training rule would assign, recomputed on this model."""
    xg = x.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        l0 = (m(xg, feat=True)[0] - phi_t).pow(2).sum()
    g = torch.autograd.grad(l0, [xg])[0].detach().flatten(1).norm(dim=1).clamp(min=1e-12)
    w = (g.mean() / g).clamp(lo, hi)
    return w * (w.numel() / w.sum()), g


def pgd(m, x, y, eps, steps=20, alpha=2 / 255):
    xa = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        xa.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(m(xa), y)
        g = torch.autograd.grad(loss, [xa])[0]
        xa = (xa.detach() + alpha * g.sign()).clamp(x - eps, x + eps).clamp(0, 1)
    return xa.detach()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', default='CIFAR100/checkpoint/clean_200ep/clean_last.pkl')
    ap.add_argument('--arms', nargs='+', required=True, help='name=checkpoint pairs')
    ap.add_argument('--eps', type=float, default=8 / 255)
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--bins', type=int, default=4)
    ap.add_argument('--deltas', type=int, default=8)
    ap.add_argument('--split-by', default='teacher_sens',
                    help="teacher_sens, or a checkpoint whose own allocation weight defines the bins")
    ap.add_argument('--out', default='writting_docs/paper/notes/radius_decompose.json')
    a = ap.parse_args()
    torch.manual_seed(0)
    ds = torchvision.datasets.CIFAR100('./data/CIFAR100', train=False, transform=T.ToTensor())
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(a.n)), batch_size=250)

    t = build(a.teacher)
    sens, Ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            f0 = t(x, feat=True)[0]
            s = 0.0
            for _ in range(a.deltas):
                d = (torch.rand_like(x) * 2 - 1) * a.eps
                f1 = t((x + d).clamp(0, 1), feat=True)[0]
                s = s + (f1 - f0).norm(dim=1) / d.flatten(1).norm(dim=1)
            sens.append((s / a.deltas).cpu()); Ys.append(y)
    sens, Ys = torch.cat(sens), torch.cat(Ys)
    if a.split_by != 'teacher_sens':
        # Bin by the multiplier the named model's own rule assigns, which is the quantity of interest --
        # who actually received a smaller radius -- at the cost of a split defined by one of the arms.
        ref = build(a.split_by)
        ws = []
        for x, _ in loader:
            x = x.cuda()
            with torch.no_grad():
                phi_t = t(x, feat=True)[0].detach()
            ws.append(alloc_weights(ref, x, phi_t)[0].detach().cpu())
        sens = -torch.cat(ws)     # negate so bin 0 is the largest radius, as with sensitivity
        del ref; torch.cuda.empty_cache()
    edges = torch.quantile(sens, torch.linspace(0, 1, a.bins + 1))
    binid = torch.bucketize(sens, edges[1:-1])

    teacher = t
    rows = {}
    for spec in a.arms:
        name, path = spec.split('=')
        m = build(path)
        cl, rb, w_all, fe_cl, fe_adv = [], [], [], [], []
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                phi_t = teacher(x, feat=True)[0].detach()
            w, _ = alloc_weights(m, x, phi_t)
            w_all.append(w.detach().cpu())
            xa_anchor = anchor_pgd(m, x, phi_t, a.eps)
            xa_ce = pgd(m, x, y, a.eps)
            with torch.no_grad():
                f_cl = m(x, feat=True)[0]
                f_adv = m(xa_anchor, feat=True)[0]
                fe_cl.append((f_cl - phi_t).norm(dim=1).cpu())
                fe_adv.append((f_adv - phi_t).norm(dim=1).cpu())
                cl.append((m(x).argmax(1) == y).cpu())
                rb.append((m(xa_ce).argmax(1) == y).cpu())
        cl, rb = torch.cat(cl), torch.cat(rb)
        w_all, fe_cl, fe_adv = torch.cat(w_all), torch.cat(fe_cl), torch.cat(fe_adv)
        per = lambda v, f=100.0: [round(f * v[binid == b].float().mean().item(), 3) for b in range(a.bins)]
        rows[name] = dict(
            clean=round(100 * cl.float().mean().item(), 2),
            pgd20=round(100 * rb.float().mean().item(), 2),
            clean_bin=per(cl), pgd20_bin=per(rb),
            w_bin=per(w_all, 1.0), anchor_clean_bin=per(fe_cl, 1.0), anchor_adv_bin=per(fe_adv, 1.0))
        print(name, json.dumps(rows[name]), flush=True)
        del m; torch.cuda.empty_cache()
    rows['_sensitivity_bin_edges'] = [round(e.item(), 4) for e in edges]
    rows['_n'] = a.n
    json.dump(rows, open(a.out, 'w'), indent=2)
    print('wrote', a.out)
