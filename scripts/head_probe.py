"""ROBUST LINEAR PROBE (2026-08-20, v2): compare the two backbone objectives with the head confound removed.

Why.  The backbone is already clean: `featdir_alpha` defaults to 0 (methods.py:2106), so the head KD
term is fully detached and never reaches the backbone.  The HEAD is not clean.  It is fitted against
z_t/16 while reading a feature whose norm is ~1 for the directional design and ~12 for raw L2, so a
difference in final clean/AA can come from the representation OR from how well the head happened to
fit it.  This refits the head, identically, on a frozen backbone, so only the representation differs.

v1 -> v2.  v1 gave every cell the same lr and weight decay, which is NOT the same optimization
problem: on a unit-norm feature the head needs ~12x larger weights to reach the same logit scale,
and weight decay fights exactly that.  It showed -- the directional probes plateaued at CE 3.48 and
4.00 against ln(100) = 4.61, i.e. near-flat logits, while *beating* the raw cells on train accuracy.
v2 fixes it with one global constant: the head input is divided by its own mean norm, measured once
on clean train data.  A constant rescale changes no design property (raw still delivers its
per-sample magnitude to the classifier, direction still delivers none); it removes the arbitrary
units so that identical hyperparameters finally mean identical problems.  Weight decay defaults to 0
for the same reason.

The scaling has to be applied to the head INPUT, not folded into the head weight: folding equalizes
only the starting logits, while the gradient wrt the weight still carries the feature norm, so the
effective learning rate would stay ~12x apart.  Hence the explicit wrapper below, which is used for
training, for the attack, and for every evaluation, so no path can disagree.

Each design keeps its OWN inference geometry -- direction reads normalize(Phi)*scale, full raw reads
Phi -- so this is not the banned partial-raw hybrid.  Only the head-fitting procedure is equalized.

Read: if direction still loses after this, the deficit is in the representation.  If the gap closes,
it was in the head fit.  The bare-regime pair matters most: that is where raw L2 wins outright
(62.40 / AA 24.34 against 61.52 / 22.90) with no stack available to explain it away.
"""
import sys, argparse, time
sys.path.insert(0, '/mnt/d/research/FAT'); sys.path.insert(0, '/mnt/d/research/FAT/scratchpad')
import torch, torch.nn as nn, torch.nn.functional as F
import dataset
from measure_cosadv import make_args
from utils import load_config, get_model, evaluate, evaluate_final_aa

DEV = 'cuda'


def enc_of(m):
    m = m.model if hasattr(m, 'model') else m
    return getattr(m, 'encoder', m)


class ProbeNet(nn.Module):
    """Frozen backbone + rescaled head input, reproducing the cell's own inference geometry.

    `raw=True`  -> h = Phi            (full-raw design: magnitude reaches the classifier)
    `raw=False` -> h = normalize(Phi) * scale   (directional design, as ResNet18_z.forward does)
    then logits = linear(h / gs).  gs is a single constant, so the geometry is untouched.
    """

    def __init__(self, student, raw, scale, gs=1.0):
        super().__init__()
        self.student, self.raw, self.scale, self.gs = student, raw, scale, gs
        self.linear = enc_of(student).linear

    def feat(self, x):
        f, _ = self.student(x, feat=True)          # both archs return the PRE-normalization feature
        return f if self.raw else F.normalize(f, dim=1) * self.scale

    def forward(self, x):
        return self.linear(self.feat(x) / self.gs)


def ce_pgd(model, x, y, eps, step, steps):
    model.eval()
    xa = x.detach() + 0.001 * torch.randn_like(x)
    for _ in range(steps):
        xa.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(model(xa), y)
        g = torch.autograd.grad(loss, [xa])[0]
        xa = xa.detach() + step * torch.sign(g.detach())
        xa = torch.min(torch.max(xa, x - eps), x + eps).clamp(0.0, 1.0)
    return xa.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--aa', type=int, default=1)
    # --adv 0 fits the head on CLEAN inputs only.  Seconds per epoch instead of ~50, since the PGD
    # that dominates the budget is skipped; the head is linear and the backbone frozen, so nothing
    # else is expensive.  Worth its own run: the backbone was trained adversarially and its features
    # move ~12 degrees under attack, so a head fitted on clean features need not be the right head
    # for attacked ones.  Running both says which data the second stage should use.
    ap.add_argument('--adv', type=int, default=1)
    # --ls: label smoothing for the head stage.  Retraining the head on hard labels came out slightly
    # BELOW leaving the teacher's head alone (NRR 36.43 adversarial / 36.06 clean against 36.53), and
    # one reading is that hard labels are simply too sharp a target for this stage.  Label smoothing
    # supplies a soft target with no teacher in it at all, so it separates "the teacher's head is
    # special" from "any adequately soft target will do".
    ap.add_argument('--ls', type=float, default=0.0)
    # --kd tau: train the head by KL to the teacher's softened logits instead of on labels, i.e.
    # reproduce the head the shipped recipe's head-KD term would have produced.  This is a valid
    # substitute for re-training the whole run with the head term switched on, because that term is
    # detached (featdir_alpha defaults to 0) and the attack is computed from the feature loss alone:
    # the backbone trajectory and the x_adv sequence are identical with the head term on or off, and
    # only the head's weights differ.  0 = off (use CE).
    ap.add_argument('--kd', type=float, default=0.0)
    ap.add_argument('--dataset', default='')
    # --reset 0 keeps the head as loaded (the teacher's) instead of re-initializing it.  Required
    # whenever --kd is used: the shipped recipe never resets the head, it warm-starts it from the
    # teacher and lets the KD term adjust it.  Starting from random against a softened target does
    # not work and is not what the recipe does -- on Tiny-ImageNet (200 classes, uniform = 0.005)
    # the KL to z_t/16 begins at 0.0132 and the head never leaves chance, clean 9.06.
    ap.add_argument('--reset', type=int, default=1)
    a = ap.parse_args()

    _args = make_args(a.cell + '.yaml')
    if a.dataset: _args.dataset = a.dataset
    cfg = load_config(_args)
    eps = float(getattr(cfg, 'train_eps', 0.0) or 0.0) or cfg.eps
    raw = not bool(getattr(cfg, 'student_norm', True))
    scale = float(getattr(cfg, 'feat_scale', 1.0) or 1.0)

    teacher, student = get_model(cfg)
    student.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
    student = student.to(DEV).eval()
    for p in student.parameters():
        p.requires_grad_(False)

    train_loader, _, test_loader = getattr(dataset, cfg.dataset)(
        root=f'./data/{cfg.dataset}', download=False, batch_size=cfg.batch_size, val=False, config=cfg)

    net = ProbeNet(student, raw, scale).to(DEV)
    if a.kd > 0:
        tsd = torch.load(cfg.checkpoint, map_location='cpu')
        teacher.load_state_dict(tsd)
        teacher = teacher.to(DEV).eval()
        for p_ in teacher.parameters():
            p_.requires_grad_(False)

    with torch.no_grad():                            # mean head-input norm on clean train data
        tot, seen = 0.0, 0
        for i, (x, _) in enumerate(train_loader):
            if i >= 16:
                break
            h = net.feat(x.to(DEV))
            tot += h.norm(dim=1).sum().item(); seen += x.size(0)
        net.gs = tot / seen

    print(f'[{a.cell}] raw={raw} scale={scale} eps_train={eps:.5f} '
          f'mean_head_input_norm={net.gs:.4f} wd={a.wd}', flush=True)

    torch.manual_seed(0)
    if a.reset:
        net.linear.reset_parameters()
    for p in net.linear.parameters():
        p.requires_grad_(True)

    opt = torch.optim.SGD(net.linear.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    for ep in range(a.epochs):
        t0, tot, corr, run = time.time(), 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEV), y.to(DEV)
            xa = ce_pgd(net, x, y, eps, cfg.step_size, cfg.steps) if a.adv else x
            net.eval()                                # backbone frozen, BN stats frozen
            logits = net(xa)
            if a.kd > 0:
                with torch.no_grad():
                    tgt = teacher(x) / a.kd
                loss = F.kl_div(F.log_softmax(logits, 1), F.softmax(tgt, 1),
                                reduction='batchmean')
            else:
                loss = F.cross_entropy(logits, y, label_smoothing=a.ls)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * y.size(0); tot += y.size(0)
            corr += (logits.argmax(1) == y).sum().item()
        sched.step()
        print(f'  ep{ep:02d} advloss {run/tot:.4f} advacc {100*corr/tot:.2f} '
              f'|W| {net.linear.weight.norm().item():.2f} ({time.time()-t0:.0f}s)', flush=True)

    net.eval()
    clean, fgsm, pgd20, pgd10, pgd50, cw = evaluate(net, test_loader, cfg)
    print(f'PROBE_RESULT {a.cell}: clean {clean:.2f} pgd20 {pgd20:.2f} cw {cw:.2f}', flush=True)
    if a.aa:
        cfg.aa_batch_size = 256
        aa = evaluate_final_aa(net, test_loader, cfg)
        print(f'PROBE_AA {a.cell}: {aa:.2f}', flush=True)


if __name__ == '__main__':
    main()
