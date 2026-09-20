"""Teacher-only metrics for the checkpoint-selection question (2026-09-19).

The teacher-ladder figure shows that a more accurate natural teacher can give a less accurate student.
It does not say which teacher to pick without training students.  This script measures, from the frozen
teacher alone, the quantities a selection rule could use, so they can be ranked against the student
results we already have for the same checkpoints.

Metrics are computed on 5000 held-out images (`--split test` by default).  The shipped teachers were
fit on the whole training set, so their train-split accuracy and margins saturate and carry no signal;
the price of using test images is that a rule read off them is not independent of the split the
students are scored on, which the new-seed teachers of step 2 are meant to fix.

  clean          top-1 accuracy
  sw_sb          within-class over between-class scatter on unit-normalized features
  margin         mean logit margin f_y - max_{k != y} f_k
  grad           mean ||d CE / dx||_2 at clean inputs
  featsens       mean ||Phi(x+delta) - Phi(x)||_2 / ||delta||_2 for uniform delta at eps 8/255
  margin_grad    margin / grad
  margin_feat    margin / featsens
"""
import argparse, json, os, sys
import torch, torch.nn.functional as F
import numpy as np
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
    # Checked in both directions since 2026-09-19: `clean_cos200ep` carries `log_s` instead of `alphas`
    # (it is the normalized-feature variant, not this network), and with only `missing` asserted it
    # loaded silently with that parameter dropped, which produced a margin of 0.22 and a feature
    # sensitivity of 0.72 for a teacher at 73% clean accuracy.  Those numbers were an artifact.
    assert not unexpected, 'checkpoint has parameters this network does not: %s' % unexpected
    assert not [k for k in missing if 'alphas' not in k], missing
    return Converter(net, MEAN, STD).cuda().eval()


def eval_split(root, split, n=5000, seed=0):
    """`train` is what these teachers were fit on, so their accuracy and margins saturate there (100%
    clean, see the 2026-09-19 run); the informative split is held-out data.  Since the shipped teachers
    were trained on the full training set, that means the test set, and a rule read off it is not
    independent of the set the students are scored on -- the new-seed teachers of step 2 are the ones
    that can be trained with 5000 images held out."""
    train = split == 'train'
    ds = torchvision.datasets.CIFAR100(root, train=train, transform=T.ToTensor())
    idx = np.random.RandomState(seed).permutation(len(ds))[:n]
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=250)


def feats(model, x):
    return model(x, feat=True)[0] if not hasattr(model, 'extract_feature') else model.extract_feature(x)[0]


def measure(model, loader, eps, n_delta=4):
    correct = n = 0
    margins, grads, sens, sens_rel, fnorm = [], [], [], [], []
    Fs, Ys = [], []
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        xg = x.clone().requires_grad_(True)
        logits = model(xg)
        loss = F.cross_entropy(logits, y)
        g = torch.autograd.grad(loss * len(y), [xg])[0].detach()   # per-sample sum -> per-sample grads
        grads.append(g.flatten(1).norm(dim=1).cpu())
        with torch.no_grad():
            z = logits.detach()
            correct += (z.argmax(1) == y).sum().item(); n += len(y)
            zy = z.gather(1, y[:, None]).squeeze(1)
            zo = z.scatter(1, y[:, None], float('-inf')).max(1).values
            margins.append((zy - zo).cpu())
            f0 = feats(model, x)
            fnorm.append(f0.norm(dim=1).cpu())
            # Averaged over `n_delta` draws, and also reported relative to the feature norm: mixup and
            # label smoothing change the scale of the representation, so an absolute displacement could
            # order the teachers for that reason alone (2026-09-20).
            s_abs = s_rel = 0.0
            for _ in range(n_delta):
                delta = (torch.rand_like(x) * 2 - 1) * eps
                f1 = feats(model, (x + delta).clamp(0, 1))
                d = (f1 - f0).norm(dim=1) / delta.flatten(1).norm(dim=1)
                s_abs = s_abs + d
                s_rel = s_rel + d / f0.norm(dim=1)
            sens.append((s_abs / n_delta).cpu())
            sens_rel.append((s_rel / n_delta).cpu())
            Fs.append(F.normalize(f0, dim=1).cpu()); Ys.append(y.cpu())
    Fs, Ys = torch.cat(Fs), torch.cat(Ys)
    mu = Fs.mean(0)
    sw = sb = 0.0
    for c in Ys.unique():
        fc = Fs[Ys == c]
        mc = fc.mean(0)
        sw += ((fc - mc) ** 2).sum(1).sum().item()
        sb += len(fc) * ((mc - mu) ** 2).sum().item()
    margin = torch.cat(margins).mean().item()
    grad = torch.cat(grads).mean().item()
    featsens = torch.cat(sens).mean().item()
    featsens_rel = torch.cat(sens_rel).mean().item()
    featnorm = torch.cat(fnorm).mean().item()
    return dict(clean=100.0 * correct / n, sw_sb=sw / sb, margin=margin, grad=grad,
                featsens=featsens, featsens_rel=featsens_rel, featnorm=featnorm,
                margin_grad=margin / grad, margin_feat=margin / featsens)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CIFAR100')
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--data-root', default='./data/CIFAR100')
    ap.add_argument('--eps', type=float, default=8 / 255)
    ap.add_argument('--split', default='test', choices=['test', 'train'])
    ap.add_argument('--out', default='writting_docs/paper/notes/teacher_metrics.json')
    a = ap.parse_args()
    torch.manual_seed(0)
    loader = eval_split(a.data_root, a.split)
    rows = {}
    for name in a.checkpoints:
        ckpt = f'{a.dataset}/checkpoint/{name}/clean_last.pkl'
        if not os.path.exists(ckpt):   # main.py names the file after the method, e.g. clean_mixup
            import glob
            cands = sorted(glob.glob(f'{a.dataset}/checkpoint/{name}/*_last.pkl'))
            assert len(cands) == 1, cands
            ckpt = cands[0]
        m = measure(build(ckpt, 100 if a.dataset == 'CIFAR100' else 10), loader, a.eps)
        rows[name] = m
        print(name, json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)
    json.dump(rows, open(a.out, 'w'), indent=2)
    print('wrote', a.out)
