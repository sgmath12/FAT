"""Reproduce IGDM's own local-linearity diagnostic (their Fig. 2a/2b) on OUR checkpoints.

IGDM rests on the first-order Taylor expansion of the TEACHER being accurate over an eps-ball:
    f(x+e) = f(x) + (df/dx)^T e + R,   remainder proportion = ||R|| / ||f(x+e)||.
Their paper measures ~0.012 for adversarially trained teachers (LTD, BDM-AT, IKL-AT), and their
Fig. 2b shows the proportion CLIMBING throughout natural training while staying small under
adversarial training.  The whole method -- matching f(x+d) - f(x-d) instead of the gradient -- is
only a gradient match when that remainder is negligible.

Our teacher is naturally trained, so this measures whether IGDM's premise holds in our setting.
"""
import sys, argparse, torch
sys.path.insert(0, '/mnt/d/research/FAT')
import dataset
from CIFAR10.models.resnet import ResNet18
from utils import Converter

MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


def remainder_proportion(model, loader, eps, n_batches, device='cuda'):
    tot, num = 0.0, 0
    for i, (x, _) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device)
        e = (torch.rand_like(x) * 2 - 1) * eps                      # uniform in the eps-ball
        x.requires_grad_(True)
        f = model(x)
        # (df/dx)^T e for every output coordinate: one JVP, exact, no per-class loop
        jvp = torch.autograd.functional.jvp(lambda z: model(z), (x.detach(),), (e,))[1]
        with torch.no_grad():
            fx = model(x.detach())
            fxe = model((x.detach() + e).clamp(0, 1))
            r = (fxe - fx - jvp).norm(dim=1) / fxe.norm(dim=1).clamp_min(1e-12)
        tot += r.sum().item(); num += x.size(0)
    return tot / num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_batches', type=int, default=8)
    ap.add_argument('--eps', type=float, default=8/255)
    a = ap.parse_args()
    import types
    cfg = types.SimpleNamespace(dataset='CIFAR100', batch_size=128)
    _, _, test_loader = dataset.CIFAR100(root='./data/CIFAR100', download=False,
                                         batch_size=cfg.batch_size, val=False, config=cfg)
    cells = [
        ("natural teacher (clean_200ep)", "CIFAR100/checkpoint/clean_200ep/clean_last.pkl"),
        ("PGD-AT ResNet-18 (ours)",       "CIFAR100/checkpoint/at_teacherinit_matched/madry_at_last.pkl"),
        ("anchor student (p=0)",          "CIFAR100/checkpoint/ladder_p0_100ep/feat_direction_last.pkl"),
        ("anchor student (shipped)",      "CIFAR100/checkpoint/l2_bestrecipe_freezehead/feat_direction_last.pkl"),
    ]
    print(f"{'checkpoint':<34}{'remainder proportion':>22}")
    for name, path in cells:
        try:
            sd = torch.load(path, map_location='cpu')
        except Exception as e:
            print(f"{name:<34}{'(missing)':>22}"); continue
        net = Converter(ResNet18(num_classes=100), MEAN, STD)
        r = net.load_state_dict(sd, strict=False)
        net = net.cuda().eval()
        prop = remainder_proportion(net, test_loader, a.eps, a.n_batches)
        print(f"{name:<34}{prop:>22.4f}")
        del net; torch.cuda.empty_cache()
    print("\nIGDM's reported values for their own robust teachers: 0.012 (LTD), 0.012 (BDM-AT), 0.016 (IKL-AT)")


if __name__ == '__main__':
    main()
