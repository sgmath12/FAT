"""Re-evaluate the original-CURE checkpoints with THIS framework's evaluation (2026-08-31).

Why.  `../CURE/repro/` already contains seven original-CURE runs, none of which reach the paper's
86.76 / 49.69 (CIFAR-10, ResNet-18).  Their numbers were produced by `repro/eval_model.py`, which
says in its own docstring that it VENDORED the attacks rather than importing this tree, and reads
CIFAR-10 out of the CURE repo's own data directory.  So the existing CURE numbers and our own table
were never produced by the same code.  This script closes that: same `utils.evaluate`, same
`evaluate_final_aa`, same `dataset.CIFAR10` loader, same eps, as every other row we report.

Architecture note.  CURE was patched to FAT's non-standard double-ReLU BasicBlock, so the
checkpoints load into `CIFAR10.models.resnet.ResNet18` with only `alphas` missing -- a 512-vector of
ones that the standard `forward` never reads (it is used solely by `forward_with_score`).  The models
take raw [0,1] input, so no Converter is applied, matching how CURE generates its adversaries.
"""
import sys, argparse, types, torch
sys.path.insert(0, '/mnt/d/research/FAT')
sys.path.insert(0, '/mnt/d/research/CURE')
import dataset
from CIFAR10.models.resnet import ResNet18
from utils import evaluate, evaluate_final_aa

R = '/mnt/d/research/CURE/repro/experiments'
CKPTS = [
    ("stdinit",       f"{R}/cure_stdinit_wd7e4/cure_stdinit_wd7e4_seed0/checkpoints_adv/final_model.pth"),
    ("mixupinit",     f"{R}/cure_mixupinit_wd7e4/cure_mixupinit_wd7e4_seed0/checkpoints_adv/final_model.pth"),
    ("eff_noinit",    f"{R}/cure_eff_noinit_wd7e4/cure_eff_noinit_wd7e4_seed0/checkpoints/final_model.pth"),
    ("absmask",       f"{R}/cure_absmask_wd7e4/cure_absmask_wd7e4_seed0/checkpoints_adv/final_model.pth"),
    ("absmask_mixup", f"{R}/cure_absmask_mixupinit_wd7e4/cure_absmask_mixupinit_wd7e4_seed0/checkpoints_adv/final_model.pth"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aa', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--aa_batch_size', type=int, default=256)
    a = ap.parse_args()

    cfg = types.SimpleNamespace(dataset='CIFAR10', eps=8/255, steps=10, step_size=2/255,
                                batch_size=a.batch_size, aa_batch_size=a.aa_batch_size,
                                convert=False, arch='ResNet18')
    _, _, test_loader = dataset.CIFAR10(root='./data/CIFAR10', download=False,
                                        batch_size=cfg.batch_size, val=False, config=cfg)
    print(f"{'cell':<16}{'clean':>8}{'FGSM':>8}{'PGD20':>8}{'PGD10':>8}{'PGD50':>8}{'CW':>8}{'AA':>8}{'NRR':>8}",
          flush=True)
    for name, path in CKPTS:
        obj = torch.load(path, map_location='cpu', weights_only=False)
        sd = obj.state_dict() if isinstance(obj, torch.nn.Module) else obj
        m = ResNet18(num_classes=10)
        r = m.load_state_dict(sd, strict=False)
        assert r.unexpected_keys == [] and r.missing_keys == ['alphas'], (name, r)
        m = m.cuda().eval()
        clean, fgsm, pgd20, pgd10, pgd50, cw = evaluate(m, test_loader, cfg)
        aa = evaluate_final_aa(m, test_loader, cfg) if a.aa else float('nan')
        nrr = 2 * clean * aa / (clean + aa) if aa == aa else float('nan')
        print(f"{name:<16}{clean:8.2f}{fgsm:8.2f}{pgd20:8.2f}{pgd10:8.2f}{pgd50:8.2f}{cw:8.2f}{aa:8.2f}{nrr:8.2f}",
              flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
