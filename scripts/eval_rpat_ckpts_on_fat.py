"""Re-evaluate our RPAT++ (ReBAT+RPAT) runs with THIS framework's evaluation (2026-08-31).

Why.  `../RPAT/run.sh` already trained RPAT++ on OUR ResNet-18 and OUR CIFAR data for 200 epochs on
both datasets, and its numbers land within 0.3 of the paper's Table 3 -- unlike CURE, RPAT
reproduces.  But those numbers came from `train_cifar_ra.py`'s own evaluation loop; run.sh only
claims to match "the same column set FAT's main.py prints", not the same code.  This closes that,
exactly as scripts/eval_cure_ckpts_on_fat.py did for CURE (where vendored and native agreed to 0.05).

Two details that must be right or the numbers are meaningless:
  * ARCHITECTURE.  `RPAT_SOTAs/networks/resnet.py` is FAT's own ResNet18, double-ReLU BasicBlock and
    all, so the state dicts load strict=True including the `alphas` buffer.
  * NORMALIZATION.  RPAT normalizes OUTSIDE the network (`model(normalize(X))`, train_cifar_ra.py:294
    etc.) with cifar10_mean/std = (0.4914, 0.4822, 0.4465) / (0.2471, 0.2435, 0.2616), applied to
    CIFAR-100 as well.  FAT's get_model instead wraps with the CIFAR-100 statistics for BOTH datasets.
    These checkpoints were trained under RPAT's constants, so we wrap with RPAT's -- using ours would
    feed the network a distribution it never saw.
"""
import sys, argparse, types, torch
sys.path.insert(0, '/mnt/d/research/FAT')
import dataset
from CIFAR10.models.resnet import ResNet18
from utils import Converter, evaluate, evaluate_final_aa

RPAT_MEAN = (0.4914, 0.4822, 0.4465)
RPAT_STD = (0.2471, 0.2435, 0.2616)
E = '/mnt/d/research/RPAT/RPAT_SOTAs/exps'
CELLS = [
    ("CIFAR100", "last wa", f"{E}/rpatpp_c100_resnet18/wa_model_199.pth", 100),
    ("CIFAR100", "best wa", f"{E}/rpatpp_c100_resnet18/wa_model_best.pth", 100),
    ("CIFAR10",  "last wa", f"{E}/rpatpp_c10_resnet18/wa_model_199.pth", 10),
    ("CIFAR10",  "best wa", f"{E}/rpatpp_c10_resnet18/wa_model_best.pth", 10),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aa', type=int, default=1)
    ap.add_argument('--aa_batch_size', type=int, default=256)
    a = ap.parse_args()
    loaders = {}
    print(f"{'dataset':<10}{'cell':<10}{'clean':>8}{'FGSM':>8}{'PGD20':>8}{'PGD10':>8}{'PGD50':>8}{'CW':>8}{'AA':>8}{'NRR':>8}",
          flush=True)
    for ds, name, path, ncls in CELLS:
        cfg = types.SimpleNamespace(dataset=ds, eps=8/255, steps=10, step_size=2/255,
                                    batch_size=128, aa_batch_size=a.aa_batch_size,
                                    convert=True, arch='ResNet18')
        if ds not in loaders:
            loaders[ds] = getattr(dataset, ds)(root=f'./data/{ds}', download=False,
                                               batch_size=cfg.batch_size, val=False, config=cfg)[2]
        sd = torch.load(path, map_location='cpu')
        net = Converter(ResNet18(num_classes=ncls), RPAT_MEAN, RPAT_STD)
        net.encoder.load_state_dict(sd, strict=True)   # Converter wraps the net as `.encoder`
        net = net.cuda().eval()
        clean, fgsm, pgd20, pgd10, pgd50, cw = evaluate(net, loaders[ds], cfg)
        aa = evaluate_final_aa(net, loaders[ds], cfg) if a.aa else float('nan')
        nrr = 2 * clean * aa / (clean + aa) if aa == aa else float('nan')
        print(f"{ds:<10}{name:<10}{clean:8.2f}{fgsm:8.2f}{pgd20:8.2f}{pgd10:8.2f}{pgd50:8.2f}{cw:8.2f}{aa:8.2f}{nrr:8.2f}",
              flush=True)
        del net; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
