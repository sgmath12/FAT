"""Verify ADR+WA+AWP's published CIFAR100/ResNet18 number (AA 28.52% / clean 57.36%) using
FAT's own eval code (utils.evaluate / utils.evaluate_final_aa), loading ADR's own model class
(NOT FAT's double-ReLU resnet_z) so the architecture exactly matches the checkpoint's training.
Usage: python scripts/eval_adr_checkpoint.py <path-to-ckpt.pt> [--ema]
"""
import sys
import argparse
import torch

sys.path.insert(0, "/mnt/d/research/ADR/src")
sys.path.insert(0, "/mnt/d/research/FAT")

from model.resnet import ResNet  # ADR's own architecture (standard single-ReLU BasicBlock)
from utils import evaluate, evaluate_final_aa
import dataset

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path")
    parser.add_argument("--ema", action="store_true", help="load model_ema (WA) weights, matches the README's 'ADR+WA+AWP' row")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--full_aa", action="store_true", help="run all 4 AutoAttack attacks instead of apgd-ce+apgd-t only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(CIFAR100_MEAN, CIFAR100_STD, num_classes=100, depth=18,
                   activation_fn=torch.nn.ReLU, adaptive_pooling=False).to(device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    key = "model_ema" if args.ema else "model"
    model.load_state_dict(ckpt[key])
    model.eval()
    print(f"Loaded {key} from {args.ckpt_path} (epoch={ckpt.get('epoch')})")

    class Config:
        pass
    config = Config()
    config.eps = 8.0 / 255.0
    config.batch_size = args.batch_size

    _, _, test_loader = dataset.CIFAR100(root="/mnt/d/research/FAT/data", download=False, batch_size=args.batch_size)

    clean_acc, fgsm_acc, pgd_acc, pgd10_acc, pgd50_acc, cw_acc = evaluate(model, test_loader, config)
    print(f"clean={clean_acc:.2f} fgsm={fgsm_acc:.2f} pgd20={pgd_acc:.2f} pgd10={pgd10_acc:.2f} pgd50={pgd50_acc:.2f} cw={cw_acc:.2f}")

    aa_acc = evaluate_final_aa(model, test_loader, config, full=args.full_aa)
    print(f"AA={aa_acc:.2f}")
    print(f"Published (README, ADR+WA+AWP, CIFAR100 ResNet18): AA=28.52 clean=57.36")


if __name__ == "__main__":
    main()
