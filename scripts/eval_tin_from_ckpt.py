"""Re-run the final evaluation for the Tiny-ImageNet run from its checkpoint (2026-08-24).

The 100-epoch run finished training and then hit CUDA OOM inside utils.evaluate -- Tiny-ImageNet is
64x64 with 200 classes, so the same batch size that fits on CIFAR does not fit here once FGSM/PGD/CW
are stacked in one function.  Nothing was lost: main.py writes `_last.pkl` before the evaluation
block, and since the 2026-08-17 fix that file is the true final-epoch model, so the evaluation can
simply be redone.

Batch size is a parameter here (default 64 against the training run's 128) and AA gets its own,
smaller still.
"""
import sys, argparse
sys.path.insert(0, '/mnt/d/research/FAT'); sys.path.insert(0, '/mnt/d/research/FAT/scratchpad')
import torch
import dataset
from measure_cosadv import make_args
from utils import load_config, get_model, evaluate, evaluate_final_aa

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='featdir_tin_100ep')
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--aa_bs', type=int, default=64)
    a = ap.parse_args()

    args = make_args(a.cell + '.yaml'); args.dataset = 'TinyImageNet'
    cfg = load_config(args)
    _, student = get_model(cfg)
    ck = 'TinyImageNet/checkpoint/%s/feat_direction_last.pkl' % a.cell
    student.load_state_dict(torch.load(ck, map_location='cpu'))
    student = student.cuda().eval()

    _, _, test_loader = getattr(dataset, cfg.dataset)(
        root='./data/%s' % cfg.dataset, download=False, batch_size=a.bs, val=False, config=cfg)

    clean, fgsm, pgd20, pgd10, pgd50, cw = evaluate(student, test_loader, cfg)
    print('EVAL %s: clean %.2f fgsm %.2f pgd20 %.2f pgd10 %.2f pgd50 %.2f cw %.2f'
          % (a.cell, clean, fgsm, pgd20, pgd10, pgd50, cw), flush=True)
    cfg.aa_batch_size = a.aa_bs
    aa = evaluate_final_aa(student, test_loader, cfg)
    print('EVAL_AA %s: %.2f' % (a.cell, aa), flush=True)
    print('NRR %s: %.2f' % (a.cell, 2 * clean * aa / (clean + aa)), flush=True)
