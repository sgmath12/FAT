"""Build a teacher instead of selecting one (2026-09-20).

The checkpoint ladder trades the two student axes against each other: the 50-epoch teacher gives the
best student clean accuracy, the 200-epoch teacher the best AutoAttack accuracy, and the measured reason
is that the two teacher quantities that track those axes peak at different times --- feature sensitivity
early, margin late.  A teacher that held both near their maxima would replace the selection question
with a construction.

The cheap candidate is weight averaging along the trajectory: the checkpoints come from one run, so they
are likely in one basin, and averaging costs nothing.  This writes the averages; `teacher_metrics.py`
then says whether any of them lands above the window on both quantities, and only then is a student
worth training.

Not to be retried: shrinking features toward their class prototype and rotating classes onto a simplex
ETF, both measured inert or harmful in 2026-08 (memory featdir-target-geometry-tricks-failed).
"""
import argparse, os, torch


def average(paths, weights=None):
    sds = [torch.load(p, map_location='cpu') for p in paths]
    w = weights or [1.0 / len(sds)] * len(sds)
    assert abs(sum(w) - 1) < 1e-6, w
    out = {}
    for k in sds[0]:
        if sds[0][k].is_floating_point():
            out[k] = sum(wi * sd[k].float() for wi, sd in zip(w, sds)).to(sds[0][k].dtype)
        else:
            out[k] = sds[0][k]          # integer buffers, e.g. num_batches_tracked
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='CIFAR100')
    ap.add_argument('--parts', nargs='+', required=True, help='checkpoint directory names')
    ap.add_argument('--weights', nargs='*', type=float)
    ap.add_argument('--name', required=True, help='directory name to write under checkpoint/')
    a = ap.parse_args()
    paths = [f'{a.dataset}/checkpoint/{p}/clean_last.pkl' for p in a.parts]
    sd = average(paths, a.weights)
    d = f'{a.dataset}/checkpoint/{a.name}'
    os.makedirs(d, exist_ok=True)
    torch.save(sd, f'{d}/clean_last.pkl')
    print('wrote %s/clean_last.pkl from %s with weights %s' % (d, a.parts, a.weights or 'uniform'))
