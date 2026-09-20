"""The candidate-window rule, fixed before the validation trajectories are trained (2026-09-20).

The teacher-selection study does not try to name the single best checkpoint. It narrows a trajectory to
a window worth training students on, from teacher metrics alone:

    keep a checkpoint if   margin >= 0.99 * max(margin over the trajectory)
                     and   feature sensitivity >= 0.85 * max(feature sensitivity over the trajectory)

The margin condition asks for a teacher whose class structure has essentially saturated; the sensitivity
condition drops the late checkpoints whose representation has gone flat. Neither threshold is claimed to
be optimal: both were read off the CIFAR-100 trajectory of tab:teacherladder_values after its students
were known, which is exactly why they are frozen here and applied unchanged to new trajectories.
"""
import argparse, json

MARGIN_FRAC = 0.99
SENS_FRAC = 0.85


def window(metrics, names):
    m = max(metrics[n]['margin'] for n in names)
    s = max(metrics[n]['featsens'] for n in names)
    return [n for n in names
            if metrics[n]['margin'] >= MARGIN_FRAC * m and metrics[n]['featsens'] >= SENS_FRAC * s]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics', default='writting_docs/paper/notes/teacher_metrics.json')
    ap.add_argument('--checkpoints', nargs='+', required=True,
                    help='one trajectory, in training order')
    a = ap.parse_args()
    met = json.load(open(a.metrics))
    names = [n for n in a.checkpoints if n in met]
    keep = window(met, names)
    mmax = max(met[n]['margin'] for n in names)
    smax = max(met[n]['featsens'] for n in names)
    print('%-18s %8s %8s %8s %8s  %s' % ('checkpoint', 'margin', '% max', 'sens.', '% max', 'kept'))
    for n in names:
        print('%-18s %8.3f %7.1f%% %8.3f %7.1f%%  %s'
              % (n, met[n]['margin'], 100 * met[n]['margin'] / mmax,
                 met[n]['featsens'], 100 * met[n]['featsens'] / smax, 'yes' if n in keep else ''))
    print('\nwindow: %s' % (', '.join(keep) if keep else 'empty'))
