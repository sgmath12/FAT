"""Rank-correlate the teacher-only metrics with the students those teachers produced (2026-09-19).

Students are the 50-epoch anchor cells of the teacher ladder, one per checkpoint, identical apart from
the teacher.  Spearman is computed by hand (rank the two lists, Pearson on the ranks) to avoid a scipy
dependency.  With nine points from one trajectory every monotone metric correlates with every other, so
the ranking here separates candidates but cannot attribute causation; `clean_cos200ep` is the one
off-trajectory teacher available and is reported separately for that reason.
"""
import json, sys

STUDENTS = {   # teacher checkpoint -> (student clean, student AA_2), 50-epoch anchor, CIFAR-100, SEED 0
    # Seed 0 throughout, which is what tab:teacherladder_values reports.  Three of these cells also
    # have seeds 1 and 2, and they bound the student noise: 10 epochs 61.15/61.56/61.04 clean and
    # 20.22/20.10/19.94 AA, 50 epochs 64.28/64.07/64.07 and 24.38/24.42/24.51, 300 epochs
    # 62.24/62.09/61.85 and 25.80/26.13/25.59.  So a spread of 0.2-0.5 clean and 0.1-0.5 AA.
    'clean_5ep': (54.54, 16.00), 'clean_10ep': (61.15, 20.22), 'clean_20ep': (63.40, 22.98),
    'clean_40ep': (63.79, 24.23), 'clean': (64.28, 24.38), 'clean_100ep': (63.65, 25.20),
    'clean_150ep': (62.93, 25.78), 'clean_200ep': (62.72, 25.88), 'clean_300ep': (62.24, 25.80),
}

def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r

def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db)

def spearman(a, b):
    return pearson(ranks(a), ranks(b))

EPOCHS = {'clean_5ep': 5, 'clean_10ep': 10, 'clean_20ep': 20, 'clean_40ep': 40, 'clean': 50,
          'clean_100ep': 100, 'clean_150ep': 150, 'clean_200ep': 200, 'clean_300ep': 300}


def fit_predict(xs, ys, held):
    """Least-squares line on the eight remaining points, evaluated at the held-out one."""
    idx = [i for i in range(len(xs)) if i != held]
    x = [xs[i] for i in idx]; y = [ys[i] for i in idx]
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    den = sum((v - mx) ** 2 for v in x)
    b = sum((v - mx) * (w - my) for v, w in zip(x, y)) / den if den else 0.0
    a = my - b * mx
    return a + b * xs[held]


def loo_mae(xs, ys):
    return sum(abs(fit_predict(xs, ys, i) - ys[i]) for i in range(len(xs))) / len(xs)


def residuals(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    den = sum((v - mx) ** 2 for v in xs)
    b = sum((v - mx) * (w - my) for v, w in zip(xs, ys)) / den if den else 0.0
    a = my - b * mx
    return [y - (a + b * x) for x, y in zip(xs, ys)]


if __name__ == '__main__':
    metrics = json.load(open(sys.argv[1] if len(sys.argv) > 1
                            else 'writting_docs/paper/notes/teacher_metrics.json'))
    names = [k for k in STUDENTS if k in metrics]
    keys = list(next(iter(metrics.values())).keys())
    col = {k: [metrics[n][k] for n in names] for k in keys}
    col['epoch'] = [EPOCHS[n] for n in names]
    sc = [STUDENTS[k][0] for k in names]
    sa = [STUDENTS[k][1] for k in names]
    snrr = [2 * c * a / (c + a) for c, a in zip(sc, sa)]

    print('n = %d checkpoints: %s\n' % (len(names), ', '.join(names)))
    print('=== Spearman with the student that teacher produced')
    print('%-14s %8s %8s %8s' % ('teacher metric', 'clean', 'AA', 'NRR'))
    for k in keys + ['epoch']:
        print('%-14s %8.3f %8.3f %8.3f' % (k, spearman(col[k], sc), spearman(col[k], sa),
                                           spearman(col[k], snrr)))

    print('\n=== Spearman between the teacher metrics themselves (collinearity along one trajectory)')
    order = ['epoch', 'clean', 'margin', 'sw_sb', 'grad', 'featsens']
    print('%-10s' % '' + ''.join('%10s' % k[:9] for k in order))
    for a_ in order:
        print('%-10s' % a_[:9] + ''.join('%10.3f' % spearman(col[a_], col[b_]) for b_ in order))

    print('\n=== Leave-one-checkpoint-out prediction of the student, mean absolute error (points)')
    print('%-14s %10s %10s' % ('predictor', 'clean MAE', 'AA MAE'))
    for k in ['featsens', 'grad', 'margin', 'sw_sb', 'clean', 'epoch']:
        print('%-14s %10.2f %10.2f' % (k, loo_mae(col[k], sc), loo_mae(col[k], sa)))
    print('%-14s %10.2f %10.2f' % ('mean only', sum(abs(v - sum(sc) / len(sc)) for v in sc) / len(sc),
                                   sum(abs(v - sum(sa) / len(sa)) for v in sa) / len(sa)))

    print('\n=== After regressing the student on teacher clean accuracy, what is left')
    rc = residuals(col['clean'], sc)
    ra = residuals(col['clean'], sa)
    print('%-14s %12s %12s' % ('teacher metric', 'clean resid.', 'AA resid.'))
    for k in ['featsens', 'grad', 'margin', 'sw_sb', 'epoch']:
        print('%-14s %12.3f %12.3f' % (k, spearman(col[k], rc), spearman(col[k], ra)))
