"""Rank-correlate the teacher-only metrics with the students those teachers produced (2026-09-19).

Students are the 50-epoch anchor cells of the teacher ladder, one per checkpoint, identical apart from
the teacher.  Spearman is computed by hand (rank the two lists, Pearson on the ranks) to avoid a scipy
dependency.  With nine points from one trajectory every monotone metric correlates with every other, so
the ranking here separates candidates but cannot attribute causation; `clean_cos200ep` is the one
off-trajectory teacher available and is reported separately for that reason.
"""
import json, sys

STUDENTS = {   # teacher checkpoint -> (student clean, student AA), 50-epoch anchor, CIFAR-100
    'clean_5ep': (54.54, 16.00), 'clean_10ep': (61.04, 19.94), 'clean_20ep': (63.40, 22.98),
    'clean_40ep': (63.79, 24.23), 'clean': (64.07, 24.51), 'clean_100ep': (63.65, 25.20),
    'clean_150ep': (62.93, 25.78), 'clean_200ep': (62.72, 25.88), 'clean_300ep': (61.85, 25.59),
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

if __name__ == '__main__':
    metrics = json.load(open(sys.argv[1] if len(sys.argv) > 1
                            else 'writting_docs/paper/notes/teacher_metrics.json'))
    names = [k for k in STUDENTS if k in metrics]
    keys = list(next(iter(metrics.values())).keys())
    sc = [STUDENTS[k][0] for k in names]
    sa = [STUDENTS[k][1] for k in names]
    snrr = [2 * c * a / (c + a) for c, a in zip(sc, sa)]
    print('n = %d checkpoints: %s' % (len(names), ', '.join(names)))
    print('%-14s %8s %8s %8s' % ('teacher metric', 'clean', 'AA', 'NRR'))
    for key in keys:
        v = [metrics[k][key] for k in names]
        print('%-14s %8.3f %8.3f %8.3f' % (key, spearman(v, sc), spearman(v, sa), spearman(v, snrr)))
    print()
    print('student clean vs AA: %.3f' % spearman(sc, sa))
    for key in keys:
        v = [metrics[k][key] for k in names]
        print('%-14s vs teacher clean: %6.3f' % (key, spearman(v, [metrics[k]['clean'] for k in names])))
