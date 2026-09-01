"""Figure 1(b): the accuracy-robustness frontier, with CFA on it (2026-09-02).

Every reference paper opens with this plot -- ARREST Fig. 1, IGDM Fig. 1b, the long-tailed
self-distillation paper Fig. 1a -- and ours had no figures at all.  Baselines are published numbers
(Appendix D of the paper), ours are measured here; the split is stated in the caption rather than
encoded in the markers, since the point of the panel is the frontier, not a controlled comparison.

The dashed curve is a least-squares line through the Pareto-optimal baselines only, i.e. those no
other baseline beats on both axes.  It is drawn for reading, not fitted for a claim, exactly as
ARREST describes its own.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 9})

C100 = [("PGD-AT", 56.56, 25.02), ("TRADES", 55.39, 24.51), ("MART", 49.83, 25.00),
        ("Consistency-AT", 58.53, 25.39), ("F$^2$AT", 54.19, 23.24),
        ("Generalist++", 62.97, 23.96), ("ADR", 56.10, 26.87), ("ADR+WA+AWP", 57.36, 28.50),
        ("Cons.-AT+RPAT", 60.33, 26.31), ("RPAT++", 55.93, 27.36)]
# OUR CURVE, not a point.  A single marker invites "Generalist++ has higher clean accuracy", and the
# answer is that the method has a trade dial: the training radius moves along the frontier at no cost.
# Shipped configuration is starred; the others differ from it only in eps_train.
OUR100 = [("$\\epsilon_{tr}\\!=\\!8$", 63.89, 27.78),
          ("CFA (ours)",                  62.65, 28.77),
          ("$10$",                        59.99, 28.77)]
STAR100 = 1

C10 = [("PGD-AT", 82.78, 44.63), ("TRADES", 82.41, 48.37), ("MART", 80.70, 47.49),
       ("ST-AT", 83.10, 50.50), ("IAD", 80.63, 50.17), ("Cons.-AT+RPAT", 84.12, 48.98),
       ("CURE", 86.76, 49.69), ("ADR", 82.41, 50.38), ("ARREST", 86.63, 46.14),
       ("Generalist++", 89.09, 46.07), ("RPAT++", 82.41, 50.75)]
OUR10 = [("$\\epsilon_{tr}\\!=\\!8$", 87.22, 51.15),
         ("CFA (ours)",                  85.58, 51.79),
         ("$10$",                        83.29, 51.79)]
STAR10 = 1


def pareto(pts):
    """Baselines no other baseline beats on both axes."""
    out = []
    for n, x, y in pts:
        if not any(x2 >= x and y2 >= y and (x2, y2) != (x, y) for _, x2, y2 in pts):
            out.append((n, x, y))
    return sorted(out, key=lambda p: p[1])


def panel(ax, pts, ours, star, title, nudge, ournudge):
    for n, x, y in pts:
        ax.scatter(x, y, s=26, c='0.55', edgecolors='0.3', linewidths=0.5, zorder=3)
        dx, dy = nudge.get(n, (0.0, 0.28))
        ax.annotate(n, (x + dx, y + dy), fontsize=6.6, color='0.25', ha='center', zorder=4)
    pf = pareto(pts)
    if len(pf) >= 2:                                   # readability curve, not a fit for a claim
        xs = np.array([p[1] for p in pf]); ys = np.array([p[2] for p in pf])
        k = np.polyfit(xs, ys, 1)
        gx = np.linspace(min(xs) - 1.2, max(xs) + 1.2, 50)
        ax.plot(gx, np.polyval(k, gx), '--', c='0.7', lw=1.0, zorder=1)
    ox = [p[1] for p in ours]; oy = [p[2] for p in ours]
    ax.plot(ox, oy, '-', c='#c81e1e', lw=1.4, zorder=4, alpha=0.85)
    for i, (n, x, y) in enumerate(ours):
        if i == star:
            ax.scatter(x, y, s=165, marker='*', c='#c81e1e', edgecolors='k', linewidths=0.6, zorder=6)
        else:
            ax.scatter(x, y, s=30, marker='o', c='#c81e1e', edgecolors='k', linewidths=0.4, zorder=6)
        dx, dy = ournudge.get(n, (0.0, 0.42))
        ax.annotate(n, (x + dx, y + dy), fontsize=7.4, color='#c81e1e', ha='center',
                    fontweight='bold' if i == star else 'normal', zorder=6)
    ax.set_xlabel('Standard accuracy (\\%)' if matplotlib.rcParams['text.usetex'] else 'Standard accuracy (%)')
    ax.set_ylabel('AutoAttack accuracy (%)')
    ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85))
panel(axes[0], C10, OUR10, STAR10, 'CIFAR-10',
      {"Generalist++": (0, -0.75), "ARREST": (0, -0.75), "RPAT++": (-1.1, 0.15),
       "ADR": (1.0, -0.1), "ST-AT": (0, 0.3), "IAD": (0, -0.75)},
      {"$\\epsilon_{tr}\\!=\\!8$": (1.0, 0.15), "CFA (ours)": (0, 0.45), "$10$": (-0.9, -0.1)})
panel(axes[1], C100, OUR100, STAR100, 'CIFAR-100',
      {"ADR+WA+AWP": (0, 0.3), "RPAT++": (-1.4, -0.15), "ADR": (1.0, -0.15),
       "Generalist++": (0, -0.55), "Cons.-AT+RPAT": (0, 0.28)},
      {"$\\epsilon_{tr}\\!=\\!8$": (1.5, -0.1), "CFA (ours)": (1.6, 0.05), "$10$": (-1.2, -0.1)})
axes[0].set_xlim(78.5, 91.5); axes[0].set_ylim(43.2, 53.4)
axes[1].set_xlim(48.0, 65.5); axes[1].set_ylim(22.4, 30.2)
fig.tight_layout(pad=0.6)
fig.savefig('/mnt/d/research/FAT/writting_docs/paper/figure/frontier.pdf', bbox_inches='tight')
print("wrote figure/frontier.pdf")
