"""Regenerate the clean--AutoAttack overview figure used in the introduction."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 7.4,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

GREY = "#787878"
RED = "#c62828"

C10_BASELINES = {
    "PGD-AT": (82.78, 44.63),
    "TRADES": (82.41, 48.37),
    "MART": (80.70, 47.49),
    "ST-AT": (83.10, 50.50),
    "IAD": (80.63, 50.17),
    "Cons.-AT+RPAT": (84.12, 48.98),
    "CURE": (86.76, 49.69),
    "ADR": (82.41, 50.38),
    "ARREST": (86.63, 46.14),
    "Generalist++": (89.09, 46.07),
    "RPAT++": (82.63, 51.00),
}

C100_BASELINES = {
    "PGD-AT": (56.56, 25.02),
    "TRADES": (55.39, 24.51),
    "MART": (49.83, 25.00),
    "Consistency-AT": (58.53, 25.39),
    "F2AT": (54.19, 23.24),
    "Generalist++": (62.97, 23.96),
    "ADR": (56.10, 26.87),
    "ADR+WA+AWP": (57.36, 28.50),
    "Cons.-AT+RPAT": (60.33, 26.31),
    "RPAT++": (56.84, 27.68),
}

C10_CURVE = [
    (89.64, 47.88, "6"),
    (88.30, 50.27, "7"),
    (87.22, 51.15, "8"),
    (84.96, 51.74, "8.8"),
    (83.29, 51.79, "10"),
]
C100_CURVE = [
    (68.07, 25.82, "6"),
    (65.61, 27.44, "7"),
    (63.89, 27.78, "8"),
    (62.17, 28.86, "8.8"),
    (59.99, 28.77, "10"),
]

C10_OFFSETS = {
    "PGD-AT": (-7, -7),
    "TRADES": (-12, -7),
    "MART": (-8, -7),
    "ST-AT": (3, 4),
    "IAD": (-8, 4),
    "Cons.-AT+RPAT": (3, -8),
    "CURE": (3, 4),
    "ADR": (-9, 4),
    "ARREST": (-12, -8),
    "Generalist++": (-17, -8),
    "RPAT++": (-10, 4),
}
C100_OFFSETS = {
    "PGD-AT": (-9, -7),
    "TRADES": (-10, 4),
    "MART": (-8, 4),
    "Consistency-AT": (3, 4),
    "F2AT": (-6, -8),
    "Generalist++": (-16, -8),
    "ADR": (3, 4),
    "ADR+WA+AWP": (3, 4),
    "Cons.-AT+RPAT": (3, -8),
    "RPAT++": (-8, 4),
}


def panel(ax, title, baselines, offsets, curve, final, xlim, ylim, xticks, yticks):
    for name, (x, y) in baselines.items():
        ax.scatter(
            x,
            y,
            s=9,
            facecolors="none",
            edgecolors=GREY,
            linewidths=0.65,
            zorder=2,
        )
        dx, dy = offsets[name]
        ax.annotate(
            name,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            color="#555555",
            fontsize=5.5,
        )

    xs = [row[0] for row in curve]
    ys = [row[1] for row in curve]
    ax.plot(
        xs,
        ys,
        color=RED,
        linewidth=0.85,
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        markeredgewidth=0.75,
        zorder=3,
    )
    for x, y, radius in curve:
        ax.annotate(
            radius,
            (x, y),
            xytext=(3, 2),
            textcoords="offset points",
            color=RED,
            fontsize=5.5,
        )

    ax.scatter(
        *final,
        marker="*",
        s=55,
        color=RED,
        edgecolor="#8e0000",
        linewidth=0.45,
        zorder=5,
    )
    ax.annotate(
        "CFA (reported)",
        final,
        xytext=(4, 5),
        textcoords="offset points",
        color=RED,
        fontsize=6.0,
        fontweight="bold",
    )
    ax.set_title(title, pad=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel("Standard accuracy (%)")
    ax.set_ylabel("AutoAttack accuracy (%)")
    ax.grid(color="#dddddd", linewidth=0.45, alpha=0.75)
    ax.tick_params(width=0.55, length=2.2)


fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.32), constrained_layout=True)
panel(
    axes[0],
    "CIFAR-10",
    C10_BASELINES,
    C10_OFFSETS,
    C10_CURVE,
    (87.37, 51.33),
    (79.7, 90.3),
    (43.5, 52.6),
    [80, 82, 84, 86, 88, 90],
    [44, 46, 48, 50, 52],
)
panel(
    axes[1],
    "CIFAR-100",
    C100_BASELINES,
    C100_OFFSETS,
    C100_CURVE,
    (64.82, 28.04),
    (48.5, 69.2),
    (22.7, 30.1),
    [50, 55, 60, 65],
    [23, 24, 25, 26, 27, 28, 29, 30],
)

out = Path(__file__).with_name("frontier.pdf")
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
