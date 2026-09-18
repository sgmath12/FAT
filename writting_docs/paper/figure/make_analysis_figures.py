"""Vector analysis figures. Run with Python, Matplotlib and NumPy installed.

Numerical data match the appendix tables. Fixed physical dimensions preserve
the manuscript layout; PDF fonts are embedded.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN, BLACK = "#0072B2", "#D55E00", "#00866B", "#333333"
# WSL can use the host's Times faces without modifying its font installation.
for font_path in Path("/mnt/c/Windows/Fonts").glob("times*.ttf"):
    font_manager.fontManager.addfont(str(font_path))
FONT = next((name for name in ("Times New Roman", "Nimbus Roman", "Liberation Serif")
             if any(f.name == name for f in font_manager.fontManager.ttflist)), "DejaVu Serif")
plt.rcParams.update({
    "font.family": "serif", "font.serif": [FONT], "font.size": 8,
    "mathtext.fontset": "stix", "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2, "ytick.major.pad": 2,
    "legend.fontsize": 7.2, "legend.frameon": False,
    "lines.linewidth": 1, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

TEACHER_ONLY = np.array([
    [1, 58.26, 20.84], [2, 59.33, 22.79], [4, 59.39, 24.48],
    [8, 59.47, 25.19], [16, 57.78, 24.00],
])
SHARED_KD = np.array([
    [4, 60.83, 25.05], [8, 60.99, 25.25],
    [16, 61.85, 25.39], [32, 61.81, 25.50],
])


def nondominated(rows):
    """Observed Pareto set, maximizing both clean and robust accuracy."""
    scores = rows[:, 1:]
    mask = [not np.any(np.all(scores >= p, axis=1) & np.any(scores > p, axis=1))
            for p in scores]
    result = rows[mask]
    return result[np.argsort(result[:, 1])]


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linewidth=0.45)


def save(fig, name):
    fig.savefig(HERE / name, metadata={"Creator": "Matplotlib", "CreationDate": None})
    plt.close(fig)


def make_target_figure():
    fig = plt.figure(figsize=(5.5, 154 / 72))
    ax = fig.add_axes([0.105, 0.285, 0.255, 0.545])
    bx = fig.add_axes([0.515, 0.285, 0.46, 0.545])
    for panel in (ax, bx):
        style_axis(panel)
    ax.set_title("(a) Target entropy", loc="left", pad=8)
    bx.set_title("(b) Best observed trade-offs", loc="left", pad=8)
    taus = [1, 2, 4, 8, 16, 32]
    entropy = [0.76, 2.40, 4.20, 4.56, 4.60, 4.60]
    ax.set_xscale("log", base=2)
    ax.plot(taus, entropy, "o-", color=BLUE, markersize=3.5,
            markerfacecolor="white", markeredgewidth=0.8)
    ax.axhline(np.log(100), color="0.45", linestyle="--", linewidth=0.7)
    ax.text(1.05, 4.7, r"$\log C$", fontsize=7, va="bottom", color="0.4")
    ax.set(xlim=(0.85, 38), ylim=(0, 5.35), xlabel=r"Temperature $\tau$",
           ylabel="Entropy (nats)")
    ax.set_xticks(taus, [str(t) for t in taus])
    ax.set_yticks([0, 2, 4])
    # Filter within each KD variant. Pooled filtering would hide every method
    # except the feature anchor, defeating the target comparison.
    for rows, color, marker in ((TEACHER_ONLY, ORANGE, "o"), (SHARED_KD, BLUE, "s")):
        frontier = nondominated(rows)
        print("KD Pareto temperatures:", frontier[:, 0].astype(int).tolist())
        bx.plot(frontier[:, 1], frontier[:, 2], color=color, marker=marker,
                markersize=4, linewidth=0.8, markerfacecolor="white", markeredgewidth=0.9)
        for tau, clean, aa in frontier:
            offset = (6, 3) if tau == 8 else (-7, -10) if tau == 16 else (-7, 5)
            bx.annotate(rf"$\tau={int(tau)}$", (clean, aa), xytext=offset,
                        textcoords="offset points", color=color, fontsize=7,
                        ha="left" if tau == 8 else "right")
    bx.plot(62.59, 25.69, "D", color=BLACK, markersize=4,
            markerfacecolor="white", markeredgewidth=0.9)
    bx.plot(62.72, 25.88, "*", color=GREEN, markersize=7, markeredgewidth=0.6)
    bx.set(xlim=(59.1, 63.05), ylim=(24.98, 26.02),
           xlabel="Clean accuracy (%)", ylabel=r"AA$_2$ (%)")
    bx.set_xticks([60, 61, 62, 63])
    bx.set_yticks([25.0, 25.4, 25.8])
    handles = [Line2D([], [], marker=marker, color=color, linestyle="none",
                      markersize=size, markerfacecolor=face, markeredgewidth=0.8, label=label)
               for label, marker, color, size, face in [
                   ("Teacher-only KL", "o", ORANGE, 4, "white"),
                   ("Shared-temp. KD", "s", BLUE, 4, "white"),
                   ("Logit MSE", "D", BLACK, 4, "white"),
                   ("Feature anchor", "*", GREEN, 7, GREEN)]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.005),
               ncol=4, columnspacing=1.1, handletextpad=0.35)
    save(fig, "analysis_targets.pdf")


def regression(ax, xs, ys, color, marker, linestyle="-"):
    slope, intercept = np.polyfit(xs, ys, 1)
    xx = np.array([min(xs), max(xs)])
    ax.plot(xx, slope * xx + intercept, color=color, linestyle=linestyle, linewidth=0.85)
    ax.plot(xs, ys, linestyle="none", marker=marker, color=color,
            markersize=3.8, markerfacecolor="white", markeredgewidth=0.85)
    return np.corrcoef(xs, ys)[0, 1]


def make_teacher_figure():
    fig = plt.figure(figsize=(5.5, 142 / 72))
    ax = fig.add_axes([0.105, 0.305, 0.34, 0.555])
    bx = fig.add_axes([0.64, 0.305, 0.33, 0.555])
    for panel in (ax, bx):
        style_axis(panel)
    ax.set_title("(a) Accuracy transfer", loc="left", pad=7)
    bx.set_title("(b) Class separation", loc="left", pad=7)
    epochs = [50, 100, 150, 200, 300]
    teacher_clean = np.array([75.81, 76.62, 77.52, 77.66, 78.32])
    student_clean = np.array([64.28, 63.65, 62.93, 62.72, 62.24])
    student_aa = np.array([24.38, 25.20, 25.78, 25.88, 25.80])
    teacher_scatter = np.array([1.100, 0.982, 0.887, 0.808, 0.712])
    student_scatter = np.array([0.955, 0.900, 0.875, 0.850, 0.829])
    delta = teacher_clean - teacher_clean[0]
    clean_r = regression(ax, delta, student_clean - student_clean[0], ORANGE, "s")
    aa_r = regression(ax, delta, student_aa - student_aa[0], GREEN, "^", "--")
    ax.set(xlim=(-0.12, 2.7), ylim=(-2.35, 1.9),
           xlabel="Teacher clean change (pp)", ylabel="Student accuracy change (pp)")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1])
    scatter_r = regression(bx, teacher_scatter, student_scatter, BLUE, "o")
    bx.set(xlim=(0.68, 1.13), ylim=(0.814, 0.975),
           xlabel=r"Teacher $S_w/S_b$", ylabel=r"Student $S_w/S_b$")
    bx.set_xticks([0.7, 0.9, 1.1])
    bx.set_yticks([0.82, 0.86, 0.90, 0.94])
    bx.text(0.045, 0.92, rf"$r={scatter_r:.3f}$", transform=bx.transAxes,
            fontsize=7.5, va="top", color=BLUE)
    for epoch, x, y in zip(epochs, teacher_scatter, student_scatter):
        bx.annotate(str(epoch), (x, y), xytext=(-5, 4) if epoch == 50 else (5, -3),
                    textcoords="offset points", ha="right" if epoch == 50 else "left",
                    fontsize=6.5, color="0.35")
    handles = [
        Line2D([], [], color=ORANGE, marker="s", markersize=3.8, markerfacecolor="white",
               label=rf"Student clean ($r={clean_r:.3f}$)"),
        Line2D([], [], color=GREEN, marker="^", linestyle="--", markersize=3.8,
               markerfacecolor="white", label=rf"Student AA$_2$ ($r={aa_r:.3f}$)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.51, 0.005),
               ncol=2, columnspacing=2, handletextpad=0.5)
    save(fig, "teacher_checkpoint_trends.pdf")


if __name__ == "__main__":
    print("Figure font:", FONT)
    make_target_figure()
    make_teacher_figure()
