"""Generate the compact vector figures used in Section 3.

The plotting code intentionally depends only on ReportLab so the figures can be
rebuilt in the paper workspace without a full scientific Python environment.
"""

from __future__ import annotations

from math import log2
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.lib import colors


HERE = Path(__file__).resolve().parent

TEXT = colors.HexColor("#202020")
GRID = colors.HexColor("#D7D7D7")
BLUE = colors.HexColor("#0072B2")
ORANGE = colors.HexColor("#D55E00")
GREEN = colors.HexColor("#009E73")
GRAY = colors.HexColor("#666666")


def _text(c, x, y, value, size=7.2, font="Times-Roman", align="left", color=TEXT):
    c.setFillColor(color)
    c.setFont(font, size)
    if align == "center":
        c.drawCentredString(x, y, value)
    elif align == "right":
        c.drawRightString(x, y, value)
    else:
        c.drawString(x, y, value)


def _math_ylabel(c, value):
    """Draw the two subscripted y-axis labels without external fonts."""
    text = c.beginText()
    if value == "AA2 (%)":
        text.setTextOrigin(-14.5, -2.5)
        text.setFont("Times-Italic", 7.0)
        text.textOut("AA")
        text.setRise(-1.5)
        text.setFont("Times-Roman", 5.3)
        text.textOut("2")
        text.setRise(0)
        text.setFont("Times-Roman", 7.0)
        text.textOut(" (%)")
    else:
        text.setTextOrigin(-9.5, -2.5)
        text.setFont("Times-Italic", 7.0)
        text.textOut("S")
        text.setRise(-1.5)
        text.setFont("Times-Italic", 5.3)
        text.textOut("w")
        text.setRise(0)
        text.setFont("Times-Roman", 7.0)
        text.textOut("/")
        text.setFont("Times-Italic", 7.0)
        text.textOut("S")
        text.setRise(-1.5)
        text.setFont("Times-Italic", 5.3)
        text.textOut("b")
    c.drawText(text)


def _line(c, points, color, width=1.25, dash=None):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.setDash(dash or [])
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.drawPath(path)
    c.setDash([])


def _marker(c, x, y, shape, color, size=2.7, fill=colors.white):
    c.setStrokeColor(color)
    c.setFillColor(fill)
    c.setLineWidth(1.05)
    if shape == "circle":
        c.circle(x, y, size, stroke=1, fill=1)
    elif shape == "square":
        c.rect(x - size, y - size, 2 * size, 2 * size, stroke=1, fill=1)
    elif shape == "diamond":
        p = c.beginPath()
        p.moveTo(x, y + size + 0.5)
        p.lineTo(x + size + 0.5, y)
        p.lineTo(x, y - size - 0.5)
        p.lineTo(x - size - 0.5, y)
        p.close()
        c.drawPath(p, stroke=1, fill=1)
    elif shape == "star":
        # A compact four-point star remains legible after ICLR-scale reduction.
        p = c.beginPath()
        p.moveTo(x, y + size + 1.0)
        p.lineTo(x + 1.1, y + 1.1)
        p.lineTo(x + size + 1.0, y)
        p.lineTo(x + 1.1, y - 1.1)
        p.lineTo(x, y - size - 1.0)
        p.lineTo(x - 1.1, y - 1.1)
        p.lineTo(x - size - 1.0, y)
        p.lineTo(x - 1.1, y + 1.1)
        p.close()
        c.drawPath(p, stroke=1, fill=1)
    elif shape == "triangle":
        p = c.beginPath()
        p.moveTo(x, y + size + 0.7)
        p.lineTo(x + size + 0.8, y - size)
        p.lineTo(x - size - 0.8, y - size)
        p.close()
        c.drawPath(p, stroke=1, fill=1)


def _axes(c, x0, y0, width, height, xlim, ylim, xticks, yticks,
          xlabel, ylabel, xlabels=None, ylabels=None):
    sx = lambda value: x0 + (value - xlim[0]) / (xlim[1] - xlim[0]) * width
    sy = lambda value: y0 + (value - ylim[0]) / (ylim[1] - ylim[0]) * height

    c.setStrokeColor(TEXT)
    c.setLineWidth(0.7)
    c.line(x0, y0, x0 + width, y0)
    c.line(x0, y0, x0, y0 + height)

    for index, tick in enumerate(yticks):
        yy = sy(tick)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.45)
        c.line(x0, yy, x0 + width, yy)
        label = ylabels[index] if ylabels else f"{tick:g}"
        _text(c, x0 - 4, yy - 2.2, label, 6.4, align="right", color=GRAY)

    for index, tick in enumerate(xticks):
        xx = sx(tick)
        c.setStrokeColor(TEXT)
        c.setLineWidth(0.55)
        c.line(xx, y0, xx, y0 - 2.5)
        label = xlabels[index] if xlabels else f"{tick:g}"
        _text(c, xx, y0 - 10.5, label, 6.4, align="center", color=GRAY)

    _text(c, x0 + width / 2, y0 - 21, xlabel, 7.0, align="center")
    c.saveState()
    c.translate(x0 - 25, y0 + height / 2)
    c.rotate(90)
    if ylabel in ("AA2 (%)", "Sw/Sb"):
        _math_ylabel(c, ylabel)
    else:
        _text(c, 0, -2.5, ylabel, 7.0, align="center")
    c.restoreState()
    return sx, sy


def _legend_item(c, x, y, label, shape, color, line=True):
    if line:
        _line(c, [(x, y + 2), (x + 15, y + 2)], color, width=1.25)
    _marker(c, x + 7.5, y + 2, shape, color, size=2.4,
            fill=colors.white if shape != "star" else color)
    _text(c, x + 19, y - 0.5, label, 6.8)


def make_target_figure():
    width, height = 396, 154
    c = canvas.Canvas(str(HERE / "analysis_targets.pdf"), pagesize=(width, height))
    c.setTitle("Target softening and metric regression")
    c.setLineJoin(1)
    c.setLineCap(1)

    _text(c, 7, 139, "(a)", 8.0, "Times-Bold")
    _text(c, 177, 139, "(b)", 8.0, "Times-Bold")
    _text(c, 25, 139, "Target concentration", 7.5, "Times-Bold")
    _text(c, 195, 139, "Clean--robust trade-off", 7.5, "Times-Bold")

    # Panel (a): temperature directly controls the entropy of the teacher target.
    x0, y0, pw, ph = 34, 30, 116, 88
    taus = [1, 2, 4, 8, 16, 32]
    entropy = [0.76, 2.40, 4.20, 4.56, 4.60, 4.60]
    xs = [log2(value) for value in taus]
    sx, sy = _axes(
        c, x0, y0, pw, ph, (0, 5), (0, 4.8),
        xs, [0, 2, 4, 4.61], "Temperature", "Entropy (nats)",
        [str(value) for value in taus], ["0", "2", "4", "log C"],
    )
    c.setStrokeColor(GRAY)
    c.setLineWidth(0.75)
    c.setDash([3, 2])
    c.line(x0, sy(4.61), x0 + pw, sy(4.61))
    c.setDash([])
    points = [(sx(x), sy(y)) for x, y in zip(xs, entropy)]
    _line(c, points, BLUE, width=1.5)
    for px, py in points:
        _marker(c, px, py, "circle", BLUE, size=2.6, fill=colors.white)

    # Panel (b): trajectories show the turnover and the regression endpoints.
    x0, y0, pw, ph = 205, 30, 167, 88
    sx, sy = _axes(
        c, x0, y0, pw, ph, (57.2, 63.15), (20.2, 26.15),
        [58, 60, 62], [21, 23, 25], "Clean accuracy (%)", "AA2 (%)",
    )
    teacher_only = [
        (1, 58.26, 20.84), (2, 59.33, 22.79), (4, 59.39, 24.48),
        (8, 59.47, 25.19), (16, 57.78, 24.00),
    ]
    shared = [
        (4, 60.83, 25.05), (8, 60.99, 25.25),
        (16, 61.85, 25.39), (32, 61.81, 25.50),
    ]
    for data, color, shape in ((teacher_only, ORANGE, "circle"), (shared, BLUE, "square")):
        points = [(sx(clean), sy(aa)) for _, clean, aa in data]
        _line(c, points, color, width=1.45)
        for temperature, clean, aa in data:
            px, py = sx(clean), sy(aa)
            _marker(c, px, py, shape, color, size=2.5, fill=colors.white)
            dx, dy = 3.5, 3.0
            if temperature == 16 and data is teacher_only:
                dx, dy = -2.5, 4.0
            elif temperature == 1:
                dx, dy = 3.5, -8.5
            elif temperature == 2:
                dx, dy = 3.5, -8.0
            elif temperature == 4 and data is shared:
                dx, dy = -8.5, -9.0
            elif temperature == 8 and data is shared:
                dx, dy = -1.5, 4.0
            elif temperature in (16, 32) and data is shared:
                dx, dy = -10.0 if temperature == 16 else 3.0, -8.0 if temperature == 16 else 3.0
            _text(c, px + dx, py + dy, str(temperature), 5.8, color=color)

    controls = [
        (62.59, 25.69, "diamond", TEXT, "L"),
        (62.72, 25.88, "star", GREEN, "F"),
    ]
    for clean, aa, shape, color, short in controls:
        px, py = sx(clean), sy(aa)
        _marker(c, px, py, shape, color, size=3.0,
                fill=color if shape == "star" else colors.white)
        _text(c, px - 2 if short == "L" else px + 4, py - 10 if short == "L" else py + 2,
              short, 6.2, "Times-Bold", color=color)

    _legend_item(c, 224, 121, "teacher-only", "circle", ORANGE)
    _legend_item(c, 305, 121, "shared KD", "square", BLUE)

    c.showPage()
    c.save()


def make_teacher_figure():
    width, height = 396, 100
    c = canvas.Canvas(str(HERE / "teacher_checkpoint_trends.pdf"), pagesize=(width, height))
    c.setTitle("Teacher checkpoint trends")
    c.setLineJoin(1)
    c.setLineCap(1)

    epochs = [50, 100, 150, 200, 300]
    teacher_clean = [75.81, 76.62, 77.52, 77.66, 78.32]
    student_clean = [64.28, 63.65, 62.93, 62.72, 62.24]
    student_aa = [24.38, 25.20, 25.78, 25.88, 25.80]
    teacher_scatter = [1.100, 0.982, 0.887, 0.808, 0.712]
    student_scatter = [0.955, 0.900, 0.875, 0.850, 0.829]

    _legend_item(c, 96, 89, "teacher", "circle", BLUE)
    _legend_item(c, 181, 89, "student", "square", ORANGE)
    _legend_item(c, 266, 89, "student AA2", "triangle", GREEN)

    specs = [
        (40, "(a)", "Accuracy change", (-2.4, 2.8), [-2, 0, 2], "Change (pp)"),
        (240, "(b)", "Class scatter", (0.68, 1.12), [0.7, 0.9, 1.1], "Sw/Sb"),
    ]

    axes = []
    for x0, panel, title, ylim, yticks, ylabel in specs:
        _text(c, x0 - 31, 77, panel, 8.0, "Times-Bold")
        _text(c, x0 + 68, 77, title, 7.3, "Times-Bold", align="center")
        axes.append(_axes(
            c, x0, 24, 136, 46, (50, 300), ylim,
            [50, 150, 300], yticks, "Teacher epoch", ylabel,
        ))

    sx, sy = axes[0]
    accuracy_changes = (
        ([value - teacher_clean[0] for value in teacher_clean], BLUE, "circle"),
        ([value - student_clean[0] for value in student_clean], ORANGE, "square"),
        ([value - student_aa[0] for value in student_aa], GREEN, "triangle"),
    )
    for values, color, shape in accuracy_changes:
        points = [(sx(epoch), sy(value)) for epoch, value in zip(epochs, values)]
        _line(c, points, color, width=1.45)
        for px, py in points:
            _marker(c, px, py, shape, color, size=2.4, fill=colors.white)

    sx, sy = axes[1]
    for values, color, shape in (
        (teacher_scatter, BLUE, "circle"),
        (student_scatter, ORANGE, "square"),
    ):
        points = [(sx(epoch), sy(value)) for epoch, value in zip(epochs, values)]
        _line(c, points, color, width=1.45)
        for px, py in points:
            _marker(c, px, py, shape, color, size=2.4, fill=colors.white)

    c.showPage()
    c.save()


if __name__ == "__main__":
    make_target_figure()
    make_teacher_figure()
