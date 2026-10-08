"""SIGMOD Chapter 4 figure style.

matplotlib + scienceplots, IEEE two-column preset (``science`` + ``ieee`` +
``no-latex``).  Serif labels, inward ticks, full spines, no grid.  Method
identity is hue + marker + linestyle together, so a figure stays readable in
grayscale and under color-vision deficiencies:

    E8Q crimson, ME8Q emerald, RaBitQ royal blue, CAQ amber, SAQ violet,
    PQ slate (recessive baseline: X marker, dotted line).

Bit rate, when a figure also varies it, is a second linestyle channel drawn
in the same method color.  Every plotted series gets its own legend entry;
nothing is left to be inferred from the caption.
"""

from __future__ import annotations

import math

import matplotlib as mpl
import matplotlib.pyplot as plt

SIGMOD_METHOD_STYLE = {
    "e8q":    ("#D32F2F", "o", "-"),
    "rabitq": ("#1976D2", "^", "--"),
    "caq":    ("#F57C00", "v", "-."),
    "pq":     ("#607D8B", "X", ":"),
    "me8q":   ("#00897B", "s", "--"),
    "saq":    ("#5E35B1", "d", ":"),
}
PALETTE_ORDER = ("e8q", "me8q", "rabitq", "caq", "saq", "pq")

# IEEE column widths (ACM sigconf): single column ~3.33 in, text width ~7.0 in.
COL_W = 3.33
TEXT_W = 7.00

# Point sizes that stay legible once the PDF is placed at column / text width.
# Scienceplots' ieee preset sizes these for a 3.3 in figure; multi-panel
# figures below override per call rather than shrinking a legend to fit.
FONT_LABEL = 8.0
FONT_TITLE = 8.0
FONT_TICK = 7.0
FONT_LEGEND = 7.0


def apply_sigmod_style() -> None:
    """IEEE style entry point for the Chapter 4 scripts.

    scienceplots resets rcParams, so it runs first.  The block after it only
    pins things the preset gets wrong for camera-ready PDF: Type 42 fonts
    (the preset emits Type 3), a serif face that is actually installed, and
    mathtext instead of a TeX install.
    """
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "no-latex"])
    except (ImportError, OSError):
        mpl.rcParams.update({
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.grid": False,
            "legend.frameon": False,
        })
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": False,
        "axes.formatter.use_mathtext": True,
        "axes.labelsize": FONT_LABEL,
        "axes.titlesize": FONT_TITLE,
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.4,
        "lines.markeredgecolor": "white",
        "legend.fontsize": FONT_LEGEND,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.2,
        "axes.prop_cycle": mpl.cycler(
            color=[SIGMOD_METHOD_STYLE[m][0] for m in PALETTE_ORDER]),
    })


# Panel fill and the line drawn on every graduation.  A light panel with a
# white line at each labelled tick reads as a ruler: a value can be taken off
# the grid instead of estimated between two black ticks, which matters on the
# logarithmic QPS axes and on the distance-ratio axes whose whole range spans
# a few thousandths.
GRID_FACE = "#EAEAF2"
GRID_LINE = "#FFFFFF"


def tick_grid(ax, face: str = GRID_FACE, line: str = GRID_LINE) -> None:
    """Fill the panel and draw a white line on every labelled graduation.

    The rule is one line per number the axis prints, so the grid can be read
    against the labels instead of becoming a stripe pattern: a plain log axis
    carries its decades, and an axis that labels its 2 and 5 graduations gets
    those too (see :func:`fine_log_axis`, which turns on its own minor grid
    once it has made those ticks meaningful).  The grid sits under the data
    (``set_axisbelow``), and the spines take the panel fill, so the boundary
    is the block itself rather than a dark frame around it.
    """
    ax.set_facecolor(face)
    ax.grid(True, which="major", color=line, linewidth=0.7)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(face)
        spine.set_linewidth(0.8)


def style_panel(ax, xlabel: str | None = None, ylabel: str | None = None,
                title: str | None = None, ylog: bool = False,
                right_axis: bool = False) -> None:
    """IEEE panel: inward ticks, tick grid, no dark frame.

    ``right_axis`` is accepted for call-site compatibility and ignored: a
    second y-axis on the right edge collides with the panel title once the
    legend moves inside the axes.
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=3)
    if ylog:
        ax.set_yscale("log")
    tick_grid(ax)
    ax.tick_params(direction="in", which="both", top=True, right=True,
                   length=2.5, width=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)


def fit_top_legend(fig, legend, max_width_in: float,
                   min_fontsize: float = 6.0) -> None:
    """Legacy hook.  Legends are no longer shrunk to fit a squeezed row.

    Kept so older call sites import cleanly; it does nothing.  A legend that
    does not fit belongs on more rows or inside the axes, not at 4 pt.
    """
    return None


# Labeled graduations the tick chooser may emit, coarsest first.  Recall is
# reported in percentage points, so every entry divides 100 exactly.
_TICK_STEPS = (10.0, 5.0, 2.5, 2.0, 1.0, 0.5, 0.25)
_TICK_MIN, _TICK_MAX = 4, 8


def fine_log_axis(ax, subs=(2.0, 5.0)) -> None:
    """Label the 2 and 5 graduations of a log axis, not just the decades.

    Matplotlib labels no minor ticks at all once a locator offers more than
    five of them, which on a QPS axis leaves two numbers a decade and a half
    apart.  Restricting the minors to 2 and 5 keeps the count under that
    threshold, so a curve between 2e3 and 5e3 can be read off the axis.
    """
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=subs))
    # The default formatter suppresses minor labels on a wide view interval,
    # so ask for them explicitly; with only 2 and 5 the rows do not collide.
    ax.yaxis.set_minor_formatter(
        LogFormatterSciNotation(minor_thresholds=(float("inf"), float("inf"))))
    # These are the axis's labelled ticks now, so they carry the panel's grid
    # too (tick_grid only grids the majors).
    ax.grid(True, axis="y", which="minor", color=GRID_LINE, linewidth=0.7)
    ax.set_axisbelow(True)


def recall_ticks(lo: float, hi: float,
                 lo_count: int = _TICK_MIN,
                 hi_count: int = _TICK_MAX) -> tuple[float, ...]:
    """Labeled graduations spanning [lo, hi], as fine as legibility allows.

    Matplotlib's default locator picks round numbers, which on a 26-point
    recall window means a label every 10 points --- too coarse to read the
    sub-percentage gaps the Chapter 4 panels exist to show.  This walks the
    step ladder until the panel carries at least ``lo_count`` labels, and
    stops before it carries more than ``hi_count``.
    """
    if not (hi > lo):
        return ()
    for step in _TICK_STEPS:
        first = math.ceil(lo / step - 1e-9) * step
        count = int(math.floor((hi - first) / step + 1e-9)) + 1
        if count < lo_count:
            continue
        ticks: list[float] = []
        value = first
        while value <= hi + 1e-9:
            ticks.append(round(value, 6))
            value += step
        if len(ticks) <= hi_count:
            return tuple(ticks)
        return tuple(ticks[:hi_count])
    return ()


def open_corner(ax, x: float = 0.02, y: float = 0.02,
                y_max: float | None = None) -> None:
    """Move each limit slightly past the data, so no marker straddles a spine.

    Panels whose windows come from the data end up with a point exactly on an
    axis corner, where half the marker is clipped away and the corner doubles
    as a sample; worse, a window that starts at the lowest measured value
    reads as if the axes were chosen around that one point.  Every limit
    moves outward by a fraction of its own span --- multiplicatively on a log
    axis, where a span fraction would be meaningless --- which costs no
    visible range and holds whatever used to sit on the edge.

    ``x`` or ``y`` may be 0 to leave that axis alone (a ratio axis that
    starts at 1.0, or a count axis that starts at zero, is already right);
    ``y_max`` clamps the padded top, for axes that cannot pass a bound.
    """
    for axis, get, set_, frac in (
        (ax.xaxis, ax.get_xlim, ax.set_xlim, x),
        (ax.yaxis, ax.get_ylim, ax.set_ylim, y),
    ):
        if frac <= 0.0:
            continue
        lo, hi = (float(v) for v in get())
        if axis.get_scale() == "log":
            if lo > 0:
                lo /= 1.0 + frac
            if hi > 0:
                hi *= 1.0 + frac
        else:
            span = hi - lo
            if span > 0:
                lo -= frac * span
                hi += frac * span
        if y_max is not None and axis is ax.yaxis:
            hi = min(hi, y_max)
        if hi > lo:
            set_(lo, hi)


def _axis_tick_candidates(axis, lo: float, hi: float, log: bool) -> list[float]:
    """Every graduation the axis would draw over a widened window."""
    from matplotlib.ticker import NullLocator
    if log:
        # The locator answers with a fixed number of ticks, so over a widened
        # window it returns the decades and drops the 2 and 5 graduations the
        # panels actually label.  Build the ladder these axes use directly.
        ladder = [m * 10.0 ** k
                  for k in range(-30, 31) for m in (1.0, 2.0, 5.0)]
        return sorted(v for v in ladder if v <= hi)
    span = hi - lo
    grid_lo = lo - 2.0 * max(span, abs(lo) * 1e-6, 1e-12)
    out: list[float] = []
    for locator in (axis.get_major_locator(), axis.get_minor_locator()):
        if locator is None or isinstance(locator, NullLocator):
            continue
        try:
            out.extend(float(v) for v in locator.tick_values(grid_lo, hi))
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
    return sorted({v for v in out if math.isfinite(v)})


def snap_corner_ticks(ax, axes: str = "xy") -> None:
    """Put the lower end of an axis on a graduation.

    Panels whose lower limit comes from the data --- a QPS floor, an adaptive
    distance-ratio window, a hand-picked recall window --- otherwise start
    between two graduations, so the corner the eye reads as the axis origin
    carries no number.  Each lower limit moves down to the next graduation;
    nothing is ever clipped, and the upper limits are left where the data put
    them.
    """
    targets = []
    if "x" in axes:
        targets.append((ax.xaxis, ax.get_xlim, ax.set_xlim,
                        ax.get_xscale() == "log"))
    if "y" in axes:
        targets.append((ax.yaxis, ax.get_ylim, ax.set_ylim,
                        ax.get_yscale() == "log"))
    for axis, get, set_, log in targets:
        lo, hi = (float(v) for v in get())
        cands = _axis_tick_candidates(axis, lo, hi, log)
        if not cands:
            continue
        below = [c for c in cands if c <= lo * (1 + 1e-9)]
        if below:
            best = max(below)
        elif len(cands) >= 2:
            # The locator only offered ticks above the limit; extend its own
            # step downward so the corner lands on the same ladder.
            step = min(b - a for a, b in zip(cands, cands[1:]) if b > a)
            best = cands[0] - step * math.ceil((cands[0] - lo) / step - 1e-9)
        else:
            continue
        if best < hi:
            set_(best, hi)
