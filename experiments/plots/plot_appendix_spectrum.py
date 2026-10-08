"""Appendix figures that expand the fixed-rate evaluation across the bitrate
spectrum b in {1,...,8}:

  fig_shaping_gain_mse            -- quantization distortion (NMSE) of the
                                     deployed codebook and of the rate-matched
                                     product quantizer, and the measured
                                     distortion reduction in dB against it
  fig_full_spectrum_trajectories  -- recall ceiling and QPS at 90% recall as a
                                     function of the rate, all six corpora
  fig_all_frontiers_grid          -- QPS--Recall@10 frontiers of every
                                     individual rate on all six corpora

Usage:
  ./experiments/plots/plot_appendix_spectrum.py --results results \
      --out figures_reproduced [--shaping results/shaping/nmse_shaping.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DISPLAY,
    METHOD_LABEL,
    load_bench_runs,
    pick_run,
    savefig,
)
from plot_chapter4 import (  # noqa: E402
    DATASETS,
    DISPLAY_NAME,
    THREADS,
    _ceiling,
    _qps_at_recall,
    _row_series,
)
from sigmod_style import (  # noqa: E402
    SIGMOD_METHOD_STYLE as M_STYLE,
    TEXT_W,
    apply_sigmod_style,
    fine_log_axis,
    open_corner,
    recall_ticks,
    snap_corner_ticks,
    style_panel,
)

METHODS = ("e8q", "rabitq", "caq")
# Corpus identity is the colour channel here (method is the linestyle), so the
# corpus palette is chosen for six-way separability, not for method identity.
CORPUS_COLOR = {
    "sift1M": "#1f4e79",
    "deep1M": "#2e7d32",
    "gist": "#8d6e00",
    "openai1536": "#b02a6b",
    "msmarco10M": "#00838f",
    "stella-trec24-1024": "#5b3fa8",
}
CORPUS_MARKER = {
    "sift1M": "o",
    "deep1M": "s",
    "gist": "^",
    "openai1536": "D",
    "msmarco10M": "v",
    "stella-trec24-1024": "P",
}
BITS = tuple(range(1, 9))


def _fixed(runs: dict, dataset: str, bits: int, method: str):
    return pick_run(runs, dataset, bits, method, threads=THREADS)


def _contiguous(values: list[int]) -> list[list[int]]:
    """Split a sorted rate list into runs of consecutive integers."""
    out: list[list[int]] = []
    for v in values:
        if out and v == out[-1][-1] + 1:
            out[-1].append(v)
        else:
            out.append([v])
    return out


# --------------------------------------------------------------------------
# (1) Quantization distortion and shaping gain
# --------------------------------------------------------------------------
def _read_shaping(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    out: dict[str, dict[int, dict[str, float]]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            ds = row["corpus"]
            bits = int(row["bits"])
            out.setdefault(ds, {})[bits] = {
                k: float(v) for k, v in row.items() if k not in ("corpus", "bits") and v != ""
            }
    return out


def plot_shaping_gain(csv_path: Path, out: Path) -> None:
    data = _read_shaping(csv_path)
    order = [d for d in DATASETS if d in data]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.35))
    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.155, top=0.965, wspace=0.20)

    ax_n, ax_g = axes
    style_panel(ax_n, xlabel="Rate (bits/dim)", ylabel="NMSE", ylog=True)
    style_panel(ax_g, xlabel="Rate (bits/dim)",
                ylabel="Reduction vs product code (dB)")
    fine_log_axis(ax_n)
    for ax in (ax_n, ax_g):
        ax.set_xlim(0.7, 8.3)
        ax.set_xticks(BITS)

    bits = sorted({b for ds in order for b in data[ds]})

    def rows(col: str) -> list[list[float]]:
        return [[data[ds][b][col] for ds in order] for b in bits]

    # Each series is drawn as the corpus mean with the corpus min-max banded:
    # the six corpora agree to well under a tenth of a dB, so six overplotted
    # curves would read as two and leave the colour legend undecodable.
    e8q = rows("nmse_e8q_deployed")
    sc = rows("nmse_scalar_best")
    total = [[10.0 * math.log10(s / e) for s, e in zip(r_s, r_e)]
             for r_s, r_e in zip(sc, e8q)]

    ink_e8q, ink_ctl = "#1f4e79", "#8d6e00"
    mean = lambda v: sum(v) / len(v)  # noqa: E731

    for series, color, ls, label in (
            (e8q, ink_e8q, "-", "E8Q codebook"),
            # The control is measured at its own best grid scale, so the gap
            # is the conservative pairing (Appendix A.5).
            (sc, ink_ctl, "--", "rate-matched product code")):
        ax_n.plot(bits, [mean(v) for v in series], color=color, linestyle=ls,
                  marker="o", markersize=2.4, linewidth=0.9, label=label)
        ax_n.fill_between(bits, [min(v) for v in series], [max(v) for v in series],
                          color=color, alpha=0.20, linewidth=0)

    ax_g.plot(bits, [mean(v) for v in total], color=ink_e8q, linestyle="-",
              marker="o", markersize=2.4, linewidth=0.9,
              label="vs rate-matched product code")
    ax_g.fill_between(bits, [min(v) for v in total], [max(v) for v in total],
                      color=ink_e8q, alpha=0.20, linewidth=0)

    # Asymptotic Voronoi shaping gain of E8 over the cubic grid at equal point
    # density, 10 log10(G(Z^8)/G(E8)); a different baseline from either curve,
    # which compare the codebook against product codes on its own level tables.
    ref = 10.0 * math.log10((1.0 / 12.0) / 0.07168)
    ax_g.axhline(ref, color="#444444", linewidth=0.7, linestyle=(0, (4, 2)))
    ax_g.annotate(f"lattice constant $G(\\mathbb{{Z}}^8)/G(E_8)={ref:.2f}$ dB",
                  (0.95, ref), xytext=(0, 2.0), textcoords="offset points",
                  ha="left", va="bottom", fontsize=6.0, color="#444444")

    ax_g.set_ylim(0.0, 1.9)
    ax_n.legend(loc="lower left", fontsize=5.8, handlelength=1.6,
                labelspacing=0.25, borderpad=0.25)
    ax_g.legend(loc="lower left", fontsize=5.8, handlelength=1.6,
                labelspacing=0.25, borderpad=0.25)
    snap_corner_ticks(ax_n, axes="y")
    snap_corner_ticks(ax_g, axes="y")
    open_corner(ax_n, x=0.0, y=0.02)
    open_corner(ax_g, x=0.0, y=0.02)
    savefig(fig, out, "fig_shaping_gain_mse")


# --------------------------------------------------------------------------
# (2) Full-spectrum trajectories
# --------------------------------------------------------------------------
def plot_trajectories(runs: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_W, 3.15), squeeze=False)
    fig.subplots_adjust(left=0.062, right=0.935, bottom=0.105, top=0.925,
                        wspace=0.62, hspace=0.52)
    for index, ds in enumerate(DATASETS):
        row, col = divmod(index, 3)
        ax = axes[row][col]
        style_panel(ax, xlabel="Rate (bits/dim)",
                    ylabel="Recall@10 ceiling (%)" if col == 0 else None,
                    title=DISPLAY[ds])
        ax.set_xlim(0.7, 8.3)
        ax.set_xticks(BITS)
        qps_ax = ax.twinx()
        qps_ax.set_yscale("log")
        qps_ax.tick_params(direction="in", which="both", colors="#555555",
                           labelsize=6.2, length=2.2, width=0.5)
        fine_log_axis(qps_ax)
        qps_ax.set_ylabel("QPS at 90% recall" if col == 2 else None,
                          color="#555555", fontsize=7.0)
        qps_ax.tick_params(labelright=col == 2, labelleft=False)
        for spine in ("top", "bottom"):
            qps_ax.spines[spine].set_visible(False)

        ceil_lo, ceil_hi = 100.0, 0.0
        qps_lo, qps_hi = float("inf"), 0.0
        for method in METHODS:
            color = M_STYLE[method][0]
            bl: list[int] = []
            rc: list[float] = []
            bq: list[int] = []
            qps: list[float] = []
            for b in BITS:
                run = _fixed(runs, ds, b, method)
                if run is None:
                    continue
                ceiling = 100.0 * _ceiling(run, method)
                bl.append(b)
                rc.append(ceiling)
                ceil_lo, ceil_hi = min(ceil_lo, ceiling), max(ceil_hi, ceiling)
                q = _qps_at_recall(run, method, 0.90)
                if math.isfinite(q):
                    bq.append(b)
                    qps.append(q)
                    qps_lo, qps_hi = min(qps_lo, q), max(qps_hi, q)
            ax.plot(bl, rc, color=color, marker=M_STYLE[method][1],
                    linestyle="-", linewidth=0.9, markersize=2.6,
                    label=METHOD_LABEL[method])
            # A dashed series is drawn only where the 90% target is reachable
            # at that rate; no line is continued across an unreachable rate,
            # so a rate with no 90% point ends a segment rather than being
            # bridged.
            for seg in _contiguous(bq):
                idx = [bq.index(b) for b in seg]
                if len(seg) > 1:
                    qps_ax.plot(seg, [qps[i] for i in idx], color=color,
                                linestyle="--", linewidth=0.8, alpha=0.8)
                else:
                    qps_ax.plot(seg, [qps[i] for i in idx], color=color,
                                linestyle="none", marker="x", markersize=3.0)
        pad = max(0.6, 0.06 * (ceil_hi - ceil_lo))
        ax.set_ylim(ceil_lo - pad, min(100.0, ceil_hi + pad))
        ax.set_yticks(recall_ticks(ax.get_ylim()[0], ax.get_ylim()[1]))
        if math.isfinite(qps_lo):
            qps_ax.set_ylim(0.55 * qps_lo, 1.7 * qps_hi)
        if index == 0:
            ax.legend(loc="lower right", fontsize=6.0, handlelength=1.5,
                      labelspacing=0.25)
    savefig(fig, out, "fig_full_spectrum_trajectories")


# --------------------------------------------------------------------------
# (3) Every-rate Pareto frontier grid
# --------------------------------------------------------------------------
def _upper_frontier(x: "np.ndarray", y: "np.ndarray") -> "np.ndarray":
    """Indices of the upper-left envelope of a QPS--recall sweep.

    An IVF sweep keeps spending probes after the recall ceiling is reached, so
    its tail wanders by a few tenths of a point of recall while throughput
    falls by a factor of two or more --- on a four-point recall window that
    prints as a vertical streak with no reachable operating point on it.  A
    point whose recall is lower than some other point's and whose throughput
    is no higher is dominated, and is dropped.
    """
    keep: list[int] = []
    best_qps = -math.inf
    for i in np.argsort(-x, kind="stable"):
        if y[i] > best_qps:
            keep.append(int(i))
            best_qps = float(y[i])
    return np.asarray(sorted(keep))


def _qps_ticks(lo: float, hi: float) -> list[float]:
    """At most three QPS graduations inside a window.

    Every panel is numbered, so the labels have to be few enough to read and
    short enough to fit the gap between two panels; the decades are thinned to
    three of the 1--2--5 ladder.
    """
    cands = [m * 10.0 ** e for e in range(-1, 8) for m in (1.0, 2.0, 5.0)]
    inside = [v for v in cands if lo <= v <= hi]
    if len(inside) <= 3:
        return inside
    step = len(inside) / 3.0
    return [inside[int(i * step)] for i in range(3)]


def _qps_label(v: float, _pos) -> str:
    if v >= 1000:
        return f"{v / 1000:g}k"
    return f"{v:g}"


def plot_all_frontiers(runs: dict, out: Path) -> None:
    """One panel per (corpus, rate): the grid is the rate sweep, per corpus.

    Columns carry the six corpora in the order --- and under the names --- of
    the body's Table 1; rows carry the eight rates.  A single figure-level
    legend replaces the per-panel ones the two-corpus version could afford.
    Every panel autoscales and carries its own QPS numbers: the corpora differ
    by two orders of magnitude and the rates by up to a factor of five, so a
    scale shared across a column would leave the low-QPS rates of a column
    squeezed into its lower third.
    """
    ncol, nrow = len(DATASETS), len(BITS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(TEXT_W, 8.05), squeeze=False)
    # The interior columns carry numbered graduations too, so the gaps have to
    # clear a QPS label rather than just a tick mark.
    fig.subplots_adjust(left=0.058, right=0.998, bottom=0.048, top=0.925,
                        wspace=0.36, hspace=0.45)
    for bi, b in enumerate(BITS):
        for ci, ds in enumerate(DATASETS):
            ax = axes[bi][ci]
            style_panel(ax, xlabel="Recall@10 (%)" if bi == nrow - 1 else None,
                        ylabel="QPS" if ci == 0 else None,
                        title=f"$b={b}$" if ci == 0 else None, ylog=True)
            ax.tick_params(labelsize=4.2)
            ax.yaxis.set_tick_params(labelleft=True, which="both", pad=1.0)
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            series: list[tuple[str, "np.ndarray", "np.ndarray"]] = []
            lo, hi = 100.0, 0.0
            for method in METHODS:
                run = _fixed(runs, ds, b, method)
                if run is None:
                    continue
                x, y = _row_series(run, method)
                keep = x >= 0.5
                if not np.any(keep):
                    continue
                x, y = 100.0 * x[keep], y[keep]
                front = _upper_frontier(x, y)
                x, y = x[front], y[front]
                lo, hi = min(lo, float(np.max(x))), max(hi, float(np.max(x)))
                series.append((method, x, y))
            if hi <= lo:
                continue
            x0 = max(30.0, math.floor((lo - 0.05 * (hi - lo)) / 5.0) * 5.0)
            x1 = min(100.0, hi + 0.03 * (hi - lo))
            # The window is set by the recall ceilings, so the QPS range has to
            # be measured inside it: a frontier can start a decade of
            # throughput above its own ceiling, and letting that tail set the
            # limit flattens the drawn curve against the floor.
            qlo, qhi = float("inf"), 0.0
            for _, x, y in series:
                inside = (x >= x0) & (x <= x1)
                if np.any(inside):
                    qlo = min(qlo, float(np.min(y[inside])))
                    qhi = max(qhi, float(np.max(y[inside])))
            for method, x, y in series:
                ax.plot(x, y, color=M_STYLE[method][0],
                        linestyle="-", linewidth=0.65,
                        marker=M_STYLE[method][1], markersize=1.6,
                        markevery=3, markeredgecolor="white",
                        markeredgewidth=0.25, label=METHOD_LABEL[method])
            ax.set_xlim(x0, x1)
            ax.set_xticks(recall_ticks(x0, x1, 3, 5))
            if math.isfinite(qlo):
                pad = 0.06 * max(math.log10(qhi / qlo), 0.10)
                ax.set_ylim(qlo * 10 ** (-pad), qhi * 10 ** pad)
            snap_corner_ticks(ax, axes="y")
            ax.set_yticks(_qps_ticks(*ax.get_ylim()))
            ax.yaxis.set_major_formatter(FuncFormatter(_qps_label))
            open_corner(ax, x=0.0, y=0.0)
    for ci, ds in enumerate(DATASETS):
        box = axes[0][ci].get_position()
        fig.text(0.5 * (box.x0 + box.x1), box.y1 + 0.022,
                 DISPLAY_NAME[ds], ha="center", va="bottom", fontsize=6.4)
    handles = [Line2D([], [], color=M_STYLE[m][0], linestyle="-",
                      linewidth=0.9, marker=M_STYLE[m][1], markersize=2.6,
                      markeredgecolor="white", markeredgewidth=0.3,
                      label=METHOD_LABEL[m]) for m in METHODS]
    # Same placement as the body's frontier figures: one figure-level legend
    # above the panels, clear of the column headers.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.998),
               ncol=len(METHODS), fontsize=6.2, frameon=False, handlelength=1.8,
               columnspacing=1.8, handletextpad=0.5, borderpad=0.0)
    savefig(fig, out, "fig_all_frontiers_grid")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shaping", type=Path,
                    default=None, help="nmse_shaping.csv from experiments/shaping")
    ap.add_argument("--only", default="all",
                    choices=("all", "shaping", "trajectories", "frontiers"))
    args = ap.parse_args()
    apply_sigmod_style()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.only in ("all", "shaping"):
        csv_path = args.shaping
        if csv_path is None:
            csv_path = Path(args.results) / "shaping" / "nmse_shaping.csv"
        if Path(csv_path).is_file():
            plot_shaping_gain(Path(csv_path), args.out)
        else:
            print(f"skip shaping figure: {csv_path} not found")

    runs = load_bench_runs(Path(args.results) / "fixed")
    if not runs:
        print("no fixed-rate ledgers")
        return
    if args.only in ("all", "trajectories"):
        plot_trajectories(runs, args.out)
    if args.only in ("all", "frontiers"):
        plot_all_frontiers(runs, args.out)


if __name__ == "__main__":
    main()
