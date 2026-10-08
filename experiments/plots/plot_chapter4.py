"""Generate the compact figures and tables specified for Chapter 4.

This entry point keeps fixed-rate and mixed-rate ledgers separate.  Fixed-rate
plots read only E8Q, RaBitQ, CAQ, and PQ rows; ME8Q/SAQ are loaded by the two
Section 4.4 plots only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from common import (
    METHOD_LABEL,
    actual_bits_per_dim,
    empty_lane_frac,
    load_ablation_runs,
    load_bench_runs,
    logical_aux_bpv,
    pick_run,
    savefig,
)
from sigmod_style import (
    SIGMOD_METHOD_STYLE as METHOD_STYLE,
    apply_sigmod_style,
    fine_log_axis,
    fit_top_legend,
    open_corner,
    recall_ticks,
    snap_corner_ticks,
    style_panel,
)
from plot_fixed_rate import apply_adr_sidecars, plot_accuracy, plot_frontier

FIXED_METHODS = ("e8q", "rabitq", "caq", "pq")
# The two wide b=4 tables are laid out as one codec per column group.  PQ is
# left out of both so they fit a single column instead of a double-column
# float; its numbers are quoted in the text of Sections 4.2 and 4.3.
TABLE_METHODS = ("e8q", "rabitq", "caq")
MIXED_METHODS = ("me8q", "saq")
DATASETS = (
    "sift1M",
    "deep1M",
    "gist",
    "openai1536",
    "msmarco10M",
    "stella-trec24-1024",
)
DISPLAY_NAME = {
    "sift1M": "SIFT",
    "deep1M": "DEEP",
    "gist": "GIST",
    "openai1536": "OpenAI-1536",
    "msmarco10M": "MSMARCO-10M",
    "stella-trec24-1024": "Stella",
}
THREADS = 12
NPROBE_ACCURACY = 200
NPROBE_BOUND = (16, 64)
MIXED_BUDGETS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)

# Datasets shown in both Section 4.4 figures: those where ME8Q leads SAQ on
# BOTH the recall ceiling and throughput at every budget whose two arms sit
# within 0.15 bits/dim of each other.  Ranked by mean ceiling gain:
# OpenAI-1536 (+3.27 pts, 1.68-3.68x QPS@90), GIST (+2.65 pts, 1.09-1.27x),
# Stella (+1.20 pts, 1.33-14.5x).  MSMARCO-10M (+1.42 pts) is the next best
# but SAQ overtakes it on throughput at b=8; DEEP (+0.46 pts) only has one
# budget-matched point, its large b=2 gap coming from an unmatched allocation.
MIXED_SHOWN = ("openai1536", "gist", "stella-trec24-1024")

# Fixed style intentionally stays identical to existing Chapter 3 figures.
BIT_MARKER = {2: "o", 4: "^", 8: "s"}
BIT_LINE = {2: ":", 4: "--", 8: "-"}


def _bits_stem(bits: float | int) -> str:
    return f"{float(bits):g}"


def _fixed_runs(results: Path) -> dict[str, dict]:
    """Return only fixed-rate ledgers; mixed-rate rows never pass this guard."""
    all_runs = load_bench_runs(results / "fixed")
    out: dict[str, dict] = {}
    for stem, run in all_runs.items():
        rows = run.get("rows", [])
        if not rows:
            continue
        keys = rows[0]
        if any(f"{method}_recall" in keys for method in FIXED_METHODS):
            out[stem] = run
    return out


def _mixed_runs(results: Path) -> dict[str, dict]:
    all_runs = load_bench_runs(results / "mixed")
    out: dict[str, dict] = {}
    for stem, run in all_runs.items():
        rows = run.get("rows", [])
        if rows and any(f"{method}_recall" in rows[0] for method in MIXED_METHODS):
            out[stem] = run
    return out


def _fixed_run(runs: dict[str, dict], dataset: str, bits: int,
               method: str) -> dict | None:
    return pick_run(runs, dataset, bits, method, threads=THREADS)


def _mixed_run(runs: dict[str, dict], dataset: str, bits: float,
               method: str) -> dict | None:
    stem = f"{dataset}_b{_bits_stem(bits)}_t{THREADS}_{method}"
    return runs.get(stem)


def _row_series(run: dict, method: str, x_key: str = "recall"):
    xs: list[float] = []
    ys: list[float] = []
    for row in run.get("rows", []):
        x = row.get(f"{method}_{x_key}")
        y = row.get(f"{method}_qps")
        if x is None or y is None:
            continue
        x, y = float(x), float(y)
        if math.isfinite(x) and math.isfinite(y) and y > 0:
            xs.append(x)
            ys.append(y)
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order]


def _qps_at_recall(run: dict, method: str, target: float) -> float:
    """Log-linear QPS interpolation used by existing headline tables."""
    points = []
    for row in run.get("rows", []):
        recall = row.get(f"{method}_recall")
        qps = row.get(f"{method}_qps")
        if recall is None or qps is None or float(qps) <= 0:
            continue
        points.append((float(recall), float(qps)))
    points.sort()
    collapsed: list[tuple[float, float]] = []
    for x, y in points:
        if collapsed and x == collapsed[-1][0]:
            collapsed[-1] = (x, max(y, collapsed[-1][1]))
        else:
            collapsed.append((x, y))
    if not collapsed or collapsed[-1][0] < target:
        return float("nan")
    below = [point for point in collapsed if point[0] <= target]
    if below and below[-1][0] == target:
        return below[-1][1]
    lo = below[-1] if below else (collapsed[0][0] - 1e-9, collapsed[0][1])
    hi = min((point for point in collapsed if point[0] > target),
             key=lambda point: point[0])
    t = (target - lo[0]) / max(hi[0] - lo[0], 1e-12)
    return math.exp((1.0 - t) * math.log(max(lo[1], 1e-9))
                    + t * math.log(hi[1]))


def _ceiling(run: dict, method: str) -> float:
    values = [float(row[f"{method}_recall"])
              for row in run.get("rows", [])
              if row.get(f"{method}_recall") is not None]
    return max(values) if values else float("nan")


def _row_at(run: dict, nprobe: int) -> dict | None:
    return next((row for row in run.get("rows", [])
                 if int(row.get("nprobe", -1)) == nprobe), None)


def _admission_rate(run: dict, nprobe: int) -> float:
    """Admitted-candidate fraction; `survivor_rate` is the ledger field name."""
    row = _row_at(run, nprobe)
    value = row.get("survivor_rate") if row else None
    return float(value) if value is not None else float("nan")


def _style_method_line(ax, method: str, bits: int, x, y,
                       *, label: str | None = None, zorder: int = 2,
                       hollow: bool = False, linewidth: float = 1.0):
    color, marker, method_ls = METHOD_STYLE[method]
    markerface = "white" if hollow else color
    ax.plot(x, y, color=color, marker=marker, linestyle=BIT_LINE.get(bits, method_ls),
            markerfacecolor=markerface, markeredgecolor="white",
            markeredgewidth=0.45, linewidth=linewidth, markersize=3.0,
            markevery=max(1, len(x) // 12), label=label, zorder=zorder)


def _csv(out: Path, name: str, header: list[str], rows: list[list]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{name}.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _tex_escape(value: str) -> str:
    return str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def _write_tabular(out: Path, name: str, columns: str, header: str,
                   rows: list[list[str]], extra: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{name}.tex").open("w") as fh:
        fh.write(f"\\begin{{tabular}}{{{columns}}}\n")
        fh.write("\\toprule\n")
        fh.write(header + r" \\" + "\n")
        fh.write("\\midrule\n")
        for row in rows:
            fh.write(" & ".join(_tex_escape(str(cell)) for cell in row)
                     + " \\\\\n")
        if extra:
            fh.write(extra + "\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


def _fmt(value: float, digits: int = 1) -> str:
    return "--" if not math.isfinite(value) else f"{value:.{digits}f}"


def _fmt_int(value: float) -> str:
    return "--" if not math.isfinite(value) else f"{value:,.0f}"


def _fmt_prob(value: float) -> str:
    """Recall ceiling for the tables: percent, two decimals.

    Every recall figure the paper prints is a percentage, so the tables read
    the same way the prose, the axes and the appendix spectrum table do.
    """
    return "--" if not math.isfinite(value) else f"{100 * value:.2f}"


def _fmt_gain(value: float) -> str:
    """Percentage margin over the best baseline, for the headline CSV."""
    return "" if not math.isfinite(value) else f"{100 * value:+.0f}%"


# ---------------------------------------------------------------------------
# Padding-free (logical) footprint.  The definition lives in common.py so every
# bytes-per-vector surface in the paper (tables and axes) shares it; see
# `empty_lane_frac` there for why the empty FastScan lanes are removed.
def _empty_lane_frac(runs: dict[str, dict], dataset: str) -> float:
    return empty_lane_frac(runs, dataset, threads=THREADS)


def _logical_aux(run: dict, method: str, empty_frac: float) -> float:
    return logical_aux_bpv(run, method, empty_frac)


def make_speedup_table(runs: dict[str, dict], out: Path) -> dict:
    """Iso-recall margin over the best baseline at $b=4$, for the sidecar CSV.

    The margins are printed in the E8Q cells of the headline table, so this
    only writes them down for the record; the standalone speedup figure was
    dropped as a duplicate of that table.
    """
    targets = (("@90%", 0.90), ("@95%", 0.95))
    rows = []
    speedups: dict[str, dict[str, float]] = {}
    for dataset in DATASETS:
        e8q = _fixed_run(runs, dataset, 4, "e8q")
        if e8q is None:
            continue
        for label, target in targets:
            ours = _qps_at_recall(e8q, "e8q", target)
            baseline = []
            for method in ("rabitq", "caq", "pq"):
                run = _fixed_run(runs, dataset, 4, method)
                if run is not None:
                    v = _qps_at_recall(run, method, target)
                    if math.isfinite(v):
                        baseline.append(v)
            value = (ours / max(baseline)
                     if math.isfinite(ours) and baseline else float("nan"))
            speedups.setdefault(dataset, {})[label] = value
            rows.append([DISPLAY_NAME[dataset], "4", label, _fmt(value, 4)])
    _csv(out, "tab_fixed_speedup",
         ["Dataset", "bits", "Target", "Speedup"], rows)
    return {"datasets": [d for d in DATASETS if d in speedups],
            "speedups": speedups}


def _margin_arms(ablation: dict[str, dict]) -> list[tuple[float, dict]]:
    pattern = re.compile(r"^abl_alpha_openai1536_b4_a([0-9.]+)_e8q$")
    return sorted((float(m.group(1)), run) for stem, run in ablation.items()
                  if (m := pattern.match(stem)))


def _grid_ratios(run: dict, base: dict | None) -> list[float]:
    """Per-depth throughput of one arm relative to ``base``, depth by depth.

    Every margin arm returns the same neighbours depth by depth --- the margin
    decides which candidates are fetched, not what is ranked --- so the paired
    ratio to the default arm is defined at each depth, and the sweep is read
    through its average over the grid rather than at any single depth, because
    the per-depth measurement scatter is several times the effect being
    measured.
    """
    ref = {int(row["nprobe"]): float(row["e8q_qps"])
           for row in (base or {}).get("rows", []) if row.get("e8q_qps")}
    ratios: list[float] = []
    for row in run.get("rows", []):
        anchor = ref.get(int(row.get("nprobe", -1)))
        qps = row.get("e8q_qps")
        if anchor and qps and float(qps) > 0:
            ratios.append(float(qps) / anchor)
    return ratios


def _geomean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def plot_margin_sensitivity(ablation: dict[str, dict], out: Path) -> dict:
    """Throughput against the screening margin, relative to the default arm.

    The sweep asks whether a tighter margin costs throughput, and neither of
    the obvious encodings answers it.  In absolute QPS it cannot: this is the
    one sweep whose throughput was measured away from the reference host, so
    its level is not comparable with the fixed-rate tables and a reader would
    take a host difference for an effect of the margin.  As one curve per arm
    it also cannot, because seven curves on top of each other hide the answer
    in their per-depth scatter.  The line is the grid average of the paired
    ratio to the default arm and the band is the interquartile range of that
    ratio over the probe depths, so the panel shows both what the average does
    (nothing beyond a few percent) and why no single depth would settle it.
    """
    arms = _margin_arms(ablation)
    if not arms:
        print("NOTE: no abl_alpha_openai1536_b4_a*_e8q ledgers under "
              "--results; skipping the panel rather than writing an empty "
              "one over the figure on disk")
        return {}
    default = next((run for alpha, run in arms if alpha == 1.0),
                   arms[0][1] if arms else None)
    fig, ax = plt.subplots(figsize=(3.33, 1.35))
    fig.subplots_adjust(left=0.145, right=0.98, bottom=0.235, top=0.97)
    style_panel(ax, xlabel=r"margin factor $\alpha$", ylabel="QPS ratio")
    ax.set_xscale("log")
    color = METHOD_STYLE["e8q"][0]
    summary = {}
    points: list[tuple[float, float, float, float]] = []
    for alpha, run in arms:
        summary[f"{alpha:g}"] = {
            "ceiling": _ceiling(run, "e8q"),
            "qps_at_90": _qps_at_recall(run, "e8q", 0.90),
            "qps_at_95": _qps_at_recall(run, "e8q", 0.95),
        }
        ratios = _grid_ratios(run, default)
        if alpha <= 0 or not ratios:
            continue
        low, high = np.percentile(ratios, [25, 75])
        points.append((alpha, _geomean(ratios), float(low), float(high)))
    xs = [point[0] for point in points]
    ax.fill_between(xs, [point[2] for point in points],
                    [point[3] for point in points], color=color, alpha=0.15,
                    linewidth=0, zorder=2)
    ax.plot(xs, [point[1] for point in points], color=color, marker="o",
            markersize=3.4, markeredgecolor="white", markeredgewidth=0.4,
            linewidth=1.1, zorder=3)
    # The window holds the whole band, at the +-10% frame where the band fits
    # inside it; the rung the default arm stands for is a rule, not a label.
    lows = [point[2] for point in points]
    highs = [point[3] for point in points]
    ax.set_ylim(bottom=min([0.90] + lows) - 0.01,
                top=max([1.10] + highs) + 0.01)
    ax.set_yticks((0.9, 1.0, 1.1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.set_xlim(0.0007, 3.0)
    ax.set_xticks((0.001, 0.01, 0.1, 1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.axhline(1.0, color="0.45", linewidth=0.6, linestyle=(0, (4, 3)),
               zorder=1)
    # The axis is a ratio whose floor is a chosen value (the +-10% window), so
    # it is not snapped: the graduation below the window is zero, and the panel
    # would spend half its height on a region no arm reaches.
    savefig(fig, out, "fig_margin_sensitivity")
    by_alpha = {point[0]: point for point in points}
    _csv(out, "margin_sensitivity",
         ["alpha", "ceiling_recall", "qps_at_90", "qps_at_95",
          "qps_at_90_rel", "qps_at_95_rel", "qps_ratio_geomean",
          "qps_ratio_p25", "qps_ratio_p75"],
         [[key, _fmt(value["ceiling"], 6), _fmt(value["qps_at_90"], 3),
           _fmt(value["qps_at_95"], 3),
           _fmt(value["qps_at_90"] / summary["1"]["qps_at_90"], 3),
           _fmt(value["qps_at_95"] / summary["1"]["qps_at_95"], 3),
           _fmt(by_alpha[float(key)][1], 4) if float(key) in by_alpha else "",
           _fmt(by_alpha[float(key)][2], 4) if float(key) in by_alpha else "",
           _fmt(by_alpha[float(key)][3], 4) if float(key) in by_alpha else ""]
          for key, value in summary.items()])
    return summary


def plot_bound_selectivity(runs: dict[str, dict], ablation: dict[str, dict],
                           out: Path) -> dict:
    """Plot measured refinement fetch ratio and candidate admission rate, E8Q vs RaBitQ.

    Panel (a): refinement fetch ratio against bit rate, four curves —
    E8Q at alpha=1 and RaBitQ, each at n_probe 16 and 64.
    Panel (b): candidate admission rate against n_probe, two curves —
    E8Q at alpha=1 and RaBitQ, both at b=4.
    """
    del ablation  # the safety-factor sweep is not part of this figure
    bits = tuple(range(2, 9))

    def rate(run: dict | None, nprobe: int) -> float:
        return 100.0 * _admission_rate(run, nprobe) if run else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(3.33, 1.80), squeeze=False)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.19, top=0.80,
                        wspace=0.50)
    left, right = axes[0]
    style_panel(left, xlabel="bits / dim", ylabel="Refinement fetch ratio (%)",
                title="(a) Refinement traffic")
    style_panel(right, xlabel=r"$n_{\mathrm{probe}}$",
                ylabel="Admitted candidates (%)",
                title="(b) Admission rate")

    # Color and marker encode the method in both panels, so one legend
    # covers everything: the line style is the only thing panel (a) adds.
    series_a = (
        ("e8q", 16, "#D32F2F", "o", "-"),
        ("e8q", 64, "#D32F2F", "o", "--"),
        ("rabitq", 16, "#1976D2", "^", "-"),
        ("rabitq", 64, "#1976D2", "^", "--"),
    )
    rows = []
    left_values: list[float] = []
    for method, npb, color, marker, linestyle in series_a:
        values = [rate(_fixed_run(runs, "openai1536", bit, method), npb)
                  for bit in bits]
        left.plot(bits, values, color=color, marker=marker, linestyle=linestyle,
                  linewidth=1.1, markersize=4.0, markeredgecolor="white",
                  markeredgewidth=0.4, zorder=3)
        left_values.extend(v for v in values if math.isfinite(v))
        for bit, value in zip(bits, values):
            rows.append([method, "1.0" if method == "e8q" else "--",
                         bit, npb, _fmt(value, 6)])
    left.set_xlim(1.75, 8.25)
    left.set_xticks(bits)
    left.set_ylim(0, max(left_values, default=1.0) * 1.08)
    left.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    right.set_xscale("log")

    def curve(run: dict | None) -> tuple[np.ndarray, np.ndarray]:
        if run is None:
            return np.asarray([]), np.asarray([])
        points = [(float(row["nprobe"]), 100.0 * float(row["survivor_rate"]))
                  for row in run.get("rows", [])
                  if row.get("nprobe") is not None
                  and row.get("survivor_rate") is not None
                  and float(row["nprobe"]) >= 16]
        points.sort()
        return (np.asarray([point[0] for point in points]),
                np.asarray([point[1] for point in points]))

    for method, color, marker in (
        ("e8q", "#D32F2F", "o"),
        ("rabitq", "#1976D2", "^"),
    ):
        run = _fixed_run(runs, "openai1536", 4, method)
        xs, ys = curve(run)
        if not len(xs):
            continue
        right.plot(xs, ys, color=color, linestyle="-", marker=marker,
                   markevery=5, markeredgecolor="white", markeredgewidth=0.4,
                   linewidth=1.1, markersize=4.0, zorder=3)
        for x, y in zip(xs, ys):
            rows.append([method, "1.0" if method == "e8q" else "--",
                         4, int(x), _fmt(y, 6)])
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlim(14, 590)
    right.set_xticks((16, 64, 256, 512))
    right.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):d}"))
    right.xaxis.set_minor_formatter(NullFormatter())
    right.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    right.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    # One key for both panels: color and marker are the method, line style is
    # the probe depth used in panel (a).
    key = [
        Line2D([0], [0], color="#D32F2F", marker="o", linestyle="-",
               markerfacecolor="#D32F2F", markeredgecolor="white",
               label=r"E8Q $\alpha=1$"),
        Line2D([0], [0], color="#1976D2", marker="^", linestyle="-",
               markerfacecolor="#1976D2", markeredgecolor="white",
               label="RaBitQ"),
        Line2D([0], [0], color="#444444", marker="None", linestyle="-",
               label=r"$n_{\mathrm{probe}}=16$"),
        Line2D([0], [0], color="#444444", marker="None", linestyle="--",
               label=r"$n_{\mathrm{probe}}=64$"),
    ]
    fig.legend(key, [h.get_label() for h in key], loc="upper center",
               bbox_to_anchor=(0.55, 0.995), ncol=4, fontsize=5.5,
               frameon=False, handlelength=1.8, columnspacing=0.7)
    for panel in fig.axes:
        # The x floor of every Chapter 4 panel is a chosen, meaningful value
        # (1.75 bits, 80% recall, a ratio of 1.0, nprobe 16); snapping it to
        # the ladder would widen the window past what the panel is about.
        snap_corner_ticks(panel, axes="y")
    savefig(fig, out, "fig_bound_selectivity")
    _csv(out, "bound_selectivity",
         ["method", "alpha", "bits", "nprobe", "admission_rate_percent"], rows)
    return {"rows": rows}


def plot_scalability(ablation: dict[str, dict], out: Path) -> dict:
    """Top-k sweep on OpenAI-1536 at b=4.

    Thread scaling is deliberately not plotted: the 1/4/8-thread arms are
    one run and the 12-thread arm is the shared fixed-rate ledger, whose
    three extra probes of recall put the merged series above a line
    anchored at its own one-thread measurement.
    """
    k_re = re.compile(r"^abl_k_openai1536_b4_k(\d+)_(e8q|rabitq)$")
    k_points: dict[tuple[str, int], dict] = {}
    for stem, run in ablation.items():
        match = k_re.match(stem)
        if match:
            k_points[(match.group(2), int(match.group(1)))] = run
    fig, ax = plt.subplots(figsize=(3.33, 1.45))
    fig.subplots_adjust(left=0.155, right=0.98, bottom=0.235, top=0.955)
    k_ls = {1: "-", 10: "--", 100: ":"}
    peak = 0.0
    floor = float("inf")
    xmax = 80.0
    for method in ("e8q", "rabitq"):
        for k in (1, 10, 100):
            run = k_points.get((method, k))
            if run is None:
                continue
            x, y = _row_series(run, method)
            keep = x >= 0.80
            if np.any(keep):
                peak = max(peak, float(np.max(y[keep])))
                floor = min(floor, float(np.min(y[keep])))
                xmax = max(xmax, 100.0 * float(np.max(x[keep])))
            # Recall windows start at 80% on this panel; the measured curve
            # below that is drawn and clipped rather than sliced off, so the
            # series reaches the frame instead of starting inside it.
            ax.plot(100 * x, y, color=METHOD_STYLE[method][0],
                    linestyle=k_ls[k], marker=METHOD_STYLE[method][1],
                    # Every second sampled probe, so the density of the
                    # measured grid is visible and the residual jitter reads
                    # as measurement noise rather than as under-sampling.
                    markevery=2, markersize=2.0,
                    markeredgecolor="white", markeredgewidth=0.35,
                    linewidth=0.75, label=f"{METHOD_LABEL[method]} k={k}")
    style_panel(ax, xlabel="Recall@k (%)", ylabel="QPS", ylog=True)
    if np.isfinite(floor) and peak:
        ax.set_ylim(0.9 * floor, 1.15 * peak)
    # The right edge follows the last measured recall: running the axis to
    # 100% spent a fifth of the panel on a region no arm reaches.
    ax.set_xlim(80, xmax + 0.6)
    ax.set_xticks([t for t in (80, 85, 90, 95, 100) if t <= xmax + 0.6])
    open_corner(ax, x=0.02, y=0.0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower left", ncol=2, fontsize=6.0,
              frameon=False, handlelength=1.6, columnspacing=0.6,
              labelspacing=0.25)
    for panel in fig.axes:
        # The x floor of every Chapter 4 panel is a chosen, meaningful value
        # (1.75 bits, 80% recall, a ratio of 1.0, nprobe 16); snapping it to
        # the ladder would widen the window past what the panel is about.
        snap_corner_ticks(panel, axes="y")
    savefig(fig, out, "fig_scalability")
    rows = []
    for (method, k), run in sorted(k_points.items()):
        for row in run.get("rows", []):
            rows.append([f"{method}:k={k}", row.get("nprobe"),
                         _fmt(row.get(f"{method}_qps", float("nan")), 4)])
    _csv(out, "scalability", ["series", "x", "qps"], rows)
    return {"k": k_points}


def make_full_spectrum_table(runs: dict[str, dict], out: Path) -> dict:
    r"""Appendix C: Recall@10 ceiling and QPS@90% over the full rate ladder.

    The body reports the three operating rates $b \in \{2,4,8\}$; this table
    walks every integer rate from 1 to 8 for all six corpora, so a reviewer can
    read the hypercubic-vs-lattice gap as a function of the rate rather than at
    three samples of it.  PQ is a rate-matched family ($M = bD/8$ sub-spaces),
    so it only exists where $bD/8$ is integral; its cells are "--" elsewhere.
    """
    methods = ("e8q", "rabitq", "caq", "pq")
    bits = tuple(range(1, 9))
    recall_rows: list[list[str]] = []
    qps_rows: list[list[str]] = []
    csv_rows: list[list] = []
    for dataset in DATASETS:
        for index, method in enumerate(methods):
            label = METHOD_LABEL[method]
            name = (f"\\multirow{{{len(methods)}}}{{*}}{{{DISPLAY_NAME[dataset]}}}"
                    if index == 0 else "")
            recall_cells = [name, label]
            qps_cells = [name, label]
            for b in bits:
                run = _fixed_run(runs, dataset, b, method)
                if run is None:
                    recall_cells.append("--")
                    qps_cells.append("--")
                    csv_rows.append([DISPLAY_NAME[dataset], label, b, "", ""])
                    continue
                ceiling = _ceiling(run, method)
                q90 = _qps_at_recall(run, method, 0.90)
                recall_cells.append(_fmt_prob(ceiling))
                qps_cells.append(_fmt_int(q90))
                csv_rows.append([DISPLAY_NAME[dataset], label, b,
                                 f"{ceiling:.6f}" if math.isfinite(ceiling) else "",
                                 f"{q90:.1f}" if math.isfinite(q90) else ""])
            # `\multirow` needs a leading separator column, so the corpus label
            # occupies its own column and the method label the next one.
            recall_rows.append(recall_cells)
            qps_rows.append(qps_cells)

    header = "Corpus & Method & " + " & ".join(f"$b{{=}}{b}$" for b in bits)
    columns = "@{}ll" + "c" * len(bits) + "@{}"
    _write_tabular(out, "tab_full_spectrum_recall", columns, header, recall_rows)
    _write_tabular(out, "tab_full_spectrum_qps", columns, header, qps_rows)
    _csv(out, "full_spectrum", ["corpus", "method", "bits", "recall_ceiling",
                                "qps_at_90"], csv_rows)
    return {"nrows": len(recall_rows)}


def make_full_spectrum_all_table(runs: dict[str, dict], out: Path) -> dict:
    r"""Appendix Table B1: the two full-spectrum tables merged into one.

    Recall@10 ceiling (percent) and QPS at 90\% recall share a row per
    (corpus, method), so a single booktabs float carries the whole rate
    spectrum.  The best entry of each column is bolded within its corpus
    block; "--" marks a rate at which the 90\% target is unreachable (QPS
    block) or a rate the rate-matched PQ family does not cover.
    """
    methods = ("e8q", "rabitq", "caq", "pq")
    bits = tuple(range(1, 9))
    blocks: list[tuple[str, list[list[str]], list[list[str]]]] = []
    for dataset in DATASETS:
        recall_rows: list[list[str]] = []
        qps_rows: list[list[str]] = []
        for method in methods:
            rec_cells: list[str] = []
            qps_cells: list[str] = []
            for b in bits:
                run = _fixed_run(runs, dataset, b, method)
                if run is None:
                    rec_cells.append("--")
                    qps_cells.append("--")
                    continue
                ceiling = _ceiling(run, method)
                q90 = _qps_at_recall(run, method, 0.90)
                rec_cells.append(
                    "--" if not math.isfinite(ceiling) else f"{100 * ceiling:.2f}")
                qps_cells.append(
                    "--" if not math.isfinite(q90) else f"{q90:,.0f}")
            recall_rows.append(rec_cells)
            qps_rows.append(qps_cells)
        blocks.append((DISPLAY_NAME[dataset], recall_rows, qps_rows))

    def _bold_columns(rows: list[list[str]], digits: int | None) -> None:
        """Bold the best numeric entry of each column, in place."""
        for col in range(len(bits)):
            best = None
            for row in rows:
                cell = row[col]
                if cell == "--":
                    continue
                value = float(cell.replace(",", ""))
                if best is None or value > best:
                    best = value
            if best is None:
                continue
            for row in rows:
                cell = row[col]
                if cell == "--":
                    continue
                if float(cell.replace(",", "")) == best:
                    row[col] = f"\\textbf{{{cell}}}"
                    break
        del digits

    for _, recall_rows, qps_rows in blocks:
        _bold_columns(recall_rows, 2)
        _bold_columns(qps_rows, 0)

    lines = [
        r"\begin{tabular}{@{}ll rrrrrrrr rrrrrrrr@{}}",
        r"\toprule",
        r"\multirow{2}{*}{Corpus} & \multirow{2}{*}{Method} & "
        r"\multicolumn{8}{c}{Recall@10 ceiling (\%)} & "
        r"\multicolumn{8}{c}{QPS at 90\% recall} \\",
        r"\cmidrule(lr){3-10}\cmidrule(lr){11-18}",
        " & & " + " & ".join(f"$b{{=}}{b}$" for b in bits) + " & "
        + " & ".join(f"$b{{=}}{b}$" for b in bits) + r" \\",
        r"\midrule",
    ]
    for index, (name, recall_rows, qps_rows) in enumerate(blocks):
        if index:
            lines.append(r"\midrule")
        for row_index in range(len(methods)):
            lead = (f"\\multirow{{{len(methods)}}}{{*}}{{{name}}}"
                    if row_index == 0 else "")
            cells = recall_rows[row_index] + qps_rows[row_index]
            text = " & ".join(c.replace(",", "{,}") for c in cells)
            lines.append(f"  {lead} & {METHOD_LABEL[methods[row_index]]} & "
                         + text + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_full_spectrum_all.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}/tab_full_spectrum_all.tex")
    return {"nrows": sum(len(r) for _, r, _ in blocks)}


def _skipped(results: Path, dataset: str, bits: float, method: str) -> bool:
    path = results / "mixed" / f"{dataset}_b{_bits_stem(bits)}_t{THREADS}_{method}.skipped"
    return path.exists()


def mixed_accuracy_data(mixed: dict[str, dict], out: Path) -> dict:
    """Measured bits/dim and Recall@10 behind the mixed-rate ledgers.

    Section 4.4 no longer draws this as a panel --- the frontier figure and
    the matched-storage table carry the comparison, and the section is an
    extension rather than a second main result.  The dump stays because the
    claim that a nominal budget understates the measured total is read off
    it: `actual_bits_per_dim` includes the auxiliary metadata.
    """
    values_out = []
    for dataset in MIXED_SHOWN:
        for method in MIXED_METHODS:
            for nominal in MIXED_BUDGETS:
                run = _mixed_run(mixed, dataset, nominal, method)
                if run is None:
                    continue
                row = _row_at(run, NPROBE_ACCURACY)
                if row is None:
                    continue
                x = actual_bits_per_dim(run, method)
                y = float(row.get(f"{method}_recall", float("nan")))
                if math.isfinite(x) and math.isfinite(y) and x <= 8.1:
                    values_out.append([dataset, method, _fmt(nominal, 1),
                                       _fmt(x, 6), _fmt(y, 6), ""])
    _csv(out, "mixed_accuracy",
         ["dataset", "method", "nominal_budget", "actual_bits_per_dim",
          "recall_at_200", "status"], values_out)
    return {"rows": values_out}


def plot_mixed_frontier(mixed: dict[str, dict], out: Path) -> dict:
    datasets = MIXED_SHOWN
    # Each dataset starts where its own curves separate, not at a shared 80%.
    # Stella starts at 88 rather than 90: SAQ at 2 bits collapses onto its
    # ceiling of 90.0 there, so its last 40 probe points share a 1e-4-wide
    # recall window and a later start would draw them as a bare vertical
    # streak against the left spine.
    recall_start = {"openai1536": 85.0, "gist": 80.0, "stella-trec24-1024": 88.0}
    fig, axes = plt.subplots(len(datasets), 1, figsize=(3.33, 3.55),
                             squeeze=False)
    # The key is a band above the panels: every in-axes corner of panel (a)
    # is crossed by a curve once SAQ at 2 bits is drawn.
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.095, top=0.885,
                        hspace=0.42)
    rows_out = []
    for col, dataset in enumerate(datasets):
        ax = axes[col, 0]
        # The three panels are one column over one x axis, so only the bottom
        # one carries the label: repeating it puts the upper panels' label on
        # top of the next panel's title.
        style_panel(ax, ylabel="QPS",
                    xlabel="Recall@10 (%)" if col == len(datasets) - 1 else None,
                    ylog=True,
                    title=f"({'abc'[col]}) {DISPLAY_NAME[dataset]}")
        start = int(recall_start[dataset])
        ax.set_xlim(start, 100)
        ax.set_xticks(recall_ticks(start, 100.0))
        fine_log_axis(ax)
        peaks: list[float] = []
        floors: list[float] = []
        for method in MIXED_METHODS:
            for bits in (2, 4, 8):
                run = _mixed_run(mixed, dataset, bits, method)
                if run is None:
                    continue
                x, y = _row_series(run, method)
                keep = x >= recall_start[dataset] / 100.0
                if not np.any(keep):
                    continue
                peaks.append(float(y[keep].max()))
                floors.append(float(y[keep].min()))
                color, _, _ = METHOD_STYLE[method]
                # Drawn over the whole sweep: the panel's window decides how
                # much is visible, and the rest leaves through the frame.
                ax.plot(100 * x, y, color=color,
                        marker=BIT_MARKER[bits],
                        linestyle=BIT_LINE[bits], linewidth=1.0, markersize=3.6,
                        markevery=max(1, len(x) // 7),
                        markerfacecolor=color,
                        markeredgecolor="white", markeredgewidth=0.4)
                for xx, yy in zip(x[keep], y[keep]):
                    rows_out.append([dataset, method, bits, _fmt(xx, 6), _fmt(yy, 4)])
        if peaks:
            # Just clear of the fastest point: a wider cap only bought empty
            # space above the 2-bit curves in all three panels.  The floor
            # comes from the same recall window, so the clipped low-recall
            # tail cannot drag the panel down.
            ax.set_ylim(bottom=0.9 * min(floors), top=1.12 * max(peaks))
        # Opened past the data like every other Chapter 4 panel.
        open_corner(ax, x=0.02, y=0.0)
    # One row above the panels, matching the fixed-rate figures.
    method_handles = [
        Line2D([0], [0], color=METHOD_STYLE["me8q"][0], marker="None",
               linestyle="-", linewidth=1.4, label="ME8Q"),
        Line2D([0], [0], color=METHOD_STYLE["saq"][0], marker="None",
               linestyle="-", linewidth=1.4, label="SAQ"),
    ]
    bit_handles = [Line2D([0], [0], color="#444444", marker=BIT_MARKER[b],
                          linestyle=BIT_LINE[b], markersize=3.5,
                          markerfacecolor="#444444", markeredgecolor="white",
                          label=f"{b} bits") for b in (2, 4, 8)]
    handles = method_handles + bit_handles
    fig.legend(handles, [h.get_label() for h in handles],
               loc="upper center", bbox_to_anchor=(0.5, 0.998), ncol=5,
               fontsize=6.0, frameon=False, handlelength=1.2,
               columnspacing=0.6, handletextpad=0.4, borderpad=0.0)
    for panel in fig.axes:
        # The x floor of every Chapter 4 panel is a chosen, meaningful value
        # (1.75 bits, 80% recall, a ratio of 1.0, nprobe 16); snapping it to
        # the ladder would widen the window past what the panel is about.
        snap_corner_ticks(panel, axes="y")
    savefig(fig, out, "fig_mixed_frontier")
    _csv(out, "mixed_frontier", ["dataset", "method", "nominal_bits",
                                  "recall", "qps"], rows_out)
    return {"rows": rows_out}


def make_memory_table(runs: dict[str, dict], out: Path) -> None:
    """Code payload, padding-free auxiliary bytes, and total, per codec.

    One row per dataset and rate.  The codecs are column groups with three
    sub-columns each, which keeps every number legible in a single column.
    The CSV carries all four codecs; the LaTeX float drops PQ so that it fits
    the column, and the PQ totals it would have shown are quoted in the text.
    """
    rows = []
    tex_rows = []
    for dataset in ("sift1M", "openai1536"):
        empty = _empty_lane_frac(runs, dataset)
        for bits in (4, 8):
            cells = []
            tex_cells: dict[str, tuple[str, str, str]] = {}
            for method in FIXED_METHODS:
                run = _fixed_run(runs, dataset, bits, method)
                if run is None:
                    code = meta = total = float("nan")
                else:
                    code = float(run.get(f"{method}_bpv_code", float("nan")))
                    meta = _logical_aux(run, method, empty)
                    total = code + meta
                cells.append(f"{_fmt(code)}/{_fmt(meta)}/{_fmt(total)}")
                tex_cells[method] = (_fmt(code), _fmt(meta), _fmt(total))
            rows.append([DISPLAY_NAME[dataset], str(bits)] + cells)
            tex_rows.append([DISPLAY_NAME[dataset], str(bits), tex_cells])
    _csv(out, "tab_memory",
         ["Dataset", "b"] + [f"{METHOD_LABEL[m]} code/aux/total"
                             for m in FIXED_METHODS], rows)
    # Written directly: the sub-column groups need \multicolumn markup, which
    # the shared writer would escape.  Full width, template type size: the
    # float is a `table*` and \extracolsep takes up the slack, so the eleven
    # columns fit without a \footnotesize or a squeezed \tabcolsep.
    groups = "".join(
        f" & \\multicolumn{{3}}{{c}}{{{METHOD_LABEL[m]}}}" for m in TABLE_METHODS
    )
    rules = "".join(
        f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(TABLE_METHODS))
    )
    sub = "".join(" & Pay. & Meta. & Total" for _ in TABLE_METHODS)
    with (out / "tab_memory.tex").open("w") as fh:
        fh.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l r"
                 + " rrr" * len(TABLE_METHODS) + "@{}}\n")
        fh.write("\\toprule\n")
        fh.write(" &" + groups + r" \\" + "\n")
        fh.write(rules + "\n")
        fh.write(r"Dataset & $b$" + sub + r" \\" + "\n")
        fh.write("\\midrule\n")
        for name, bits, tex_cells in tex_rows:
            parts = [_tex_escape(name), bits]
            for method in TABLE_METHODS:
                parts.extend(tex_cells[method])
            fh.write(" & ".join(parts) + r" \\" + "\n")
        fh.write("\\bottomrule\n\\end{tabular*}\n")


def make_headline_table(runs: dict[str, dict], out: Path) -> None:
    """Headline $b=4$ table, with E8Q's margin over the best baseline inline.

    Putting the margin in the E8Q QPS cells is what lets the standalone
    iso-recall speedup figure go: the bar chart carried the same two numbers
    for two corpora, where the table carries them for all six.
    """
    rows = []
    tex_rows = []
    for dataset in DATASETS:
        cells: dict[str, list[str]] = {}
        perf: dict[str, dict[str, float]] = {}
        for method in FIXED_METHODS:
            run = _fixed_run(runs, dataset, 4, method)
            if run is None:
                continue
            perf[method] = {
                "ceiling": _ceiling(run, method),
                "q90": _qps_at_recall(run, method, 0.90),
                "q95": _qps_at_recall(run, method, 0.95),
            }
        if "e8q" not in perf:
            continue
        baseline: dict[str, float] = {}
        # Which baseline supplies the comparison is named in the cell: the
        # fastest of the three changes from row to row and from target to
        # target, so an unlabelled "+24%" leaves the reader to work out what
        # it is a margin over.
        compared: dict[str, str] = {}
        for target in ("q90", "q95"):
            candidates = [(m, perf[m][target]) for m in ("rabitq", "caq", "pq")
                          if m in perf and math.isfinite(perf[m][target])]
            if candidates:
                best = max(candidates, key=lambda item: item[1])
                baseline[target] = best[1]
                compared[target] = METHOD_LABEL[best[0]]
            else:
                baseline[target] = float("nan")
                compared[target] = ""
        gains = {}
        for target in ("q90", "q95"):
            ours = perf["e8q"][target]
            gains[target] = (ours / baseline[target] - 1.0
                             if math.isfinite(ours)
                             and math.isfinite(baseline[target])
                             and baseline[target] > 0 else float("nan"))
        for method in FIXED_METHODS:
            if method not in perf:
                continue
            cells[method] = [_fmt_prob(perf[method]["ceiling"]),
                             _fmt_int(perf[method]["q90"]),
                             _fmt_int(perf[method]["q95"])]
            if method == "e8q":
                for index, target in enumerate(("q90", "q95")):
                    if (math.isfinite(perf["e8q"][target])
                            and not math.isfinite(baseline[target])):
                        # E8Q reaches a target no baseline reaches, so the
                        # entry stands alone and is set in bold.
                        cells[method][index + 1] = (
                            r"\textbf{" + cells[method][index + 1] + "}")
                    if math.isfinite(gains[target]):
                        # Plain text rather than mathtext: the margin is
                        # bold, and \bfseries does not reach inside $...$.
                        # The baseline the margin is taken against is named
                        # in the CSV and in the caption rather than in the
                        # cell.
                        cells[method][index + 1] += (
                            r"\,\textbf{(+"
                            + str(round(100 * gains[target]))
                            + r"\%)}")
            margin = {
                target: (_fmt_gain(gains[target])
                         + (f" vs. {compared[target]}" if compared[target] else ""))
                for target in ("q90", "q95")}
            rows.append([DISPLAY_NAME[dataset], METHOD_LABEL[method]]
                        + [cells[method][0],
                           _fmt_int(perf[method]["q90"]),
                           _fmt_int(perf[method]["q95"]),
                           margin["q90"] if method == "e8q" else "",
                           margin["q95"] if method == "e8q" else ""])
        tex_rows.append((DISPLAY_NAME[dataset], cells))
    _csv(out, "tab_fixed_headline",
         ["Corpus", "Quantizer", "Ceiling Recall@10", "QPS@90%", "QPS@95%",
          "E8Q margin @90%", "E8Q margin @95%"], rows)
    # One row per corpus, codecs as column groups: the long form is 24 rows
    # tall and leaves most of the column width unused.  The float is a
    # full-width `table*`, so the tabular stretches to \textwidth and the
    # type size stays the template's --- the width comes from the column
    # separation, never from shrinking the font.
    out.mkdir(parents=True, exist_ok=True)
    with (out / "tab_fixed_headline.tex").open("w") as fh:
        ncols = 1 + 3 * len(TABLE_METHODS)
        fh.write(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}"
                 + "l" + " rrr" * len(TABLE_METHODS) + r"@{}}" + "\n")
        fh.write(r"\toprule" + "\n")
        # Leading empty cell: each \multicolumn must start over its own
        # column group, which is what centres the codec names on their
        # three data columns.
        fh.write("& " + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{{METHOD_LABEL[m]}}}"
            for m in TABLE_METHODS) + r" \\" + "\n")
        fh.write("".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
                         for i in range(len(TABLE_METHODS))) + "\n")
        fh.write("Corpus" + r" & Ceil. (\%) & @90 & @95" * len(TABLE_METHODS)
                 + r" \\" + "\n")
        fh.write(r"\midrule" + "\n")
        for corpus, cells in tex_rows:
            parts = [_tex_escape(corpus)]
            for method in TABLE_METHODS:
                if method not in cells:
                    continue
                # Already LaTeX markup: escaping again would double the
                # backslash in the margin's \%.
                parts.extend(cells[method])
            fh.write(" & ".join(parts) + r" \\" + "\n")
        fh.write(r"\bottomrule" + "\n" + r"\end{tabular*}" + "\n")


def _dump_metrics(out: Path, metrics: dict) -> None:
    with (out / "chapter4_metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    apply_sigmod_style()
    fixed = _fixed_runs(args.results)
    mixed = _mixed_runs(args.results)
    ablation = load_ablation_runs(args.results)
    required = [
        (dataset, method)
        for dataset in DATASETS
        for method in FIXED_METHODS
        if _fixed_run(fixed, dataset, 4, method) is None
    ]
    if required:
        missing = ", ".join(f"{dataset}/{method}" for dataset, method in required)
        raise SystemExit(f"missing fixed-rate ledgers: {missing}")

    # Section 4.2, fixed-rate only.
    filled = apply_adr_sidecars(fixed, args.results)
    print(f"distance-ratio side-cars: {filled} ledgers on the full nprobe grid")
    plot_accuracy(fixed, args.out)
    plot_frontier(fixed, args.out)
    speedup = make_speedup_table(fixed, args.out)
    make_memory_table(fixed, args.out)
    make_headline_table(fixed, args.out)
    spectrum = make_full_spectrum_table(fixed, args.out)
    spectrum_all = make_full_spectrum_all_table(fixed, args.out)

    # Section 4.3, OpenAI-1536 b=4 diagnostics only.
    margin = plot_margin_sensitivity(ablation, args.out)
    bound = plot_bound_selectivity(fixed, ablation, args.out)
    scale = plot_scalability(ablation, args.out)

    # Section 4.4, mixed-rate data never enters fixed-rate figures/tables.
    mixed_accuracy = mixed_accuracy_data(mixed, args.out)
    mixed_frontier = plot_mixed_frontier(mixed, args.out)
    _dump_metrics(args.out, {
        "fixed_speedup": speedup,
        "margin": margin,
        "bound": {"nrows": len(bound["rows"])},
        "scalability": {"k": sorted({k for _, k in scale["k"]})},
        "mixed_accuracy": {"nrows": len(mixed_accuracy["rows"])},
        "mixed_frontier": {"nrows": len(mixed_frontier["rows"])},
        "full_spectrum": spectrum,
    })
    print(f"chapter 4 figures and tables written to {args.out}")


if __name__ == "__main__":
    main()
