"""Fixed-rate figures and summary table for Section 4.2.

Only E8Q, RaBitQ, CAQ, and PQ are loaded from the fixed-rate campaign.
Mixed-rate ledgers are deliberately excluded from this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DISPLAY,
    METHOD_LABEL,
    empty_lane_frac,
    load_bench_runs,
    logical_bpv,
    pick_run,
    savefig,
)
from sigmod_style import (  # noqa: E402
    SIGMOD_METHOD_STYLE as METHOD_STYLE,
    apply_sigmod_style,
    fine_log_axis as _fine_log_axis,
    fit_top_legend,
    open_corner,
    recall_ticks as _recall_ticks,
    snap_corner_ticks,
    style_panel,
)

DATASETS = (
    "sift1M",
    "deep1M",
    "gist",
    "openai1536",
    "msmarco10M",
    "stella-trec24-1024",
)
METHODS = ("e8q", "rabitq", "caq", "pq")
BITS = tuple(range(1, 9))
THREADS = 12
ACCURACY_NPROBE = 200
SUMMARY_BITS = 4
QPS_FLOOR_FRACTION = 0.9
# Panels are opened a little past their data on every side (sigmod_style's
# open_corner): a marker whose centre is the axis minimum is half clipped and
# makes the corner read as a sample, and a curve whose first point sits on the
# left spine looks cut off rather than started.
X_OPEN_FRAC = 0.02
Y_OPEN_FRAC = 0.02
# Distance-ratio panels read `<m>_adr`, the sum-ratio form README.md
# defines --- the same quantity the vendored SAQ/CAQ binary prints as its
# `ratio` column (on openai1536 b=4 at nprobe 64: CAQ ratio 1.0040, CAQ adr
# 1.00395, E8Q adr 1.00370), which is why the two are comparable.  The
# per-row `<m>_ratio` field most methods also carry is a DIFFERENT statistic
# --- mean squared L2 of the returned k over the squared L2 to the exact NN
# --- and it does not approach 1.0 even at perfect recall (E8Q reads 1.33-1.39
# where the ratio above reads 1.001), so it cannot share an axis with CAQ.
#
# E8Q and RaBitQ record `_adr` on the full grid only after the re-measurement
# in experiments/_adr_full.sh, which lands in results/adr_full/.  Older
# ledgers carry it at four Table-5 depths; the side-car overlays the full
# column when it is present, so the panels are complete without the frozen
# QPS being retimed.
ADR_DIRNAME = "adr_full"

DISPLAY_NAME = {
    "sift1M": "SIFT",
    "deep1M": "DEEP",
    "gist": "GIST",
    "openai1536": "OpenAI-1536",
    "msmarco10M": "MSMARCO-10M",
    "stella-trec24-1024": "Stella",
}


def _method_runs(results: Path) -> dict[str, dict]:
    """Load only fixed-rate method ledgers, even if directory has other arms.

    Accepts either a results root with a ``fixed/`` subdirectory (the
    ``results/{fixed,mixed,ablation}`` package layout) or a directory that
    already holds the ledgers.
    """
    root = results / "fixed" if (results / "fixed").is_dir() else results
    all_runs = load_bench_runs(root)
    fixed: dict[str, dict] = {}
    for stem, run in all_runs.items():
        rows = run.get("rows", [])
        if not rows:
            continue
        if any(f"{method}_recall" in rows[0] for method in METHODS):
            method = next(method for method in METHODS
                          if f"{method}_recall" in rows[0])
            fixed[stem] = run
            # Keep method variable explicit: this guard prevents a future
            # mixed-rate ledger with a fixed-rate-looking filename from being
            # accidentally plotted.
            if method not in METHODS:
                fixed.pop(stem, None)
    return fixed


def _available_bits(runs: dict[str, dict], dataset: str, method: str) -> list[int]:
    out: list[int] = []
    for bits in BITS:
        if pick_run(runs, dataset, bits, method, threads=THREADS) is not None:
            out.append(bits)
    return out


def _header(run: dict, method: str, key: str) -> float:
    value = run.get(f"{method}_{key}")
    return float(value) if value is not None else float("nan")


def _row_values(run: dict, method: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    x: list[float] = []
    y: list[float] = []
    for row in run.get("rows", []):
        xv = row.get(f"{method}_{key}")
        qps = row.get(f"{method}_qps")
        if xv is None or qps is None:
            continue
        xv, qps = float(xv), float(qps)
        if math.isfinite(xv) and math.isfinite(qps) and qps > 0:
            x.append(xv)
            y.append(qps)
    return np.asarray(x), np.asarray(y)


def _frontier_values(
    run: dict, method: str, key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x, QPS, Recall@10 and nprobe for one method ledger."""
    x: list[float] = []
    qps: list[float] = []
    recall: list[float] = []
    probes: list[float] = []
    for row in run.get("rows", []):
        xv = row.get(f"{method}_{key}")
        qv = row.get(f"{method}_qps")
        rv = row.get(f"{method}_recall")
        nv = row.get("nprobe")
        if xv is None or qv is None or rv is None or nv is None:
            continue
        xv, qv, rv, nv = float(xv), float(qv), float(rv), float(nv)
        if (math.isfinite(xv) and math.isfinite(qv)
                and math.isfinite(rv) and qv > 0):
            x.append(xv)
            qps.append(qv)
            recall.append(rv)
            probes.append(nv)
    return (np.asarray(x), np.asarray(qps), np.asarray(recall),
            np.asarray(probes))


def _adr_dir(results: Path) -> Path | None:
    """Locate the full-grid distance-ratio side-cars, if they exist."""
    here = Path(__file__).resolve().parents[2] / "results" / ADR_DIRNAME
    for candidate in (results / ADR_DIRNAME, results.parent / ADR_DIRNAME, here):
        if candidate.is_dir():
            return candidate
    return None


def apply_adr_sidecars(runs: dict[str, dict], results: Path) -> int:
    """Overlay the full-grid distance ratios onto the ledgers, in place.

    Each side-car holds one ledger's ``<m>_adr`` on every nprobe of the grid,
    re-measured under the same seed and checked against the frozen recall
    depth by depth (``experiments/adr_fill.py``), so the published QPS is
    untouched and the ratio still belongs to the plotted operating point.
    Returns how many ledgers were filled; a missing directory is not an
    error --- the panels then show whatever depths the ledger itself carries.
    """
    adr_dir = _adr_dir(results)
    if adr_dir is None:
        return 0
    filled = 0
    for path in sorted(adr_dir.glob("*.json")):
        run = runs.get(path.stem)
        if run is None:
            continue
        try:
            sidecar = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {path}: {exc}")
            continue
        method = sidecar.get("method")
        key = f"{method}_adr"
        by_probe = {
            int(row["nprobe"]): float(row[key])
            for row in sidecar.get("rows", [])
            if isinstance(row, dict) and row.get(key) is not None
        }
        if not by_probe:
            continue
        for row in run.get("rows", []):
            value = by_probe.get(int(row.get("nprobe", -1)))
            if value is not None:
                row[key] = value
        filled += 1
    return filled


def _qps_at_recall(run: dict, method: str, target: float) -> float:
    points = []
    for row in run.get("rows", []):
        recall = row.get(f"{method}_recall")
        qps = row.get(f"{method}_qps")
        if recall is None or qps is None or float(qps) <= 0:
            continue
        points.append((float(recall), float(qps)))
    points.sort()
    if not points or points[-1][0] < target:
        return float("nan")
    below = [point for point in points if point[0] <= target]
    if below and below[-1][0] == target:
        return below[-1][1]
    lo = below[-1] if below else (points[0][0] - 1e-9, points[0][1])
    hi = min((point for point in points if point[0] > target),
             key=lambda point: point[0])
    t = (target - lo[0]) / max(hi[0] - lo[0], 1e-12)
    return math.exp(math.log(max(lo[1], 1e-9)) * (1.0 - t)
                     + math.log(hi[1]) * t)


def _plot_style(method: str) -> tuple[str, str, str]:
    return METHOD_STYLE[method]


def _tight_ylim(values: list[float], minimum: float = 0.25) -> tuple[float, float]:
    """Give each recall panel a data-focused range with visible headroom."""
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1.0)
    pad = max(minimum, 0.08 * span)
    return max(0.0, lo - pad), min(100.0, hi + pad)


def plot_accuracy(runs: dict[str, dict], out: Path) -> None:
    """Plot full-rate recall and a high-rate recall zoom for each dataset.

    Relative distance error is intentionally not shown in the main figure:
    it is not the ANNS quality objective, and its ordering does not track
    Recall@10.  The lower panel in each dataset pair instead magnifies the
    measured high-rate interval (b >= 4), where sub-percentage recall gaps
    matter operationally.
    """
    # Dataset pairs are vertical: full measured rate range above, high-rate
    # zoom below.  The legend sits in its own band above the panels.  The
    # row gap is kept tight --- three of them at the old 0.55 spent most of
    # a printed inch on whitespace, and the panels do not need it to read as
    # pairs.
    fig, axes = plt.subplots(4, 3, figsize=(7.00, 4.45), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.055, top=0.92,
                        wspace=0.32, hspace=0.40)

    # Per-dataset y windows, chosen from the measured n_probe=200 recalls so
    # the E8Q gap stays visible instead of being flattened by one outlier.
    # Row 1 keeps the whole 1-8 bit curve but drops the low-rate tail; row 2
    # (b >= 4) zooms to the high-recall band where the codecs separate.
    y_windows = {
        "sift1M":              ((74.0, 100.0), (91.5, 99.6)),
        "deep1M":              ((77.0, 100.0), (91.8, 99.2)),
        "gist":                ((81.0, 99.0),  (90.2, 98.0)),
        "openai1536":          ((83.0, 97.5),  (93.7, 96.8)),
        "msmarco10M":          ((84.5, 96.5),  (91.4, 95.6)),
        "stella-trec24-1024":  ((90.5, 100.0), (96.4, 100.0)),
    }

    for index, dataset in enumerate(DATASETS):
        pair_row, col = divmod(index, 3)
        full_ax = axes[2 * pair_row][col]
        zoom_ax = axes[2 * pair_row + 1][col]
        style_panel(full_ax,
                 ylabel="Recall@10 (%)" if col == 0 else None,
                 title=DISPLAY_NAME[dataset])
        style_panel(zoom_ax,
                 xlabel="bits / dim" if pair_row == 1 else None)
        # Every panel keeps its tick numbers: the zoom row starts at 4, so a
        # reader cannot borrow the scale from the panel above it.  Only the
        # bottom row repeats the axis title.  The bit-rate window is opened
        # past 1 and 8 (see the loop below), so the first and last markers
        # sit inside the box instead of on its spines.
        full_ax.set_xlim(1.0, 8.0)
        zoom_ax.set_xlim(4.0, 8.0)
        full_ax.set_xticks(BITS)
        zoom_bits = tuple(bits for bits in BITS if bits >= 4)
        zoom_ax.set_xticks(zoom_bits)
        series: dict[str, tuple[list[float], list[float]]] = {}
        full_values: list[float] = []
        zoom_values: list[float] = []
        for method in METHODS:
            xs: list[float] = []
            ys: list[float] = []
            for bits in _available_bits(runs, dataset, method):
                run = pick_run(runs, dataset, bits, method, threads=THREADS)
                if run is None:
                    continue
                row = next((r for r in run.get("rows", [])
                            if r.get("nprobe") == ACCURACY_NPROBE), None)
                if row is None:
                    continue
                recall = row.get(f"{method}_recall")
                if recall is None:
                    continue
                recall = float(recall) * 100.0
                if not math.isfinite(recall):
                    continue
                xs.append(float(bits))
                ys.append(recall)
                full_values.append(recall)
                if bits >= 4:
                    zoom_values.append(recall)
            if xs:
                series[method] = (xs, ys)

        if not full_values:
            continue
        # The window is the key region, not the data's extent: each corpus is
        # read where its codecs separate, so a one-point gap is still a
        # visible gap.  Every measured point is still drawn --- the curve
        # simply leaves the box at the frame, which is what keeps the panel's
        # whole area in use instead of starting the curve in its middle.
        full_ylim, zoom_ylim = y_windows[dataset]
        for panel, (lo, hi) in ((full_ax, full_ylim), (zoom_ax, zoom_ylim)):
            panel.set_ylim(lo, hi)
            open_corner(panel, x=X_OPEN_FRAC, y=Y_OPEN_FRAC)
            # Labelled graduations as fine as the panel height allows, taken
            # from the window the panel is about: the point of the zoom row is
            # sub-percentage gaps, which a 10-point ladder cannot resolve.
            panel.set_yticks(_recall_ticks(lo, hi))
        for method in METHODS:
            if method not in series:
                continue
            xs, ys = series[method]
            color, marker, linestyle = _plot_style(method)
            # PQ is measured only at b in {4, 8}; a connecting line would
            # assert an unmeasured interval at b = 5..7, so it is drawn as
            # markers only.
            if method == "pq":
                linestyle = "none"
            label = METHOD_LABEL[method] if index == 0 else None
            full_ax.plot(xs, ys, color=color, marker=marker,
                         linestyle=linestyle, label=label,
                         markevery=1, linewidth=1.0, markersize=3.5,
                         markeredgecolor="white", markeredgewidth=0.4)
            high = [(x, y) for x, y in zip(xs, ys) if x >= 4]
            if high:
                high_x, high_y = zip(*high)
                zoom_ax.plot(high_x, high_y, color=color, marker=marker,
                             linestyle=linestyle, markevery=1,
                             linewidth=1.0, markersize=3.5,
                             markeredgecolor="white", markeredgewidth=0.4)

    for panel in axes.ravel():
        # Only the x scale is opened here: the recall windows were already
        # opened per panel above, and snapping their floor to a graduation
        # afterwards would drag it back onto the lowest measured point.
        open_corner(panel, x=X_OPEN_FRAC, y=0.0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   bbox_to_anchor=(0.5, 0.995), ncol=4,
                   fontsize=7.5, frameon=False, handlelength=2.0,
                   columnspacing=1.2)
    savefig(fig, out, "fig_fixed_accuracy")


def plot_frontier(runs: dict[str, dict], out: Path) -> None:
    """Plot high-recall QPS--Recall and QPS--distance-ratio frontiers."""
    # PQ sits a decade or more below the other three codecs in this recall
    # window --- on SIFT its whole retained range is 5.5e3-1.6e4 against
    # 1.2e4-9.2e4 for E8Q --- so drawing it would set the panel floor and
    # squeeze the comparison the figure exists to make.  It stays in the
    # accuracy figure, where it is a single marker per bit rate.
    frontier_bits = {
        "e8q": (2, 4, 8),
        "rabitq": (2, 4, 8),
        "caq": (2, 4, 8),
    }
    # Bit rate is the marker, codec is the color.  Linestyle alone cannot
    # carry the rate: the curves overlap for long stretches and a dash gap
    # is invisible there.
    bit_marker = {2: "o", 4: "^", 8: "s"}
    bit_ls = {2: ":", 4: "--", 8: "-"}
    # Where each dataset's curves actually live.  A shared 80% start leaves
    # Stella's panel half empty and hides the gap between the codecs.
    recall_start = {
        "sift1M": 75.0, "deep1M": 75.0, "gist": 80.0,
        "openai1536": 85.0, "msmarco10M": 85.0, "stella-trec24-1024": 90.0,
    }
    # No legend band: each legend row lives inside the one panel of its row
    # that has empty space, so the panels can use the full figure height.
    # Wider and shorter: the two legend rows share one band above the panels,
    # so no legend box sits on top of a curve.
    fig, axes = plt.subplots(4, 3, figsize=(7.00, 5.40), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.045, top=0.925,
                        wspace=0.30, hspace=0.52)

    for index, dataset in enumerate(DATASETS):
        pair_row, col = divmod(index, 3)
        recall_ax = axes[2 * pair_row][col]
        adr_ax = axes[2 * pair_row + 1][col]
        style_panel(recall_ax,
                 ylabel="QPS" if col == 0 else None,
                 xlabel="Recall@10 (%)",
                 title=DISPLAY_NAME[dataset], ylog=True)
        style_panel(adr_ax,
                 xlabel="Distance ratio", ylog=True)
        # Decades plus the 2 and 5 graduations, so a curve can be placed
        # between 2e3 and 5e3 instead of guessed at.
        for ax in (recall_ax, adr_ax):
            _fine_log_axis(ax)
        recall_ax.set_xlim(recall_start[dataset], 100.0)
        recall_ax.set_xticks(_recall_ticks(recall_start[dataset], 100.0))
        adr_ax.set_xlim(1.000, 1.025)

        # Collect the curves first, then set each panel's window from what is
        # in it: the QPS floor follows the slowest measured point, and the
        # distance-ratio axis stops just past the last one.
        series: list[dict] = []
        for method, rates in frontier_bits.items():
            color, _, _ = _plot_style(method)
            for bits in rates:
                run = pick_run(runs, dataset, bits, method,
                               threads=THREADS)
                if run is None:
                    continue
                _, qps, recall, _ = _frontier_values(run, method, "recall")
                keep = (recall >= recall_start[dataset] / 100.0) & (qps > 0)
                ratio_x, ratio_qps, ratio_recall, _ = (
                    _frontier_values(run, method, "adr"))
                rkeep = ((ratio_recall >= recall_start[dataset] / 100.0)
                         & (ratio_x >= 1.0) & (ratio_qps > 0))
                # Both the whole curve and the mask that the panel's window is
                # derived from: every measured point is drawn, and the part
                # that falls outside the window is clipped by the axes rather
                # than dropped from the series.
                series.append({
                    "color": color,
                    "marker": bit_marker[bits],
                    "linestyle": bit_ls[bits],
                    "zorder": 5 if method == "e8q" else 2,
                    "recall": (100.0 * recall, qps),
                    "recall_window": keep,
                    "ratio": (ratio_x, ratio_qps),
                    "ratio_window": rkeep,
                })

        recall_floor_values: list[float] = []
        recall_peak_values: list[float] = []
        adr_floor_values: list[float] = []
        adr_peak_values: list[float] = []
        adr_xmax: list[float] = []
        adr_xmin: list[float] = []
        for item in series:
            recall_x, qps = item["recall"]
            in_window = item["recall_window"]
            if qps.size:
                recall_floor_values.extend(qps[in_window].tolist())
                if qps[in_window].size:
                    recall_peak_values.append(float(qps[in_window].max()))
                recall_ax.plot(
                    recall_x, qps,
                    color=item["color"], marker=item["marker"],
                    linestyle=item["linestyle"],
                    markevery=max(1, int(qps.size) // 6),
                    linewidth=1.0, markersize=3.2,
                    markeredgecolor="white", markeredgewidth=0.4,
                    zorder=item["zorder"])

            adr_x, adr_qps = item["ratio"]
            shown = item["ratio_window"]
            if adr_qps.size:
                adr_floor_values.extend(adr_qps[shown].tolist())
                if adr_x[shown].size:
                    adr_xmax.append(float(adr_x[shown].max()))
                    adr_xmin.append(float(adr_x[shown].min()))
                    adr_peak_values.append(float(adr_qps[shown].max()))
                adr_ax.plot(
                    adr_x, adr_qps,
                    color=item["color"], marker=item["marker"],
                    linestyle=item["linestyle"],
                    markevery=max(1, int(adr_qps.size) // 6),
                    linewidth=1.0, markersize=3.2,
                    markeredgecolor="white", markeredgewidth=0.4,
                    zorder=item["zorder"])

        # The codecs differ by well under 2x at the top of the curves, so cap
        # the panel just above its fastest point.
        if recall_peak_values:
            recall_ax.set_ylim(top=1.2 * max(recall_peak_values))
        if recall_floor_values:
            recall_ax.set_ylim(
                bottom=QPS_FLOOR_FRACTION * min(recall_floor_values))
        if adr_peak_values:
            adr_ax.set_ylim(top=1.3 * max(adr_peak_values))
        if adr_floor_values:
            adr_ax.set_ylim(bottom=0.75 * min(adr_floor_values))
        # A fixed cap left the right of most panels empty and piled every
        # curve against the left spine.  End each panel just past its own
        # last measured point; the lower limit is left to the corner-tick
        # pass, which runs once every panel has its final window.
        if adr_xmax:
            hi = max(adr_xmax)
            adr_ax.set_xlim(min(adr_xmin) if adr_xmin else 1.0,
                            hi + max(0.08 * (hi - 1.0), 0.0004))

    # The codec key goes in the GIST Recall panel, whose curves stay in the
    # upper-right and leave the lower-left free; the bit-rate key goes in the
    # DEEP distance-ratio panel, which is empty on the left.  Windows open
    # past the data first: the highest probe of the fastest codec otherwise
    # touches the top spine and the first point of every curve starts on the
    # left one.  A distance-ratio axis keeps its floor at 1.0 --- a ratio
    # below one is not a value this experiment can produce --- so only its
    # QPS axis is opened.
    for position, panel in enumerate(axes.ravel()):
        row, _ = divmod(position, 3)
        open_corner(panel, x=X_OPEN_FRAC if row % 2 == 0 else 0.0,
                    y=Y_OPEN_FRAC)
        snap_corner_ticks(panel)
    codec_handles = [Line2D(
        [0], [0], color=_plot_style(method)[0], marker=bit_marker[4],
        linestyle="-", linewidth=1.0, markersize=3.5,
        markerfacecolor=_plot_style(method)[0], markeredgecolor="white",
        label=METHOD_LABEL[method]) for method in frontier_bits]
    bit_handles = [Line2D(
        [0], [0], color="#444444", marker=bit_marker[bits],
        linestyle=bit_ls[bits], linewidth=1.0, markersize=3.5,
        markerfacecolor="#444444", markeredgecolor="white",
        label=f"{bits} bits / dim") for bits in (2, 4, 8)]
    fig.legend(handles=codec_handles + bit_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=7, fontsize=7.0,
               frameon=False, handlelength=2.2, columnspacing=0.9)
    savefig(fig, out, "fig_fixed_frontier")


def write_summary_table(runs: dict[str, dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        # Same padding-free convention as Table 5: report the metadata that
        # is actually stored, not the allocator's 32-lane rounding.
        empty = empty_lane_frac(runs, dataset, threads=THREADS)
        for method in METHODS:
            run = pick_run(runs, dataset, SUMMARY_BITS, method,
                           threads=THREADS)
            if run is None:
                continue
            ceiling = max(
                float(row[f"{method}_recall"])
                for row in run.get("rows", [])
                if row.get(f"{method}_recall") is not None
            )
            q90 = _qps_at_recall(run, method, 0.90)
            q95 = _qps_at_recall(run, method, 0.95)
            rows.append({
                "Dataset": DISPLAY_NAME[dataset],
                "Method": METHOD_LABEL[method],
                "Ceiling": f"{ceiling:.4f}",
                "QPS@90": "--" if not math.isfinite(q90) else f"{q90:,.0f}",
                "QPS@95": "--" if not math.isfinite(q95) else f"{q95:,.0f}",
                "B/vec": f"{logical_bpv(run, method, empty):.1f}",
            })

    with (out / "tab_fixed_main.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tex = out / "tab_fixed_main.tex"
    with tex.open("w") as fh:
        fh.write("\\begin{tabular}{@{}llrrrr@{}}\n")
        fh.write("\\toprule\n")
        fh.write("Corpus & Quantizer & Ceiling & QPS@90 & QPS@95 & B/vec \\\\\n")
        fh.write("\\midrule\n")
        start = 0
        for index, row in enumerate(rows):
            if index == 0 or row["Dataset"] != rows[index - 1]["Dataset"]:
                if index:
                    fh.write("\\midrule\n")
                fh.write(f"\\multirow{{4}}{{*}}{{{row['Dataset']}}} & ")
            else:
                fh.write(" & ")
            fh.write(" & ".join(row[key] for key in
                                  ("Method", "Ceiling", "QPS@90",
                                   "QPS@95", "B/vec")))
            fh.write(" \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    apply_sigmod_style()
    runs = _method_runs(args.results)
    filled = apply_adr_sidecars(runs, args.results)
    print(f"distance-ratio side-cars: {filled} ledgers on the full nprobe grid")
    required = [
        (dataset, method)
        for dataset in DATASETS
        for method in METHODS
        if pick_run(runs, dataset, SUMMARY_BITS, method,
                    threads=THREADS) is None
    ]
    if required:
        missing = ", ".join(f"{dataset}/{method}" for dataset, method in required)
        raise SystemExit(f"missing fixed-rate ledgers: {missing}")
    plot_accuracy(runs, args.out)
    plot_frontier(runs, args.out)
    write_summary_table(runs, args.out)


if __name__ == "__main__":
    main()
