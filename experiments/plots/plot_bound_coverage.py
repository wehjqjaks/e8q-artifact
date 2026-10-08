"""Figure A1: empirical lower-bound coverage vs. theoretical risk.

Reads the curve CSV written by ``experiments/bound_coverage.py`` and plots,
for each bit budget, the measured violation rate of the ZPR screening bound
against the theoretical per-candidate risk ``delta = 1 - Phi(lambda)``.  The
diagonal is the guarantee: a series below it means the deployed bound violates
less often than the theorem allows.

    <python> analysis/plot_bound_coverage.py --results results/bound_coverage \
        --out figures_reproduced
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import savefig  # noqa: E402
from sigmod_style import (  # noqa: E402
    FONT_LABEL, FONT_LEGEND, FONT_TICK, TEXT_W, apply_sigmod_style,
    style_panel,
)

# Corpora, ordered by dimension so the legend reads like the dataset table.
# Hue + marker together, so the panel survives greyscale print.
CORPORA = [
    ("sift1M", "SIFT", "#D32F2F", "o"),
    ("deep1M", "DEEP", "#1976D2", "^"),
    ("gist", "GIST", "#F57C00", "v"),
    ("openai1536", "OpenAI-1536", "#00897B", "s"),
    ("msmarco10M", "MSMARCO-10M", "#5E35B1", "d"),
    ("stella-trec24-1024", "Stella", "#607D8B", "P"),
]
BITS = (2, 4, 8)


def load_curve(path: Path) -> dict[tuple[str, int], list[tuple[float, float]]]:
    data: dict[tuple[str, int], list[tuple[float, float]]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            key = (row["dataset"], int(row["bits"]))
            data.setdefault(key, []).append(
                (float(row["delta"]), float(row["empirical_rate"]))
            )
    for points in data.values():
        points.sort()
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/bound_coverage"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=10_000_000)
    args = ap.parse_args(argv)

    apply_sigmod_style()
    data = load_curve(args.results / "bound_coverage_curve.csv")
    floor = 0.5 / args.pairs

    fig, axes = plt.subplots(
        1, len(BITS), figsize=(TEXT_W, 2.15), sharey=True,
        gridspec_kw={"wspace": 0.10},
    )
    for ax, bits in zip(axes, BITS):
        ref = next((pts for (ds, b), pts in data.items() if b == bits), None)
        if not ref:
            continue
        grid = np.asarray([d for d, _ in ref])
        ax.plot(grid, grid, color="#9aa4ac", linestyle="--", linewidth=0.9,
                zorder=2, label="theory")
        for ds, label, color, marker in CORPORA:
            pts = data.get((ds, bits))
            if not pts:
                continue
            x = np.asarray([p[0] for p in pts])
            y = np.asarray([p[1] for p in pts])
            ax.plot(x, np.maximum(y, floor), color=color, marker=marker,
                    linestyle="-", linewidth=1.0, markersize=3.0,
                    markeredgecolor="white", markeredgewidth=0.35, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.0008, 0.26)
        # The measured rate tracks delta from below across the whole grid, so
        # the panel only needs the band the sweep actually occupies; a floor
        # four decades below the smallest delta would leave most of the panel
        # empty.
        ax.set_ylim(4e-4, 0.62)
        ax.set_xticks((0.001, 0.01, 0.1))
        ax.set_xticklabels(("0.001", "0.01", "0.1"))
        style_panel(ax, xlabel=r"theoretical risk $\delta$",
                    title=f"$b={bits}$")
    axes[0].set_ylabel("empirical violation rate")
    axes[0].set_yticks((1e-3, 1e-2, 1e-1))
    axes[0].set_yticklabels((r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"))
    axes[0].tick_params(labelsize=FONT_TICK)

    key = [Line2D([0], [0], color=c, marker=m, linestyle="-", markersize=3.4,
                  markeredgecolor="white", markeredgewidth=0.35, label=lab)
           for _ds, lab, c, m in CORPORA]
    key.append(Line2D([0], [0], color="#9aa4ac", linestyle="--", linewidth=0.9,
                      label=r"theory ($y=\delta$)"))
    fig.legend(key, [h.get_label() for h in key], loc="upper center",
               bbox_to_anchor=(0.5, 0.925), ncol=7, fontsize=FONT_LEGEND,
               frameon=False, handlelength=1.7, columnspacing=0.8)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.715, bottom=0.195)
    for ax in axes:
        ax.tick_params(labelsize=FONT_TICK)
        ax.xaxis.label.set_size(FONT_LABEL)
    savefig(fig, args.out, "fig_bound_coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
