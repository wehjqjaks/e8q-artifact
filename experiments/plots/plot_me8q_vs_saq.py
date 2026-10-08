"""ME8Q vs SAQ quantization accuracy (SAQ-paper Figure-8 analogue).

2xN panels, one per dataset (default: the validation trio). Row 1:
Recall@10 at nprobe=200 versus bits per dimension; Row 2: average
relative error (log scale) versus bits per dimension. Only me8q and saq.

X-axis is the ACTUAL total memory in bits per dimension (`<m>_bits_per_dim`;
ledgers without it fall back to bytes_per_vector/dim), so every point sits
at or under its budget b. Budgets too low for a method's budget search to
find a coded plan have no ledger at all — the driver writes a
`<stem>.skipped` marker — and are annotated per panel instead of drawn.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DISPLAY, METHOD_LABEL, METHOD_STYLE, actual_bits_per_dim, apply_style,
    compact_subplots, field_of, load_bench_runs, parse_stem, row_field,
    savefig, style_ax,
)

METHODS = ("me8q", "saq")
RELERR_PROBE = 200
_MARKER_RE = re.compile(r"^(.+)_b([\d.]+)_t(\d+)_(\w+)\.skipped$")


def _collect(runs: dict[str, dict], method: str) -> dict[str, dict]:
    """dataset -> {bits: {bpd, recall, relerr}}."""
    out: dict[str, dict] = {}
    for stem, run in runs.items():
        parsed = parse_stem(stem)
        if parsed is None:
            continue
        ds, m, bits, _ = parsed
        if m != method:
            continue
        rec = row_field(run, method, RELERR_PROBE, "recall")
        rel = field_of(run, method, "relerr_avg")
        bpd = actual_bits_per_dim(run, method)
        out.setdefault(ds, {})[float(bits)] = {
            "bpd": bpd,
            "recall": rec,
            "relerr": rel,
        }
    return out


def _skipped(out_dir: Path) -> dict[str, dict[str, list]]:
    """method -> dataset -> [bits] with a `.skipped` marker (no plan)."""
    out: dict[str, dict[str, list]] = {}
    for p in sorted(out_dir.glob("*.skipped")):
        m = _MARKER_RE.match(p.name)
        if m is None:
            continue
        ds, bits, _, method = m.groups()
        try:
            doc = json.loads(p.read_text())
        except json.JSONDecodeError:
            doc = {}
        entry = {"bits": float(bits), "reason": doc.get("reason", "skipped")}
        out.setdefault(method, {}).setdefault(ds, []).append(entry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True,
                    help="run directory containing stable/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+",
                    default=["deep1M", "msong", "openai1536"])
    args = ap.parse_args()

    apply_style()
    runs = load_bench_runs(args.results / "stable")
    data = {m: _collect(runs, m) for m in METHODS}
    skipped = _skipped(args.results / "stable")
    n = len(args.datasets)
    fig, axes = compact_subplots(2, n, col_w=2.33, row_h=1.7)

    for col, ds in enumerate(args.datasets):
        ax_r = axes[0, col]
        ax_e = axes[1, col]
        notes = []
        xs_all: list = []
        for method in METHODS:
            color, marker, ls = METHOD_STYLE[method]
            cells = dict(sorted(data[method].get(ds, {}).items()))
            xs_ok = [c["bpd"] for c in cells.values()
                     if np.isfinite(c["bpd"]) and np.isfinite(c["recall"])]
            ys_ok = [100 * c["recall"] for c in cells.values()
                     if np.isfinite(c["bpd"]) and np.isfinite(c["recall"])]
            ax_r.plot(xs_ok, ys_ok, ls, color=color, marker=marker,
                      markersize=4, markeredgecolor="white", markeredgewidth=0.5,
                      label=METHOD_LABEL[method], linewidth=1.2)
            es = [(c["bpd"], 100 * c["relerr"]) for c in cells.values()
                  if np.isfinite(c["bpd"]) and c["relerr"] > 0]
            if es:
                ax_e.plot([e[0] for e in es], [e[1] for e in es], ls,
                          color=color, marker=marker, markersize=4,
                          markeredgecolor="white", markeredgewidth=0.5,
                          linewidth=1.2)
            xs_all.extend(xs_ok)
            xs_all.extend(e[0] for e in es)
            sk = skipped.get(method, {}).get(ds, [])
            if sk:
                hi = max(e["bits"] for e in sk)
                notes.append(f"{METHOD_LABEL[method]}: no plan b≤{hi:g}")
        if notes:
            ax_r.text(0.03, 0.97, "\n".join(notes), transform=ax_r.transAxes,
                      fontsize=6.5, va="top", ha="left", color="0.25")
        for ax in (ax_r, ax_e):
            style_ax(ax)
            ax.set_xlabel("bits / dim (actual total)")
        if xs_all:
            # One shared x window per column so the two rows read together.
            lo, hi_x = min(xs_all), max(xs_all)
            pad = 0.05 * max(hi_x - lo, 0.5)
            for ax in (ax_r, ax_e):
                ax.set_xlim(lo - pad, hi_x + pad)
        ax_r.set_title(DISPLAY.get(ds, ds), fontsize=9)
        ax_e.set_yscale("log")
    axes[0, 0].set_ylabel("Recall@10 (%)")
    axes[1, 0].set_ylabel("Avg rel. error (%)")
    axes[0, 0].legend(loc="lower right", fontsize=7.5, frameon=False)
    savefig(fig, args.out, "fig_me8q_vs_saq_accuracy")


if __name__ == "__main__":
    main()
