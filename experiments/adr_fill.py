#!/usr/bin/env python3
"""Fold a full-grid distance-ratio re-measurement into a side-car ledger.

The fixed-rate runners recorded the sum-ratio distance ratio (README.md)
only at the four Table-5 depths, while the vendored CAQ binary emits
the same quantity at every grid point.  Figure 4's QPS--distance-ratio
panels therefore showed four isolated markers for E8Q and RaBitQ next to a
53-point CAQ curve.  Re-running an arm with ``--adr`` fills the column over
the whole grid; this script keeps that column plus the recall it was scored
from, so the published QPS is left exactly as frozen.

Recall must match the frozen ledger depth by depth.  The search is
deterministic for a fixed seed, so a mismatch means the fresh ratio
describes a different operating point than the one the figure plots, and
the side-car is refused rather than silently mixed in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows = data.get("rows") if isinstance(data, dict) else None
    if not (isinstance(rows, list) and rows):
        raise SystemExit(f"{path}: no rows")
    return rows


def method_of(path: Path) -> str:
    stem = path.stem
    for method in ("me8q", "rabitq", "e8q", "saq", "caq", "pq"):
        if stem.endswith("_" + method):
            return method
    raise SystemExit(f"{path}: cannot read the method off the filename")


def build_sidecar(frozen: list[dict], fresh: list[dict], method: str,
                  tol: float, label: str = "") -> list[dict]:
    """The side-car's rows: the fresh ratio at the frozen operating points.

    ``frozen`` is the published ledger -- the one whose QPS the figure plots
    -- and ``fresh`` the ``--adr`` re-run.  The two must agree on recall
    depth by depth, because the search is deterministic for a fixed seed: a
    mismatch means the fresh ratio describes a different operating point
    than the one being plotted, and silence would mix the two.
    """
    adr_key = f"{method}_adr"
    recall_key = f"{method}_recall"
    ref = {int(r["nprobe"]): r for r in frozen}
    got = {int(r["nprobe"]): r for r in fresh}

    missing = sorted(n for n in ref if n not in got)
    if missing:
        raise SystemExit(f"{label or 'fresh ledger'}: grid misses depths {missing}")

    out_rows: list[dict] = []
    mismatched: list[str] = []
    for nprobe in sorted(ref):
        f, g = ref[nprobe], got[nprobe]
        if adr_key not in g:
            raise SystemExit(
                f"{label or 'fresh ledger'}: nprobe={nprobe} has no {adr_key}")
        r_frozen = float(f[recall_key])
        r_fresh = float(g[recall_key])
        if abs(r_frozen - r_fresh) > tol:
            mismatched.append(
                f"nprobe={nprobe}: frozen {r_frozen:.6f} vs fresh {r_fresh:.6f}")
        out_rows.append({
            "nprobe": int(nprobe),
            recall_key: round(r_fresh, 6),
            adr_key: float(g[adr_key]),
        })
    if mismatched:
        print("recall mismatch against the frozen ledger:", file=sys.stderr)
        for line in mismatched[:10]:
            print("  " + line, file=sys.stderr)
        raise SystemExit(
            f"{len(mismatched)} of {len(ref)} depths differ; the fresh "
            "ratio does not describe the published operating points")
    return out_rows


def sidecar_doc(rows: list[dict], method: str, frozen_name: str,
                fresh_name: str) -> dict:
    return {
        "method": method,
        "source": frozen_name,
        "fresh": fresh_name,
        "metric": "adr_sum_ratio (sum-ratio form)",
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fresh", type=Path, required=True,
                    help="ledger from the --adr re-run (full grid)")
    ap.add_argument("--frozen", type=Path, required=True,
                    help="published ledger the figure is drawn from")
    ap.add_argument("--out", type=Path, required=True,
                    help="side-car JSON to write")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="per-depth recall tolerance (default 1e-9)")
    args = ap.parse_args()

    method = method_of(args.frozen)
    if method_of(args.fresh) != method:
        raise SystemExit("fresh and frozen ledgers disagree on the method")

    out_rows = build_sidecar(_rows(args.frozen), _rows(args.fresh), method,
                             args.tol, label=str(args.fresh))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        sidecar_doc(out_rows, method, args.frozen.name, args.fresh.name),
        indent=2) + "\n")
    adr_key = f"{method}_adr"
    print(f"{args.out}: {len(out_rows)} depths, "
          f"{adr_key} {out_rows[0][adr_key]:.6f}..{out_rows[-1][adr_key]:.6f}, "
          "recall identical to the frozen ledger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
