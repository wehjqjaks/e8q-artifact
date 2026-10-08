"""Shared loaders and SciencePlots style for paper figures/tables.

Color cycle is Okabe–Ito (Wong, Nat. Methods 2011) — the palette SIGMOD /
VLDB camera-ready figures use when they need 7+ colorblind-safe series.
E8Q is vermillion (the strongest hue); baselines sit on blue / green /
sky / orange / grey so greyscale print still separates by marker.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protocol import METHODS as _PROTOCOL_METHODS  # noqa: E402
from protocol import parse_stem  # noqa: E402

# Paper corpora order and display names (README.md; sync with
# experiments/lib.sh CORE_DATASETS).
CORE = ["sift1M", "sift10m", "deep1M", "msong", "word2vec", "gist",
        "glove200", "imagenet", "youtube", "vogue768",
        "arxiv-abstracts768", "openai1536", "openai3072", "msmarco10M",
        "stella-trec24-1024"]
# Body/appendix split (README.md): 5 body datasets — low/mid/high dim,
# 1M and 10M scale; final selection re-checked against measured margins.
# Override with BODY_DATASETS (space-separated) before plotting.
BODY = [d for d in (os.environ.get("BODY_DATASETS",
                                   "sift1M msong gist openai3072 sift10m").split())
        if d in CORE]
APPENDIX = [d for d in CORE if d not in BODY]

# On-disk stems sometimes use a lower-case 'm' (sift1m); dataset keys use sift1M.
_DS_ALIAS = {"sift1m": "sift1M"}


def canon_ds(name: str | None) -> str | None:
    if not name:
        return name
    return _DS_ALIAS.get(name, name)


def ds_stems(ds: str) -> tuple[str, ...]:
    """Filename prefixes to try for a dataset key."""
    c = canon_ds(ds) or ds
    if c == "sift1M":
        return ("sift1M", "sift1m")
    return (c,)


DISPLAY = {
    "sift1M": "SIFT-1M", "sift1m": "SIFT-1M",
    "sift10m": "SIFT-10M", "deep1M": "DEEP-1M",
    "msong": "Msong", "word2vec": "Word2Vec", "gist": "GIST",
    "glove200": "GloVe-200", "imagenet": "ImageNet", "youtube": "YouTube",
    "vogue768": "Vogue-768", "arxiv-abstracts768": "ArXiv-768",
    "openai1536": "OpenAI-1536", "openai3072": "OpenAI-3072",
    "msmarco10M": "MSMARCO-10M", "stella-trec24-1024": "Stella-TREC24",
}

# Fixed-rate paper palette: E8Q = cardinal red; RaBitQ = blue; CAQ = amber;
# PQ = purple.  Marker and line style remain part of the encoding so figures
# stay readable in grayscale and under color-vision deficiencies.
METHOD_STYLE = {
    "e8q":    ("#DC2626", "o", "-"),
    "rabitq": ("#2563EB", "^", "--"),
    "me8q":   ("#059669", "s", "--"),
    "caq":    ("#D97706", "v", "-."),
    "saq":    ("#6366F1", "d", ":"),
    "pq":     ("#7A5195", "X", ":"),
}
METHOD_LABEL = {
    "e8q": "E8Q", "me8q": "ME8Q", "rabitq": "RaBitQ", "saq": "SAQ",
    "pq": "PQ", "caq": "CAQ",
}
# Unified ANNS-overlay encoding: hue + marker + linestyle identify the
# method; the bit width is a lightness ramp INSIDE the method's camp
# (low bits light, high bits dark). Camps with ramps: E8Q warm red, RaBitQ
# cold blue, CAQ amber; ME8Q/SAQ/PQ stay flat (they only appear at anchor
# bitrates). Anchors are linearly interpolated in RGB between bit stops.
CAMP_RAMP = {
    "e8q":    [(1, "#FCA5A5"), (2, "#F87171"), (4, "#DC2626"), (8, "#7F1D1D")],
    "rabitq": [(1, "#BFDBFE"), (2, "#93C5FD"), (4, "#2563EB"), (8, "#1E3A8A")],
    "caq":    [(1, "#FEF3C7"), (2, "#FDE68A"), (4, "#D97706"), (8, "#78350F")],
}


def _ramp_color(stops: list[tuple[int, str]], bits: float) -> str:
    b0 = max(int(np.floor(bits)), stops[0][0])
    lo = hi = stops[-1]
    for i, s in enumerate(stops):
        if s[0] >= b0:
            hi = s
            lo = stops[max(i - 1, 0)]
            break
    if hi[0] == lo[0]:
        return lo[1]
    t = (bits - lo[0]) / (hi[0] - lo[0])
    c0 = np.array(mpl.colors.to_rgb(lo[1]))
    c1 = np.array(mpl.colors.to_rgb(hi[1]))
    return mpl.colors.to_hex(tuple(c0 + t * (c1 - c0)))


def bits_color(method: str, bits: int) -> str:
    """Camp color at a bit width: ramp for the three camped methods, the
    flat method color otherwise."""
    stops = CAMP_RAMP.get(method)
    if stops is None or bits not in (1, 2, 3, 4, 5, 6, 7, 8):
        return METHOD_STYLE[method][0]
    return _ramp_color(stops, bits)
METHODS = _PROTOCOL_METHODS


def apply_style() -> None:
    """Compose Seaborn's paper context with SciencePlots' IEEE geometry.

    Seaborn supplies the restrained white-grid context and a stable palette
    hook; SciencePlots then applies the compact two-column paper defaults.
    Method colours below are set explicitly, so changing either library's
    default cycle cannot change the meaning of a curve.

    usetex is off by default so figures render on machines without a full
    TeXLive (cm-super); set E8Q_PLOT_TEX=1 on a machine that has it.
    """
    try:
        import seaborn as sns
        sns.set_theme(context="paper", style="whitegrid", font_scale=0.82)
    except ImportError:
        sns = None

    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee"])
    except (ImportError, OSError):
        mpl.rcParams.update({
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.5,
        })
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # SciencePlots requests Times; use the installed TeX Gyre metric
        # equivalent so rendering never falls back to a sans-serif font.
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "DejaVu Serif"],
        # SIGMOD-style base: serif CM math, charcoal spines, faint grid,
        # thin lines and white-edged markers so dense families un-stick
        "mathtext.fontset": "cm",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        # thin lines/markers so dense method families stay separable; the
        # white marker edge keeps overlapped points readable
        "lines.linewidth": 0.9,
        "lines.markersize": 3.0,
        "lines.markeredgewidth": 0.5,
        "lines.markeredgecolor": "white",
        "text.usetex": os.environ.get("E8Q_PLOT_TEX", "0") == "1",
        "axes.formatter.use_mathtext": True,
        "legend.frameon": False,
        "legend.borderaxespad": 0.2,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        "axes.prop_cycle": mpl.cycler(
            color=["#D55E00", "#0072B2", "#009E73", "#CC79A7",
                   "#E69F00", "#56B4E9", "#999999"]
        ),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    if sns is not None:
        # Keep Seaborn in the rendering path while preserving the paper's
        # method-specific hue assignment and grayscale-safe markers.
        sns.set_palette([METHOD_STYLE[m][0]
                         for m in ("e8q", "rabitq", "caq", "pq")])


def load_bench_runs(*exp_dirs: Path) -> dict[str, dict]:
    """Load every bench JSON under the given folders, keyed by stem."""
    runs: dict[str, dict] = {}
    for exp_dir in exp_dirs:
        if exp_dir is None or not exp_dir.is_dir():
            continue
        for p in sorted(exp_dir.rglob("*.json")):
            if p.name in {"not_exposed.json"}:
                continue
            try:
                runs[p.stem] = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                print(f"warning: skipping {p}: {exc}")
    return runs


def pick_run(runs: dict, ds: str, bits: int, method: str,
             threads: int | None = None):
    """Resolve `<ds>_b<bits>_t<T>_<method>` (T defaults to 8)."""
    aliases = ds_stems(ds)
    if threads is None:
        for alias in aliases:
            gold = f"{alias}_b{bits}_t8_{method}"
            if gold in runs:
                return runs[gold]
            legacy = f"{alias}_{method}_b{bits}_t8"
            if legacy in runs:
                return runs[legacy]
        want = set(aliases)
        for stem, run in runs.items():
            parsed = parse_stem(stem)
            if not parsed:
                continue
            if canon_ds(parsed[0]) in want and parsed[1] == method and parsed[2] == bits:
                return run
        return None
    for alias in aliases:
        hit = runs.get(f"{alias}_b{bits}_t{threads}_{method}") or \
              runs.get(f"{alias}_{method}_b{bits}_t{threads}")
        if hit is not None:
            return hit
    return None


def datasets_in(runs: dict, preferred: list[str] | None = None) -> list[str]:
    found: list[str] = []
    order = list(preferred or CORE)
    extra: list[str] = []
    seen = set()
    for stem, run in runs.items():
        ds = canon_ds(run.get("dataset"))
        if not ds:
            parsed = parse_stem(stem)
            ds = canon_ds(parsed[0]) if parsed else None
        if not ds or ds in seen:
            continue
        seen.add(ds)
        extra.append(ds)
    for d in order:
        if d in seen:
            found.append(d)
    for d in extra:
        if d not in found:
            found.append(d)
    return found


def bits_in(runs: dict, ds: str, method: str) -> list[int]:
    """Bits values with a ledger for (ds, method), sorted."""
    out = set()
    for stem, run in runs.items():
        parsed = parse_stem(stem)
        if parsed and canon_ds(parsed[0]) in ds_stems(ds) and parsed[1] == method:
            out.add(parsed[2])
    return sorted(out)


def load_ablation_runs(run_dir: Path) -> dict[str, dict]:
    """Load abl_*.json from <run>/ablation (plus legacy extras shapes)."""
    runs: dict[str, dict] = {}
    for sub in ("ablation", "param", "extras"):
        d = run_dir / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.json")):
            if p.name in {"not_exposed.json"}:
                continue
            try:
                runs[p.stem] = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                print(f"warning: skipping {p}: {exc}")
    return runs


def qps_at_recall(rows: list[dict], recall_key: str, qps_key: str,
                  target: float = 0.95) -> float:
    """QPS at a recall target by log-QPS interpolation; NaN if never reached."""
    pts = sorted(
        (r[recall_key], r[qps_key]) for r in rows
        if r.get(recall_key, -1) >= 0 and r.get(qps_key, -1) > 0
    )
    if not pts or max(r for r, _ in pts) < target:
        return float("nan")
    below = [p for p in pts if p[0] <= target]
    if below and below[-1][0] == target:
        return float(below[-1][1])
    lo = below[-1] if below else (pts[0][0] - 1e-9, pts[0][1])
    hi = min((p for p in pts if p[0] > target), key=lambda p: p[0])
    t = (target - lo[0]) / max(hi[0] - lo[0], 1e-12)
    import math
    return float(math.exp(math.log(max(lo[1], 1e-9)) * (1 - t) + math.log(hi[1]) * t))


def series_of(run: dict, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(recall, qps, ratio) arrays for one method of one bench run."""
    rec = np.array([r.get(f"{method}_recall", float("nan")) for r in run["rows"]], dtype=float)
    qps = np.array([r.get(f"{method}_qps", float("nan")) for r in run["rows"]], dtype=float)
    ratio = np.array([r.get(f"{method}_ratio", float("nan")) for r in run["rows"]], dtype=float)
    return rec, qps, ratio


def relerr_of(run: dict, method: str) -> tuple[float, float]:
    avg = run.get(f"{method}_relerr_avg", float("nan"))
    mx = run.get(f"{method}_relerr_max", float("nan"))
    return float(avg) if avg is not None else float("nan"), \
           float(mx) if mx is not None else float("nan")


def build_of(run: dict, method: str) -> float:
    v = run.get(f"{method}_build_s")
    return float(v) if v is not None else float("nan")


def mem_of(run: dict, method: str) -> tuple[float, float]:
    """(mem_gb, bytes_per_vector); derive bpv from mem_gb/n when missing."""
    gb = run.get(f"{method}_mem_gb")
    bpv = run.get(f"{method}_bytes_per_vector")
    mem_gb = float(gb) if gb is not None else float("nan")
    bytes_pv = float(bpv) if bpv is not None else float("nan")
    n = run.get("n")
    if (not np.isfinite(bytes_pv) or bytes_pv <= 0) and n and np.isfinite(mem_gb) and mem_gb > 0:
        bytes_pv = mem_gb * 1e9 / float(n)
    return mem_gb, bytes_pv


def actual_bits_per_dim(run: dict, method: str,
                        fallback: float = float("nan")) -> float:
    """Total memory in bits per dim, from `<m>_bits_per_dim`; ledgers
    without it fall back to bytes_per_vector/dim."""
    v = run.get(f"{method}_bits_per_dim")
    if v is not None:
        return float(v)
    bpv = run.get(f"{method}_bytes_per_vector")
    dim = float(run.get("dim") or 0)
    if bpv is not None and dim:
        return float(bpv) * 8.0 / dim
    return fallback


# ---------------------------------------------------------------------------
# Logical (padding-free) footprint.
#
# E8Q and RaBitQ pack codes into per-list 32-vector FastScan batches, so every
# inverted list is rounded up to a whole batch and the tail lanes are dummy
# slots that carry no data.  Paper tables and bytes-per-vector axes report what
# a method has to store, not how its allocator rounds, so those empty lanes
# come out of the reported footprint.  The number of empty lanes is a property
# of the shared IVF partition alone, so it is fitted once per dataset from the
# E8Q rate series (the only series measured at every integer rate) and reused
# for every method on that partition.  PQ, CAQ, SAQ, and ME8Q store codes flat
# and pay no such rounding.
FASTSCAN_BATCH = 32


def empty_lane_frac(runs: dict[str, dict], dataset: str,
                    threads: int | None = None) -> float:
    """Empty FastScan lanes per stored vector, fitted from the E8Q series.

    E8Q's per-lane arena cost is `blocks*b + 12` bytes: the scan record holds
    one code byte per 8-dim block plus a 128 B header shared by 32 lanes, and
    the refine record adds `(b-1)` bytes per block plus two fp32 per lane.  The
    auxiliary term is therefore affine in that quantity and the slope is the
    empty-lane fraction.  Returns 0.0 when the series is too short to fit, so
    callers degrade to the physical number rather than to a guess.
    """
    xs: list[float] = []
    ys: list[float] = []
    for bits in range(1, 9):
        run = pick_run(runs, dataset, bits, "e8q", threads=threads)
        if run is None:
            continue
        aux = run.get("e8q_bpv_factor")
        dim = run.get("dim")
        if aux is None or dim is None:
            continue
        blocks = (int(dim) + 7) // 8
        xs.append(float(blocks * bits + 12))
        ys.append(float(aux))
    if len(xs) < 2:
        return 0.0
    slope, _ = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    return max(float(slope), 0.0)


def padding_bpv(run: dict, method: str, empty_frac: float) -> float:
    """Bytes per vector that per-list batch rounding wastes for `method`."""
    dim = int(run.get("dim") or 0)
    bits = int(run.get("bits") or 0)
    if method == "e8q":
        blocks = (dim + 7) // 8
        per_lane = (blocks + 4) + (blocks * (bits - 1) + 8)
    elif method == "rabitq":
        # Batched plane only (1-bit code + three fp32 factors); the extra-bit
        # store is flat and pays no batch rounding.
        per_lane = dim / 8.0 + 12.0
    else:
        return 0.0
    return empty_frac * per_lane


def logical_bpv(run: dict, method: str, empty_frac: float = 0.0) -> float:
    """`<method>_bytes_per_vector` with FastScan batch padding removed."""
    bpv = run.get(f"{method}_bytes_per_vector")
    if bpv is None:
        bpv = mem_of(run, method)[1]
    if not np.isfinite(bpv):
        return float("nan")
    return float(bpv) - padding_bpv(run, method, empty_frac)


def logical_aux_bpv(run: dict, method: str, empty_frac: float = 0.0) -> float:
    """Auxiliary bytes per vector (footprint minus code payload), padding-free."""
    aux = run.get(f"{method}_bpv_factor")
    if aux is None:
        aux = run.get(f"{method}_bpv_overhead")
    if aux is None:
        return float("nan")
    return float(aux) - padding_bpv(run, method, empty_frac)


def logical_mem_of(run: dict, method: str,
                   empty_frac: float = 0.0) -> tuple[float, float]:
    """(GB, bytes/vector) with FastScan batch padding removed."""
    bpv = logical_bpv(run, method, empty_frac)
    n = run.get("n")
    gb = bpv * float(n) / 1e9 if n and np.isfinite(bpv) else float("nan")
    return gb, bpv


def x_bits(run: dict, method: str, nominal) -> float:
    """x for accuracy plots. The mixed-precision arms (saq/me8q) plot at
    their ACTUAL total bits/dim — under the true-bit-budget protocol the
    nominal b is a cap, and the achieved point is what compares fairly.
    Fixed-rate arms plot at their nominal b: their protocol is the
    code-quota convention."""
    if method in ("saq", "me8q"):
        v = actual_bits_per_dim(run, method)
        if np.isfinite(v):
            return v
    return float(nominal)


def fmt6(x, *, percent: bool = False) -> str:
    """Table cell: 6 decimal places, or `--`. `percent=True` prints 100·x."""
    if x is None:
        return "--"
    v = float(x)
    if not np.isfinite(v):
        return "--"
    if percent:
        v *= 100.0
    return f"{v:.6f}"


def clip_xy(x, y, xmin: float | None = None, xmax: float | None = None):
    """Keep points inside [xmin, xmax] on x (inclusive)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if xmin is not None:
        m &= x >= xmin
    if xmax is not None:
        m &= x <= xmax
    return x[m], y[m]


def row_at(run: dict, nprobe: int) -> dict | None:
    for r in run.get("rows", []):
        if r.get("nprobe") == nprobe:
            return r
    return None


def field_of(run: dict, method: str, key: str, default: float = float("nan")) -> float:
    """Header field `<method>_<key>` as float; NaN when absent."""
    v = run.get(f"{method}_{key}")
    return float(v) if v is not None else default


def row_field(run: dict, method: str, nprobe: int, key: str) -> float:
    r = row_at(run, nprobe)
    if r is None:
        return float("nan")
    v = r.get(f"{method}_{key}")
    return float(v) if v is not None else float("nan")


def recall_ceiling(run: dict, method: str) -> float:
    """Max Recall@10 over the whole grid (§5.3)."""
    vals = [r.get(f"{method}_recall") for r in run.get("rows", [])]
    vals = [float(v) for v in vals if v is not None]
    return max(vals) if vals else float("nan")


def qps_at_x(rows: list[dict], x_key: str, qps_key: str, target: float,
             *, lower_is_better: bool = False) -> float:
    """QPS where a monotone x-series crosses `target` (log-QPS interpolation).

    `lower_is_better` flips the reach test (ADR: reached when min ≤ target).
    NaN when the curve never crosses."""
    pts = sorted(
        (r[x_key], r[qps_key]) for r in rows
        if r.get(x_key) is not None and r.get(qps_key, -1) > 0
    )
    if not pts:
        return float("nan")
    xs = [p[0] for p in pts]
    reached = (min(xs) <= target) if lower_is_better else (max(xs) >= target)
    if not reached:
        return float("nan")
    side = (lambda p: p[0] >= target) if lower_is_better else (lambda p: p[0] <= target)
    near = [p for p in pts if side(p)]
    anchor = near[-1] if not lower_is_better else near[0]
    if anchor[0] == target:
        return float(anchor[1])
    others = [p for p in pts if not side(p)]
    if not others:
        return float("nan")
    mate = min(others, key=lambda p: abs(p[0] - target))
    lo, hi = (anchor, mate) if anchor[0] < mate[0] else (mate, anchor)
    t = (target - lo[0]) / max(hi[0] - lo[0], 1e-12)
    import math
    return float(math.exp(math.log(max(lo[1], 1e-9)) * (1 - t) + math.log(hi[1]) * t))


def qps_at_nprobe(run: dict, method: str, nprobe: int) -> float:
    for r in run.get("rows", []):
        if r.get("nprobe") == nprobe:
            v = r.get(f"{method}_qps")
            return float(v) if v is not None else float("nan")
    return float("nan")


def recall_at_nprobe(run: dict, method: str, nprobe: int) -> float:
    for r in run.get("rows", []):
        if r.get("nprobe") == nprobe:
            v = r.get(f"{method}_recall")
            return float(v) if v is not None else float("nan")
    return float("nan")


def compact_subplots(nrows: int, ncols: int, col_w: float = 2.15, row_h: float = 1.85):
    """Tight grid. constrained_layout keeps labels off the axes."""
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(col_w * max(ncols, 1), row_h * max(nrows, 1)),
        squeeze=False,
        constrained_layout=True,
    )
    return fig, axes


def style_ax(ax, xlabel: str | None = None, ylabel: str | None = None,
             title: str | None = None, ylog: bool = False) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=2)
    if ylog:
        ax.set_yscale("log")
    ax.tick_params(direction="in", which="both", length=2.5, width=0.5)
    ax.grid(True, which="major", ls=":", lw=0.4, alpha=0.5)


def savefig(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"wrote {out}/{name}.pdf")


def write_table(out_dir: Path, name: str, header: list[str],
                rows: list[list]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{name}.csv"
    with csv.open("w") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(str(c) for c in row) + "\n")
    tex = out_dir / f"{name}.tex"
    with tex.open("w") as fh:
        fh.write("\\begin{tabular}{" + "l" * len(header) + "}\n")
        fh.write("\\toprule\n" + " & ".join(header) + " \\\\\n\\midrule\n")
        for row in rows:
            fh.write(" & ".join(str(c) for c in row) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {csv} and {tex}")
    return csv
