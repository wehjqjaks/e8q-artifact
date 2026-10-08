"""Re-derive the abstract's headline claims straight from the frozen ledgers.

Uses experiments/plots/common.py::qps_at_recall (log-QPS interpolation in
recall) — the same function the paper's tables/figures are built from.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments/plots"))
from common import qps_at_recall  # noqa: E402

# Reads the ledgers run_all.sh wrote; pass another tree as argv[1].
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
FIX = RESULTS / "fixed"
MIX = RESULTS / "mixed"

CORPORA = {
    "sift1M": "SIFT",
    "deep1M": "DEEP",
    "gist": "GIST",
    "openai1536": "OpenAI-1536",
    "msmarco10M": "MSMARCO-10M",
    "stella-trec24-1024": "Stella",
}


def load(path: Path, stem: str):
    p = path / f"{stem}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def qps(run, m, t):
    return qps_at_recall(run["rows"], f"{m}_recall", f"{m}_qps", t)


def ceiling(run, m):
    return max(r[f"{m}_recall"] for r in run["rows"]
               if r.get(f"{m}_recall") is not None)


def peers(run, m):
    """Keys present in a row for method m."""
    return sorted(k for k in run["rows"][0] if k.startswith(m + "_"))


print("=" * 78)
print("A. b=4, QPS at 90% recall (log-QPS interp) — fixed-rate headline")
print("=" * 78)
hdr = f"{'corpus':<13}{'E8Q':>10}{'RaBitQ':>10}{'CAQ':>10}   {'E8Q/RaBitQ':>11}{'E8Q/CAQ':>9}"
print(hdr)
best_rq = best_caq = None
for stem, disp in CORPORA.items():
    e = load(FIX, f"{stem}_b4_t12_e8q")
    r = load(FIX, f"{stem}_b4_t12_rabitq")
    c = load(FIX, f"{stem}_b4_t12_caq")
    if e is None:
        print(f"{disp:<13}  (no e8q ledger)")
        continue
    qe, qr, qc = (qps(e, "e8q", 0.90), qps(r, "rabitq", 0.90) if r else float("nan"),
                  qps(c, "caq", 0.90) if c else float("nan"))
    rr = qe / qr if qr and math.isfinite(qr) and qr > 0 else float("nan")
    rc = qe / qc if qc and math.isfinite(qc) and qc > 0 else float("nan")
    print(f"{disp:<13}{qe:>10.0f}{qr:>10.0f}{qc:>10.0f}   "
          f"{rr:>10.4f}x{rc:>8.4f}x")
    if math.isfinite(rr) and (best_rq is None or rr > best_rq[1]):
        best_rq = (disp, rr, qe, qr)
    if math.isfinite(rc) and (best_caq is None or rc > best_caq[1]):
        best_caq = (disp, rc, qe, qc)
print(f"\n  max E8Q/RaBitQ @90% = {best_rq[1]:.4f}x on {best_rq[0]} "
      f"({best_rq[2]:.0f} vs {best_rq[3]:.0f} QPS) -> +{100*(best_rq[1]-1):.1f}%")
print(f"  max E8Q/CAQ    @90% = {best_caq[1]:.4f}x on {best_caq[0]} "
      f"({best_caq[2]:.0f} vs {best_caq[3]:.0f} QPS) -> +{100*(best_caq[1]-1):.1f}%")

print()
print("=" * 78)
print("A2. same, at 95% recall (for the '68.3% at 95%' sentence)")
print("=" * 78)
for stem, disp in CORPORA.items():
    e = load(FIX, f"{stem}_b4_t12_e8q")
    r = load(FIX, f"{stem}_b4_t12_rabitq")
    c = load(FIX, f"{stem}_b4_t12_caq")
    qe, qr, qc = (qps(e, "e8q", 0.95), qps(r, "rabitq", 0.95) if r else float("nan"),
                  qps(c, "caq", 0.95) if c else float("nan"))
    f = lambda v: f"{v:>9.0f}" if math.isfinite(v) else "        --"
    rr = qe / qr if math.isfinite(qr) and qr > 0 else float("nan")
    rc = qe / qc if math.isfinite(qc) and qc > 0 else float("nan")
    print(f"{disp:<13}{f(qe)}{f(qr)}{f(qc)}   "
          f"{rr if math.isfinite(rr) else float('nan'):>10.4f} {rc if math.isfinite(rc) else float('nan'):>9.4f}")

print()
print("=" * 78)
print("B. Recall ceiling = max recall over the 53-point grid, every b")
print("=" * 78)
for bits in (2, 4, 8):
    print(f"\n-- b={bits} --")
    print(f"{'corpus':<13}{'E8Q':>9}{'RaBitQ':>9}{'CAQ':>9}   {'d(RaBitQ)':>10}{'d(CAQ)':>9}")
    for stem, disp in CORPORA.items():
        e = load(FIX, f"{stem}_b{bits}_t12_e8q")
        r = load(FIX, f"{stem}_b{bits}_t12_rabitq")
        c = load(FIX, f"{stem}_b{bits}_t12_caq")
        if e is None:
            continue
        ce = ceiling(e, "e8q")
        cr = ceiling(r, "rabitq") if r else float("nan")
        cc = ceiling(c, "caq") if c else float("nan")
        dr, dc = 100 * (ce - cr), 100 * (ce - cc)
        print(f"{disp:<13}{ce:>9.4f}{cr:>9.4f}{cc:>9.4f}   {dr:>+9.2f}{dc:>+9.2f}")

print()
print("=" * 78)
print("C. 'only codec to reach 99% recall on GIST' — max recall vs 0.99")
print("=" * 78)
for bits in range(1, 9):
    e = load(FIX, f"gist_b{bits}_t12_e8q")
    r = load(FIX, f"gist_b{bits}_t12_rabitq")
    c = load(FIX, f"gist_b{bits}_t12_caq")
    p = load(FIX, f"gist_b{bits}_t12_pq")
    if e is None:
        continue
    ce, cr, cc = ceiling(e, "e8q"), ceiling(r, "rabitq"), ceiling(c, "caq")
    cp = ceiling(p, "pq") if p else float("nan")
    flag = "  <-- E8Q>=99%" if ce >= 0.99 else ""
    print(f"gist b={bits}  E8Q={ce:.4f}  RaBitQ={cr:.4f}  CAQ={cc:.4f}"
          + (f"  PQ={cp:.4f}" if math.isfinite(cp) else "") + flag)
e8 = load(FIX, "gist_b8_t12_e8q")
row = [r for r in e8["rows"] if r["e8q_recall"] >= 0.99]
if row:
    r0 = min(row, key=lambda r: r["e8q_recall"])
    print(f"  lowest-nprobe point at >=99% recall: nprobe={r0['nprobe']} "
          f"recall={r0['e8q_recall']:.4f} qps={r0['e8q_qps']:.0f}")

print()
print("=" * 78)
print("C2. GIST b=8 top grid rows, all codecs (tail of the sweep)")
print("=" * 78)
runs = {"E8Q": e8, "RaBitQ": load(FIX, "gist_b8_t12_rabitq"),
        "CAQ": load(FIX, "gist_b8_t12_caq")}
print(f"{'nprobe':>7}" + "".join(f"{k+'_rec':>12}{k+'_qps':>11}" for k in runs))
for i in range(len(e8["rows"]) - 1, max(len(e8["rows"]) - 9, -1), -1):
    line = f"{e8['rows'][i]['nprobe']:>7}"
    for k, run in runs.items():
        if run is None:
            line += f"{'--':>12}{'--':>11}"
            continue
        rr = run["rows"][i]
        line += f"{rr[f'{k.lower()}_recall']:>12.4f}{rr[f'{k.lower()}_qps']:>11.0f}"
    print(line)

print()
print("=" * 78)
print("D. ME8Q vs SAQ at matched total physical storage (results/mixed)")
print("=" * 78)
print(f"{'corpus':<13}{'b':>3}{'ME8Q@90':>10}{'SAQ@90':>10}{'ratio':>9}"
      f"   {'ME8Q ceil':>10}{'SAQ ceil':>9}{'d pts':>8}")
best = None
for stem, disp in CORPORA.items():
    for b in (2, 4, 8):
        e = load(MIX, f"{stem}_b{b}_t12_me8q")
        s = load(MIX, f"{stem}_b{b}_t12_saq")
        if e is None or s is None:
            continue
        qe, qs = qps(e, "me8q", 0.90), qps(s, "saq", 0.90)
        ce, cs = ceiling(e, "me8q"), ceiling(s, "saq")
        ratio = qe / qs if math.isfinite(qe) and math.isfinite(qs) and qs > 0 else float("nan")
        fe = f"{qe:>10.0f}" if math.isfinite(qe) else f"{'--':>10}"
        fs = f"{qs:>10.0f}" if math.isfinite(qs) else f"{'--':>10}"
        fr = f"{ratio:>8.2f}x" if math.isfinite(ratio) else f"{'--':>9}"
        print(f"{disp:<13}{b:>3}{fe}{fs}{fr}   {ce:>10.4f}{cs:>9.4f}"
              f"{100*(ce-cs):>+8.2f}")
        if math.isfinite(ratio) and (best is None or ratio > best[0]):
            best = (ratio, disp, b, qe, qs)
if best:
    print(f"\n  max ME8Q/SAQ @90% = {best[0]:.4f}x on {best[1]} b={best[2]} "
          f"({best[3]:.0f} vs {best[4]:.0f} QPS)")

print()
print("=" * 78)
print("D2. matched storage? physical bytes/vector as recorded in the ledgers")
print("=" * 78)
for stem, disp in CORPORA.items():
    for b in (2, 4, 8):
        e = load(MIX, f"{stem}_b{b}_t12_me8q")
        s = load(MIX, f"{stem}_b{b}_t12_saq")
        if e is None or s is None:
            continue
        eb = e.get("me8q_bytes_per_vector", float("nan"))
        sb = s.get("saq_bytes_per_vector", float("nan"))
        print(f"{disp:<13} b={b}  me8q={eb:10.3f} B/vec   saq={sb:10.3f} B/vec"
              f"   delta={100*(eb-sb)/sb:+6.2f}%   plan={e.get('me8q_plan')}"
              f"  code={e.get('me8q_bpv_code')} factor={e.get('me8q_bpv_factor')}")
