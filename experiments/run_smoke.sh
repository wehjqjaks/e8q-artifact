#!/usr/bin/env bash
# AVX-512 smoke test — run BEFORE the full campaign (README.md).
#
# sift1M × b in 1..8 × {e8q, me8q, rabitq, saq, caq} on the full 53-point
# nprobe grid, one trial, identical protocol to the MAIN grid (same seed per
# (dataset, bits) cell, same pin, same thread count). Cheap (~hours, not days)
# and decisive: it checks that the AVX-512 e8qlib build keeps the advantage
# the AVX2 build showed against RaBitQ. A regression here means the full
# campaign is not worth running.
#
#   ./experiments/run_smoke.sh [--run-id ID] [--data-root DIR] [--out DIR]
#
# Env knobs are the same as run_all.sh (E8Q_CPUS, E8Q_THREADS, E8Q_NQ,
# E8Q_SAVE_INDEXES, SEED_OVERRIDE, BASELINES). saq/caq are included only when
# the host has AVX-512 and the vendored binaries build.
#
# Verdict (printed at the end, exit code 0 = pass, 2 = fail):
#   per bits, QPS of e8q and me8q interpolated at 0.95 × that method's own
#   recall ceiling, divided by RaBitQ's QPS at the same recall point.
#   PASS when e8q beats RaBitQ at every bits where both curves reach the
#   point. The per-bits table is the thing to read; the gate is only a floor.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
source "$HERE/lib.sh"
lib_init

DATA_ROOT="$REPO/datasets"
OUT_ROOT="results"
RUN_ID="smoke_$(date +%Y%m%d_%H%M%S)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --out) OUT_ROOT="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done

if ! grep -q avx512f /proc/cpuinfo 2>/dev/null; then
    echo "smoke: host has no AVX-512 (avx512f missing in /proc/cpuinfo)." >&2
    echo "  this check is meaningless on an AVX2 machine — run it on the server." >&2
    exit 1
fi
if [[ "${E8Q_ISA}" != "avx512" ]]; then
    echo "smoke: e8qlib is built for ${E8Q_ISA}, not avx512." >&2
    echo "  rebuild: pip install . -Ccmake.define.E8Q_ISA=avx512" >&2
    exit 1
fi

# saq/caq binaries, same auto-build rule as run_all.sh step 0.
if { [[ ! -x "$REPO/baselines/SAQ/bin/test_qps" ]] || \
     [[ ! -x "$REPO/baselines/SAQ/bin/create_index" ]]; } && \
   command -v cmake >/dev/null 2>&1; then
    echo "smoke: building vendored SAQ"
    cmake -S "$REPO/baselines/SAQ" -B "$REPO/baselines/SAQ/build" \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build "$REPO/baselines/SAQ/build" -j"$(nproc)" || \
        echo "smoke: SAQ build failed; saq/caq arms will be skipped"
fi

export RESULTS_ROOT="$OUT_ROOT"
export DATASETS_OVERRIDE="sift1M"
export E8Q_BITS="${E8Q_BITS:-1 2 3 4 5 6 7 8}"
export E8Q_TRIALS=1
export BASELINES="${BASELINES:-all}"
export E8Q_METHODS="e8q rabitq me8q"

echo "smoke: sift1M bits=[$E8Q_BITS] methods=[$E8Q_METHODS + baselines]"
echo "  pin=${E8Q_CPUS:-none} threads=$E8Q_THREADS isa=$E8Q_ISA -> $OUT_ROOT/$RUN_ID/stable"
bash "$HERE/exp_stable.sh" "$DATA_ROOT" "$RUN_ID"

OUT="$(new_run "$OUT_ROOT" "$RUN_ID" stable)"
"$E8Q_PYTHON" - "$OUT" "$E8Q_THREADS" <<'PY'
import json, math, sys
from pathlib import Path

out, threads = Path(sys.argv[1]), sys.argv[2]
# QPS where the recall curve crosses 0.95 x the method's own ceiling
# (log-QPS interpolation). NaN when the curve never reaches it.
def qps_at(rows, method, target):
    pts = sorted((r.get(f"{method}_recall"), r.get(f"{method}_qps"))
                 for r in rows)
    pts = [(r, q) for r, q in pts if r is not None and q is not None and q > 0]
    if not pts or max(r for r, _ in pts) < target:
        return float("nan")
    below = [(r, q) for r, q in pts if r <= target]
    anchor = below[-1] if below else None
    if anchor and anchor[0] == target:
        return anchor[1]
    above = [(r, q) for r, q in pts if r >= target]
    if not anchor or not above:
        return float("nan")
    lo, hi = anchor, above[0]
    t = (target - lo[0]) / max(hi[0] - lo[0], 1e-12)
    return math.exp(math.log(lo[1]) + t * (math.log(hi[1]) - math.log(lo[1])))

print()
print(f"{'bits':>4}  {'e8q ceiling':>11}  {'e8q/rabitq':>10}  "
      f"{'me8q/rabitq':>11}  {'saq/rabitq':>10}  {'caq/rabitq':>10}")
fail = []
seen = 0
for bits in range(1, 9):
    files = {m: out / f"sift1M_b{bits}_t{threads}_{m}.json"
             for m in ("e8q", "me8q", "rabitq", "saq", "caq")}
    if not files["e8q"].is_file() or not files["rabitq"].is_file():
        print(f"{bits:>4}  MISSING ledger "
              f"(e8q={files['e8q'].is_file()} rabitq={files['rabitq'].is_file()})")
        fail.append(bits)
        continue
    seen += 1
    runs = {m: json.loads(p.read_text()) for m, p in files.items() if p.is_file()}

    def ceiling(m):
        vals = [r.get(f"{m}_recall") for r in runs[m]["rows"]]
        vals = [float(v) for v in vals if v is not None]
        return max(vals) if vals else float("nan")

    def ratio(m):
        if m not in runs:
            return None
        tgt = 0.95 * ceiling(m)
        a, b = qps_at(runs[m]["rows"], m, tgt), qps_at(runs["rabitq"]["rows"], "rabitq", tgt)
        return a / b if b and b == b and a == a else float("nan")

    ce = ceiling("e8q")
    ratios = {m: ratio(m) for m in ("e8q", "me8q", "saq", "caq")}
    cell = lambda v: "     n/a" if v is None else (f"{v:10.2f}" if v == v else "       nan")
    print(f"{bits:>4}  {ce:11.4f}  {cell(ratios['e8q'])}  {cell(ratios['me8q'])}  "
          f"{cell(ratios['saq'])}  {cell(ratios['caq'])}")
    if not (ratios["e8q"] == ratios["e8q"] and ratios["e8q"] > 1.0):
        fail.append(bits)

print()
if not seen:
    print("VERDICT: FAIL — no complete e8q/rabitq ledgers")
    sys.exit(2)
if fail:
    print(f"VERDICT: FAIL — e8q does not beat rabitq at bits {fail}")
    print("  do NOT start the full campaign; the AVX-512 build regressed.")
    sys.exit(2)
print("VERDICT: PASS — e8q beats rabitq at 0.95 x recall ceiling on sift1M, b1-8.")
print("  full campaign: SEED_OVERRIDE=<seed> E8Q_TRIALS=3 nohup ./run_all.sh "
      "--run-id <id> > run.log 2>&1 &")
PY
