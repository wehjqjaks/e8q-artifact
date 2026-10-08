#!/usr/bin/env bash
# One-shot construction-cost sweep behind the two tables that report build
# cost, on the construction corpora and rates both tables use
# (MSMARCO-10M and Stella at b = 4 and 8):
#
#   main      Table 3 (tab:exp-build):  E8Q / RaBitQ / CAQ
#   appendix  the appendix construction table: ME8Q / SAQ, with memory
#
# Both tables need only the BUILD.  Every arm runs with --build-only, which
# stops it the moment the index exists: no nprobe sweep, no warm-up rounds, no
# repeated rounds.  A cell therefore costs one corpus read plus one build, and
# `all` costs 5 arms x 2 datasets x 2 rates = 20 builds and nothing else.
#
# Each arm records, in one ledger per cell:
#   <m>_build_s             wall seconds around the build call
#   <m>_build_kvps          n / build_s, in thousands of vectors/s
#   <m>_mem_gb              index size after construction (the artifact)
#   <m>_bytes_per_vector    the same per stored vector
#   <m>_build_peak_gb       peak process RSS during construction
#
#   bash experiments/table3_build.sh [main|appendix|all] [outdir]
#
# Re-running skips any cell whose ledger is already complete, so an interrupted
# or partially-failed sweep resumes; an me8q ledger built before run_cell
# passed --match-total-budget is rebuilt rather than kept (see
# me8q_protocol_stale).  Outputs: the raw ledgers under <outdir>/<method>/, a
# JSON summary at <outdir>/table3_build.json, the table bodies at
# <outdir>/tab_build.tex, tab_build.csv and tab_build_appendix.tex, and a
# printed report.  Table 3's two bodies are copied into the paper by hand,
# because this sweep -- not the fixed-rate campaign -- is what its numbers
# come from:
#
#   cp <outdir>/tab_build.tex <outdir>/tab_build.csv <paper>/figures/
#
# The appendix arm (ME8Q / SAQ) is collected as evidence; the paper carries no
# construction table for the mixed-rate methods.
#
# On the server (Alibaba Cloud ECS r7.6xlarge, Xeon 8369B, 12 physical cores,
# AVX-512) -- the machine Table 3's numbers were taken on:
#
#   conda activate e8q
#   pip install . --force-reinstall --no-deps        # the optimized codec
#   cd <repo> && CUDA_VISIBLE_DEVICES= \
#     bash experiments/table3_build.sh all results/table3_build_server
#
# Each cell is pinned to the 12 physical cores and gets README.md's
# per-dataset batch seed (batch_seed "$ds"), so every method and bit width of
# a dataset shares one rotation -- the same seeds the campaign ledgers carry.
# Build threads: every arm, rabitq's construction included, builds at
# E8Q_THREADS (12) -- the script sets the thread env below, no export needed.
#
# `all` is 5 arms x 2 datasets x 2 rates = 20 builds, each one corpus read plus
# one construction and no search.  Both corpora together are ~100 GB of fvecs,
# so run it on the machine that holds them.  The script defaults to the server's
# own geometry (12 threads, the 0,2,4.. pin list); override E8Q_THREADS /
# E8Q_CPUS to reproduce a different reference.  Paste the printed report and
# results/table3_build_server/table3_build.json back, then copy the three table
# bodies into the paper as above.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
lib_init

TABLE="${1:-all}"
OUT="${2:-$REPO/results/table3_build}"
DATA_ROOT="${DATA_ROOT:-$REPO/datasets}"

DATASETS="${TB_DATASETS-msmarco10M stella-trec24-1024}"
TB_BITS="${TB_BITS-4 8}"
export E8Q_THREADS="${E8Q_THREADS:-12}"
# Build thread width, belt and braces. run_rabitq follows --threads now and
# records the effective value as rabitq_build_threads, but pin the knob here
# too so a sweep can never silently oversubscribe the pinned cores. Note that
# OMP_NUM_THREADS is not what controls the vendored arms -- e8qlib and RaBitQ
# carry explicit num_threads clauses and RaBitQ clamps against
# hardware_concurrency(), which reads no environment -- it only keeps a
# library that falls back to the OpenMP default inside the pin.
export RABITQ_CONSTRUCT_THREADS="${RABITQ_CONSTRUCT_THREADS:-$E8Q_THREADS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$E8Q_THREADS}"
# One grid point. --build-only never reaches the sweep; SAQ's arm keeps a
# single point so its ledger keeps the row shape the report reads.
export E8Q_GRID="${E8Q_GRID:-1}"
export E8Q_WARMUP=0 E8Q_REPEATS=1 E8Q_MINMS=0 E8Q_NQ="${E8Q_NQ:-100}"

# The reference pin list is one entry per physical core (0,2,4,...). On a host
# with fewer CPUs the out-of-range entries make taskset fail, which would abort
# every run, so keep only the CPUs that exist.
if [[ -n "${E8Q_CPUS:-}" ]]; then
    # Count online CPUs with getconf, not `nproc`: coreutils >= 9.4 makes plain
    # `nproc` report OMP_NUM_THREADS, and this script exports that below, so a
    # `-lt "$(nproc)"` filter silently keeps only the first E8Q_THREADS CPUs --
    # the reference list collapses to 0,2,4,6,8,10 and 12 threads land on the
    # first 6 physical cores.  That is a 2x build-time measurement error.
    _ncpu="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc --all)"
    _pin=()
    IFS=, read -ra _cpus <<< "$E8Q_CPUS"
    for _c in "${_cpus[@]}"; do
        [[ "$_c" =~ ^[0-9]+$ && "$_c" -lt "$_ncpu" ]] && _pin+=("$_c")
    done
    if [[ ${#_pin[@]} -gt 0 ]]; then
        E8Q_CPUS="$(IFS=,; printf '%s' "${_pin[*]}")"
        PIN=(taskset -c "$E8Q_CPUS")
    else
        PIN=()
    fi
    # Loud, because under-pinning does not fail on its own: it just makes every
    # cell slower and every method's build time incomparable to the paper's.
    if [[ ${#_pin[@]} -lt "$E8Q_THREADS" ]]; then
        echo "WARN: pinned ${#_pin[@]} CPUs ($E8Q_CPUS) but E8Q_THREADS=$E8Q_THREADS" >&2
    fi
fi

mkdir -p "$OUT"
echo "[$(date '+%F %T')] table=$TABLE out=$OUT isa=${E8Q_ISA:-?} threads=$E8Q_THREADS"
echo "  datasets: $DATASETS   rates: $TB_BITS   CPUs: ${E8Q_CPUS:-all}"

me8q_protocol_stale() {  # me8q_protocol_stale <method> <ledger>
    # An me8q ledger describes the index the paper evaluates only if it was
    # built under the true-bit-budget protocol: without the budget fit,
    # run_me8q records no unpadded footprint at all.  A ledger that lacks one
    # predates run_cell's --match-total-budget and must be rebuilt -- the
    # resume gate alone would happily keep it.
    [[ "$1" == "me8q" ]] || return 1
    "${E8Q_PYTHON:-python3}" -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)          # unreadable here: treat as current
sys.exit(0 if d.get("me8q_unpadded_bpv", -1.0) <= 0 else 1)
' "$2"
}

run_cell() {  # run_cell <method> <dataset> <bits>
    local m="$1" ds="$2" b="$3"
    local json
    json="$(ledger_json "$OUT/$m" "$ds" "$m" "$b")"
    if ledger_ok "$json"; then
        if me8q_protocol_stale "$m" "$json"; then
            echo "  rebuild $(basename "$json"): built without the" \
                 "true-bit-budget protocol"
        else
            echo "  skip $(basename "$json")"
            return 0
        fi
    fi
    rm -f "$json"
    local extra=()
    case "$m" in
        # SAQ/CAQ own their relative-error probe; it is not a construction
        # column, so it is switched off rather than paid for.
        saq|caq) extra=(--relerr 0) ;;
        # ME8Q's build cell must describe the index the body evaluates.  The
        # mixed-rate arm of the campaign runs the true-bit-budget protocol
        # (exp_stable.sh passes --match-total-budget), which charges ME8Q's
        # per-vector overhead against the b*dim/8 ceiling; a nominal build
        # lands ~3% over it (527.4 against 512 B/vec on MSMARCO b=4) at a
        # different DP plan.  Its build_s is the whole budget fit -- the
        # nominal build plus the rebuilds that reach the deployed plan.
        me8q)    extra=(--match-total-budget) ;;
    esac
    # README.md: one batch seed per dataset drives every method and
    # every bit width, so the rotation is identical across a dataset's cells.
    # Same cksum draw the campaign drivers pass; SEED_OVERRIDE still wins.
    # Without this the runners fall back to their --seed default of 0.
    echo "[$(date '+%F %T')] build-only $m $ds b$b seed=$(batch_seed "$ds")"
    run_py "$m" "$DATA_ROOT" "$ds" "$b" "$OUT/$m" --build-only \
        --seed "$(batch_seed "$ds")" ${extra[@]+"${extra[@]}"} \
        || echo "  FAILED $m $ds b$b"
}

run_table() {  # run_table <main|appendix>
    local arms
    if [[ "$1" == "main" ]]; then arms="e8q rabitq caq"; else arms="me8q saq"; fi
    # Resume: run_cell skips any cell whose ledger is already complete (see
    # ledger_ok), so an interrupted or partly-failed sweep continues where it
    # stopped instead of rebuilding what it measured. Say how much that is.
    local n_done=0 n_total=0 json
    for m in $arms; do
        if [[ ( "$m" == "saq" || "$m" == "caq" ) ]] && ! have_saq_runtime; then
            continue
        fi
        for b in $TB_BITS; do
            for ds in $DATASETS; do
                n_total=$((n_total + 1))
                json="$(ledger_json "$OUT/$m" "$ds" "$m" "$b")"
                if ledger_ok "$json"; then n_done=$((n_done + 1)); fi
            done
        done
    done
    echo "[$(date '+%F %T')] resume: $((n_total - n_done)) of $n_total cells to run ($n_done already complete)"
    # The saq/caq arms shell out to the vendored binaries; a stale one is a
    # silent measurement error (see saq_binaries_stale in lib.sh).
    if saq_binaries_stale; then
        echo "WARN: baselines/SAQ/bin is older than the vendored sources;" \
             "rebuild it on this host before trusting saq/caq numbers" >&2
    fi
    for m in $arms; do
        if [[ ( "$m" == "saq" || "$m" == "caq" ) ]] && ! have_saq_runtime; then
            echo "  $m: SAQ runtime not built (baselines/SAQ/bin) -- skipping"
            continue
        fi
        for b in $TB_BITS; do
            for ds in $DATASETS; do
                run_cell "$m" "$ds" "$b"
            done
        done
    done
}

case "$TABLE" in
    main)     run_table main ;;
    appendix) run_table appendix ;;
    all)      run_table main; run_table appendix ;;
    *)        echo "usage: table3_build.sh [main|appendix|all] [outdir]" >&2; exit 2 ;;
esac

# ------------------------------------------------------------------ report
# Reads the ledgers just written and prints the two tables plus the memory
# columns, in the paper's own units (seconds / KVPS, GB, bytes per vector).
"${E8Q_PYTHON:-python3}" - "$OUT" "$DATASETS" "$TB_BITS" "$TABLE" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
datasets = sys.argv[2].split()
bits = [int(b) for b in sys.argv[3].split()]
table = sys.argv[4]

# Dataset labels as the paper writes them.
LABEL = {
    "msmarco10M": "MSMARCO-10M",
    "stella-trec24-1024": "Stella",
}
MNAME = {"e8q": "E8Q", "rabitq": "RaBitQ", "caq": "CAQ",
         "me8q": "ME8Q", "saq": "SAQ"}


def cell(method, ds, b):
    # The ledger name carries the thread count; the table does not care which
    # it was, and a re-run under a different -threads must not read as missing,
    # so take whatever cell exists (newest wins if several do).
    hits = sorted(
        (out / method).glob(f"{ds}_b{b}_t*_{method}.json"),
        key=lambda q: q.stat().st_mtime,
    )
    if not hits:
        return None
    p = hits[-1]
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    build_s = d.get(f"{method}_build_s", -1.0)
    # ME8Q reaches the protocol's footprint in two construction passes: a
    # nominal build, then the one whose plan the evaluation searches.  The
    # table reports the deployed pass, since that is the index the search
    # cost of Section 4.4 belongs to; <m>_build_s keeps the fit total, which
    # is what a user pays to land under a stated ceiling.
    last_s = d.get(f"{method}_build_last_s", -1.0)
    if last_s > 0:
        build_s = last_s
    n = d.get("n", 0)
    return {
        "build_s": build_s,
        "build_fit_s": d.get(f"{method}_build_s", -1.0),
        "kvps": (n / build_s / 1000.0) if build_s > 0 else -1.0,
        "mem_gb": d.get(f"{method}_mem_gb"),
        "bpv": d.get(f"{method}_bytes_per_vector"),
        "peak_gb": d.get(f"{method}_build_peak_gb"),
        "path": str(p),
    }


def fmt(v, spec, dash="--"):
    return dash if v is None or v < 0 else format(v, spec)


def pair(c):
    """The seconds / KVPS cell, or a single dash when the arm never ran."""
    if c is None or c["build_s"] < 0:
        return "--"
    return f"{fmt(c['build_s'], '.1f')}" + r"\,\,/\,\," + fmt(c["kvps"], ".1f")


def pair_csv(c):
    if c is None or c["build_s"] < 0:
        return "--, --"
    return f"{fmt(c['build_s'], '.1f')},{fmt(c['kvps'], '.1f')}"


def load_arms(arms):
    cells = {}
    for m in arms:
        for b in bits:
            for ds in datasets:
                cells[(m, ds, b)] = cell(m, ds, b)
    return cells


main_arms = ["e8q", "rabitq", "caq"]
app_arms = ["me8q", "saq"]
summary = {}

if table in ("main", "all"):
    cs = load_arms(main_arms)
    print("\n=== Table 3 (tab:exp-build): build seconds / KVPS ===")
    lines = []
    for ds in datasets:
        for b in bits:
            cells = [cs[(m, ds, b)] for m in main_arms]
            print(f"  {LABEL.get(ds, ds):12s} b={b}  " + "   ".join(
                "--" if c is None
                else f"{fmt(c['build_s'], '6.1f')} / {fmt(c['kvps'], '5.1f')}"
                for c in cells
            ))
            lines.append((ds, b, cells))
    tex = [
        r"\begin{tabular}{@{}l c rrr@{}}",
        r"\toprule",
        r"Dataset & $b$ & E8Q & RaBitQ & CAQ \\",
        r"\midrule",
    ]
    csv = ["Dataset,b," + ",".join(
        f"{MNAME[m]} time (s),{MNAME[m]} KVPS" for m in main_arms)]
    for ds, b, cells in lines:
        tex.append(f"{LABEL.get(ds, ds)} & {b} & "
                   + " & ".join(pair(c) for c in cells) + r" \\")
        csv.append(",".join([LABEL.get(ds, ds), str(b)]
                            + [pair_csv(c) for c in cells]))
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_build.tex").write_text("\n".join(tex) + "\n")
    (out / "tab_build.csv").write_text("\n".join(csv) + "\n")
    summary["table3"] = {
        f"{ds}_b{b}_{m}": cs[(m, ds, b)] for m in main_arms for ds in datasets
        for b in bits
    }

if table in ("appendix", "all"):
    cs = load_arms(app_arms)
    print("\n=== Appendix construction table: build seconds / KVPS, "
          "index GB, bytes/vector, peak RSS GB ===")
    lines = []
    for ds in datasets:
        for b in bits:
            for m in app_arms:
                c = cs[(m, ds, b)]
                if c is None:
                    print(f"  {LABEL.get(ds, ds):12s} b={b} {m:6s}  --")
                    lines.append((ds, b, m, None))
                    continue
                print(
                    f"  {LABEL.get(ds, ds):12s} b={b} {m:6s}  "
                    f"{fmt(c['build_s'], '6.1f')} s / {fmt(c['kvps'], '5.1f')} KVPS"
                    f" / {fmt(c['mem_gb'], '5.2f')} GB"
                    f" / {fmt(c['bpv'], '7.2f')} B/vec"
                    f" / {fmt(c['peak_gb'], '5.2f')} GB peak"
                    f" / {fmt(c['build_fit_s'], '6.1f')} s fit"
                )
                lines.append((ds, b, m, c))
    tex = [
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c rr r rr r@{}}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{ME8Q} & \multicolumn{3}{c}{SAQ} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Dataset & $b$ & s / KVPS & GB & B/vec"
        r" & s / KVPS & GB & B/vec \\",
        r"\midrule",
    ]
    for ds in datasets:
        for b in bits:
            cols = []
            for m in app_arms:
                c = cs[(m, ds, b)]
                if c is None or c["build_s"] < 0:
                    cols += ["--", "--", "--"]
                else:
                    cols += [
                        pair(c),
                        fmt(c["mem_gb"], ".1f"),
                        fmt(c["bpv"], ".1f"),
                    ]
            tex.append(f"{LABEL.get(ds, ds)} & {b} & " + " & ".join(cols) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular*}"]
    (out / "tab_build_appendix.tex").write_text("\n".join(tex) + "\n")
    summary["appendix"] = {
        f"{ds}_b{b}_{m}": cs[(m, ds, b)] for m in app_arms for ds in datasets
        for b in bits
    }

(out / "table3_build.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
missing = [k for grp in summary.values() for k, v in grp.items() if v is None]
print(f"\nwrote {out}/table3_build.json")
if missing:
    print("MISSING cells: " + ", ".join(sorted(missing)))
PY
echo "[$(date '+%F %T')] done"
