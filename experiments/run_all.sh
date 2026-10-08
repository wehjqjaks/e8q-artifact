#!/usr/bin/env bash
# Paper data acquisition — every experiment the submitted figures and tables
# read, written into the layout plot_all.sh consumes.
#
#   ./experiments/run_all.sh [--data-root DIR] [--results DIR] [--run-id ID]
#                            [--stages LIST] [--no-prepare] [--baselines all|none]
#
# Stages (default `main adr bound shaping`; `diag` is opt-in):
#
#   main     results/fixed/, results/mixed/     the rate grid (exp_stable.sh)
#            results/ablation/                  the two plotted search-knob sweeps
#   adr      results/adr_full/                  full-grid distance-ratio side-cars
#            results/_adr_stage/indexes/        the .idx files bound/ probes
#   bound    results/bound_coverage/            Appendix A.4 screening-bound coverage
#   shaping  results/shaping/                   Appendix B distortion/shaping spectrum
#   diag     results/{phases,latency,perf}/     README.md micro-benchmarks
#
# `main` is what the body figures need:
#   * fixed rate — E8Q / RaBitQ / CAQ at every integer bit 1..8 and PQ at
#     b in {2,4,8} on the six paper corpora.  The appendix's full-spectrum
#     table walks all eight rates for every corpus, so the sparse {2,4,8}
#     grid the campaign used is NOT enough here.
#   * mixed rate — ME8Q / SAQ on the nine nominal budgets the mixed figures
#     and the matched-storage table read (0.5 … 8).
#   * ablation — the screening-margin (alpha) sweep and the top-k sweep on
#     openai1536 b=4, the two Section 4.3 panels that are not the main grid.
# The alpha and top-k arms are search-time knobs and share one index each,
# which is why they run here rather than through exp_stable's per-cell loop.
#
# Datasets default to the six the paper evaluates; the RAM-guarded arms
# (msmarco10M, stella-trec24-1024) skip themselves on a small host with a
# `ram-skip:` line and no ledger, and are picked up when re-invoked on a
# bigger one (README.md).  E8Q_MMAP_BASE=1 streams those two corpora
# from disk instead: recall is bit-identical, QPS is not, so it is for the
# ratio/coverage stages and never for a QPS-bearing comparison.
#
# Every stage is resume-safe: a complete ledger (or an already-written
# side-car) is skipped, so re-invoking with the same --run-id fills holes.
#
# Env knobs (see lib.sh): DATASETS_OVERRIDE, E8Q_THREADS, E8Q_GRID, E8Q_NQ,
# E8Q_TRIALS, E8Q_CPU/E8Q_CPUS, E8Q_MMAP_BASE, E8Q_RAM_MARGIN, BASELINES.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
source "$HERE/lib.sh"
lib_init

# The six corpora the paper evaluates (README.md).
PAPER_DATASETS="sift1M deep1M gist openai1536 msmarco10M stella-trec24-1024"
# Nominal bit budgets of the mixed-rate figures (plot_chapter4.MIXED_BUDGETS).
PAPER_MIXED_BITS="0.5 1 1.5 2 3 4 5 6 8"
# The two Section 4.3 knob sweeps, both on openai1536 b=4.
SWEEP_DS="openai1536"
SWEEP_BITS="4"
SWEEP_ALPHAS="0.001 0.01 0.1 0.5 1 1.5 2"
SWEEP_KS="1 10 100"

DATA_ROOT="$REPO/datasets"
RESULTS="$REPO/results"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
STAGES="main adr bound shaping"
PREPARE=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --results) RESULTS="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --baselines) export BASELINES="$2"; shift 2 ;;
        --no-prepare) PREPARE=0; shift ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done
export BASELINES="${BASELINES:-all}"
export RESULTS_ROOT="$RESULTS"
# The paper's stages all sweep the full nprobe ladder, so a ledger with fewer
# rows than that is a crash leftover, not a result.  The diag suite uses short
# grids and lowers this per arm.
export E8Q_MIN_ROWS="${E8Q_MIN_ROWS:-50}"

want() { [[ " $STAGES " == *" $1 "* ]]; }
mkdir -p "$RESULTS"
echo "run_all: data=$DATA_ROOT results=$RESULTS run=$RUN_ID stages=[$STAGES]"
echo "  (ram guard margin ${E8Q_RAM_MARGIN:-6} GiB; skipped arms are retried on"
echo "   bigger machines — see README.md)"

# ---------------------------------------------------------------------------
# 0. corpus readiness
# ---------------------------------------------------------------------------
if [[ "$PREPARE" == "1" ]] && want main; then
    echo "==================================================================="
    echo "dataset readiness"
    for ds in ${DATASETS_OVERRIDE:-$PAPER_DATASETS}; do
        if [[ -f "$DATA_ROOT/$ds/${ds}_base.fvecs" ]]; then
            echo "  $ds: ready"
            continue
        fi
        echo "  $ds: missing -> datasets/prepare.sh $ds"
        bash "$REPO/datasets/prepare.sh" "$ds" || \
            echo "  $ds: prepare failed (manual acquisition?); skipping"
    done
fi

# ---------------------------------------------------------------------------
# main — the rate grid, then the two knob sweeps
# ---------------------------------------------------------------------------
if want main; then
    echo "==================================================================="
    echo "main: rate grid"
    # Every corpus gets the full integer ladder: the appendix's full-spectrum
    # table reads b = 1..8 on all six, so E8Q_BITS_FULL_DS covers the set.
    export DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-$PAPER_DATASETS}"
    export E8Q_BITS="${E8Q_BITS:-1 2 3 4 5 6 7 8}"
    export E8Q_BITS_FULL_DS="${E8Q_BITS_FULL_DS:-$DATASETS_OVERRIDE}"
    export E8Q_MIXED_BITS="${E8Q_MIXED_BITS:-$PAPER_MIXED_BITS}"
    bash "$HERE/exp_stable.sh" "$DATA_ROOT" "$RUN_ID"

    # exp_stable writes one directory per method family; the plotters read the
    # split, so copy each ledger to the side it belongs to.  The work tree
    # stays put: it is what exp_stable's resume gate reads on a re-invocation.
    echo "== sorting $RESULTS/$RUN_ID/stable -> fixed/ mixed/"
    mkdir -p "$RESULTS/fixed" "$RESULTS/mixed"
    for f in "$RESULTS/$RUN_ID/stable"/*.json "$RESULTS/$RUN_ID/stable"/*.skipped; do
        [[ -f "$f" ]] || continue
        case "$(basename "$f")" in
            *_me8q.*|*_saq.*) cp -f "$f" "$RESULTS/mixed/" ;;
            *) cp -f "$f" "$RESULTS/fixed/" ;;
        esac
    done
    echo "  fixed: $(ls "$RESULTS/fixed"/*.json 2>/dev/null | wc -l) ledgers" \
         "mixed: $(ls "$RESULTS/mixed"/*.json 2>/dev/null | wc -l) ledgers"

    echo "== ablation: screening margin (alpha) and top-k sweeps"
    DEST="$RESULTS/ablation" STAGE="$RESULTS/$RUN_ID/ablation_stage" \
        DS="$SWEEP_DS" BITS="$SWEEP_BITS" ALPHAS="$SWEEP_ALPHAS" \
        bash "$HERE/_alpha_local.sh"

    OUT="$RESULTS/ablation"
    IDX="$RESULTS/$RUN_ID/ablation_stage/indexes/${SWEEP_DS}_b${SWEEP_BITS}_t${E8Q_THREADS}"
    mkdir -p "$OUT" "$(dirname "$IDX")"
    for method in e8q rabitq; do
        for k in $SWEEP_KS; do
            dst="$OUT/abl_k_${SWEEP_DS}_b${SWEEP_BITS}_k${k}_${method}.json"
            if ledger_ok "$dst"; then
                echo "  (k=$k $method) skip, complete"
                continue
            fi
            rm -f "$dst"
            echo "== k=$k $method -> $dst"
            run_py "$method" "$DATA_ROOT" "$SWEEP_DS" "$SWEEP_BITS" "$OUT" \
                --seed "$(batch_seed "$SWEEP_DS")" --k "$k" \
                --index-path "${IDX}_${method}.idx" \
                || echo "  k=$k $method: FAILED (logged)"
            src="$(ledger_json "$OUT" "$SWEEP_DS" "$method" "$SWEEP_BITS")"
            if [[ -f "$src" ]]; then
                mv "$src" "$dst"
            else
                echo "  k=$k $method: no ledger (ram guard or missing corpus)"
            fi
        done
    done
fi

# ---------------------------------------------------------------------------
# adr — full-grid distance-ratio side-cars, and the indexes bound/ probes
# ---------------------------------------------------------------------------
if want adr; then
    echo "==================================================================="
    echo "adr: distance-ratio fill (also leaves the .idx files bound/ reads)"
    DEST="$RESULTS/adr_full" STAGE="$RESULTS/_adr_stage" \
        DATA_ROOT="$DATA_ROOT" bash "$HERE/_adr_full.sh"
fi

# ---------------------------------------------------------------------------
# bound — Appendix A.4 empirical coverage of the screening bound
# ---------------------------------------------------------------------------
if want bound; then
    echo "==================================================================="
    echo "bound: screening-bound coverage"
    if [[ -f "$RESULTS/bound_coverage/bound_coverage.json" && "${FORCE:-0}" != "1" ]]; then
        echo "  already present (FORCE=1 to re-measure)"
    else
        mkdir -p "$RESULTS/bound_coverage"
        # It probes the stored codes of the E8Q indexes, so it reads the same
        # staging tree the adr stage leaves behind.
        "$E8Q_PYTHON" "$HERE/bound_coverage.py" \
            --data-root "$DATA_ROOT" \
            --index-root "${INDEX_ROOT:-$RESULTS/_adr_stage/indexes}" \
            --out "$RESULTS/bound_coverage"
    fi
fi

# ---------------------------------------------------------------------------
# shaping — Appendix B distortion / shaping-gain spectrum
# ---------------------------------------------------------------------------
if want shaping; then
    echo "==================================================================="
    echo "shaping: distortion spectrum"
    BIN="$HERE/shaping/bin/measure_shaping"
    if [[ ! -x "$BIN" ]]; then
        echo "  building measure_shaping"
        # Self-contained recipe from the source header; the codec, rotator and
        # SIMD kernels come from src/ like any other tool.
        (cd "$HERE/shaping" && g++ -O2 -std=c++17 -I"$REPO/include" -I"$REPO/src" \
            measure_shaping.cpp "$REPO/src/quantization/e8q_codec.cpp" \
            "$REPO/src/utils/rotator.cpp" "$REPO/src/simd/kernels_generic.cpp" \
            "$REPO/src/simd/kernels_dispatch.cpp" "$REPO/src/simd/dispatch.cpp" \
            -o measure_shaping)
    fi
    bash "$HERE/shaping/run_all.sh" --out "$RESULTS/shaping"
fi

# ---------------------------------------------------------------------------
# diag — README.md micro-benchmarks (not part of the submitted figures)
# ---------------------------------------------------------------------------
if want diag; then
    echo "==================================================================="
    echo "diag: phases / latency / perf (README.md; not plotted by the paper)"
    run_arm() {
        # run_arm <method> <ds> <bits> <out> <seed> [extra runner flags...]
        local method="$1" ds="$2" bits="$3" out="$4" seed="$5"
        shift 5
        mkdir -p "$out"
        local json
        json="$(ledger_json "$out" "$ds" "$method" "$bits")"
        if ledger_ok "$json"; then
            echo "  ($method $ds b$bits) skip, complete"
            return 0
        fi
        rm -f "$json"
        run_py "$method" "$DATA_ROOT" "$ds" "$bits" "$out" --seed "$seed" "$@" \
            || echo "  $method $ds b$bits: FAILED (logged)"
    }
    E8Q_MIN_ROWS=1 E8Q_GRID="16,64,128,256" \
        run_arm e8q "$SWEEP_DS" "$SWEEP_BITS" "$RESULTS/phases" \
        "$(batch_seed "$SWEEP_DS")" --phases
    for method in e8q rabitq; do
        E8Q_MIN_ROWS=1 E8Q_GRID="64" \
            run_arm "$method" "$SWEEP_DS" "$SWEEP_BITS" "$RESULTS/latency" \
            "$(batch_seed "$SWEEP_DS")" --latency-nq 10000
    done
    if command -v perf >/dev/null 2>&1; then
        mkdir -p "$RESULTS/perf"
        for method in e8q rabitq caq pq; do
            dest="$RESULTS/perf/perf_${method}_${SWEEP_DS}_b${SWEEP_BITS}.txt"
            [[ -s "$dest" ]] && { echo "  ($method) skip, complete"; continue; }
            E8Q_MIN_ROWS=1 E8Q_GRID="64" E8Q_REPEATS=3 E8Q_MINMS=0 \
                perf stat -e instructions,cycles,L1-dcache-load-misses,\
L1-dcache-loads,LLC-load-misses,LLC-loads,branch-misses,branches \
                -o "$dest" -- \
                "${PIN[@]}" "$E8Q_PYTHON" "$HERE/run.py" --method "$method" \
                --data-root "$DATA_ROOT" --dataset "$SWEEP_DS" \
                --bits "$SWEEP_BITS" --grid 64 --threads "$E8Q_THREADS" \
                --nq "$E8Q_NQ" --warmup 1 --repeats 3 --min-ms 0 \
                --out "$RESULTS/perf" --seed "$(batch_seed "$SWEEP_DS")" \
                || echo "  perf $method: FAILED (logged)"
        done
    fi
fi

echo "run_all done: $RESULTS"
if [[ -d "$RESULTS/_adr_stage/indexes" ]]; then
    echo "  index staging tree: $RESULTS/_adr_stage/indexes ($(du -sh "$RESULTS/_adr_stage/indexes" 2>/dev/null | cut -f1))"
    echo "  bound/ needs it; delete it to reclaim the space and re-run --stages adr."
fi
echo "next: ./plot_all.sh --results $RESULTS"
