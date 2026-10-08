#!/usr/bin/env bash
# Shared configuration for every paper experiment script.
#
# Measurement protocol: warmup 3 untimed passes, >= 9 timed rounds per method
# AND >= 400 ms of timed signal, median QPS with p25/p75 IQR, k = 10,
# squared L2, no raw rerank, 12 search threads (one per physical core),
# shared IVF4096 centroids.
# Full data contract: README.md.
#
# JSON / index contract (every arm, every script):
#   <out>/<dataset>_b<bits>_t<threads>_<method>.json
#   <run>/indexes/<dataset>_b<bits>_t<threads>_<method>.idx   (pq: .faiss)
# method in {e8q,me8q,rabitq,pq,saq,caq}.
#
# Environment overrides:
#   E8Q_PYTHON     python of the repo's conda env `e8q` (faiss / e8qlib)
#   E8Q_PAPER_PYTHON  python of conda env `paper` (SciencePlots figures)
#   E8Q_CPUS       taskset CPU list (default 0,2,4,...,22 = one hardware
#                  thread per physical core on the reference ECS host, whose
#                  guest exposes each core's hyper-thread pair as adjacent
#                  vCPUs (cpu0/cpu1 are siblings); verify with
#                  /sys/devices/system/cpu/cpu*/topology/thread_siblings_list;
#                  empty disables)
#   E8Q_THREADS    search threads (default 12 = physical-core count)
#   E8Q_GRID       nprobe grid (default = paper 53-point grid: [1,512],
#                  includes nprobe=200; see README.md)
#   E8Q_BITS       bit budgets (default "1 2 3 4 5 6 7 8")
#   E8Q_NQ         query count (default 1000)
#   E8Q_SAVE_INDEXES  1 = persist indexes under <run>/indexes/
#   E8Q_MMAP_BASE  1 = pass --mmap-base (corpus streamed from a packed
#                  on-disk copy; for corpora larger than RAM)
#   E8Q_RAM_MARGIN RAM guard margin in GiB (default 6; runners skip arms
#                  that cannot fit and are retried on bigger machines)
#   DATASETS_OVERRIDE  space-separated dataset names
#   SEED_OVERRIDE  pin every cell to this seed
#   BASELINES      all (default) | none
lib_init() {
    set -euo pipefail
    REPO="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"
    E8Q_PYTHON="${E8Q_PYTHON:-$(ls -d "$HOME"/environment/miniconda3/envs/e8q/bin/python \
        "$HOME"/miniconda3/envs/e8q/bin/python 2>/dev/null | head -1 || true)}"
    if [[ -z "${E8Q_PYTHON}" ]]; then
        E8Q_PYTHON="$(command -v python3)"
    fi
    E8Q_PAPER_PYTHON="${E8Q_PAPER_PYTHON:-$(ls -d "$HOME"/environment/miniconda3/envs/paper/bin/python \
        "$HOME"/miniconda3/envs/paper/bin/python 2>/dev/null | head -1 || true)}"
    E8Q_THREADS="${E8Q_THREADS:-12}"
    E8Q_WARMUP="${E8Q_WARMUP:-3}"
    E8Q_REPEATS="${E8Q_REPEATS:-9}"
    E8Q_MINMS="${E8Q_MINMS:-400}"
    E8Q_NQ="${E8Q_NQ:-1000}"
    E8Q_BITS="${E8Q_BITS:-1 2 3 4 5 6 7 8}"
    E8Q_GRID="${E8Q_GRID:-1,2,3,4,5,6,7,8,10,12,14,16,18,20,22,24,28,32,36,40,44,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,240,256,272,304,336,368,400,432,464,496,512}"
    E8Q_CPUS="${E8Q_CPUS-0,2,4,6,8,10,12,14,16,18,20,22}"
    E8Q_SAVE_INDEXES="${E8Q_SAVE_INDEXES:-0}"
    BASELINES="${BASELINES:-all}"
    if [[ -n "${E8Q_CPUS}" ]] && command -v taskset >/dev/null 2>&1; then
        PIN=(taskset -c "$E8Q_CPUS")
    else
        PIN=()
    fi
    if [[ -f "$REPO/build/CMakeCache.txt" ]]; then
        E8Q_ISA="$(awk -F= '/^E8Q_ISA:STRING=/{print tolower($2)}' "$REPO/build/CMakeCache.txt" | head -1)"
    fi
    E8Q_ISA="${E8Q_ISA:-avx2}"
}

new_run() {
    local root="$1" run_id="$2" name="$3"
    local dir="$root/$run_id/$name"
    mkdir -p "$dir"
    echo "$dir"
}

# <ds>_b<bits>_t<T>[_r<m>]_<method>.json (trial > 1 gains _r<m>)
ledger_json() {
    local out="$1" ds="$2" method="$3" bits="$4"
    local t="${5:-$E8Q_THREADS}" trial="${6:-1}"
    local suffix=""
    [[ "$trial" -gt 1 ]] && suffix="_r${trial}"
    echo "$out/${ds}_b${bits}_t${t}${suffix}_${method}.json"
}

# Resume gate: skip only a parseable JSON with a non-empty rows list.
# Truncated / empty / non-JSON leftovers from a crash are deleted so the
# arm is rerun. Mirrors protocol.ledger_complete. Ledgers of the
# prune-stats methods (e8q, me8q, rabitq) must also carry survivor_rate on
# every row; a ledger without it predates the stats patch and is rerun --
# unless it is a --build-only ledger, which has no search rows to carry it.
ledger_ok() {
    local f="$1"
    [[ -f "$f" ]] || return 1
    "$E8Q_PYTHON" -c "
import json, os, sys
from pathlib import Path
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception:
    sys.exit(1)
rows = d.get('rows') if isinstance(d, dict) else None
# E8Q_MIN_ROWS guards truncated ledgers (crash mid-search). The main-grid
# drivers set 50; ablation/latency arms use small grids and keep 1.
if not (isinstance(rows, list)
        and len(rows) >= int(os.environ.get('E8Q_MIN_ROWS', '1'))):
    sys.exit(1)
stem = Path(p).stem
method = next((m for m in ('me8q', 'rabitq', 'e8q', 'saq', 'caq', 'pq')
               if stem.endswith('_' + m)), None)
# --build-only ledgers (the construction sweeps) hold a single placeholder
# row and no search results at all; the construction tables read only
# <method>_build_s. They are complete as soon as that timing exists, so the
# survivor-rate gate below must not send them back through a rebuild.
build_only = (bool(rows)
              and all(isinstance(r, dict) and r.get('nprobe', 0) < 0
                      for r in rows)
              and method is not None
              and (d.get(f'{method}_build_s') or -1) > 0)
if method in ('e8q', 'me8q', 'rabitq'):
    done = all(isinstance(r, dict) and 'survivor_rate' in r for r in rows)
    sys.exit(0 if done or build_only else 1)
sys.exit(0)
" "$f"
}

index_path() {
    local out="$1" ds="$2" method="$3" bits="$4"
    local t="${5:-$E8Q_THREADS}"
    local ext="idx"
    [[ "$method" == "pq" ]] && ext="faiss"
    echo "$out/indexes/${ds}_b${bits}_t${t}_${method}.${ext}"
}

cpu_has_avx512() {
    grep -q -E '(avx512f|avx512)' /proc/cpuinfo 2>/dev/null
}

# SAQ/CAQ only when the binary was compiled for AVX-512 AND the CPU has it.
have_saq_runtime() {
    [[ "${E8Q_ISA}" == "avx512" ]] && cpu_has_avx512 && \
        [[ -x "$REPO/baselines/SAQ/bin/test_qps" ]] && \
        [[ -x "$REPO/baselines/SAQ/bin/create_index" ]]
}

# Stale vendored binary: `bin/` is gitignored, so a tree copied from another
# machine brings its binaries along. rsync preserves mtimes, CMake then reports
# every target up to date, and the arm silently measures code compiled
# elsewhere -- an older revision of the vendored patch, a different CPU's
# -march=native, or a different libstdc++. Rebuild in place on the host that
# runs the sweep (`rm -rf baselines/SAQ/build baselines/SAQ/bin` first).
saq_binaries_stale() {
    local bin
    for bin in "$REPO"/baselines/SAQ/bin/*; do
        [[ -f "$bin" ]] || continue
        if [[ -n "$(find "$REPO/baselines/SAQ/saqlib" "$REPO/baselines/SAQ/src" \
                    -newer "$bin" -print -quit 2>/dev/null)" ]]; then
            return 0
        fi
    done
    return 1
}

# Unified Python dispatch. Extra flags after the four positional args.
run_py() {
    local method="$1" root="$2" ds="$3" bits="$4" out="$5"
    shift 5
    if [[ ! -f "$root/$ds/${ds}_base.fvecs" ]]; then
        echo "run_py: $root/$ds not prepared; skipping (datasets/prepare.sh)"
        return 0
    fi
    local idx_args=()
    if [[ "$E8Q_SAVE_INDEXES" == "1" ]]; then
        mkdir -p "$out/indexes"
        idx_args=(--index-path "$(index_path "$out" "$ds" "$method" "$bits")")
    fi
    local mem_args=()
    # E8Q_MMAP_BASE=1 streams a corpus bigger than RAM through a packed
    # on-disk copy (protocol.ensure_packed_matrix) instead of holding it:
    # the runners then cost the index plus the rows a query touches.
    # Recall is bit-identical to the resident path (verified on sift1M b=4,
    # 53/53 depths); only memory and QPS change, so it is used for msmarco10M
    # and stella-trec24-1024 and never for a QPS-bearing comparison.
    [[ "${E8Q_MMAP_BASE:-0}" == "1" ]] && mem_args=(--mmap-base)
    "${PIN[@]}" env ${E8Q_PROJECTION:+E8Q_PROJECTION="$E8Q_PROJECTION"} \
        "$E8Q_PYTHON" "$REPO/experiments/run.py" --method "$method" \
        --data-root "$root" --dataset "$ds" --bits "$bits" \
        --grid "$E8Q_GRID" --threads "$E8Q_THREADS" --nq "$E8Q_NQ" \
        --warmup "$E8Q_WARMUP" --repeats "$E8Q_REPEATS" --min-ms "$E8Q_MINMS" \
        --out "$out" "${idx_args[@]}" "${mem_args[@]}" "$@"
}

# Paper corpora — the 14 registered (youtube retired).
CORE_DATASETS=(
    sift1M sift10m deep1M msong word2vec gist glove200 imagenet
    vogue768 arxiv-abstracts768 openai1536 openai3072 msmarco10M
    stella-trec24-1024
)

baseline_available() {
    case "$1" in
        saq|caq) have_saq_runtime ;;
        pq) "$E8Q_PYTHON" -c "import faiss" >/dev/null 2>&1 ;;
        *) return 1 ;;
    esac
}

# Shared-seed helper: a DETERMINISTIC seed per (dataset, bits) cell so every
# method, arm, and resumed run of the same cell sees the same rotation.
# SEED_OVERRIDE still wins (pins the whole campaign to one seed).
# Optional third arg: trial index (> 1 keys the seed so trials are
# independent draws; trial 1 keeps the historical two-part key).
# Batch master seed. A batch is (dataset, trial): ONE seed drives the whole
# dataset sweep — every method AND every bit width inside it — so rotation,
# IVF init and query order are identical across the bits, and the recall/QPS
# curves are comparable point to point. Independent batches (other datasets,
# other trials) get independent arbitrary draws; the draw is a cksum of the
# batch key, so a resumed run reuses the exact same seed. SEED_OVERRIDE pins
# an entire campaign to one seed (debugging, A/B).
batch_seed() { # ds [trial]
    if [[ -n "${SEED_OVERRIDE:-}" ]]; then
        echo "$SEED_OVERRIDE"
    else
        local key="${1:-}"
        if [[ "${2:-1}" -gt 1 ]]; then
            key="${key}_r${2}"
        fi
        cksum <<< "$key" | cut -d' ' -f1
    fi
}
