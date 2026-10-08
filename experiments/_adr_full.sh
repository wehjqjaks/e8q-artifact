#!/usr/bin/env bash
# Fill the distance-ratio column over the FULL 53-point nprobe grid for the
# E8Q and RaBitQ fixed-rate arms, and fold it into results/adr_full/.
#
# Why this exists: the sum-ratio distance ratio (README.md) is the
# quantity the SAQ/CAQ binary reports as its `ratio` column, so it is the
# only distance ratio the codecs can be compared on.  CAQ emits it on every
# grid point; the E8Q/RaBitQ runners used to record it at four Table-5
# depths, which left those two as four isolated markers against a 53-point
# CAQ curve in Figure 4.  This driver re-runs the arms with --adr to fill
# the column everywhere.
#
# QPS is NOT re-timed and is NOT taken from these runs: warmup 0 / one timed
# round / min-ms 0, and experiments/adr_fill.py keeps only the ratio column
# after checking the recall against the published ledger depth by depth.  The
# paper's QPS stays exactly as frozen.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
export E8Q_WARMUP=0 E8Q_REPEATS=1 E8Q_MINMS=0
lib_init

DATASETS="${DATASETS:-sift1M deep1M gist openai1536 msmarco10M stella-trec24-1024}"
BITS="${BITS:-2 4 8}"
METHODS="${METHODS:-e8q rabitq}"
DATA_ROOT="${DATA_ROOT:-$REPO/datasets}"
DEST="${DEST:-$REPO/results/adr_full}"
STAGE="${STAGE:-$REPO/results/adr_full_stage}"
# MMAP_BASE=1 keeps the corpus on disk (experiments/protocol.py --mmap-base).
# Required for msmarco10M (38 GiB) and stella-trec24-1024 (68 GiB), whose raw
# base does not co-reside with the index in a small host's RAM.  DEST/STAGE
# are overridable so ./run_all.sh can drive this stage into its own results
# tree; the defaults are the frozen package's paths.
MMAP_ARGS=()
[[ "${MMAP_BASE:-0}" == "1" ]] && MMAP_ARGS=(--mmap-base)
mkdir -p "$DEST" "$STAGE" "$STAGE/indexes"

for ds in $DATASETS; do
    for method in $METHODS; do
        for bits in $BITS; do
            stem="${ds}_b${bits}_t${E8Q_THREADS}_${method}"
            frozen="$REPO/results/fixed/${stem}.json"
            dest="$DEST/${stem}.json"
            fresh="$(ledger_json "$STAGE" "$ds" "$method" "$bits")"
            # A side-car whose index has been reclaimed is NOT complete:
            # bound_coverage.py probes those .idx files directly, so the arm
            # has to run again to leave both behind.
            if [[ -f "$dest" && -f "$(index_path "$STAGE" "$ds" "$method" "$bits")" ]]; then
                echo "== $stem: already filled"
                continue
            fi
            if [[ ! -f "$frozen" ]]; then
                echo "== $stem: no frozen ledger, skipped"
                continue
            fi
            rm -f "$fresh"
            echo "== $stem: measuring -> $dest"
            run_py "$method" "$DATA_ROOT" "$ds" "$bits" "$STAGE" \
                --seed "$(batch_seed "$ds")" \
                --index-path "$(index_path "$STAGE" "$ds" "$method" "$bits")" \
                --adr "${MMAP_ARGS[@]}"
            if [[ ! -f "$fresh" ]]; then
                # ram_guard declined the arm (or its corpus is not prepared).
                echo "== $stem: arm produced no ledger, skipped"
                continue
            fi
            "$E8Q_PYTHON" "$REPO/experiments/adr_fill.py" \
                --fresh "$fresh" --frozen "$frozen" --out "$dest"
            rm -f "$fresh"
        done
    done
done
echo "distance-ratio fill complete"
