#!/usr/bin/env bash
# Reproduces the appendix's quantization-distortion figure data:
#   ./experiments/shaping/run_all.sh [--sample N] [--out DIR]
# Writes one CSV per corpus plus the concatenated nmse_shaping.csv.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# Default matches the appendix protocol: a stride sample of ~3x10^4 vectors.
SAMPLE=30000
OUT="$REPO/results/shaping"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample) SAMPLE="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done
BIN="$HERE/bin/measure_shaping"
[[ -x "$BIN" ]] || { echo "build $BIN first (see its header)"; exit 1; }
mkdir -p "$OUT"
CORPORA="sift1M deep1M gist openai1536 msmarco10M stella-trec24-1024"
for ds in $CORPORA; do
    echo "== $ds"
    "$BIN" --data-root "$REPO/datasets" --dataset "$ds" --sample "$SAMPLE" \
        --out "$OUT/nmse_shaping_$ds.csv"
done
head -1 "$OUT/nmse_shaping_sift1M.csv" > "$OUT/nmse_shaping.csv"
for ds in $CORPORA; do tail -n +2 "$OUT/nmse_shaping_$ds.csv" >> "$OUT/nmse_shaping.csv"; done
echo "wrote $OUT/nmse_shaping.csv"
