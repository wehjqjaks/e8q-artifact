#!/usr/bin/env bash
# One-click download + preprocessing for the E8Q paper corpora.
#
#   ./datasets/prepare.sh                                 # everything
#   ./datasets/prepare.sh msong sift                      # a subset
#
# Each corpus lands in datasets/<name>/ (base/query/gt/ivf/pca). Raw
# archives stay in datasets/<name>/raw/. Override with --data-root DIR.
#
# Steps per corpus: download -> convert -> exact top-100 ground truth ->
# shared IVF k-means (nlist=4096, all threads) -> full-rank PCA artifacts.
# Existing outputs are kept (idempotent). Raw staging under
# datasets/<name>/raw/ is deleted once the processed files are complete
# (--keep-raw in prepare.py preserves it). MSMARCO-10M streams the Cohere
# passage shards through the same driver, and `--refill` restores only the
# base files of a pruned tree. `sift10m` downloads the UCI archive 353 zip
# (slow server, no Range support — a cut download restarts); `youtube`
# has no public mirror left and is skipped with a notice.
#
# Python: the repo's conda env `e8q` (set E8Q_PYTHON to override).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${E8Q_PYTHON:-}" ]]; then
    PY="$E8Q_PYTHON"
elif command -v conda >/dev/null 2>&1; then
    PY="$(conda info --base 2>/dev/null)/envs/e8q/bin/python"
else
    PY="$(ls -d "$HOME"/environment/miniconda3/envs/e8q/bin/python \
              "$HOME"/miniconda3/envs/e8q/bin/python 2>/dev/null | head -1 || true)"
fi
if [[ -z "${PY:-}" || ! -x "$PY" ]]; then
    echo "prepare.sh: conda env 'e8q' not found; set E8Q_PYTHON to a python with numpy+faiss" >&2
    exit 1
fi

"$PY" "$HERE/prepare.py" "$@"
