#!/usr/bin/env bash
# Local re-measurement of the screening-margin (alpha) sweep behind Figure 6's
# sensitivity panel: openai1536, b=4, full 53-point paper nprobe grid, one
# shared index across all arms (alpha is a search-time knob).  Ledgers are
# written straight into results/ablation/ under the name
# plot_chapter4.plot_margin_sensitivity looks for.  DEST/STAGE are
# overridable so ./run_all.sh can drive this stage into its own results tree.
#
# Local host note: this machine is a 10-core/20-thread desktop, not the
# 12-physical-core reference host, so absolute QPS is not comparable with
# the fixed-rate tables; the sweep is read for its shape across alpha.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
lib_init

DS="${DS:-openai1536}"
BITS="${BITS:-4}"
ALPHAS="${ALPHAS:-0.001 0.01 0.1 0.5 1 1.5 2}"
DATA_ROOT="${DATA_ROOT:-$REPO/datasets}"
DEST="${DEST:-$REPO/results/ablation}"
STAGE="${STAGE:-$REPO/results/alpha_local_stage}"
mkdir -p "$STAGE" "$DEST" "$STAGE/indexes"
IDX="$STAGE/indexes/${DS}_b${BITS}_t${E8Q_THREADS}_e8q.idx"

for alpha in $ALPHAS; do
    dest="$DEST/abl_alpha_${DS}_b${BITS}_a${alpha}_e8q.json"
    if [[ -f "$dest" ]]; then
        echo "== alpha=$alpha: already present"
        continue
    fi
    rm -f "$(ledger_json "$STAGE" "$DS" e8q "$BITS")"
    echo "== alpha=$alpha -> $dest"
    run_py e8q "$DATA_ROOT" "$DS" "$BITS" "$STAGE" \
        --seed "$(batch_seed "$DS")" --index-path "$IDX" --prune-alpha "$alpha"
    src="$(ledger_json "$STAGE" "$DS" e8q "$BITS")"
    if [[ -f "$src" ]]; then
        mv "$src" "$dest"
    else
        # run_py exits 0 on a missing corpus and the RAM guard exits before
        # writing one, so an absent ledger is a skip, not a failure.
        echo "  alpha=$alpha: no ledger (ram guard or missing corpus)"
    fi
done
rm -rf "$STAGE"
echo "alpha sweep complete"
