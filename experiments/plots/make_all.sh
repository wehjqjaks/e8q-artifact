#!/usr/bin/env bash
# One-click reproduction of every submitted figure and table
# (plot_all.sh at the repo root delegates here).
#
#   ./experiments/plots/make_all.sh [--results DIR] [--out DIR] [--only LIST]
#
# Defaults read the results tree ./run_all.sh writes (./results) and write into
# ./figures_reproduced.  Every artifact is written as PDF + PNG, tables as
# CSV + LaTeX.
#
# \input'ed by the paper:
#   fig_fixed_accuracy, fig_fixed_frontier      plot_chapter4 (via plot_fixed_rate)
#   fig_margin_sensitivity, fig_bound_selectivity,
#   fig_scalability, fig_mixed_frontier         plot_chapter4
#   fig_shaping_gain_mse,
#   fig_all_frontiers_grid                      plot_appendix_spectrum
#     (plot_appendix_spectrum also writes fig_full_spectrum_trajectories,
#      which the submitted appendix no longer includes)
#   fig_bound_coverage                          plot_bound_coverage
#   tab_fixed_headline, tab_memory,
#   tab_full_spectrum_all                       plot_chapter4
#   tab_build (Table 3)                         experiments/table3_build.sh
# Figure 1 (architecture diagram) and Figure 2 (ZPR schematic) are
# hand-supplied; they are not part of this pipeline.
#
# Table 3 is the exception to "one producer per artifact": its cells are
# wall-clock build seconds, which only the machine holding the corpora can
# take, so the construction sweep writes the body and the fixed-rate ledgers'
# own build_s (from whichever code version built them) never reaches the
# paper.  The same sweep can also emit an ME8Q/SAQ construction table; the
# paper does not carry one.
#
# --only picks a subset (comma-separated: chapter4, spectrum, coverage,
#        or tables for both table producers).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

PY="${E8Q_PAPER_PYTHON:-$(ls -d "$HOME"/environment/miniconda3/envs/paper/bin/python \
    "$HOME"/miniconda3/envs/paper/bin/python 2>/dev/null | head -1 || true)}"
[[ -n "$PY" && -x "$PY" ]] || PY="$(command -v python3)"

RESULTS="$REPO/results"
# Used only for the final "every \includegraphics exists" check; point --paper
# at the paper's LaTeX directory when reproducing against the manuscript.
PAPER="$REPO/paper"
OUT="$REPO/figures_reproduced"
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --results) RESULTS="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --paper) PAPER="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 1 ;;
    esac
done
mkdir -p "$OUT"
echo "plots: python=$PY"
echo "       results=$RESULTS"
echo "       out=$OUT"
echo "       paper=$PAPER"

want() {
    [[ -z "$ONLY" ]] && return 0
    [[ ",$ONLY," == *",$1,"* ]]
}

# The plotters are cumulative: plot_chapter4 draws the accuracy and frontier
# panels itself (imported from plot_fixed_rate), so running it alone produces
# the whole fixed-rate set and there is one producer per artifact.
stage_chapter4=""
for s in chapter4 tables; do
    if want "$s"; then stage_chapter4="1"; fi
done
if [[ -n "$stage_chapter4" ]]; then
    echo "== plot_chapter4 (fixed-rate accuracy + frontier, Section 4.3 panels,"
    echo "                  Section 4.4 mixed panes, tables)"
    "$PY" "$HERE/plot_chapter4.py" --results "$RESULTS" --out "$OUT" || \
        echo "  plot_chapter4: FAILED (missing fixed-rate ledgers?)"
fi

stage_spectrum=""
for s in spectrum tables; do
    if want "$s"; then stage_spectrum="1"; fi
done
if [[ -n "$stage_spectrum" ]]; then
    echo "== plot_appendix_spectrum (Appendix B distortion spectrum)"
    "$PY" "$HERE/plot_appendix_spectrum.py" --results "$RESULTS" --out "$OUT" || \
        echo "  plot_appendix_spectrum: FAILED (missing shaping CSVs?)"
fi

if want coverage; then
    echo "== plot_bound_coverage (Appendix A.4)"
    "$PY" "$HERE/plot_bound_coverage.py" \
        --results "${BOUND_RESULTS:-$RESULTS/bound_coverage}" --out "$OUT" || \
        echo "  plot_bound_coverage: FAILED (missing bound_coverage.json?)"
fi

# ---------------------------------------------------------------------------
# Every figure the paper includes must now exist in $OUT.  Missing ones are a
# hard error: a silent hole here is a figure that ships as an undefined
# reference.
# ---------------------------------------------------------------------------
PRODUCED="fig_fixed_accuracy fig_fixed_frontier fig_margin_sensitivity \
fig_bound_selectivity fig_scalability fig_mixed_frontier fig_shaping_gain_mse \
fig_full_spectrum_trajectories fig_all_frontiers_grid fig_bound_coverage"

missing=0
declared=0
while read -r rel; do
    [[ -n "$rel" ]] || continue
    declared=$((declared + 1))
    stem="${rel#figures/}"; stem="${stem%.*}"
    if [[ -f "$OUT/$stem.pdf" || -f "$OUT/$stem.png" ]]; then
        continue
    fi
    if [[ " $PRODUCED " == *" $stem "* ]]; then
        echo "MISSING (should have been generated): $stem" >&2
        missing=$((missing + 1))
    else
        echo "  hand-supplied, not generated here: $stem (see the paper's figures/)"
    fi
done < <(grep -h 'includegraphics' "$PAPER"/*.tex 2>/dev/null \
         | grep -o 'figures/[A-Za-z0-9_./-]*' | sort -u)

if (( missing > 0 )); then
    echo "plot_all: $missing figure(s) missing — the paper will not build" >&2
    exit 1
fi
echo "figures+tables done: $OUT ($declared figures referenced by the paper)"
