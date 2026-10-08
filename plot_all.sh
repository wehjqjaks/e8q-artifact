#!/usr/bin/env bash
# One-click paper figures + tables — repo-root entry point.
#
#   ./plot_all.sh [--results DIR] [--out DIR] [--only LIST]
#
# Regenerates every figure and table the paper \inputs (PDF + PNG, CSV +
# LaTeX) from a results tree.  Defaults read ./results and overwrite
# ./figures_reproduced; it exits non-zero if any figure the
# LaTeX includes is still missing afterwards.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/experiments/plots/make_all.sh" "$@"
