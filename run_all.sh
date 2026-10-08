#!/usr/bin/env bash
# One-click paper data acquisition — repo-root entry point.
#
#   ./run_all.sh [--data-root DIR] [--results DIR] [--run-id ID]
#                [--stages LIST] [--no-prepare] [--baselines all|none]
#
# Runs every experiment the submitted figures and tables read, into the
# results tree ./plot_all.sh consumes:
#
#   main     results/fixed/  results/mixed/  results/ablation/
#   adr      results/adr_full/  results/_adr_stage/indexes/
#   bound    results/bound_coverage/
#   shaping  results/shaping/
#
# Defaults are the six paper corpora, the integer rate ladder 1..8 for the
# fixed-rate arms and the nine nominal budgets of the mixed-rate figures.
# Data contract: README.md.  Figures: ./plot_all.sh.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/experiments/run_all.sh" "$@"
