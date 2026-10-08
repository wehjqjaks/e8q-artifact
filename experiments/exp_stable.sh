#!/usr/bin/env bash
# Paper MAIN ledger (README.md): every method (e8q,
# me8q, rabitq, pq; saq/caq iff AVX-512 build AND CPU) × 14 datasets ×
# b grid × the 53-point nprobe grid. One JSON per (dataset, method,
# bits, trial) holds every per-cell stat (recall@10, qps + IQR, distance
# ratio, e8q/me8q/rabitq prune internals) and every per-arm stat (build
# time, memory, nprobe=200 relative error, GT-top-100 estimator probe).
#
# E8Q_TRIALS repeats each dataset with an independent master seed
# (batch_seed appends _r<trial>); plotters take the median across trials.
# The reference server run sets E8Q_TRIALS=3 for the body datasets.
#
# RAM guard: runners compare a per-arm estimate against /proc/meminfo and
# exit 0 with a `ram-skip:` line (no ledger file → retried on a bigger
# machine). PQ skips budgets that 4/8-bit sub-quantizers cannot express.
#
# JSON: <ds>_b<bits>_t<T>[_r<m>]_<method>.json
# Seed: one RANDOM draw per (dataset, bits, trial) cell, shared by every
# method. PCA and k-means are precomputed files and are not part of the
# seed scheme.
#
#   DATASETS_OVERRIDE=sift1m ./experiments/exp_stable.sh <data-root> <run-id>
#   E8Q_METHODS="e8q rabitq" ...    # core-method filter (default all three)
#   BASELINES=none  ...          # skip pq/saq/caq
#   SEED_OVERRIDE=123 ...
#   E8Q_TRIALS=3 ...
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
lib_init

DATA_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/datasets}"
RUN_ID="${2:-run_$(date +%Y%m%d_%H%M%S)}"
EXTRA="${EXTRA:-}"
# Which core methods run (default = paper grid). Baselines pq/saq/caq are
# still gated by BASELINES.
E8Q_METHODS="${E8Q_METHODS:-e8q rabitq me8q}"
E8Q_TRIALS="${E8Q_TRIALS:-1}"
# A ledger only counts as complete when it carries the full 53-point grid.
export E8Q_MIN_ROWS="${E8Q_MIN_ROWS:-50}"
# Mixed-precision arms (me8q, saq) sweep a sparse non-uniform bit ladder
# (README.md): half-bit steps only through the metadata-dominated zone
# (0.5-2), integers through the application zone (3-6), and 8/10 as the
# saturation anchors. The budget is fractional at the DP level only —
# per-segment quantizer rates stay integers. Fixed-rate arms keep the
# integer 1-8 grid. (A denser or higher ladder is one env change:
# E8Q_MIXED_BITS="0.5 1 1.5 ... 13".)
E8Q_PQ_BITS="${E8Q_PQ_BITS:-4 8}"
E8Q_MIXED_BITS="${E8Q_MIXED_BITS:-0.5 1 1.5 2 3 4 5 6 8 10}"
# Fixed-rate grid is PER DATASET (README.md): only the two Fig.-1 anchors
# plot every bit (its 2x8 panel is sift1M x openai1536), tab_relerr needs b=6
# on stella, and every other dataset is read at {2,4,8} only — the fixed-rate
# points that any figure or table consumes. A bit no panel plots is a run that
# never needed to happen.
E8Q_BITS_FULL_DS="${E8Q_BITS_FULL_DS:-sift1M openai1536}"
E8Q_BITS_MID_DS="${E8Q_BITS_MID_DS:-stella-trec24-1024}"
E8Q_BITS_MID="${E8Q_BITS_MID:-2 4 6 8}"
E8Q_BITS_SPARSE="${E8Q_BITS_SPARSE:-2 4 8}"
ds_fixed_bits() {
    case " $E8Q_BITS_FULL_DS " in *" $1 "*) printf '%s' "$E8Q_BITS"; return ;; esac
    case " $E8Q_BITS_MID_DS " in *" $1 "*) printf '%s' "$E8Q_BITS_MID"; return ;; esac
    printf '%s' "$E8Q_BITS_SPARSE"
}

OUT="$(new_run "${RESULTS_ROOT:-results}" "$RUN_ID" stable)"
echo "stable -> $OUT  pin=${E8Q_CPUS:-none} threads=$E8Q_THREADS isa=$E8Q_ISA"
echo "  json: <ds>_b<bits>_t${E8Q_THREADS}[_r<m>]_<method>.json"
echo "  fixed: full=[${E8Q_BITS_FULL_DS}] mid=[${E8Q_BITS_MID_DS}: ${E8Q_BITS_MID}] sparse=[${E8Q_BITS_SPARSE}]"
echo "  mixed=[$E8Q_MIXED_BITS]  trials=$E8Q_TRIALS  BASELINES=$BASELINES  E8Q_SAVE_INDEXES=$E8Q_SAVE_INDEXES"
if have_saq_runtime; then
    echo "  saq/caq: ON (AVX-512 build + CPU)"
else
    echo "  saq/caq: skipped (need E8Q_ISA=avx512 and avx512 CPU)"
fi

for ds in ${DATASETS_OVERRIDE:-${CORE_DATASETS[*]}}; do
    DS_FIXED="$(ds_fixed_bits "$ds")"
    DS_BITS="$(printf '%s\n' $DS_FIXED $E8Q_MIXED_BITS | sort -g -u | tr '\n' ' ')"
    echo "-- $ds fixed=[$DS_FIXED]"
    for bits in $DS_BITS; do
        for trial in $(seq 1 "$E8Q_TRIALS"); do
            SEED="$(batch_seed "$ds" "$trial")"
            echo "== $ds b=$bits trial=$trial seed=$SEED"

            case " $DS_FIXED " in
                *" $bits "*)
                    for method in e8q rabitq; do
                        [[ " $E8Q_METHODS " == *" $method "* ]] || continue
                        json="$(ledger_json "$OUT" "$ds" "$method" "$bits" "$E8Q_THREADS" "$trial")"
                        if ledger_ok "$json"; then
                            echo "  ($method) skip, complete"
                        else
                            rm -f "$json"
                            run_py "$method" "$DATA_ROOT" "$ds" "$bits" "$OUT" \
                                --seed "$SEED" --trial "$trial" --adr $EXTRA \
                                || echo "  $method $ds b$bits t$trial: FAILED (logged)"
                        fi
                    done
                    ;;
            esac

            # me8q runs under the true-bit-budget protocol: the DP budget is
            # padded — negatively if needed — so total memory = b*dim/8 per
            # vector (README.md memory accounting). 0.5-bit budgets.
            if [[ " $E8Q_METHODS " == *" me8q "* ]]; then
                case " $E8Q_MIXED_BITS " in
                    *" $bits "*)
                        me8q_json="$(ledger_json "$OUT" "$ds" me8q "$bits" "$E8Q_THREADS" "$trial")"
                        if [[ -f "${me8q_json%.json}.skipped" ]]; then
                            echo "  (me8q) skipped: budget too low (no plan)"
                        elif ledger_ok "$me8q_json"; then
                            echo "  (me8q) skip, complete"
                        else
                            rm -f "$me8q_json"
                            run_py me8q "$DATA_ROOT" "$ds" "$bits" "$OUT" \
                                --seed "$SEED" --trial "$trial" --adr \
                                --match-total-budget $EXTRA \
                                || echo "  me8q $ds b$bits t$trial: FAILED (logged)"
                        fi
                        ;;
                esac
            fi

            [[ "$BASELINES" == "all" ]] || continue

            # README.md: pq is the classic anchor at b = 4 and 8 only
            # (its 8-bit sub-quantizers express those budgets exactly).
            case " $E8Q_PQ_BITS " in
                *" $bits "*)
                    if baseline_available pq; then
                        pq_json="$(ledger_json "$OUT" "$ds" pq "$bits" "$E8Q_THREADS" "$trial")"
                        if ledger_ok "$pq_json"; then
                            echo "  (pq) skip, complete"
                        else
                            rm -f "$pq_json"
                            run_py pq "$DATA_ROOT" "$ds" "$bits" "$OUT" \
                                --seed "$SEED" --trial "$trial" --adr $EXTRA \
                                || echo "  pq $ds b$bits t$trial: FAILED (logged)"
                        fi
                    else
                        echo "  pq: skipped (faiss not importable)"
                    fi
                    ;;
            esac

            if have_saq_runtime; then
                # E8Q_CAQ=0 skips the CAQ arms only (they sit on the fixed-rate
                # ladder); the mixed-ladder SAQ arm below always runs.
                if [[ "${E8Q_CAQ:-1}" == "1" ]]; then
                case " $DS_FIXED " in
                    *" $bits "*)
                        arm_json="$(ledger_json "$OUT" "$ds" caq "$bits" "$E8Q_THREADS" "$trial")"
                        if ledger_ok "$arm_json"; then
                            echo "  (caq) skip, complete"
                        else
                            rm -f "$arm_json"
                            run_py caq "$DATA_ROOT" "$ds" "$bits" "$OUT" \
                                --seed "$SEED" --trial "$trial" --adr $EXTRA \
                                || echo "  caq $ds b$bits t$trial: FAILED (logged)"
                        fi
                        ;;
                esac
                fi
                case " $E8Q_MIXED_BITS " in
                    *" $bits "*)
                        arm_json="$(ledger_json "$OUT" "$ds" saq "$bits" "$E8Q_THREADS" "$trial")"
                        if [[ -f "${arm_json%.json}.skipped" ]]; then
                            echo "  (saq) skipped: budget too low (no plan)"
                        elif ledger_ok "$arm_json"; then
                            echo "  (saq) skip, complete"
                        else
                            rm -f "$arm_json"
                            run_py saq "$DATA_ROOT" "$ds" "$bits" "$OUT" \
                                --seed "$SEED" --trial "$trial" --adr $EXTRA \
                                || echo "  saq $ds b$bits t$trial: FAILED (logged)"
                        fi
                        ;;
                esac
            fi
        done
    done
done
echo "stable done: $OUT"
