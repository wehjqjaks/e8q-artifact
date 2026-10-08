#include "index/e8q_scan_common.hpp"
#include "simd/e8q_search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

// Tolerance contract: this backend accumulates in fp64 where the AVX2/AVX512
// kernels accumulate in fp32, so cross-backend comparisons (tests, oracles)
// must use a small relative tolerance — the two are equal in exact arithmetic
// but differ in the last few ulps of fp32. Search decisions derived from these
// scores (heap ordering at near-ties) may flip on ties within that tolerance.

namespace e8qlib::search_simd {

using A = Access;

void QuantizeU8Generic(
    const float* src, size_t n, float /*qinv*/, float /*qoff*/, float hc, float delta,
    uint8_t* dst
) {
    QuantizeU8Tail(src, 0, n, hc, delta, dst);
}

// b == 1 exact score: the rowA byte is the whole code — digits 0..6 are the
// sign bits (bit 7-i), digit 7 is their parity (the E8 shell), bit 0 is the
// branch. Affine twin of the vectorized backends (their per-block gather):
// level = t0 + d*(t1 - t0), so the block costs two table loads, not eight.
float ExactScoreB1Generic(
    const E8qIndex& idx, const uint8_t* plane, uint32_t lane, const float* q
) {
    const int blocks = A::blocks(idx);
    const float* tab0 = A::refine_tab(idx);
    double acc = 0.0;
    for (int blk = 0; blk < blocks; ++blk) {
        const int xb = plane[static_cast<size_t>(blk) * 32 + lane];
        const int par = __builtin_popcount(static_cast<unsigned>(xb & 0xFE)) & 1;
        const float* tab = tab0 + static_cast<size_t>(xb & 1) * 256;
        const double t0 = static_cast<double>(tab[0]);
        const double dlt = static_cast<double>(tab[1]) - t0;
        double sq = 0.0;
        double sdq = 0.0;
        for (int i = 0; i < 7; ++i) {
            const double qi = static_cast<double>(
                q[static_cast<size_t>(blk) * 8 + static_cast<size_t>(i)]
            );
            sq += qi;
            sdq += qi * static_cast<double>((xb >> (7 - i)) & 1);
        }
        const double q7 = static_cast<double>(q[static_cast<size_t>(blk) * 8 + 7]);
        sq += q7;
        sdq += q7 * static_cast<double>(par);
        acc += t0 * sq + dlt * sdq;
    }
    return static_cast<float>(acc);
}

void RefineBatchGeneric(
    const E8qIndex& idx, const SurvivorRow* rows, size_t n, const float* q, QueryScratch& s
) {
    for (size_t i = 0; i < n; ++i) {
        const SurvivorRow& row = rows[i];
        const float S = A::ExactScore(idx, row.refine, row.plane, row.lane, q);
        A::InsertHeap(idx, row.h3 - 2.0F * row.g1 * S, row.id, s);
    }
}

void ScanClusterGeneric(
    const E8qIndex& idx, size_t cid, float d2, const float* q, QueryScratch& s,
    SearchStats* stats
) {
    const uint64_t vb = A::cluster_begin(idx, cid);
    const size_t cnt = static_cast<size_t>(A::cluster_begin(idx, cid + 1) - vb);
    if (cnt == 0) {
        return;
    }
    const uint8_t* batchp = A::arena(idx) + A::arena_begin(idx)[cid];
    const size_t batch_bytes = A::batch_bytes(idx);
    const size_t ref_batch_bytes = A::ref_batch_bytes(idx);
    const int bits = A::bits(idx);
    const int blocks = A::blocks(idx);
    const uint8_t* refbase = A::ref_arena(idx) + A::ref_arena_begin(idx)[cid];
    const uint32_t* ids = A::ids(idx);
    auto flush = [&]() { A::FlushSurvivors(idx, s, q, stats); };
    const float sqrt_d2 = std::sqrt(std::max(0.0F, d2));

    size_t off = 0;
    {
        auto finish_c = [&](const int32_t* accC, const uint8_t* bp, size_t off_cur,
                            size_t n_logic) {
            if (stats != nullptr) {
                stats->candidates += n_logic;
            }
            const uint8_t* refrec = refbase + (off_cur / 32) * ref_batch_bytes;
            const float* g1b = reinterpret_cast<const float*>(refrec + A::ref_g1_off(idx));
            const float* h3b = reinterpret_cast<const float*>(refrec + A::ref_h3_off(idx));
            const float distk0 = HeapLimit(s);
            const float r_horizon =
                (distk0 < d2) ? std::sqrt(std::max(0.0F, distk0)) : sqrt_d2;
            float dlo_lanes[32];
            float batch_min = std::numeric_limits<float>::infinity();
            {
                // Orthogonal-projection bound; scalar mirror of the SIMD
                // branches (see OrthoBatchHeader for the additive pad).
                float base_min = 0.0F, base_scale_f = 0.0F;
                float weight_min_f = 0.0F, weight_scale_f = 0.0F;
                float perp_min = 0.0F, perp_scale_f = 0.0F;
                float pad_w_lut = 0.0F, pad_w_round = 0.0F;
                OrthoBatchHeader(
                    bp, s.ortho_lut_err, s.ortho_score_max, base_min, base_scale_f,
                    weight_min_f, weight_scale_f, perp_min, perp_scale_f, pad_w_lut,
                    pad_w_round
                );
                if (!s.disable_prune && distk0 < std::numeric_limits<float>::infinity()) {
                    const float w_amax =
                        reinterpret_cast<const float*>(bp + E8qIndex::kScalesOff)[6];
                    const float pad_add = w_amax * pad_w_lut + pad_w_round;
                    const float perp_amax = perp_min + 255.0F * perp_scale_f;
                    // W is signed (ZPR): the upper bound of W*S over the batch
                    // needs max |S| in affine units, not the signed max.
                    int32_t s_max_raw = accC[0];
                    int32_t s_min_raw = accC[0];
                    for (int lane = 1; lane < 32; ++lane) {
                        s_max_raw = std::max(s_max_raw, accC[lane]);
                        s_min_raw = std::min(s_min_raw, accC[lane]);
                    }
                    const float s_hi =
                        static_cast<float>(s_max_raw) * s.prep_delta_c + s.prep_base_c;
                    const float s_lo =
                        static_cast<float>(s_min_raw) * s.prep_delta_c + s.prep_base_c;
                    const float s_absmax = std::max(std::abs(s_hi), std::abs(s_lo));
                    const float batch_dlo_lb = (d2 + base_min - pad_add) -
                                               w_amax * s_absmax -
                                               (s.ortho_z * r_horizon) * perp_amax;
                    if (batch_dlo_lb >= distk0) {
                        return;
                    }
                }
                const float base_floor = base_min + d2 - pad_w_round;
                const float slack_scale = s.ortho_z * r_horizon;
                const uint8_t* base_bank = bp + E8qIndex::kH3QOff;
                const uint8_t* weight_bank = bp + E8qIndex::kCqOff;
                const uint8_t* perp_bank = bp + E8qIndex::kAqOff;
                for (int lane = 0; lane < 32; ++lane) {
                    const float scv =
                        static_cast<float>(accC[lane]) * s.prep_delta_c + s.prep_base_c;
                    const float base_term = base_floor +
                                            static_cast<float>(base_bank[lane]) *
                                                base_scale_f;
                    const float weight = weight_min_f +
                                         static_cast<float>(weight_bank[lane]) *
                                             weight_scale_f;
                    const float perp = perp_min +
                                        static_cast<float>(perp_bank[lane]) *
                                            perp_scale_f;
                    const float dc = base_term - weight * scv;
                    const float dlo = dc - std::abs(weight) * pad_w_lut - slack_scale * perp;
                    dlo_lanes[lane] = dlo;
                    if (dlo < batch_min) {
                        batch_min = dlo;
                    }
                }
            }
            if (!s.disable_prune && batch_min >= distk0) {
                return;
            }
            uint32_t mask = s.disable_prune ? 0xFFFFFFFFu : 0;
            if (!s.disable_prune) {
                for (size_t lane = 0; lane < n_logic; ++lane) {
                    if (dlo_lanes[lane] < distk0) {
                        mask |= 1u << lane;
                    }
                }
            }
            CollectLanes(
                mask, n_logic, dlo_lanes, s, s.disable_prune, stats, flush,
                [&](int lane) {
                    return SurvivorRow{
                        bits >= 2
                            ? refbase + (off_cur / 32) * ref_batch_bytes +
                                  A::ref_refine_off(idx) +
                                  static_cast<size_t>(lane) *
                                      (static_cast<size_t>(blocks) *
                                       static_cast<size_t>(bits - 1))
                            : nullptr,
                        bp + A::p1_off(idx),
                        static_cast<uint32_t>(lane),
                        g1b[lane],
                        d2 + h3b[lane],
                        ids[static_cast<size_t>(vb) + off_cur + static_cast<size_t>(lane)]
                    };
                }
            );
        };
        while (off < cnt) {
            const size_t n_logic = std::min<size_t>(32, cnt - off);
            const bool pair = (n_logic == 32) && (cnt - off >= 64);
            alignas(32) int32_t accC[32];
            alignas(32) int32_t accB[32];
            if (pair) {
                AccumulateCoarseHiPair(batchp + A::p1_off(idx), batchp + batch_bytes + A::p1_off(idx),
                    s.lut_c.data(), s.lut_pair.data(), static_cast<size_t>(blocks), accC, accB
                );
            } else {
                AccumulateCoarseHi(batchp + A::p1_off(idx), s.lut_c.data(), s.lut_pair.data(), static_cast<size_t>(blocks),
                    accC
                );
            }
            finish_c(accC, batchp, off, n_logic);
            if (pair) {
                finish_c(accB, batchp + batch_bytes, off + 32, 32);
                batchp += 2 * batch_bytes;
                off += 64;
            } else {
                batchp += batch_bytes;
                off += 32;
            }
        }
        flush();
        return;
    }
}

}  // namespace e8qlib::search_simd
