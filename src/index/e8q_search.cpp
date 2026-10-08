#include "e8qlib/index/e8q.hpp"

#include "index/e8q_internal.hpp"
#include "simd/e8q_search.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

using e8qlib::detail::kMaxK;
using e8qlib::detail::kSurvCapMax;
using e8qlib::detail::Better;

namespace e8qlib {

void E8qIndex::PrepareQuery(const float* rot_q, QueryScratch& s) const {
    if (s.staged_rows.size() != kSurvCapMax) {
        s.staged_rows.resize(kSurvCapMax);
    }
    const double* const mid = gate_mid_.data();
    s.cf_buf.resize(static_cast<size_t>(blocks_) * 64);
    std::vector<float>& cf = s.cf_buf;
    for (int blk = 0; blk < blocks_; ++blk) {
        const float* qb = rot_q + static_cast<size_t>(blk) * 8;
        float* A = cf.data() + static_cast<size_t>(blk) * 64;
        float* B = A + 32;
        for (int v = 0; v < 16; ++v) {
            double av = 0.0;
            double bv = 0.0;
            for (int h = 0; h < 4; ++h) {
                const int bit = (v >> (3 - h)) & 1;
                av += static_cast<double>(qb[4 + h]) *
                      mid[static_cast<size_t>(2 * (4 + h) + bit)];
                bv += static_cast<double>(qb[h]) * mid[static_cast<size_t>(2 * h + bit)];
            }
            A[v] = static_cast<float>(av);
            B[v] = static_cast<float>(bv);
        }
        for (int c = 0; c < 16; ++c) {
            A[16 + c] = A[c];
            B[16 + c] = B[c];
        }
    }
    float hc = 0.0F;
    for (float v : cf) {
        hc = std::max(hc, std::abs(v));
    }
    s.prep_delta_c = hc > 0.0F ? 2.0F * hc / 255.0F : 1.0F;
    // No clear() before this resize: the full buffer is overwritten by the
    // quantizer right below, and clear()+resize() would re-zero every byte.
    if (s.lut_c.size() != cf.size()) {
        s.lut_c.resize(cf.size());
    }
    const float qinv = 1.0F / s.prep_delta_c;
    const float qoff = hc / s.prep_delta_c;
    search_simd::QuantizeU8(
        cf.data(), cf.size(), qinv, qoff, hc, s.prep_delta_c, s.lut_c.data()
    );
    s.lut_pair.resize(cf.size());
    FillPairLut(s.lut_c.data(), static_cast<size_t>(blocks_), s.lut_pair.data());
    s.prep_base_c = -hc * static_cast<float>(2 * blocks_);
    // LUT pad: per-block rounding is U[-delta, delta]; the sum concentrates
    // as sqrt(M). Two-sigma CLT cover is 2 sqrt(M) delta. |S| stays the
    // linear envelope (a tighter score_max ate recall; L1 revert).
    const float sqrt_blocks = std::sqrt(static_cast<float>(blocks_));
    s.ortho_lut_err = detail::LutErrClt(sqrt_blocks, s.prep_delta_c);
    s.ortho_score_max =
        static_cast<float>(blocks_) * (255.0F * s.prep_delta_c + 2.0F * hc);
}

void E8qIndex::RouteQuery(const float* rot_q, size_t nprobe, QueryScratch& s) const {
    const size_t kc = num_clusters_;
    s.probe_dist.resize(kc);
    s.probe_key.resize(kc);
    const size_t take = std::min(nprobe, kc);
    s.probe_id.resize(take);
    ivf::RouteL2(
        rot_q, centroid_rot_.data(), centroid_n2_.data(), kc, padded_dim_, take,
        s.probe_dist.data(), s.probe_id.data(), s.probe_key.data()
    );
}

float E8qIndex::ExactScore(
    const uint8_t* ref_vec, const uint8_t* plane_vec, uint32_t lane, const float* q
) const {
    if (bits_ == 1) {
        return search_simd::ExactScoreB1(*this, plane_vec, lane, q);
    }
    if (bits_ >= 2 && bits_ <= 8) {
        double acc = 0.0;
        for (int blk = 0; blk < blocks_; ++blk) {
            uint8_t dg[8];
            int br = 0;
            DecodeSplitBlockLane(bits_, plane_vec, ref_vec, blk, lane, dg, &br);
            const float* tab = refine_tab_.data() + static_cast<size_t>(br) * 256;
            for (int d = 0; d < 8; ++d) {
                acc += static_cast<double>(
                           q[static_cast<size_t>(blk) * 8 + static_cast<size_t>(d)]
                       ) *
                       static_cast<double>(tab[static_cast<size_t>(dg[d])]);
            }
        }
        return static_cast<float>(acc);
    }
    if (bits_ >= 9) {
        const size_t stride = static_cast<size_t>(1) << bits_;
        double acc = 0.0;
        for (int blk = 0; blk < blocks_; ++blk) {
            uint16_t dg[8];
            int br = 0;
            DecodeSplitBlockLane(bits_, plane_vec, ref_vec, blk, lane, dg, &br);
            const float* tab = refine_tab_.data() + static_cast<size_t>(br) * stride;
            for (int d = 0; d < 8; ++d) {
                acc += static_cast<double>(
                           q[static_cast<size_t>(blk) * 8 + static_cast<size_t>(d)]
                       ) *
                       static_cast<double>(tab[static_cast<size_t>(dg[d])]);
            }
        }
        return static_cast<float>(acc);
    }
    throw std::runtime_error("ExactScore: unsupported bits");
}

void E8qIndex::RefineBatch(
    const SurvivorRow* rows, size_t n, const float* q, QueryScratch& s
) const {
    search_simd::RefineBatch(*this, rows, n, q, s);
}

void E8qIndex::InsertHeap(float dist, uint32_t id, QueryScratch& s) const {
    float* hd = s.heap_dist.data();
    uint32_t* hi = s.heap_id.data();
    const size_t k = s.heap_dist.size();
    size_t count = s.heap_count;
    if (count < k) {
        size_t i = count++;
        hd[i] = dist;
        hi[i] = id;
        while (i > 0) {
            const size_t parent = (i - 1) / 2;
            if (Better(hd[parent], hi[parent], hd[i], hi[i])) {
                std::swap(hd[i], hd[parent]);
                std::swap(hi[i], hi[parent]);
                i = parent;
            } else {
                break;
            }
        }
        s.heap_count = count;
        return;
    }
    if (!Better(dist, id, hd[0], hi[0])) {
        return;
    }
    hd[0] = dist;
    hi[0] = id;
    size_t i = 0;
    while (true) {
        const size_t l = 2 * i + 1;
        const size_t r = l + 1;
        size_t largest = i;
        if (l < k && Better(hd[largest], hi[largest], hd[l], hi[l])) {
            largest = l;
        }
        if (r < k && Better(hd[largest], hi[largest], hd[r], hi[r])) {
            largest = r;
        }
        if (largest == i) {
            break;
        }
        std::swap(hd[i], hd[largest]);
        std::swap(hi[i], hi[largest]);
        i = largest;
    }
}

void E8qIndex::FlushSurvivors(QueryScratch& s, const float* q, SearchStats* stats) const {
    SurvivorRow* const surv_rows = s.staged_rows.data();
    size_t& surv_n = s.staged_n;
    if (surv_n == 0) {
        return;
    }
    // Every staged row is refined, unconditionally: while the heap is
    // unfilled the threshold is +inf, so no drop can be sound, and once it
    // fills the dlo filter has already removed everything the bound can
    // reject. (A cold-start cap ranked by the coarse point estimate pruned
    // true top-k members at low bit rates — removed.)
    auto do_refine = [&]() { RefineBatch(surv_rows, surv_n, q, s); };
    if (stats != nullptr) {
        const auto r0 = std::chrono::steady_clock::now();
        do_refine();
        s.refine_ns += static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - r0
            )
                .count()
        );
    } else {
        do_refine();
    }
    surv_n = 0;
}

void E8qIndex::ScanCluster(
    size_t cid, float d2, const float* q, QueryScratch& s, SearchStats* stats
) const {
    search_simd::ScanCluster(*this, cid, d2, q, s, stats);
}

void E8qIndex::Search(
    const float* query, size_t k, size_t nprobe, uint32_t* out_ids, QueryScratch& s,
    SearchStats* stats, float* out_dists
) const {
    if (k > kMaxK) {
        throw std::invalid_argument(
            "k must be <= " + std::to_string(kMaxK)
        );
    }
    static const bool phases_on = std::getenv("E8Q_PHASES") != nullptr;
    // Counters are written whenever the caller passes a stats sink; the
    // per-phase microsecond timers stay opt-in behind E8Q_PHASES so a stats
    // request alone cannot perturb reported QPS.
    SearchStats* const ph = stats;
    auto now_us = []() {
        return phases_on ? static_cast<double>(
                               std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch()
                               )
                                   .count()
                           ) /
                               1000.0
                         : 0.0;
    };
    const double t0 = now_us();
    // Rotate writes every element (memcpy of dim_ + zero pad, then the
    // in-place transforms), so this only has to guarantee the size.
    s.rot_q.resize(padded_dim_);
    rotator_.Rotate(query, s.rot_q.data());
    for (size_t d = 0; d < padded_dim_; ++d) {
        s.rot_q[d] -= mean_rot_[d];
    }
    s.heap_dist.assign(k, 0.0F);
    s.heap_id.assign(k, 0);
    s.heap_count = 0;
    s.staged_n = 0;
    s.disable_prune = disable_prune_;
    const double t1 = now_us();
    PrepareQuery(s.rot_q.data(), s);
    RouteQuery(s.rot_q.data(), nprobe, s);
    const double t2 = now_us();
    const size_t take = std::min(nprobe, num_clusters_);
    {
        // FWER over the top-k, not over every scanned point: a false prune
        // only matters if it drops a true neighbour of the k-set. k_eff =
        // max(4k, 32) leaves a kurtosis cushion; lambda then stays ~3.3-3.5
        // at alpha=0.01 instead of growing with nprobe (PhiInv(1-alpha/N_scan)
        // hits ~5.4 at np=512). Still a disclosed closed-form union bound,
        // no calibration, no dataset term.
        const size_t effective_k = std::max<size_t>(k * 4, 32);
        s.ortho_z = detail::OrthoQuantile(ortho_alpha_, effective_k);
    }
    double refine_acc = 0.0;
    for (size_t p = 0; p < take; ++p) {
        const uint32_t cid = s.probe_id[p];
        if (p + 1 < take) {
            const uint8_t* na = arena_.data() + arena_begin_[s.probe_id[p + 1]];
            __builtin_prefetch(na, 0, 3);
            __builtin_prefetch(na + 64, 0, 3);
            __builtin_prefetch(na + 128, 0, 3);
            __builtin_prefetch(na + 192, 0, 3);
        }
        const float d2 = s.probe_dist[cid];
        s.refine_ns = 0.0;
        ScanCluster(cid, d2, s.rot_q.data(), s, ph);
        refine_acc += s.refine_ns;
    }
    s.refine_ns = 0.0;
    FlushSurvivors(s, s.rot_q.data(), ph);
    refine_acc += s.refine_ns;
    if (ph != nullptr) {
        const double t3 = now_us();
        ph->rotate_us += t1 - t0;
        ph->prepare_route_us += t2 - t1;
        ph->scan_us += (t3 - t2) - refine_acc / 1000.0;
        ph->refine_us += refine_acc / 1000.0;
    }
    ExtractTopK(s, k, out_ids, out_dists);
}

void E8qIndex::ExtractTopK(
    QueryScratch& s, size_t k, uint32_t* out_ids, float* out_dists
) const {
    uint32_t order[kMaxK];
    for (size_t i = 0; i < s.heap_count; ++i) {
        order[i] = static_cast<uint32_t>(i);
    }
    std::sort(order, order + s.heap_count, [&](uint32_t a, uint32_t b) {
        return Better(s.heap_dist[a], s.heap_id[a], s.heap_dist[b], s.heap_id[b]);
    });
    for (size_t i = 0; i < k; ++i) {
        if (i < s.heap_count) {
            out_ids[i] = s.heap_id[order[i]];
            if (out_dists != nullptr) {
                out_dists[i] = s.heap_dist[order[i]];
            }
        } else {
            out_ids[i] = std::numeric_limits<uint32_t>::max();
            if (out_dists != nullptr) {
                out_dists[i] = std::numeric_limits<float>::infinity();
            }
        }
    }
}

}  // namespace e8qlib
