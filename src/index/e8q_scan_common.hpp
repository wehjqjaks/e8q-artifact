#pragma once

// Shared scan helpers (no SIMD types). Used by every ScanCluster backend so
// the survivor-push and scalar dequant paths stay one copy.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "e8qlib/index/e8q.hpp"
#include "index/e8q_internal.hpp"

namespace e8qlib::search_simd {

struct Access {
    static int bits(const E8qIndex& i) { return i.bits_; }
    static int blocks(const E8qIndex& i) { return i.blocks_; }
    static size_t batch_bytes(const E8qIndex& i) { return i.batch_bytes_; }
    static size_t ref_batch_bytes(const E8qIndex& i) { return i.ref_batch_bytes_; }
    static size_t p1_off(const E8qIndex& i) { return i.P1Off(); }
    static size_t ref_refine_off(const E8qIndex& i) { return i.RefRefineOff(); }
    static size_t ref_g1_off(const E8qIndex& i) { return i.RefG1Off(); }
    static size_t ref_h3_off(const E8qIndex& i) { return i.RefH3Off(); }
    static const uint8_t* arena(const E8qIndex& i) { return i.arena_.data(); }
    static const uint64_t* arena_begin(const E8qIndex& i) { return i.arena_begin_.data(); }
    static const uint8_t* ref_arena(const E8qIndex& i) { return i.ref_arena_.data(); }
    static const uint64_t* ref_arena_begin(const E8qIndex& i) {
        return i.ref_arena_begin_.data();
    }
    static const uint32_t* ids(const E8qIndex& i) { return i.ids_.data(); }
    static uint64_t cluster_begin(const E8qIndex& i, size_t cid) {
        return i.cluster_begin_[cid];
    }
    static const float* refine_tab(const E8qIndex& i) { return i.refine_tab_.data(); }
    static void InsertHeap(const E8qIndex& i, float dist, uint32_t id, QueryScratch& s) {
        i.InsertHeap(dist, id, s);
    }
    static void FlushSurvivors(
        const E8qIndex& i, QueryScratch& s, const float* q, SearchStats* stats
    ) {
        i.FlushSurvivors(s, q, stats);
    }
    static float ExactScore(
        const E8qIndex& i, const uint8_t* ref, const uint8_t* plane, uint32_t lane,
        const float* q
    ) {
        return i.ExactScore(ref, plane, lane, q);
    }
};

inline float HeapLimit(const QueryScratch& s) {
    return s.heap_count == s.heap_dist.size() ? s.heap_dist[0]
                                              : std::numeric_limits<float>::infinity();
}

// How many staged survivors to hold before refining them. Every staged row is
// refined, and the prune bound compares against the *current* k-heap
// threshold, so the threshold is only as useful as it is fresh: flushing at
// the staging cap keeps it at +inf for a whole cluster (the heap is still
// empty when those lanes are tested) and every lane of that cluster has to be
// refined. Flushing at a block's worth of rows fills the heap after the first
// ten lanes and lets the same bound reject the rest of the cluster.
// E8Q_FLUSH_EVERY overrides it (measurement knob, and the ablation's
// "flush at the cap" arm).
inline size_t FlushEvery() {
    static const size_t every = [] {
        const char* e = std::getenv("E8Q_FLUSH_EVERY");
        const long v = e != nullptr ? std::strtol(e, nullptr, 10) : 0;
        return v > 0 ? static_cast<size_t>(v) : detail::kSurvFlushEvery;
    }();
    return every;
}

inline void QuantizeU8Tail(
    const float* src, size_t i, size_t n, float hc, float delta, uint8_t* dst
) {
    for (; i < n; ++i) {
        // RNE to match the vectorized LUT fill (cvtps_epi32 under default
        // MXCSR); lround's half-away-from-zero diverges from it at exact .5.
        const int v = static_cast<int>(std::nearbyintf((src[i] + hc) / delta));
        dst[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
    }
}

template <class Flush, class MakeRow>
inline void CollectLanes(
    uint32_t mask, size_t n_logic, const float* dlo, QueryScratch& s,
    bool disable_prune, SearchStats* stats, Flush&& flush, MakeRow&& make_row
) {
    if (n_logic < 32) {
        mask &= (1u << n_logic) - 1u;
    }
    const size_t flush_every = FlushEvery();
    size_t surv_cnt = 0;
    SurvivorRow* const rows = s.staged_rows.data();
    size_t& n = s.staged_n;
    while (mask != 0) {
        const int lane = __builtin_ctz(mask);
        mask &= mask - 1;
        if (!disable_prune && !(dlo[lane] < HeapLimit(s))) {
            continue;
        }
        if (n >= flush_every) {
            flush();
        }
        ++surv_cnt;
        rows[n++] = make_row(lane);
    }
    if (stats != nullptr) {
        stats->survivors += surv_cnt;
    }
}

inline const uint64_t* PlaneExpandLut() {
    static const std::vector<uint64_t> lut = [] {
        std::vector<uint64_t> t(256, 0);
        for (int v = 0; v < 256; ++v) {
            uint64_t e = 0;
            for (int i = 0; i < 8; ++i) {
                e |= static_cast<uint64_t>((v >> (7 - i)) & 1) << (8 * i);
            }
            t[static_cast<size_t>(v)] = e;
        }
        return t;
    }();
    return lut.data();
}

// Ortho scan header (WriteOrthoBanks): u8 banks base/W/Omega, scales
// sc[0..5], sc[6] = |W|_max (batch skip only; LUT pad is per-lane |W|).
// W_x = 2*rn/<u,w_scan> so <u, R_x> == 0. Slack is
// lambda * r_horizon * Omega with Omega = ||R_x||/sqrt(dim-1) and
// r_horizon = min(sqrt(d2), sqrt(distk0)). Per-lane:
//   base_min    = rn^2 + W*<c',w_scan>   (caller adds d2)
//   perp_min    = Omega_min
//   pad_w_lut   = CLT LUT bound (caller multiplies by |W_lane|)
//   pad_w_round = half W quantum * (|S| bound + LUT bound)
// Pads are deterministic u8-rounding covers, additive, outside the
// Gaussian perp term.
inline void OrthoBatchHeader(
    const uint8_t* bp, float lut_err_bound, float score_max,
    float& base_min, float& base_scale, float& wmin_eff, float& weight_scale,
    float& perp_min, float& perp_scale, float& pad_w_lut, float& pad_w_round
) {
    const float* sc = reinterpret_cast<const float*>(bp + E8qIndex::kScalesOff);
    base_min = sc[0];
    base_scale = sc[1];
    wmin_eff = sc[2];
    weight_scale = sc[3];
    perp_min = sc[4];
    perp_scale = sc[5];
    (void)sc[6];  // |W|_max, kept on disk; LUT pad is now per-lane |W|
    pad_w_lut = lut_err_bound;
    pad_w_round = 0.5F * weight_scale * (score_max + lut_err_bound);
}

}  // namespace e8qlib::search_simd
