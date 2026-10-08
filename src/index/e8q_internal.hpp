#pragma once

// Private index internals. Not installed.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "e8qlib/quantization/e8q_codec.hpp"
#include "e8qlib/fastscan/fastscan.hpp"
#include "e8qlib/index/e8q.hpp"
#include "e8qlib/index/ivf_common.hpp"
#include "simd/kernels.hpp"
#include "simd/scan_kernels.hpp"

namespace e8qlib {
namespace detail {

constexpr uint32_t kMagic = 0x58413845;  // "E8AX" — current on-disk layout
constexpr uint32_t kOptMagic = 0x514F5045;  // "EPOQ" build-options trailer
constexpr size_t kMaxK = e8qlib::kMaxK;
// Survivor staging capacity. ScanCluster's overflow flush threshold must stay
// <= this.
constexpr size_t kSurvCapMax = 1024;
// How many staged survivors to hold before refining them. Every staged row is
// refined, and the prune bound compares against the *current* k-heap
// threshold, so the threshold is only as useful as it is fresh: flushing at
// the arena cap keeps it at +inf for the whole first cluster (the heap is
// still empty when those lanes are tested) and every lane of that cluster has
// to be refined. Flushing at one block's worth of rows fills the heap after
// the first ten lanes and lets the same bound reject the rest of the cluster.
// The bound itself is unchanged — the extra rejections ride the same
// alpha-tail guarantee the later clusters already rely on.
constexpr size_t kSurvFlushEvery = 32;

// Total heap order: smaller distance first, id as the tie-break so equal
// distances compare consistently across backends.
inline bool Better(float d1, uint32_t i1, float d2, uint32_t i2) {
    return d1 < d2 || (d1 == d2 && i1 < i2);
}

inline void AdviseHugePages(void* p, size_t bytes) {
#if defined(__linux__)
    constexpr uintptr_t kHuge = uintptr_t{2} << 20;
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uintptr_t lo = (a + kHuge - 1) & ~(kHuge - 1);
    const uintptr_t hi = (a + bytes) & ~(kHuge - 1);
    if (hi > lo) {
        madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_HUGEPAGE);
    }
#else
    (void)p;
    (void)bytes;
#endif
}

template <class T, class Alloc>
void AllocHuge(std::vector<T, Alloc>& v, size_t count, T fill) {
    v.clear();
    v.reserve(count);
    AdviseHugePages(v.data(), count * sizeof(T));
    v.assign(count, fill);
}

// Inverse standard-normal CDF via a few Newton steps on Phi (std::erf).
// Called once per query (not per point), so correctness beats constant
// tables; converges from the asymptotic start in <= 12 iterations.
inline double PhiInv(double p) {
    if (p <= 0.0) {
        return -8.0;
    }
    if (p >= 1.0) {
        return 8.0;
    }
    // Asymptotic start (exact in both tails, off in the middle where
    // Newton still converges in a few steps).
    const double t = std::min(p, 1.0 - p);
    double x = std::sqrt(-2.0 * std::log(t));
    if (p < 0.5) {
        x = -x;
    }
    for (int it = 0; it < 12; ++it) {
        const double phi = 0.3989422804014327 * std::exp(-0.5 * x * x);
        const double Phi = 0.5 * (1.0 + std::erf(x * 0.7071067811865476));
        x -= (Phi - p) / phi;
    }
    return x;
}

// Per-query concentration quantile: lambda = PhiInv(1 - alpha/n).
// alpha is the disclosed per-query prune-loss budget (a disclosed constant,
// not a calibration). Callers pass k_eff = max(4k, 32), not N_scan.
inline float OrthoQuantile(double alpha, size_t k_eff) {
    const double tail = std::min(alpha / static_cast<double>(std::max<size_t>(k_eff, 1)), 0.49);
    return static_cast<float>(std::max(PhiInv(1.0 - tail), 1.0));
}

inline float LutErrClt(float sqrt_blocks, float delta) {
    return 2.0F * sqrt_blocks * delta;
}

struct ClusterBuffers {
    std::vector<uint8_t> p1_rows;   // per-vector rowA bytes (blocks each)
    std::vector<uint8_t> refine;    // per-vector low-digit staging ((b-1) B/block)
    std::vector<float> g1;
    std::vector<float> g2;          // h3 accumulation target
    // Orthogonal-projection bound scalars: h3~, W, Omega per vector.
    std::vector<float> ortho_base;
    std::vector<float> ortho_weight;
    std::vector<float> ortho_perp;
    std::vector<uint32_t> ids;
};

}  // namespace detail
}  // namespace e8qlib
