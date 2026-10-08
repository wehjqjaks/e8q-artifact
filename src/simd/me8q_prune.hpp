#pragma once

// ME8Q FastScan coarse prune. Generic TUs include this with no SIMD types.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace e8qlib::me8q {

struct CoarsePruneArgs {
    size_t nrun = 0;
    size_t lane_cnt = 0;
    float d2 = 0.0F;
    float lut_delta = 0.0F;
    float lambda = 0.0F;
    float distk = 0.0F;
    const int32_t* acc = nullptr;
    const double* lut_min_x = nullptr;
    const double* sqrt_d2_r = nullptr;  // sqrt(d2_r); r_horizon = min(this, sqrt(distk))
    const double* qerr_r = nullptr;     // CLT LUT pad: 2 sqrt(nblk) * delta
    const uint8_t* banks = nullptr;
    size_t bank_run_bytes = 160;
    const double* score_max_r = nullptr;
    float* lane_dc = nullptr;
    float* lane_pool = nullptr;
    float* lane_qa = nullptr;
    float* lane_sc = nullptr;
    float* lane_tr = nullptr;
    std::vector<uint32_t>* keep_lanes = nullptr;
    size_t* candidates = nullptr;
    size_t* bits_total = nullptr;
    size_t bits_row = 0;
};

// Per-run scalars shared by the AVX2 and AVX-512 filters (no intrinsics, so
// both ISA TUs include this header). Fills rh/qerr/wpad/lutmin.
struct PruneRunScalars {
    std::vector<float> rh;
    std::vector<float> qerr;
    std::vector<float> wpad;
    std::vector<float> lutmin;
};

inline void FillPruneRunScalars(const CoarsePruneArgs& a, PruneRunScalars& out) {
    const float sqrt_distk =
        (a.distk < std::numeric_limits<float>::infinity())
            ? std::sqrt(std::max(0.0F, a.distk))
            : std::numeric_limits<float>::infinity();
    out.rh.resize(a.nrun);
    out.qerr.resize(a.nrun);
    out.wpad.resize(a.nrun);
    out.lutmin.resize(a.nrun);
    for (size_t r = 0; r < a.nrun; ++r) {
        out.rh[r] = std::min(static_cast<float>(a.sqrt_d2_r[r]), sqrt_distk);
        out.qerr[r] = static_cast<float>(a.qerr_r[r]);
        out.lutmin[r] = static_cast<float>(a.lut_min_x[r]);
        const float* sc = reinterpret_cast<const float*>(
            a.banks + r * a.bank_run_bytes + 128
        );
        out.wpad[r] = static_cast<float>(0.5 * sc[3] * (a.score_max_r[r] + a.qerr_r[r]));
    }
}

// Returns lanes covered by the SIMD filter (0 = none; caller scores every lane).
size_t CoarsePruneGeneric(const CoarsePruneArgs& a);

#if defined(E8Q_HAVE_AVX2)
size_t CoarsePruneAvx2(const CoarsePruneArgs& a);
#endif
#if defined(E8Q_HAVE_AVX512)
size_t CoarsePruneAvx512(const CoarsePruneArgs& a);
#endif

size_t CoarsePrune(const CoarsePruneArgs& a);

}  // namespace e8qlib::me8q
