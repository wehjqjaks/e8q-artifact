// ME8Q mixed-precision planning. SAQ DP allocator: cost(seg, B) =
// sum var_i / 2^B, 64-dim segments, B in [1, 8] (E8Q codec rates),
// 64-bit factor overhead per coded segment. The plan is a per-dimension-run
// list covering the padded dimension; the only 0-bit run is the trailing one.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace e8qlib::me8q {

// SAQ constants (saqlib/defines.hpp, saqlib/quantization/saq_data.hpp).
// Rate cap matches CAQ: KMaxQuantizeBits = 13. Uniform E8Q IVF stays 1..8.
inline constexpr size_t kSegDim = 64;      // kDimPaddingSize
inline constexpr int kMaxBits = 13;        // SAQ CAQ / E8Q codec rates
inline constexpr size_t kFactorBits = 64;  // kNumShortFactors * 32

struct PlanSeg {
    size_t len = 0;  // dimensions in this run (multiple of kSegDim)
    int bits = 0;    // bits per dimension, 0 = dropped
};

[[nodiscard]] inline bool operator==(const PlanSeg& a, const PlanSeg& b) {
    return a.len == b.len && a.bits == b.bits;
}

using Plan = std::vector<PlanSeg>;

// ceil(dim / kSegDim) * kSegDim — the padded planning/search domain. SAQ pads
// the tail segment with zeros; me8q does the same in the PCA domain.
[[nodiscard]] size_t PaddedDim(size_t dim);

struct PlanInput {
    const float* var = nullptr;  // per-dim variance, length = padded dim
    size_t dim = 0;              // padded dim (multiple of kSegDim)
    float avg_bits = 4.0F;
    int max_bits = kMaxBits;
    // Byte-matching pad: bits per vector added to the SAQ budget so the
    // plan's total spend (payload + factors) matches a target method's
    // actual per-vector memory. SAQ's DP leaves its long factors and
    // amortized codebooks unbudgeted; charging them HERE (not inside the
    // vendored baseline) keeps the baseline paper-faithful while the
    // comparison happens at equal total bytes.
    double extra_bits = 0.0;
};

// SAQ's DP (budget / overhead / 1%-better acceptance), cost = var / 2^B.
[[nodiscard]] Plan Allocate(const PlanInput& in);

struct PlanSummary {
    double bits_per_dim = 0.0;   // (payload + factor) bits per original dim
    size_t factor_bits = 0;
    size_t payload_bits = 0;
    int num_segments = 0;
    int max_bits = 0;
    size_t dropped_dims = 0;
    size_t dim = 0;
};

[[nodiscard]] PlanSummary Summarize(const Plan& plan, size_t dim);

}  // namespace e8qlib::me8q
