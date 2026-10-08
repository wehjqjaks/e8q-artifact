// Verbatim transcription of SAQ's allocator, kept next to the optimized one
// (`e8qlib::me8q::Allocate`) purely as an oracle.
//
// Source: saq-repository/saqlib/quantization/saq_data.hpp,
// `SaqDataMaker::dynamic_programming` (lines 147-227 in the checkout), with
// only the mechanical substitutions needed to drop Eigen/glog:
//   FloatVec               -> const float*  + dim
//   data_variance.segment(off, len).sum() -> manual sum over [off, off+len)
//   QuantPlanT             -> e8qlib::me8q::Plan
// Everything else — the budget, the 64-bit factor overhead, the segment cap,
// the 1.01 acceptance rule, the tail-drop action and the backtrack — is the
// original control flow, including its loop order (which decides ties).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "e8qlib/quantization/me8q_plan.hpp"

namespace e8qlib::me8q {

inline Plan ReferenceSaqDp(const float* data_variance, size_t num_dim_padded, float avg_bits) {
    const auto kNumShortFactors = 2;
    const auto kMaxQuantBit = static_cast<size_t>(kMaxBits);
    const auto num_bit_factors = kNumShortFactors * sizeof(float) * 8;  // 64
    const size_t tot_bits =
        static_cast<size_t>(avg_bits * static_cast<float>(num_dim_padded)) + num_bit_factors;
    const size_t max_num_segs = avg_bits < 2 ? num_dim_padded / kSegDim
                                             : num_dim_padded / kSegDim / 2;
    constexpr auto valid_lmt = std::numeric_limits<double>::max();
    auto f = std::vector<std::vector<std::vector<std::pair<double, size_t>>>>(
        max_num_segs + 1,
        std::vector<std::vector<std::pair<double, size_t>>>(
            num_dim_padded / kSegDim + 1,
            std::vector<std::pair<double, size_t>>(tot_bits + 1, {valid_lmt, 0})
        )
    );
    const size_t i_end = num_dim_padded / kSegDim;
    size_t ans_ns = 0;
    size_t ans_i = i_end;
    size_t ans_b = tot_bits;
    f[0][0][0] = {0, 0};
    for (size_t ns = 0; ns <= max_num_segs; ns++) {
        for (size_t i = 0; i <= i_end; i++) {
            for (size_t used_bits = 0; used_bits <= tot_bits; ++used_bits) {
                if (f[ns][i][used_bits].first < valid_lmt) {
                    if (i == i_end) {
                        if (f[ns][i][used_bits].first * 1.01 < f[ans_ns][ans_i][ans_b].first) {
                            ans_ns = ns;
                            ans_i = i;
                            ans_b = used_bits;
                        }
                        continue;
                    }
                    if (ns == max_num_segs) {
                        continue;
                    }
                    double var_sum = 0;
                    for (size_t j = 1; (i + j) * kSegDim <= num_dim_padded; j++) {
                        for (size_t k = 0; k < kSegDim; ++k) {
                            var_sum += static_cast<double>(
                                data_variance[(i + j - 1) * kSegDim + k]
                            );
                        }
                        for (size_t b = 1; b <= kMaxQuantBit; ++b) {
                            const auto b_new =
                                used_bits + b * j * kSegDim + num_bit_factors;
                            if (b_new > tot_bits) {
                                break;
                            }
                            const auto v =
                                var_sum / static_cast<double>(size_t{1} << b);
                            auto& f_to = f[ns + 1][i + j][b_new];
                            if (f_to.first > f[ns][i][used_bits].first + v) {
                                f_to.first = f[ns][i][used_bits].first + v;
                                f_to.second = (i << 4) + b;
                            }
                        }
                    }
                    const auto err0 = var_sum;
                    if (f[ns][i][used_bits].first + err0 < f[1 + ns][i_end][used_bits].first) {
                        f[1 + ns][i_end][used_bits].first =
                            f[ns][i][used_bits].first + err0;
                        f[1 + ns][i_end][used_bits].second = (i << 4) + 0;
                    }
                }
            }
        }
    }
    Plan quant_plan;
    size_t ns = ans_ns;
    size_t i = ans_i;
    size_t b_used = ans_b;
    while (i > 0) {
        auto& f_cur = f[ns][i][b_used];
        auto prev_i = (f_cur.second >> 4);
        auto curr_bits = static_cast<int>(f_cur.second & 0xf);
        auto curr_dim_len = (i - prev_i) * kSegDim;
        quant_plan.push_back({curr_dim_len, curr_bits});
        ns--;
        i = prev_i;
        if (curr_bits != 0) {
            b_used -= static_cast<size_t>(curr_bits) * curr_dim_len + num_bit_factors;
        }
    }
    std::reverse(quant_plan.begin(), quant_plan.end());
    return quant_plan;
}

// SAQ's objective evaluated on a plan: sum over runs of var_sum / 2^B, with a
// dropped run (B == 0) charged its full variance.
inline double SaqPlanCost(const Plan& plan, const float* var, size_t dim) {
    double cost = 0.0;
    size_t off = 0;
    for (const PlanSeg& seg : plan) {
        double vsum = 0.0;
        for (size_t k = 0; k < seg.len; ++k) {
            vsum += static_cast<double>(var[off + k]);
        }
        cost += seg.bits == 0 ? vsum : vsum / static_cast<double>(size_t{1} << seg.bits);
        off += seg.len;
    }
    (void)dim;
    return cost;
}

}  // namespace e8qlib::me8q

