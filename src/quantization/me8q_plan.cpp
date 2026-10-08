#include "e8qlib/quantization/me8q_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace e8qlib::me8q {
namespace {

constexpr double kInf = std::numeric_limits<double>::max();

// Cost of every candidate run (block start i, j blocks long) at every rate.
// Indexed [(i * (nblocks + 1) + j) * (kMaxBits + 1) + b]. SAQ: var / 2^B.
std::vector<double> BuildCostTable(const PlanInput& in) {
    const size_t nblocks = in.dim / kSegDim;
    std::vector<double> cost(static_cast<size_t>(nblocks + 1) * (nblocks + 1) *
                                 (kMaxBits + 1),
                             0.0);
    std::vector<double> p1(nblocks + 1, 0.0);
    for (size_t b = 0; b < nblocks; ++b) {
        double s1 = 0.0;
        for (size_t k = 0; k < kSegDim; ++k) {
            s1 += static_cast<double>(in.var[b * kSegDim + k]);
        }
        p1[b + 1] = p1[b] + s1;
    }
    for (size_t i = 0; i + 1 <= nblocks; ++i) {
        for (size_t j = 1; i + j <= nblocks; ++j) {
            const double vsum = p1[i + j] - p1[i];
            double* row = &cost[(i * (nblocks + 1) + j) * (kMaxBits + 1)];
            for (int b = 1; b <= kMaxBits; ++b) {
                row[b] = vsum / static_cast<double>(int64_t{1} << b);
            }
        }
    }
    return cost;
}

}  // namespace

size_t PaddedDim(size_t dim) {
    return ((dim + kSegDim - 1) / kSegDim) * kSegDim;
}

Plan Allocate(const PlanInput& in) {
    if (in.var == nullptr || in.dim == 0 || in.dim % kSegDim != 0) {
        throw std::invalid_argument("Allocate: dim must be a positive multiple of 64");
    }
    if (in.max_bits < 1 || in.max_bits > kMaxBits) {
        throw std::invalid_argument("Allocate: max_bits must be in [1, 13]");
    }
    const size_t nblocks = in.dim / kSegDim;
    // True-bit-budget protocol: extra_bits may be NEGATIVE (charging me8q's
    // own per-vector overhead against the b*dim total), so the sum can hit
    // zero or below; clamp instead of casting a negative double to size_t.
    const double budget_bits =
        static_cast<double>(in.avg_bits) * static_cast<double>(in.dim) +
        in.extra_bits + static_cast<double>(kFactorBits);
    const size_t tot_bits = budget_bits > 0.0 ? static_cast<size_t>(budget_bits) : 0;
    // Every reachable bit total is a multiple of kSegDim (each run adds
    // b*j*64 + 64 bits), so the budget can be tracked in units of kSegDim
    // bits without changing which plans are feasible.
    const size_t tot_units = tot_bits / kSegDim;
    const size_t max_segs = in.avg_bits < 2.0F ? nblocks : nblocks / 2;

    const std::vector<double> cost = BuildCostTable(in);
    std::vector<double> p1(nblocks + 1, 0.0);
    for (size_t b = 0; b < nblocks; ++b) {
        double s = 0.0;
        for (size_t k = 0; k < kSegDim; ++k) {
            s += static_cast<double>(in.var[b * kSegDim + k]);
        }
        p1[b + 1] = p1[b] + s;
    }
    const double total_var = p1[nblocks];

    const size_t span_i = nblocks + 1;
    const size_t span_t = tot_units + 1;
    const size_t layer = span_i * span_t;
    std::vector<double> f((max_segs + 1) * layer, kInf);
    std::vector<uint32_t> pred((max_segs + 1) * layer, 0);
    auto at = [&](size_t ns, size_t i, size_t t) { return (ns * span_i + i) * span_t + t; };

    f[at(0, 0, 0)] = 0.0;
    double best = kInf;
    size_t ans_ns = 0;
    size_t ans_t = 0;
    for (size_t ns = 0; ns <= max_segs; ++ns) {
        for (size_t i = 0; i <= nblocks; ++i) {
            for (size_t t = 0; t <= tot_units; ++t) {
                const size_t cur_idx = at(ns, i, t);
                const double cur = f[cur_idx];
                if (cur >= kInf) {
                    continue;
                }
                if (i == nblocks) {
                    if (cur * 1.01 < best) {  // SAQ's acceptance rule
                        best = cur;
                        ans_ns = ns;
                        ans_t = t;
                    }
                    continue;
                }
                if (ns == max_segs) {
                    continue;
                }
                for (size_t j = 1; i + j <= nblocks; ++j) {
                    const double* row = &cost[(i * span_i + j) * (kMaxBits + 1)];
                    for (int b = 1; b <= in.max_bits; ++b) {
                        const size_t t2 = t + static_cast<size_t>(b) * j + 1;
                        if (t2 > tot_units) {
                            break;  // monotone in b, as in SAQ
                        }
                        const double v = cur + row[b];
                        const size_t to = at(ns + 1, i + j, t2);
                        if (v < f[to]) {
                            f[to] = v;
                            pred[to] = static_cast<uint32_t>((i << 4) | static_cast<size_t>(b));
                        }
                    }
                }
                // B == 0: drop every remaining dimension, priced at its full
                // variance (SAQ's tail-truncation action).
                const double rest = total_var - p1[i];
                const size_t to = at(ns + 1, nblocks, t);
                if (cur + rest < f[to]) {
                    f[to] = cur + rest;
                    pred[to] = static_cast<uint32_t>(i << 4);
                }
            }
        }
    }

    if (best >= kInf) {
        // SAQ's DP is infeasible when its segment cap is zero (padded dim
        // 64 with avg_bits >= 2 caps the plan at dim/128 segments); the
        // reference implementation then walks back from an untouched cell,
        // which yields the single all-dropped run below. Replicated, with a
        // warning, so the baseline matches SAQ cell for cell.
        std::fprintf(
            stderr,
            "e8qlib::me8q::Allocate: SAQ cap admits no plan (dim=%zu, avg_bits=%g); "
            "emitting the all-dropped run as the reference does\n",
            in.dim, static_cast<double>(in.avg_bits)
        );
        return Plan{{in.dim, 0}};
    }
    Plan plan;
    size_t ns = ans_ns;
    size_t i = nblocks;
    size_t t = ans_t;
    while (i > 0) {
        const uint32_t p = pred[at(ns, i, t)];
        const size_t prev_i = p >> 4;
        const int bits = static_cast<int>(p & 0xfU);
        plan.push_back({(i - prev_i) * kSegDim, bits});
        ns--;
        if (bits != 0) {
            t -= static_cast<size_t>(bits) * (i - prev_i) + 1;
        }
        i = prev_i;
    }
    std::reverse(plan.begin(), plan.end());
    return plan;
}

PlanSummary Summarize(const Plan& plan, size_t dim) {
    PlanSummary s;
    s.dim = dim;
    for (const PlanSeg& seg : plan) {
        if (seg.bits == 0) {
            s.dropped_dims += seg.len;
            continue;
        }
        ++s.num_segments;
        s.max_bits = std::max(s.max_bits, seg.bits);
        s.payload_bits += seg.len * static_cast<size_t>(seg.bits);
        s.factor_bits += kFactorBits;
    }
    s.bits_per_dim = dim == 0 ? 0.0
                              : static_cast<double>(s.payload_bits + s.factor_bits) /
                                    static_cast<double>(dim);
    return s;
}

}  // namespace e8qlib::me8q
