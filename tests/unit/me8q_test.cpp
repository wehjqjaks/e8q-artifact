// ME8Q unit + invariant tests.
//
// Coverage:
//   * SAQ allocator replica vs the library DP
//   * plan invariants (tail-drop, budget, padded coverage)
//   * bit-tight segment packing round-trip at rates 1..13
//   * codec distortion monotone in rate
//   * scorer agrees with unpack-and-lookup
//   * segment encoder matches E8Q EncodeVectorAtScale at rates 1..13
//   * index build / exhaustive estimator / IVF / save-load
//   * certified prefix prune and cascade identity
//   * two-level FastScan equivalence at huge eps
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "e8qlib/quantization/e8q_codec.hpp"
#include "e8qlib/quantization/me8q_codec.hpp"
#include "e8qlib/quantization/me8q_plan.hpp"
#include "e8qlib/index/me8q.hpp"
#include "e8qlib/utils/rotator.hpp"

#include "me8q_saq_reference.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                          \
    do {                                                                               \
        const double a_ = (a);                                                         \
        const double b_ = (b);                                                         \
        if (!(std::abs(a_ - b_) <= (tol))) {                                           \
            std::printf("FAIL %s:%d: %s = %.17g != %s = %.17g\n", __FILE__, __LINE__,  \
                        #a, a_, #b, b_);                                               \
            ++g_failures;                                                              \
        }                                                                              \
    } while (0)

using e8qlib::me8q::Plan;
using e8qlib::me8q::PlanSeg;
using e8qlib::me8q::PlanInput;
using e8qlib::me8q::kSegDim;
using e8qlib::me8q::kMaxBits;
using e8qlib::me8q::kFactorBits;

size_t PlanBits(const Plan& plan) {
    size_t bits = 0;
    for (const PlanSeg& seg : plan) {
        if (seg.bits > 0) {
            bits += static_cast<size_t>(seg.bits) * seg.len + kFactorBits;
        }
    }
    return bits;
}

void TestPlanReplica() {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(0.0F, 1.0F);
    const size_t dims[] = {64, 128, 192, 256, 320};
    const float bits[] = {1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    for (size_t dim : dims) {
        for (float b : bits) {
            std::vector<float> var(dim);
            for (size_t j = 0; j < dim; ++j) {
                var[j] = static_cast<float>((1.0 + 0.9 * u(rng)) * std::pow(0.97, static_cast<double>(j)));
            }
            const Plan ref = e8qlib::me8q::ReferenceSaqDp(var.data(), dim, b);
            PlanInput in;
            in.var = var.data();
            in.dim = dim;
            in.avg_bits = b;
            in.max_bits = kMaxBits;
            const Plan got = e8qlib::me8q::Allocate(in);
            if (ref == got) {
                continue;
            }
            const double c_ref = SaqPlanCost(ref, var.data(), dim);
            const double c_got = SaqPlanCost(got, var.data(), dim);
            CHECK_NEAR(c_ref, c_got, 1e-9 * std::max(1.0, std::abs(c_ref)));
            CHECK(PlanBits(ref) == PlanBits(got) || c_ref > 0.0);
        }
    }
}

void TestPlanInvariants() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(0.0F, 1.0F);
    const float bits[] = {0.5F, 1.0F, 2.0F, 4.0F, 8.0F};
    for (size_t dim : {size_t{64}, size_t{192}, size_t{960}}) {
        for (float b : bits) {
            std::vector<float> var(dim);
            for (size_t j = 0; j < dim; ++j) {
                var[j] = static_cast<float>((1.0 + u(rng)) * std::pow(0.98, static_cast<double>(j)));
            }
            PlanInput in;
            in.var = var.data();
            in.dim = dim;
            in.avg_bits = b;
            const Plan plan = e8qlib::me8q::Allocate(in);
            size_t total = 0;
            bool tail_dropped = false;
            for (const PlanSeg& seg : plan) {
                CHECK(seg.len % kSegDim == 0);
                CHECK(seg.bits >= 0 && seg.bits <= kMaxBits);
                if (seg.bits == 0) {
                    tail_dropped = true;
                } else {
                    CHECK(!tail_dropped);
                }
                total += seg.len;
            }
            CHECK(total == dim);
            const size_t budget =
                static_cast<size_t>(b * static_cast<float>(dim)) + kFactorBits;
            CHECK(PlanBits(plan) <= budget);
            CHECK(e8qlib::me8q::Summarize(plan, dim).bits_per_dim <=
                  static_cast<double>(budget) / static_cast<double>(dim));
        }
    }
}

void TestPackRoundTrip() {
    std::mt19937 rng(99);
    for (int bits = 1; bits <= kMaxBits; ++bits) {
        const int d_s = 64;
        e8qlib::me8q::SegCode code;
        code.digits.resize(static_cast<size_t>(d_s));
        code.branch.resize(static_cast<size_t>(d_s) / 8);
        const uint32_t digit_mask = (1U << bits) - 1U;
        for (size_t blk = 0; blk < code.branch.size(); ++blk) {
            code.branch[blk] = static_cast<uint8_t>(rng() & 1U);
            int parity = 0;
            for (int i = 0; i < 7; ++i) {
                const uint16_t d = static_cast<uint16_t>(rng() & digit_mask);
                code.digits[blk * 8 + static_cast<size_t>(i)] = d;
                parity ^= static_cast<int>(d & 1U);
            }
            code.digits[blk * 8 + 7] = static_cast<uint16_t>(parity);
        }
        std::vector<uint8_t> packed(static_cast<size_t>(e8qlib::me8q::SegmentBytes(bits, d_s)));
        e8qlib::me8q::PackSegment(code, d_s, bits, packed.data());
        std::vector<uint16_t> digits(static_cast<size_t>(d_s));
        std::vector<uint8_t> branch(static_cast<size_t>(d_s) / 8);
        e8qlib::me8q::UnpackSegment(packed.data(), d_s, bits, digits.data(), branch.data());
        for (size_t blk = 0; blk < branch.size(); ++blk) {
            CHECK(branch[blk] == code.branch[blk]);
            for (int i = 0; i < 8; ++i) {
                CHECK(digits[blk * 8 + static_cast<size_t>(i)] ==
                      code.digits[blk * 8 + static_cast<size_t>(i)]);
            }
        }
        CHECK(e8qlib::me8q::SegmentBytes(bits, d_s) == bits * d_s / 8);
    }
}

void TestCodecDistortionMonotone() {
    std::mt19937 rng(4242);
    std::normal_distribution<float> g(0.0F, 1.0F);
    const int d_s = 256;
    const int n = 2048;
    std::vector<float> vec(static_cast<size_t>(d_s) * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d_s; ++j) {
            const double scale = std::pow(0.97, j);
            vec[static_cast<size_t>(i) * d_s + j] = static_cast<float>(g(rng) * scale);
        }
    }
    e8qlib::me8q::SegmentCodec codec;
    double prev = std::numeric_limits<double>::max();
    for (int bits = 1; bits <= kMaxBits; bits += 3) {
        double err = 0.0;
        double energy = 0.0;
        std::vector<uint8_t> packed(static_cast<size_t>(e8qlib::me8q::SegmentBytes(bits, d_s)));
        std::vector<float> levels(static_cast<size_t>(d_s));
        for (int i = 0; i < n; ++i) {
            const float* r = vec.data() + static_cast<size_t>(i) * d_s;
            double norm2 = 0.0;
            for (int j = 0; j < d_s; ++j) {
                norm2 += static_cast<double>(r[j]) * static_cast<double>(r[j]);
            }
            const double norm = std::sqrt(norm2);
            const e8qlib::me8q::SegCode code = codec.Encode(r, d_s, bits, norm);
            e8qlib::me8q::PackSegment(code, d_s, bits, packed.data());
            double n2 = 0.0;
            const double dot = codec.ScoreAndNorm2(packed.data(), d_s, bits, r, &n2);
            e8qlib::me8q::ReconstructLevels(packed.data(), d_s, bits, codec, levels.data());
            const double alpha = n2 > 0.0 ? dot / n2 : 0.0;
            for (int j = 0; j < d_s; ++j) {
                const double d = static_cast<double>(r[j]) - alpha * levels[static_cast<size_t>(j)];
                err += d * d;
            }
            energy += norm2;
        }
        const double rel = err / energy;
        std::printf("  [mono] bits=%d rel=%.6g\n", bits, rel);
        CHECK(rel < prev);
        prev = rel;
    }
}

void TestScoringMatchesUnpacking() {
    std::mt19937 rng(31337);
    std::normal_distribution<float> g(0.0F, 1.0F);
    const int d_s = 128;
    e8qlib::me8q::SegmentCodec codec;
    for (int bits = 1; bits <= kMaxBits; ++bits) {
        std::vector<float> r(static_cast<size_t>(d_s));
        std::vector<float> q(static_cast<size_t>(d_s));
        for (float& v : r) {
            v = g(rng);
        }
        for (float& v : q) {
            v = g(rng);
        }
        double norm2 = 0.0;
        for (float v : r) {
            norm2 += static_cast<double>(v) * v;
        }
        const e8qlib::me8q::SegCode code = codec.Encode(r.data(), d_s, bits, std::sqrt(norm2));
        std::vector<uint8_t> packed(static_cast<size_t>(e8qlib::me8q::SegmentBytes(bits, d_s)));
        e8qlib::me8q::PackSegment(code, d_s, bits, packed.data());
        double n2 = 0.0;
        const double score = codec.ScoreAndNorm2(packed.data(), d_s, bits, q.data(), &n2);
        std::vector<uint16_t> digits(static_cast<size_t>(d_s));
        std::vector<uint8_t> branch(static_cast<size_t>(d_s) / 8);
        e8qlib::me8q::UnpackSegment(packed.data(), d_s, bits, digits.data(), branch.data());
        double score_ref = 0.0;
        double n2_ref = 0.0;
        for (int blk = 0; blk < d_s / 8; ++blk) {
            const float* lev = codec.Levels(bits, branch[static_cast<size_t>(blk)]);
            for (int i = 0; i < 8; ++i) {
                const double lv = lev[digits[static_cast<size_t>(blk) * 8 + i]];
                score_ref += static_cast<double>(q[static_cast<size_t>(blk) * 8 + i]) * lv;
                n2_ref += lv * lv;
            }
        }
        CHECK_NEAR(score, score_ref, 1e-6 * std::max(1.0, std::abs(score_ref)));
        CHECK_NEAR(n2, n2_ref, 1e-6 * std::max(1.0, std::abs(n2_ref)));
        const double score_fast = static_cast<double>(codec.ScoreFast(packed.data(), d_s, bits, q.data()));
        CHECK_NEAR(score_fast, score_ref, 1e-4 * std::max(1.0, std::abs(score_ref)));
        const uint8_t* ins[4] = {packed.data(), packed.data(), packed.data(), packed.data()};
        float s4[4];
        codec.ScoreFast4(ins, d_s, bits, q.data(), s4);
        for (int j = 0; j < 4; ++j) {
            CHECK_NEAR(static_cast<double>(s4[j]), score_fast, 1e-6);
        }
        {
            const int nblk = d_s / 8;
            std::vector<uint8_t> plane(static_cast<size_t>(nblk) * 32, 0);
            std::vector<uint8_t> refine;
            if (bits >= 2) {
                refine.resize(static_cast<size_t>((bits - 1) * d_s / 8));
                e8qlib::me8q::PackSegment(code, d_s, bits - 1, refine.data());
            }
            for (int blk = 0; blk < nblk; ++blk) {
                uint8_t byte = 0;
                for (int t = 0; t < 8; ++t) {
                    const int d = code.digits[static_cast<size_t>(blk) * 8 + t];
                    const int top = bits == 1 && t == 7
                                        ? (code.branch[static_cast<size_t>(blk)] & 1)
                                        : ((d >> (bits - 1)) & 1);
                    if (top != 0) {
                        byte = static_cast<uint8_t>(byte | (1u << (7 - t)));
                    }
                }
                plane[static_cast<size_t>(blk) * 32] = byte;
            }
            const uint8_t* refp = bits >= 2 ? refine.data() : nullptr;
            const float s1 = codec.ScoreRunSplit(refp, plane.data(), 0, nblk, bits, q.data());
            CHECK_NEAR(static_cast<double>(s1), score_fast, 1e-4 * std::max(1.0, std::abs(score_fast)));
            const uint8_t* refs[4] = {refp, refp, refp, refp};
            const uint8_t* pbs[4] = {plane.data(), plane.data(), plane.data(), plane.data()};
            float rs4[4];
            codec.ScoreRunSplit4(refs, pbs, 0, nblk, bits, q.data(), 4, rs4);
            for (int j = 0; j < 4; ++j) {
                CHECK_NEAR(static_cast<double>(rs4[j]), static_cast<double>(s1), 1e-6);
            }
        }
        for (size_t i = 0; i < digits.size(); ++i) {
            CHECK(digits[i] == code.digits[i]);
        }
        for (size_t blk = 0; blk < branch.size(); ++blk) {
            CHECK(branch[blk] == code.branch[blk]);
        }
        double n2r = 0.0;
        const double dot_r = codec.ScoreAndNorm2(packed.data(), d_s, bits, r.data(), &n2r);
        const double rho = dot_r / std::sqrt(norm2 * n2r);
        std::printf("  [corr] bits=%d rho=%.4f\n", bits, rho);
        CHECK(rho > (bits >= 6 ? 0.80 : 0.35));
    }
}

void TestEncoderMatchesE8q() {
    std::mt19937 rng(90210);
    std::normal_distribution<double> g(0.0, 1.0);
    const int d_s = 256;
    e8qlib::me8q::SegmentCodec codec;
    for (int trial = 0; trial < 4; ++trial) {
        std::vector<float> r(static_cast<size_t>(d_s));
        std::vector<float> unit(static_cast<size_t>(d_s));
        double norm2 = 0.0;
        for (int j = 0; j < d_s; ++j) {
            const double scale = std::pow(0.95, static_cast<double>(j));
            r[static_cast<size_t>(j)] = static_cast<float>(g(rng) * scale);
            norm2 += static_cast<double>(r[static_cast<size_t>(j)]) * r[static_cast<size_t>(j)];
        }
        const double norm = std::sqrt(norm2);
        for (int j = 0; j < d_s; ++j) {
            unit[static_cast<size_t>(j)] = static_cast<float>(r[static_cast<size_t>(j)] / norm);
        }
        for (int bits = 1; bits <= kMaxBits; ++bits) {
            const e8qlib::me8q::SegCode mine = codec.Encode(r.data(), d_s, bits, norm);
            const e8qlib::VectorCode ref = e8qlib::EncodeVectorAtScale(
                unit.data(), d_s, bits, e8qlib::FixedScale(bits)
            );
            for (size_t blk = 0; blk < mine.branch.size(); ++blk) {
                CHECK(mine.branch[blk] == ref.branch[blk]);
                for (int i = 0; i < 8; ++i) {
                    CHECK(mine.digits[blk * 8 + static_cast<size_t>(i)] ==
                          static_cast<uint16_t>(ref.digits[blk][static_cast<size_t>(i)]));
                }
            }
        }
    }
}

void TestTiledFht() {
    std::mt19937 rng(11);
    std::normal_distribution<float> g(0.0F, 1.0F);
    e8qlib::FhtKac4Rotator rot(64, 0xBEEFULL);
    for (const size_t dim : {size_t{64}, size_t{128}, size_t{192}, size_t{960}, size_t{1536}}) {
        std::vector<float> x(dim);
        std::vector<float> y(dim);
        for (size_t i = 0; i < dim; ++i) {
            x[i] = g(rng);
        }
        rot.RotateTiles(x.data(), y.data(), dim);
        double nx = 0.0;
        double ny = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            nx += static_cast<double>(x[i]) * x[i];
            ny += static_cast<double>(y[i]) * y[i];
        }
        CHECK_NEAR(std::sqrt(ny / nx), 1.0, 1e-4);
    }
}

struct Synth {
    size_t n = 3000;
    size_t dim = 128;
    size_t nq = 40;
    size_t k = 10;
    size_t clusters = 8;
    std::vector<float> base;
    std::vector<float> query;
    std::vector<float> centroids;
    std::vector<uint32_t> cluster_ids;
    std::vector<float> variance;
};

Synth MakeSynth() {
    Synth s;
    std::mt19937 rng(2026);
    std::normal_distribution<float> g(0.0F, 1.0F);
    s.base.resize(s.n * s.dim);
    for (size_t i = 0; i < s.n; ++i) {
        for (size_t j = 0; j < s.dim; ++j) {
            const double scale = std::pow(0.98, static_cast<double>(j));
            s.base[i * s.dim + j] = static_cast<float>(g(rng) * scale);
        }
    }
    s.cluster_ids.resize(s.n);
    s.clusters = 8;
    s.centroids.assign(s.clusters * s.dim, 0.0F);
    for (size_t i = 0; i < s.n; ++i) {
        const size_t c = static_cast<size_t>(s.base[i * s.dim] > 0.0F) |
                         (static_cast<size_t>(s.base[i * s.dim + 1] > 0.0F) << 1) |
                         (static_cast<size_t>(s.base[i * s.dim + 2] > 0.0F) << 2);
        s.cluster_ids[i] = static_cast<uint32_t>(c);
        for (size_t j = 0; j < s.dim; ++j) {
            s.centroids[c * s.dim + j] += s.base[i * s.dim + j];
        }
    }
    std::vector<size_t> counts(s.clusters, 0);
    for (size_t i = 0; i < s.n; ++i) {
        ++counts[s.cluster_ids[i]];
    }
    for (size_t c = 0; c < s.clusters; ++c) {
        for (size_t j = 0; j < s.dim; ++j) {
            s.centroids[c * s.dim + j] /= static_cast<float>(std::max<size_t>(counts[c], 1));
        }
    }
    s.query.resize(s.nq * s.dim);
    for (size_t i = 0; i < s.nq; ++i) {
        for (size_t j = 0; j < s.dim; ++j) {
            const double scale = std::pow(0.98, static_cast<double>(j));
            s.query[i * s.dim + j] = static_cast<float>(g(rng) * scale);
        }
    }
    s.variance.assign(s.dim, 0.0F);
    for (size_t i = 0; i < s.n; ++i) {
        for (size_t j = 0; j < s.dim; ++j) {
            s.variance[j] += s.base[i * s.dim + j] * s.base[i * s.dim + j];
        }
    }
    for (size_t j = 0; j < s.dim; ++j) {
        s.variance[j] /= static_cast<float>(s.n);
    }
    return s;
}

std::vector<uint32_t> ExactTopK(const Synth& s, const float* q, size_t k) {
    std::vector<std::pair<float, uint32_t>> all(s.n);
    for (size_t i = 0; i < s.n; ++i) {
        double d = 0.0;
        for (size_t j = 0; j < s.dim; ++j) {
            const double df = static_cast<double>(q[j]) - s.base[i * s.dim + j];
            d += df * df;
        }
        all[i] = {static_cast<float>(d), static_cast<uint32_t>(i)};
    }
    std::partial_sort(all.begin(), all.begin() + static_cast<long>(k), all.end());
    std::vector<uint32_t> out(k);
    for (size_t i = 0; i < k; ++i) {
        out[i] = all[i].second;
    }
    return out;
}

void TestIndexBuildSearchSaveLoad() {
    const Synth s = MakeSynth();
    e8qlib::me8q::BuildOptions opt;
    opt.avg_bits = 2.0F;
    opt.threads = 2;
    e8qlib::me8q::Me8qIndex idx;
    idx.Build(
        s.base.data(), s.dim, s.centroids.data(), s.dim, s.cluster_ids.data(), s.n, s.dim,
        s.clusters, opt, s.variance.data()
    );
    CHECK(idx.n() == s.n);
    CHECK(idx.padded_dim() == e8qlib::me8q::PaddedDim(s.dim));
    CHECK(idx.bits_per_dim() > 0.0 && idx.bits_per_dim() <= 2.0 + 1.0);

    std::vector<std::pair<float, uint32_t>> top;
    idx.ExhaustiveTopK(s.query.data(), s.k, &top);
    CHECK(top.size() == s.k);
    CHECK(top[0].first <= top[s.k - 1].first);
    for (const auto& item : top) {
        CHECK_NEAR(idx.Estimate(s.query.data(), item.second), item.first, 1e-3);
    }

    size_t hits = 0;
    const std::vector<uint32_t> exact = ExactTopK(s, s.query.data(), s.k);
    for (const auto& item : top) {
        if (std::find(exact.begin(), exact.end(), item.second) != exact.end()) {
            ++hits;
        }
    }
    std::printf("  synthetic exhaustive top-10 overlap: %zu/%zu\n", hits, s.k);
    CHECK(hits >= 6);

    const std::string path = "me8q_test_index.bin";
    idx.Save(path.c_str());
    e8qlib::me8q::Me8qIndex loaded;
    loaded.Load(path.c_str());
    std::vector<std::pair<float, uint32_t>> top2;
    loaded.ExhaustiveTopK(s.query.data(), s.k, &top2);
    CHECK(top2.size() == top.size());
    for (size_t i = 0; i < top.size(); ++i) {
        CHECK(top2[i].second == top[i].second);
        CHECK_NEAR(top2[i].first, top[i].first, 1e-3);
    }
    std::remove(path.c_str());

    std::vector<uint32_t> out_ids(s.k);
    e8qlib::me8q::SearchStats stats;
    idx.SearchIVF(s.query.data(), s.k, s.clusters, out_ids.data(), &stats);
    CHECK(stats.candidates == s.n);
    for (size_t i = 0; i < s.k; ++i) {
        CHECK(out_ids[i] == top[i].second);
    }
}

void TestLevelBounds() {
    e8qlib::me8q::SegmentCodec codec;
    for (int bits = 1; bits <= 8; ++bits) {
        const float mx = codec.LevelAbsMax(bits);
        CHECK(mx > 0.0F);
        for (int br = 0; br < 2; ++br) {
            const float* lev = codec.Levels(bits, br);
            const int q = 1 << bits;
            for (int d = 0; d < q; ++d) {
                CHECK(std::abs(lev[d]) <= mx + 1e-6F);
            }
        }
    }
}

void TestFastscanEquivalence() {
    const Synth s = MakeSynth();
    e8qlib::me8q::BuildOptions opt;
    opt.avg_bits = 3.0F;
    opt.threads = 2;
    e8qlib::me8q::Me8qIndex idx;
    idx.Build(
        s.base.data(), s.dim, s.centroids.data(), s.dim, s.cluster_ids.data(), s.n, s.dim,
        s.clusters, opt, s.variance.data()
    );
    // Tiny prune budget: the orthogonal bound then refines (essentially)
    // every lane, so the fast path must reproduce the exact top-k.
    e8qlib::me8q::Me8qIndex idx_full;
    e8qlib::me8q::BuildOptions opt_full = opt;
    opt_full.prune_alpha = 1e-9F;
    idx_full.Build(
        s.base.data(), s.dim, s.centroids.data(), s.dim, s.cluster_ids.data(), s.n, s.dim,
        s.clusters, opt_full, s.variance.data()
    );
    idx.BuildFastLayout();
    std::vector<uint32_t> a(s.k), b(s.k);
    e8qlib::me8q::SearchStats sa, sb;
    idx.SearchIVF(s.query.data(), s.k, s.clusters, a.data(), &sa);
    idx_full.SearchIVFFast(s.query.data(), s.k, s.clusters, b.data(), &sb);
    for (size_t i = 0; i < s.k; ++i) {
        CHECK(a[i] == b[i]);
    }
    // The union-bound slack is nonzero at every alpha, so the cascade may
    // still skip provably-lost tail runs; the top-k must stay exact.
    CHECK(sb.bits_scored <= sb.bits_total);
    std::printf("  fastscan alpha=1e-9 scored %zu/%zu bits (same top-k)\n", sb.bits_scored,
                sb.bits_total);
}

// The batched route sweeps the centroid matrix in blocks of ~128 KiB and
// must read each block's own centroid norms. Regression for the route bug
// that made the AVX2 and generic builds disagree with the AVX-512 build:
// RouteGemvAvx2/RouteGemvGeneric indexed cn2 by the block-local offset while
// StoreRoute512 indexed cn2 by the global cluster id, so every block past
// the first scored against the first block's norms. Invisible below ~257
// clusters (a single block) — this test uses 512.
void TestBatchRouteMatchesSingle() {
    const size_t n = 20000, dim = 128, clusters = 512, nq = 9, k = 10, nprobe = 16;
    std::mt19937 rng(4242);
    std::normal_distribution<float> g(0.0F, 1.0F);
    std::vector<float> base(n * dim);
    for (size_t i = 0; i < n * dim; ++i) {
        base[i] = g(rng);
    }
    std::vector<uint32_t> cids(n);
    for (size_t i = 0; i < n; ++i) {
        cids[i] = static_cast<uint32_t>(i % clusters);
    }
    std::vector<float> cent(clusters * dim, 0.0F);
    std::vector<size_t> cnt(clusters, 0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            cent[cids[i] * dim + j] += base[i * dim + j];
        }
        ++cnt[cids[i]];
    }
    for (size_t c = 0; c < clusters; ++c) {
        for (size_t j = 0; j < dim; ++j) {
            cent[c * dim + j] /= static_cast<float>(std::max<size_t>(cnt[c], 1));
        }
    }
    std::vector<float> variance(dim, 0.0F);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            variance[j] += base[i * dim + j] * base[i * dim + j];
        }
    }
    for (size_t j = 0; j < dim; ++j) {
        variance[j] /= static_cast<float>(n);
    }
    e8qlib::me8q::BuildOptions opt;
    opt.avg_bits = 3.0F;
    opt.threads = 2;
    opt.prune_alpha = 1e-9F;
    e8qlib::me8q::Me8qIndex idx;
    idx.Build(
        base.data(), dim, cent.data(), dim, cids.data(), n, dim, clusters, opt,
        variance.data()
    );
    std::vector<float> queries(nq * dim);
    for (size_t i = 0; i < nq * dim; ++i) {
        queries[i] = g(rng);
    }
    std::vector<uint32_t> b_ids(nq * k);
    std::vector<float> b_d(nq * k);
    e8qlib::me8q::SearchStats st;
    idx.SearchIVFFastBatch(
        queries.data(), nq, k, nprobe, b_ids.data(), &st, b_d.data(), 2
    );
    size_t mismatches = 0;
    for (size_t i = 0; i < nq; ++i) {
        std::vector<uint32_t> s_ids(k);
        std::vector<float> s_d(k);
        idx.SearchIVFFast(
            queries.data() + i * dim, k, nprobe, s_ids.data(), nullptr, s_d.data()
        );
        for (size_t j = 0; j < k; ++j) {
            if (s_ids[j] != b_ids[i * k + j]) {
                ++mismatches;
            }
            CHECK_NEAR(
                static_cast<double>(s_d[j]), static_cast<double>(b_d[i * k + j]), 1e-4
            );
        }
    }
    std::printf(
        "  batched route vs single query: %zu/%zu ids differ\n", mismatches, nq * k
    );
    CHECK(mismatches == 0);
}

// The allocator's degenerate fallback, end to end.  Padded dim 64 gives
// nblocks = 1, so the segment cap `avg_bits < 2 ? nblocks : nblocks / 2` is
// zero and the DP is infeasible; Allocate then emits the single all-dropped
// run, which is what the SAQ reference does at the same rate.  ME8Q routes
// over the coded prefix of the plan, so that index routes over nothing:
// route_dim() is 0 and the centroid sweep is a column of constants.
//
// Constants still have to be swept.  The group route divides the centroid
// matrix into L2-sized blocks and dim == 0 made that `131072 / (dim * 4)` a
// division by zero (SIGFPE, hit by tests/python/test_indexes.py's dim-64
// case); ivf::RouteL2Group now guards it.  This pins the path: both route
// forms must return, agree, and survive a save/load.
void TestDegenerateAllDroppedPlan() {
    const size_t n = 512, dim = 64, clusters = 8, nq = 6, k = 10, nprobe = 8;
    std::mt19937 rng(20261002);
    std::normal_distribution<float> g(0.0F, 1.0F);
    std::vector<float> base(n * dim);
    for (float& x : base) {
        x = g(rng);
    }
    std::vector<uint32_t> cids(n);
    for (size_t i = 0; i < n; ++i) {
        cids[i] = static_cast<uint32_t>(i % clusters);
    }
    std::vector<float> cent(clusters * dim, 0.0F);
    std::vector<size_t> cnt(clusters, 0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            cent[cids[i] * dim + j] += base[i * dim + j];
        }
        ++cnt[cids[i]];
    }
    for (size_t c = 0; c < clusters; ++c) {
        for (size_t j = 0; j < dim; ++j) {
            cent[c * dim + j] /= static_cast<float>(std::max<size_t>(cnt[c], 1));
        }
    }
    std::vector<float> variance(dim, 0.0F);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            variance[j] += base[i * dim + j] * base[i * dim + j];
        }
    }
    for (size_t j = 0; j < dim; ++j) {
        variance[j] /= static_cast<float>(n);
    }
    e8qlib::me8q::BuildOptions opt;
    opt.avg_bits = 3.0F;
    opt.threads = 2;

    e8qlib::me8q::Me8qIndex idx;
    // Allocate prints its "no plan" warning here; it is the point of the test.
    idx.Build(
        base.data(), dim, cent.data(), dim, cids.data(), n, dim, clusters, opt,
        variance.data()
    );
    CHECK(idx.route_dim() == 0);

    std::vector<float> queries(nq * dim);
    for (float& x : queries) {
        x = g(rng);
    }
    std::vector<uint32_t> b_ids(nq * k);
    std::vector<float> b_d(nq * k);
    e8qlib::me8q::SearchStats st{};
    idx.SearchIVFFastBatch(
        queries.data(), nq, k, nprobe, b_ids.data(), &st, b_d.data(), 2
    );
    for (size_t i = 0; i < nq; ++i) {
        std::vector<uint32_t> s_ids(k);
        std::vector<float> s_d(k);
        idx.SearchIVFFast(
            queries.data() + i * dim, k, nprobe, s_ids.data(), nullptr, s_d.data()
        );
        for (size_t j = 0; j < k; ++j) {
            CHECK(s_ids[j] < n);
            CHECK(b_ids[i * k + j] < n);
            CHECK(s_ids[j] == b_ids[i * k + j]);
            CHECK_NEAR(
                static_cast<double>(s_d[j]), static_cast<double>(b_d[i * k + j]), 1e-4
            );
        }
    }
    // No code means nothing to prune against: every row of every probed list
    // is a candidate and reaches the heap.
    CHECK(st.candidates == nq * n);
    CHECK(st.survivors == nq * n);
    CHECK(st.pruned == 0);

    const std::string path = "me8q_test_all_dropped.bin";
    idx.Save(path.c_str());
    e8qlib::me8q::Me8qIndex loaded;
    loaded.Load(path.c_str());
    std::vector<uint32_t> l_ids(nq * k);
    std::vector<float> l_d(nq * k);
    loaded.SearchIVFFastBatch(
        queries.data(), nq, k, nprobe, l_ids.data(), nullptr, l_d.data(), 2
    );
    for (size_t i = 0; i < nq * k; ++i) {
        CHECK(l_ids[i] == b_ids[i]);
    }
    std::remove(path.c_str());
}

}  // namespace

int main() {
    TestPlanReplica();
    TestPlanInvariants();
    TestPackRoundTrip();
    TestCodecDistortionMonotone();
    TestScoringMatchesUnpacking();
    TestEncoderMatchesE8q();
    TestTiledFht();
    TestIndexBuildSearchSaveLoad();
    TestLevelBounds();
    TestFastscanEquivalence();
    TestBatchRouteMatchesSingle();
    TestDegenerateAllDroppedPlan();
    if (g_failures == 0) {
        std::printf("all me8q checks passed\n");
        return 0;
    }
    std::printf("%d me8q checks failed\n", g_failures);
    return 1;
}
