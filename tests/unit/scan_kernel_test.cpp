// Scan-kernel ISA parity: generic / AVX2 / AVX-512 produce identical i32
// lane sums (integer-exact by construction; this pins the layout contracts).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "e8qlib/index/e8q.hpp"
#include "e8qlib/simd/dispatch.hpp"
#include "simd/scan_kernels.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

void CheckEqual32(const int32_t* a, const int32_t* b, const char* tag) {
    for (int t = 0; t < 32; ++t) {
        if (a[t] != b[t]) {
            std::printf("FAIL %s lane %d: %d != %d\n", tag, t, a[t], b[t]);
            ++g_failures;
            return;
        }
    }
}

void TestAccumulateCoarseHi() {
    std::mt19937 rng(20260920);
    std::uniform_int_distribution<int> ui(0, 255);
    for (size_t blocks : {3u, 64u, 200u}) {
        std::vector<uint8_t> payload(blocks * 32);
        std::vector<uint8_t> lut_c(blocks * 64);
        for (auto& c : payload) c = static_cast<uint8_t>(ui(rng));
        for (auto& c : lut_c) c = static_cast<uint8_t>(ui(rng));
        alignas(32) int32_t g[32];
        e8qlib::scan_simd::AccumulateCoarseHiGeneric(
            payload.data(), lut_c.data(), blocks, g
        );
#if defined(E8Q_HAVE_AVX2)
        alignas(32) int32_t a[32];
        e8qlib::scan_simd::AccumulateCoarseHiAvx2(
            payload.data(), lut_c.data(), blocks, a
        );
        CheckEqual32(g, a, "AccumulateCoarseHi generic vs avx2");
#endif
#if defined(E8Q_HAVE_AVX512)
        if (e8qlib::simd::Avx512Enabled()) {
            // The AVX-512 kernel reads the pre-permuted LUT the scan builds
            // once per query (FillPairLut), not lut_c directly.
            std::vector<uint8_t> lut_pair(blocks * 64, 0);
            e8qlib::FillPairLut(lut_c.data(), blocks, lut_pair.data());
            alignas(32) int32_t f[32];
            e8qlib::scan_simd::AccumulateCoarseHiAvx512(
                payload.data(), lut_c.data(), lut_pair.data(), blocks, f
            );
            CheckEqual32(g, f, "AccumulateCoarseHi generic vs avx512");
        }
#endif
    }
}

void TestAccumulateCoarseHiPair() {
    std::mt19937 rng(20260921);
    std::uniform_int_distribution<int> ui(0, 255);
    for (size_t blocks : {5u, 129u}) {
        std::vector<uint8_t> p1(blocks * 32);
        std::vector<uint8_t> p2(blocks * 32);
        std::vector<uint8_t> lut_c(blocks * 64);
        for (auto& c : p1) c = static_cast<uint8_t>(ui(rng));
        for (auto& c : p2) c = static_cast<uint8_t>(ui(rng));
        for (auto& c : lut_c) c = static_cast<uint8_t>(ui(rng));
        alignas(32) int32_t g1[32];
        alignas(32) int32_t g2[32];
        e8qlib::scan_simd::AccumulateCoarseHiPairGeneric(
            p1.data(), p2.data(), lut_c.data(), blocks, g1, g2
        );
#if defined(E8Q_HAVE_AVX2)
        alignas(32) int32_t a1[32];
        alignas(32) int32_t a2[32];
        e8qlib::scan_simd::AccumulateCoarseHiPairAvx2(
            p1.data(), p2.data(), lut_c.data(), blocks, a1, a2
        );
        CheckEqual32(g1, a1, "AccumulateCoarseHiPair out1 generic vs avx2");
        CheckEqual32(g2, a2, "AccumulateCoarseHiPair out2 generic vs avx2");
#endif
#if defined(E8Q_HAVE_AVX512)
        if (e8qlib::simd::Avx512Enabled()) {
            std::vector<uint8_t> lut_pair(blocks * 64, 0);
            e8qlib::FillPairLut(lut_c.data(), blocks, lut_pair.data());
            alignas(32) int32_t f1[32];
            alignas(32) int32_t f2[32];
            e8qlib::scan_simd::AccumulateCoarseHiPairAvx512(
                p1.data(), p2.data(), lut_c.data(), lut_pair.data(), blocks, f1, f2
            );
            CheckEqual32(g1, f1, "AccumulateCoarseHiPair out1 generic vs avx512");
            CheckEqual32(g2, f2, "AccumulateCoarseHiPair out2 generic vs avx512");
        }
#endif
    }
}

}  // namespace

int main() {
    TestAccumulateCoarseHi();
    TestAccumulateCoarseHiPair();
    if (g_failures == 0) {
        std::printf("all scan kernel checks passed\n");
        return 0;
    }
    std::printf("%d scan kernel checks failed\n", g_failures);
    return 1;
}
