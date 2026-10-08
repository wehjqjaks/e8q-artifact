#include "simd/scan_kernels.hpp"

#include <cstring>

#include "e8qlib/index/e8q.hpp"
#include "e8qlib/simd/dispatch.hpp"

namespace e8qlib {

void AccumulateCoarseHi(
    const uint8_t* payload, const uint8_t* lut_c, const uint8_t* lut_pair, size_t blocks,
    int32_t out[32]
) {
#if defined(E8Q_HAVE_AVX512)
    if (simd::Avx512Enabled()) {
        scan_simd::AccumulateCoarseHiAvx512(payload, lut_c, lut_pair, blocks, out);
        return;
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (simd::Avx2Enabled()) {
        scan_simd::AccumulateCoarseHiAvx2(payload, lut_c, blocks, out);
        return;
    }
#endif
    scan_simd::AccumulateCoarseHiGeneric(payload, lut_c, blocks, out);
}

void AccumulateCoarseHiPair(
    const uint8_t* payload1, const uint8_t* payload2, const uint8_t* lut_c,
    const uint8_t* lut_pair, size_t blocks, int32_t out1[32], int32_t out2[32]
) {
#if defined(E8Q_HAVE_AVX512)
    if (simd::Avx512Enabled()) {
        scan_simd::AccumulateCoarseHiPairAvx512(
            payload1, payload2, lut_c, lut_pair, blocks, out1, out2
        );
        return;
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (simd::Avx2Enabled()) {
        scan_simd::AccumulateCoarseHiPairAvx2(
            payload1, payload2, lut_c, blocks, out1, out2
        );
        return;
    }
#endif
    scan_simd::AccumulateCoarseHiPairGeneric(
        payload1, payload2, lut_c, blocks, out1, out2
    );
}

void FillPairLut(const uint8_t* lut_c, size_t blocks, uint8_t* lut_pair) {
#if defined(E8Q_HAVE_AVX512)
    if (simd::Avx512Enabled()) {
        scan_simd::FillPairLutAvx512(lut_c, blocks, lut_pair);
        return;
    }
#endif
    for (size_t bi = 0; bi + 2 <= blocks; bi += 2) {
        std::memcpy(lut_pair + bi * 64, lut_c + bi * 64, 32);
        std::memcpy(lut_pair + bi * 64 + 32, lut_c + bi * 64 + 64, 32);
        std::memcpy(lut_pair + bi * 64 + 64, lut_c + bi * 64 + 32, 32);
        std::memcpy(lut_pair + bi * 64 + 96, lut_c + bi * 64 + 96, 32);
    }
}

}  // namespace e8qlib
