#pragma once

// ISA rowA scan kernels (private, not installed). The public entry points
// keep the names declared in e8qlib/index/e8q.hpp and dispatch at runtime;
// the ISA variants live in scan_kernels_{generic,avx2,avx512}.cpp. All
// kernels are integer-exact across backends (same u8 LUTs, same u16 lane
// bounds), so dispatch only changes speed, never scores.

#include <cstddef>
#include <cstdint>

namespace e8qlib {
namespace scan_simd {

void AccumulateCoarseHiGeneric(
    const uint8_t* __restrict__ payload, const uint8_t* __restrict__ lut_c,
    size_t blocks, int32_t out[32]
);
void AccumulateCoarseHiPairGeneric(
    const uint8_t* __restrict__ payload1, const uint8_t* __restrict__ payload2,
    const uint8_t* __restrict__ lut_c, size_t blocks, int32_t out1[32], int32_t out2[32]
);

#if defined(E8Q_HAVE_AVX2)
void AccumulateCoarseHiAvx2(
    const uint8_t* __restrict__ payload, const uint8_t* __restrict__ lut_c,
    size_t blocks, int32_t out[32]
);
void AccumulateCoarseHiPairAvx2(
    const uint8_t* __restrict__ payload1, const uint8_t* __restrict__ payload2,
    const uint8_t* __restrict__ lut_c, size_t blocks, int32_t out1[32], int32_t out2[32]
);
#endif

#if defined(E8Q_HAVE_AVX512)
void AccumulateCoarseHiAvx512(
    const uint8_t* __restrict__ payload, const uint8_t* __restrict__ lut_c,
    const uint8_t* __restrict__ lut_pair, size_t blocks, int32_t out[32]
);
void AccumulateCoarseHiPairAvx512(
    const uint8_t* __restrict__ payload1, const uint8_t* __restrict__ payload2,
    const uint8_t* __restrict__ lut_c, const uint8_t* __restrict__ lut_pair, size_t blocks,
    int32_t out1[32], int32_t out2[32]
);
void FillPairLutAvx512(
    const uint8_t* __restrict__ lut_c, size_t blocks, uint8_t* __restrict__ lut_pair
);
#endif

}  // namespace scan_simd
}  // namespace e8qlib
