// Scalar reference twins of the rowA scan kernels plus the runtime
// dispatchers. Integer-exact with the SIMD variants: the same u8 LUTs and
// the same per-lane sums, evaluated lane by lane (the u16 chunk bounds make
// every overflow-free summation order equivalent).
#include "simd/scan_kernels.hpp"

namespace e8qlib {
namespace scan_simd {

namespace {

// One lane's coarse contribution from one block: the low nibble of byte t
// prices against the lane's LUT half [0,16), the high nibble against [32,64).
inline int CoarseLaneByte(const uint8_t* lc, uint8_t h, int t) {
    const size_t half = static_cast<size_t>(t >> 4) * 16;
    return lc[half + (h & 0xF)] + lc[32 + half + (h >> 4)];
}

}  // namespace

void AccumulateCoarseHiGeneric(
    const uint8_t* __restrict__ payload, const uint8_t* __restrict__ lut_c,
    size_t blocks, int32_t out[32]
) {
    for (int t = 0; t < 32; ++t) {
        out[t] = 0;
    }
    for (size_t bi = 0; bi < blocks; ++bi) {
        const uint8_t* h = payload + bi * 32;
        const uint8_t* lc = lut_c + bi * 64;
        for (int t = 0; t < 32; ++t) {
            out[t] += CoarseLaneByte(lc, h[t], t);
        }
    }
}

void AccumulateCoarseHiPairGeneric(
    const uint8_t* __restrict__ payload1, const uint8_t* __restrict__ payload2,
    const uint8_t* __restrict__ lut_c, size_t blocks, int32_t out1[32], int32_t out2[32]
) {
    for (int t = 0; t < 32; ++t) {
        out1[t] = 0;
        out2[t] = 0;
    }
    for (size_t bi = 0; bi < blocks; ++bi) {
        const uint8_t* h1 = payload1 + bi * 32;
        const uint8_t* h2 = payload2 + bi * 32;
        const uint8_t* lc = lut_c + bi * 64;
        for (int t = 0; t < 32; ++t) {
            out1[t] += CoarseLaneByte(lc, h1[t], t);
            out2[t] += CoarseLaneByte(lc, h2[t], t);
        }
    }
}

}  // namespace scan_simd
}  // namespace e8qlib
