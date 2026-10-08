// AVX2 rowA scan kernels: one u8 LUT stream (64 B per block) over the
// byte-per-lane rows; the per-lane u8 sums come back i32-exact.
#include "simd/scan_kernels.hpp"
#include "simd/simd_helpers.hpp"

#include <immintrin.h>

#include <algorithm>

namespace e8qlib {
namespace scan_simd {

void AccumulateCoarseHiAvx2(
    const uint8_t* __restrict__ payload, const uint8_t* __restrict__ lut_c,
    size_t blocks, int32_t out[32]
) {
    // Chunk bound: 128 x 2 x 255 < 2^16 per u16 lane (two lookups per block).
    constexpr size_t kChunkBlocks = 128;
    const __m256i nib_mask = _mm256_set1_epi8(0xF);
    // Natural byte order (byte j = lane j): the even/odd byte split
    // stays inside the u16 lanes (masked to the low byte), so no per-block
    // 128-bit extracts are needed — one widen per 128-block chunk.
    __m256i evA = _mm256_setzero_si256();
    __m256i evB = _mm256_setzero_si256();
    __m256i odA = _mm256_setzero_si256();
    __m256i odB = _mm256_setzero_si256();
    for (size_t b0 = 0; b0 < blocks; b0 += kChunkBlocks) {
        const size_t bn = std::min(kChunkBlocks, blocks - b0);
        __m256i accE = _mm256_setzero_si256();
        __m256i accO = _mm256_setzero_si256();
        for (size_t bi = b0; bi < b0 + bn; ++bi) {
            const __m256i H = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(payload + bi * 32)
            );
            const uint8_t* lc = lut_c + bi * 64;
            __m256i rl = _mm256_shuffle_epi8(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lc)),
                _mm256_and_si256(H, nib_mask)
            );
            __m256i rh = _mm256_shuffle_epi8(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lc + 32)),
                _mm256_and_si256(_mm256_srli_epi16(H, 4), nib_mask)
            );
            // Mod-2^16 split: W = (hi << 8) + lo, so sum(lo) is recovered
            // once per chunk as sum(W) - (sum(hi) << 8) — two ANDs per block
            // saved (chunk x 2 x 255 < 2^16, no carry between bytes).
            accE = _mm256_add_epi16(accE, rl);
            accE = _mm256_add_epi16(accE, rh);
            accO = _mm256_add_epi16(accO, _mm256_srli_epi16(rl, 8));
            accO = _mm256_add_epi16(accO, _mm256_srli_epi16(rh, 8));
        }
        accE = _mm256_sub_epi16(accE, _mm256_slli_epi16(accO, 8));
        evA = _mm256_add_epi32(evA, _mm256_cvtepu16_epi32(_mm256_castsi256_si128(accE)));
        evB = _mm256_add_epi32(evB, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(accE, 1)));
        odA = _mm256_add_epi32(odA, _mm256_cvtepu16_epi32(_mm256_castsi256_si128(accO)));
        odB = _mm256_add_epi32(odB, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(accO, 1)));
    }
    detail::UnpackAcc(detail::AccYmm{evA, evB, odA, odB}, out);
}

// Two-batch form of AccumulateCoarseHi: one block loop scans two consecutive
// 32-lane batches of the same cluster and shares the per-block LUT loads (the
// LUT stream is 2/3 of the single-batch kernel's loads, so pairing cuts the
// load-port pressure on the scan's hot loop). The chunk bounds apply per
// accumulator chain exactly as in the single-batch form.
void AccumulateCoarseHiPairAvx2(
    const uint8_t* __restrict__ payload1, const uint8_t* __restrict__ payload2,
    const uint8_t* __restrict__ lut_c, size_t blocks, int32_t out1[32], int32_t out2[32]
) {
    constexpr size_t kChunkBlocks = 128;
    const __m256i nib_mask = _mm256_set1_epi8(0xF);
    __m256i evA1 = _mm256_setzero_si256();
    __m256i evB1 = _mm256_setzero_si256();
    __m256i odA1 = _mm256_setzero_si256();
    __m256i odB1 = _mm256_setzero_si256();
    __m256i evA2 = _mm256_setzero_si256();
    __m256i evB2 = _mm256_setzero_si256();
    __m256i odA2 = _mm256_setzero_si256();
    __m256i odB2 = _mm256_setzero_si256();
    for (size_t b0 = 0; b0 < blocks; b0 += kChunkBlocks) {
        const size_t bn = std::min(kChunkBlocks, blocks - b0);
        __m256i accE1 = _mm256_setzero_si256();
        __m256i accO1 = _mm256_setzero_si256();
        __m256i accE2 = _mm256_setzero_si256();
        __m256i accO2 = _mm256_setzero_si256();
        for (size_t bi = b0; bi < b0 + bn; ++bi) {
            const __m256i H1 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(payload1 + bi * 32)
            );
            const __m256i H2 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(payload2 + bi * 32)
            );
            const uint8_t* lc = lut_c + bi * 64;
            const __m256i lc0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lc));
            const __m256i lc1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lc + 32));
            __m256i rlA = _mm256_shuffle_epi8(lc0, _mm256_and_si256(H1, nib_mask));
            __m256i rhA = _mm256_shuffle_epi8(
                lc1, _mm256_and_si256(_mm256_srli_epi16(H1, 4), nib_mask)
            );
            __m256i rlB = _mm256_shuffle_epi8(lc0, _mm256_and_si256(H2, nib_mask));
            __m256i rhB = _mm256_shuffle_epi8(
                lc1, _mm256_and_si256(_mm256_srli_epi16(H2, 4), nib_mask)
            );
            accE1 = _mm256_add_epi16(accE1, rlA);
            accE1 = _mm256_add_epi16(accE1, rhA);
            accO1 = _mm256_add_epi16(accO1, _mm256_srli_epi16(rlA, 8));
            accO1 = _mm256_add_epi16(accO1, _mm256_srli_epi16(rhA, 8));
            accE2 = _mm256_add_epi16(accE2, rlB);
            accE2 = _mm256_add_epi16(accE2, rhB);
            accO2 = _mm256_add_epi16(accO2, _mm256_srli_epi16(rlB, 8));
            accO2 = _mm256_add_epi16(accO2, _mm256_srli_epi16(rhB, 8));
        }
        accE1 = _mm256_sub_epi16(accE1, _mm256_slli_epi16(accO1, 8));
        accE2 = _mm256_sub_epi16(accE2, _mm256_slli_epi16(accO2, 8));
        evA1 = _mm256_add_epi32(evA1, _mm256_cvtepu16_epi32(_mm256_castsi256_si128(accE1)));
        evB1 = _mm256_add_epi32(evB1, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(accE1, 1)));
        odA1 = _mm256_add_epi32(odA1, _mm256_cvtepu16_epi32(_mm256_castsi256_si128(accO1)));
        odB1 = _mm256_add_epi32(odB1, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(accO1, 1)));
        evA2 = _mm256_add_epi32(evA2, _mm256_cvtepu16_epi32(_mm256_castsi256_si128(accE2)));
        evB2 = _mm256_add_epi32(evB2, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(accE2, 1)));
        odA2 = _mm256_add_epi32(odA2, _mm256_cvtepu16_epi32(_mm256_castsi256_si128(accO2)));
        odB2 = _mm256_add_epi32(odB2, _mm256_cvtepu16_epi32(_mm256_extracti128_si256(accO2, 1)));
    }
    detail::UnpackAcc(detail::AccYmm{evA1, evB1, odA1, odB1}, out1);
    detail::UnpackAcc(detail::AccYmm{evA2, evB2, odA2, odB2}, out2);
}

}  // namespace scan_simd
}  // namespace e8qlib
