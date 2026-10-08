// AVX-512 rowA scan kernels. Same integer semantics as the AVX2 twins
// (identical u8 LUTs, identical u16 lane bounds, exact arithmetic): blocks
// are processed in pairs so one 512-bit shuffle covers two 32-lane blocks.
#include "simd/scan_kernels.hpp"
#include "simd/simd_helpers.hpp"

#include <immintrin.h>

#include <algorithm>

namespace e8qlib {
namespace scan_simd {
namespace {

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
#endif
// Per block the 64-byte LUT line holds two 32-byte tables: the low nibble
// prices against bytes [0,32), the high nibble against [32,64) — each lane
// half (16 bytes) against its own 16-byte quarter. The two 512-bit builds
// below gather the matching quarters of blocks bi and bi+1: Lo from the
// first table, Hi from the second.
inline __m512i PairLut512Lo(const uint8_t* lc, size_t bi) {
    // dst lanes 0/1 draw from src1, lanes 2/3 from src2 (vshufi64x2 field
    // semantics), so 0x44 gathers [bi.l0, bi.l1, bi+1.l0, bi+1.l1].
    const __m512i a = _mm512_loadu_si512(lc + bi * 64);
    const __m512i b = _mm512_loadu_si512(lc + bi * 64 + 64);
    return _mm512_shuffle_i64x2(a, b, 0x44);
}

inline __m512i PairLut512Hi(const uint8_t* lc, size_t bi) {
    // dst lanes 0/1 draw from src1 (fields 2,3 = its l2/l3), lanes 2/3 from
    // src2 the same way: 0xEE gathers [bi.l2, bi.l3, bi+1.l2, bi+1.l3].
    const __m512i a = _mm512_loadu_si512(lc + bi * 64);
    const __m512i b = _mm512_loadu_si512(lc + bi * 64 + 64);
    return _mm512_shuffle_i64x2(a, b, 0xEE);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
#endif
inline void
AddPairBlock(
    __m512i h, __m512i lut_lo, __m512i lut_hi, __m512i* acc_e, __m512i* acc_o,
    const __m512i nib_mask
) {
    const __m512i rl = _mm512_shuffle_epi8(lut_lo, _mm512_and_si512(h, nib_mask));
    const __m512i rh = _mm512_shuffle_epi8(lut_hi, _mm512_and_si512(_mm512_srli_epi16(h, 4), nib_mask));
    *acc_e = _mm512_add_epi16(*acc_e, rl);
    *acc_e = _mm512_add_epi16(*acc_e, rh);
    *acc_o = _mm512_add_epi16(*acc_o, _mm512_srli_epi16(rl, 8));
    *acc_o = _mm512_add_epi16(*acc_o, _mm512_srli_epi16(rh, 8));
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
#endif
inline void
UnpackAcc256Add(__m512i acc_e, __m512i acc_o, int32_t out[32]) {
    // Post-fold, the low 256 bits hold the batch lanes exactly like an AVX2
    // chain (128 lane 0 = lanes 0..15, lane 1 = lanes 16..31); the upper 256
    // bits are the fold's duplicate. Chunks accumulate into `out`.
    // Register-only un-interleave: widen both u16 banks to u32 (lanes 0..7 =
    // e/o of batch lanes 0..15, lanes 8..15 = lanes 16..31) and pick the
    // even/odd interleave with one per-lane permute per half.
    acc_e = _mm512_sub_epi16(acc_e, _mm512_slli_epi16(acc_o, 8));
    const __m512i e32 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(acc_e));
    const __m512i o32 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(acc_o));
    const __m512i lo = _mm512_permutex2var_epi32(
        e32,
        _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23),
        o32
    );
    const __m512i hi = _mm512_permutex2var_epi32(
        e32,
        _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31),
        o32
    );
    _mm512_store_si512(
        reinterpret_cast<__m512i*>(out),
        _mm512_add_epi32(_mm512_load_si512(reinterpret_cast<const __m512i*>(out)), lo)
    );
    _mm512_store_si512(
        reinterpret_cast<__m512i*>(out + 16),
        _mm512_add_epi32(_mm512_load_si512(reinterpret_cast<const __m512i*>(out + 16)), hi)
    );
}

}  // namespace

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"), noinline))
#endif
void
AccumulateCoarseHiAvx512(
    const uint8_t* __restrict__ payload, const uint8_t* __restrict__ lut_c,
    const uint8_t* __restrict__ lut_pair, size_t blocks, int32_t out[32]
) {
    // Chunk bound: 128 x 2 x 255 < 2^16 per u16 lane.
    constexpr size_t kChunkBlocks = 128;
    const __m512i nib_mask = _mm512_set1_epi8(0xF);
    for (int t = 0; t < 32; ++t) {
        out[t] = 0;
    }
    for (size_t b0 = 0; b0 < blocks; b0 += kChunkBlocks) {
        const size_t bn = std::min(kChunkBlocks, blocks - b0);
        __m512i acc_e = _mm512_setzero_si512();
        __m512i acc_o = _mm512_setzero_si512();
        // Odd tail block staged in 256-bit chains (exact: per-lane chunk sums
        // stay below the u16 bound, so splitting the summation is free).
        __m256i tail_e = _mm256_setzero_si256();
        __m256i tail_o = _mm256_setzero_si256();
        size_t bi = b0;
        for (; bi + 2 <= b0 + bn; bi += 2) {
            const __m512i h = _mm512_loadu_si512(payload + bi * 32);
            AddPairBlock(
                h,
                lut_pair ? _mm512_loadu_si512(lut_pair + bi * 64)
                         : PairLut512Lo(lut_c, bi),
                lut_pair ? _mm512_loadu_si512(lut_pair + bi * 64 + 64)
                         : PairLut512Hi(lut_c, bi),
                &acc_e, &acc_o,
                nib_mask
            );
        }
        if (bi < b0 + bn) {
            const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(payload + bi * 32));
            const uint8_t* lc = lut_c + bi * 64;
            const __m256i rl = _mm256_shuffle_epi8(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lc)),
                _mm256_and_si256(h, _mm512_castsi512_si256(nib_mask))
            );
            const __m256i rh = _mm256_shuffle_epi8(
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lc + 32)),
                _mm256_and_si256(_mm256_srli_epi16(h, 4), _mm512_castsi512_si256(nib_mask))
            );
            tail_e = _mm256_add_epi16(tail_e, rl);
            tail_e = _mm256_add_epi16(tail_e, rh);
            tail_o = _mm256_add_epi16(tail_o, _mm256_srli_epi16(rl, 8));
            tail_o = _mm256_add_epi16(tail_o, _mm256_srli_epi16(rh, 8));
        }
        acc_e = _mm512_add_epi16(
            acc_e,
            _mm512_inserti64x4(_mm512_castsi256_si512(tail_e), _mm256_setzero_si256(), 1)
        );
        acc_o = _mm512_add_epi16(
            acc_o,
            _mm512_inserti64x4(_mm512_castsi256_si512(tail_o), _mm256_setzero_si256(), 1)
        );
        // The two blocks of every pair occupy 128-bit lanes 0/1 and 2/3; both
        // must land in the same batch-lane slots, so fold the pair dimension
        // (block 0 + block 2, block 1 + block 3) before the unpack.
        const __m512i swap_e = _mm512_shuffle_i64x2(acc_e, acc_e, 0x4E);
        const __m512i swap_o = _mm512_shuffle_i64x2(acc_o, acc_o, 0x4E);
        acc_e = _mm512_add_epi16(acc_e, swap_e);
        acc_o = _mm512_add_epi16(acc_o, swap_o);
        UnpackAcc256Add(acc_e, acc_o, out);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"), noinline))
#endif
void
AccumulateCoarseHiPairAvx512(
    const uint8_t* __restrict__ payload1, const uint8_t* __restrict__ payload2,
    const uint8_t* __restrict__ lut_c, const uint8_t* __restrict__ lut_pair, size_t blocks,
    int32_t out1[32], int32_t out2[32]
) {
    constexpr size_t kChunkBlocks = 128;
    const __m512i nib_mask = _mm512_set1_epi8(0xF);
    for (int t = 0; t < 32; ++t) {
        out1[t] = 0;
        out2[t] = 0;
    }
    for (size_t b0 = 0; b0 < blocks; b0 += kChunkBlocks) {
        const size_t bn = std::min(kChunkBlocks, blocks - b0);
        __m512i acc_e1 = _mm512_setzero_si512();
        __m512i acc_o1 = _mm512_setzero_si512();
        __m512i acc_e2 = _mm512_setzero_si512();
        __m512i acc_o2 = _mm512_setzero_si512();
        __m256i tail_e1 = _mm256_setzero_si256();
        __m256i tail_o1 = _mm256_setzero_si256();
        __m256i tail_e2 = _mm256_setzero_si256();
        __m256i tail_o2 = _mm256_setzero_si256();
        size_t bi = b0;
        for (; bi + 2 <= b0 + bn; bi += 2) {
            // One shared LUT pair prices both batches' block pairs.
            // lut_pair != nullptr: the pre-permuted layout (plain loads).
            // null: me8q's run-sliced calls where block offsets don't line
            // up with the pair stride — rebuild the registers as before.
            const __m512i lut_lo = lut_pair
                                       ? _mm512_loadu_si512(lut_pair + bi * 64)
                                       : PairLut512Lo(lut_c, bi);
            const __m512i lut_hi = lut_pair
                                       ? _mm512_loadu_si512(lut_pair + bi * 64 + 64)
                                       : PairLut512Hi(lut_c, bi);
            AddPairBlock(
                _mm512_loadu_si512(payload1 + bi * 32), lut_lo, lut_hi, &acc_e1, &acc_o1,
                nib_mask
            );
            AddPairBlock(
                _mm512_loadu_si512(payload2 + bi * 32), lut_lo, lut_hi, &acc_e2, &acc_o2,
                nib_mask
            );
        }
        if (bi < b0 + bn) {
            const __m256i nib256 = _mm512_castsi512_si256(nib_mask);
            const uint8_t* lc = lut_c + bi * 64;
            const __m256i l0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lc));
            const __m256i l1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lc + 32));
            for (int b = 0; b < 2; ++b) {
                const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    (b == 0 ? payload1 : payload2) + bi * 32
                ));
                const __m256i rl = _mm256_shuffle_epi8(l0, _mm256_and_si256(h, nib256));
                const __m256i rh = _mm256_shuffle_epi8(l1, _mm256_and_si256(_mm256_srli_epi16(h, 4), nib256));
                __m256i* te = b == 0 ? &tail_e1 : &tail_e2;
                __m256i* to = b == 0 ? &tail_o1 : &tail_o2;
                *te = _mm256_add_epi16(*te, rl);
                *te = _mm256_add_epi16(*te, rh);
                *to = _mm256_add_epi16(*to, _mm256_srli_epi16(rl, 8));
                *to = _mm256_add_epi16(*to, _mm256_srli_epi16(rh, 8));
            }
        }
        acc_e1 = _mm512_add_epi16(
            acc_e1,
            _mm512_inserti64x4(_mm512_castsi256_si512(tail_e1), _mm256_setzero_si256(), 1)
        );
        acc_o1 = _mm512_add_epi16(
            acc_o1,
            _mm512_inserti64x4(_mm512_castsi256_si512(tail_o1), _mm256_setzero_si256(), 1)
        );
        acc_e2 = _mm512_add_epi16(
            acc_e2,
            _mm512_inserti64x4(_mm512_castsi256_si512(tail_e2), _mm256_setzero_si256(), 1)
        );
        acc_o2 = _mm512_add_epi16(
            acc_o2,
            _mm512_inserti64x4(_mm512_castsi256_si512(tail_o2), _mm256_setzero_si256(), 1)
        );
        // Fold the pair dimension (see the single-batch kernel).
        acc_e1 = _mm512_add_epi16(acc_e1, _mm512_shuffle_i64x2(acc_e1, acc_e1, 0x4E));
        acc_o1 = _mm512_add_epi16(acc_o1, _mm512_shuffle_i64x2(acc_o1, acc_o1, 0x4E));
        acc_e2 = _mm512_add_epi16(acc_e2, _mm512_shuffle_i64x2(acc_e2, acc_e2, 0x4E));
        acc_o2 = _mm512_add_epi16(acc_o2, _mm512_shuffle_i64x2(acc_o2, acc_o2, 0x4E));
        UnpackAcc256Add(acc_e1, acc_o1, out1);
        UnpackAcc256Add(acc_e2, acc_o2, out2);
    }
}


#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq")))
#endif
void
FillPairLutAvx512(const uint8_t* __restrict__ lut_c, size_t blocks, uint8_t* __restrict__ lut_pair) {
    for (size_t bi = 0; bi + 2 <= blocks; bi += 2) {
        const __m512i a = _mm512_loadu_si512(lut_c + bi * 64);
        const __m512i b = _mm512_loadu_si512(lut_c + bi * 64 + 64);
        _mm512_storeu_si512(lut_pair + bi * 64, _mm512_shuffle_i64x2(a, b, 0x44));
        _mm512_storeu_si512(lut_pair + bi * 64 + 64, _mm512_shuffle_i64x2(a, b, 0xEE));
    }
}

}  // namespace scan_simd
}  // namespace e8qlib
