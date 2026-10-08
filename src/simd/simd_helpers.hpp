#pragma once

// SIMD helpers for TUs compiled at the selected ISA (-mavx2 / -mavx512*).
// Never include from generic TUs: E8Q_HAVE_* is a public compile definition
// even when this translation unit has no AVX flags.

#include <cstdint>
#include <immintrin.h>

namespace e8qlib::simd {

inline float HsumPs(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    const __m128 s4 = _mm_add_ps(lo, hi);
    const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    return _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)));
}

inline int32_t HsumEpi32(__m256i v) {
    const __m256i h1 = _mm256_hadd_epi32(v, v);
    const __m256i h2 = _mm256_hadd_epi32(h1, h1);
    return _mm256_extract_epi32(h2, 0) + _mm256_extract_epi32(h2, 4);
}

inline float HminPs(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_min_ps(lo, hi);
    m = _mm_min_ps(m, _mm_movehl_ps(m, m));
    return _mm_cvtss_f32(_mm_min_ss(m, _mm_shuffle_ps(m, m, 1)));
}

inline int32_t HmaxEpi32(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i m = _mm_max_epi32(lo, hi);
    m = _mm_max_epi32(m, _mm_shuffle_epi32(m, 0x4E));
    m = _mm_max_epi32(m, _mm_shuffle_epi32(m, 0xB1));
    return _mm_cvtsi128_si32(m);
}

inline int32_t HminEpi32(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i m = _mm_min_epi32(lo, hi);
    m = _mm_min_epi32(m, _mm_shuffle_epi32(m, 0x4E));
    m = _mm_min_epi32(m, _mm_shuffle_epi32(m, 0xB1));
    return _mm_cvtsi128_si32(m);
}

#if defined(E8Q_HAVE_AVX512)
// 16-entry-at-a-time level lookup from a stride-256 table pair, 8 useful
// lanes. Defined here, not in a kernel TU: at b >= 6 the refine core calls it
// once per (row, block) — a cross-TU call costs the call itself, the ABI
// spills, and a reload of every one of the 8..16 table registers, which
// measured ~4x the whole b = 8 refine time.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi"), always_inline))
#endif
inline __m512
RefineLut512Full(const float* tab, __m512i d, int entries) {
    const __m512i lo5 = _mm512_and_si512(d, _mm512_set1_epi32(31));
    const __mmask16 m5 = _mm512_movepi32_mask(_mm512_slli_epi32(d, 26));
    const __mmask16 m6 = _mm512_movepi32_mask(_mm512_slli_epi32(d, 25));
    const __mmask16 m7 = _mm512_movepi32_mask(_mm512_slli_epi32(d, 24));
    const __m512 c0 =
        _mm512_permutex2var_ps(_mm512_loadu_ps(tab), lo5, _mm512_loadu_ps(tab + 16));
    const __m512 c1 =
        _mm512_permutex2var_ps(_mm512_loadu_ps(tab + 32), lo5, _mm512_loadu_ps(tab + 48));
    const __m512 p01 = _mm512_mask_blend_ps(m5, c0, c1);
    if (entries <= 64) {
        return p01;
    }
    const __m512 c2 =
        _mm512_permutex2var_ps(_mm512_loadu_ps(tab + 64), lo5, _mm512_loadu_ps(tab + 80));
    const __m512 c3 =
        _mm512_permutex2var_ps(_mm512_loadu_ps(tab + 96), lo5, _mm512_loadu_ps(tab + 112));
    const __m512 p23 = _mm512_mask_blend_ps(m5, c2, c3);
    const __m512 q03 = _mm512_mask_blend_ps(m6, p01, p23);
    if (entries <= 128) {
        return q03;
    }
    const __m512 c4 =
        _mm512_permutex2var_ps(_mm512_loadu_ps(tab + 128), lo5, _mm512_loadu_ps(tab + 144));
    const __m512 c5 =
        _mm512_permutex2var_ps(_mm512_loadu_ps(tab + 160), lo5, _mm512_loadu_ps(tab + 176));
    const __m512 p45 = _mm512_mask_blend_ps(m5, c4, c5);
    const __m512 c6 =
        _mm512_permutex2var_ps(_mm512_loadu_ps(tab + 192), lo5, _mm512_loadu_ps(tab + 208));
    const __m512 c7 =
        _mm512_permutex2var_ps(_mm512_loadu_ps(tab + 224), lo5, _mm512_loadu_ps(tab + 240));
    const __m512 p67 = _mm512_mask_blend_ps(m5, c6, c7);
    const __m512 q47 = _mm512_mask_blend_ps(m6, p45, p67);
    return _mm512_mask_blend_ps(m7, q03, q47);
}

// 8-lane twin: lookup through the single branch-0 table (callers that
// canonicalize branch-1 digits with the reflection L1[d] == -L0[q-1-d]).
inline __m256
RefineLut512(const float* tab, __m256i dig, int entries) {
    return _mm512_castps512_ps256(
        RefineLut512Full(tab, _mm512_castsi256_si512(dig), entries)
    );
}
#endif

}  // namespace e8qlib::simd

namespace e8qlib::detail {

struct AccYmm {
    __m256i evA;
    __m256i evB;
    __m256i odA;
    __m256i odB;
};

inline void UnpackAcc(const AccYmm& a, int32_t out[32]) {
    alignas(32) uint32_t e0[8];
    alignas(32) uint32_t e1[8];
    alignas(32) uint32_t o0[8];
    alignas(32) uint32_t o1[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(e0), a.evA);
    _mm256_store_si256(reinterpret_cast<__m256i*>(e1), a.evB);
    _mm256_store_si256(reinterpret_cast<__m256i*>(o0), a.odA);
    _mm256_store_si256(reinterpret_cast<__m256i*>(o1), a.odB);
    for (int k = 0; k < 8; ++k) {
        out[2 * k] = static_cast<int32_t>(e0[k]);
        out[2 * k + 1] = static_cast<int32_t>(o0[k]);
        out[16 + 2 * k] = static_cast<int32_t>(e1[k]);
        out[16 + 2 * k + 1] = static_cast<int32_t>(o1[k]);
    }
}

}  // namespace e8qlib::detail
