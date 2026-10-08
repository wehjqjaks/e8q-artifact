#include "simd/fastscan_kernels.hpp"

#include <immintrin.h>

#include <algorithm>

namespace e8qlib::fastscan {
void QuantizeLutAvx2(
    uint8_t* dst, const float* src, size_t len, float* min_out, float* delta_out
) {
    // Scalar (min, max) pass so the affine scale matches the generic kernel
    // bit for bit; the per-lane quantize below is exact IEEE (div, add, trunc).
    float lo = src[0];
    float hi = src[0];
    for (size_t k = 0; k < len; ++k) {
        lo = std::min(lo, src[k]);
        hi = std::max(hi, src[k]);
    }
    const float delta = hi > lo ? (hi - lo) / 255.0F : 1.0F;
    const __m256 vlo = _mm256_set1_ps(lo);
    const __m256 vdelta = _mm256_set1_ps(delta);
    const __m256 half = _mm256_set1_ps(0.5F);
    const __m256i zeroi = _mm256_setzero_si256();
    const __m256i c255 = _mm256_set1_epi32(255);
    // packs of two 8-lane groups interleave in 4-byte runs; the shuffle
    // restores [n0 | n1] order for a 16-byte store.
    const __m128i pick16 =
        _mm_setr_epi8(0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15);
    size_t j = 0;
    for (; j + 16 <= len; j += 16) {
        const __m256 t0 = _mm256_div_ps(_mm256_sub_ps(_mm256_loadu_ps(src + j), vlo), vdelta);
        const __m256 t1 =
            _mm256_div_ps(_mm256_sub_ps(_mm256_loadu_ps(src + j + 8), vlo), vdelta);
        __m256i n0 = _mm256_cvttps_epi32(_mm256_add_ps(t0, half));
        __m256i n1 = _mm256_cvttps_epi32(_mm256_add_ps(t1, half));
        n0 = _mm256_min_epi32(_mm256_max_epi32(n0, zeroi), c255);
        n1 = _mm256_min_epi32(_mm256_max_epi32(n1, zeroi), c255);
        const __m256i s16 = _mm256_packs_epi32(n0, n1);
        const __m256i s8 = _mm256_packus_epi16(s16, s16);
        // Bring the upper 128 bits (n0/n1 high halves) down before the pick.
        const __m128i lo128 =
            _mm256_castsi256_si128(_mm256_permute4x64_epi64(s8, 0b01001000));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + j), _mm_shuffle_epi8(lo128, pick16));
    }
    for (; j < len; ++j) {
        const float q = (src[j] - lo) / delta;
        const int v = static_cast<int>(q + 0.5F);
        dst[j] = static_cast<uint8_t>(std::clamp(v, 0, 255));
    }
    *min_out = lo;
    *delta_out = delta;
}

}  // namespace e8qlib::fastscan
