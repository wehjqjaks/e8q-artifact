#include "simd/fastscan_kernels.hpp"

#include <immintrin.h>

#include <algorithm>

namespace e8qlib::fastscan {
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi"), noinline))
#endif
void
QuantizeLutAvx512(
    uint8_t* dst, const float* src, size_t len, float* min_out, float* delta_out
) {
    float lo = src[0];
    float hi = src[0];
    for (size_t k = 0; k < len; ++k) {
        lo = std::min(lo, src[k]);
        hi = std::max(hi, src[k]);
    }
    const float delta = hi > lo ? (hi - lo) / 255.0F : 1.0F;
    const __m512 vlo = _mm512_set1_ps(lo);
    const __m512 vdelta = _mm512_set1_ps(delta);
    const __m512 half = _mm512_set1_ps(0.5F);
    size_t j = 0;
    for (; j + 16 <= len; j += 16) {
        const __m512 t = _mm512_div_ps(_mm512_sub_ps(_mm512_loadu_ps(src + j), vlo), vdelta);
        const __m512i n = _mm512_cvttps_epi32(_mm512_add_ps(t, half));
        // Unsigned saturate covers the [0, 255] clamp in one instruction.
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + j), _mm512_cvtusepi32_epi8(n));
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
