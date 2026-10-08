#include "simd/kernels.hpp"
#include "simd/simd_helpers.hpp"

#include <immintrin.h>

namespace e8qlib::simd {

float QueryNorm2Avx2(const float* q, size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    size_t d = 0;
    for (; d + 8 <= dim; d += 8) {
        const __m256 v = _mm256_loadu_ps(q + d);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    float s = HsumPs(acc);
    for (; d < dim; ++d) {
        s += q[d] * q[d];
    }
    return s;
}

void RouteGemvAvx2(
    const float* q, const float* centroids, const float* cn2, size_t nlist, size_t dim,
    float qn2, float* dist, uint64_t* keys, size_t base, size_t stride
) {
    const size_t pitch = stride != 0 ? stride : dim;
    // Centroid pairs with independent accumulator chains: the per-centroid
    // horizontal sum (7 dependent shuffles) no longer stalls the FMA stream
    // between centroids. Each centroid's dot keeps the exact reduction tree
    // of the single-centroid form, so every distance is bit-identical.
    size_t c = 0;
    for (; c + 2 <= nlist; c += 2) {
        const float* cc0 = centroids + c * pitch;
        const float* cc1 = cc0 + pitch;
        _mm_prefetch(reinterpret_cast<const char*>(cc0 + 2 * pitch), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(cc1 + 2 * pitch), _MM_HINT_T0);
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps();
        __m256 b1 = _mm256_setzero_ps();
        __m256 b2 = _mm256_setzero_ps();
        __m256 b3 = _mm256_setzero_ps();
        size_t d = 0;
        for (; d + 32 <= dim; d += 32) {
            const __m256 q0 = _mm256_loadu_ps(q + d);
            const __m256 q1 = _mm256_loadu_ps(q + d + 8);
            const __m256 q2 = _mm256_loadu_ps(q + d + 16);
            const __m256 q3 = _mm256_loadu_ps(q + d + 24);
            a0 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(cc0 + d), a0);
            a1 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(cc0 + d + 8), a1);
            a2 = _mm256_fmadd_ps(q2, _mm256_loadu_ps(cc0 + d + 16), a2);
            a3 = _mm256_fmadd_ps(q3, _mm256_loadu_ps(cc0 + d + 24), a3);
            b0 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(cc1 + d), b0);
            b1 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(cc1 + d + 8), b1);
            b2 = _mm256_fmadd_ps(q2, _mm256_loadu_ps(cc1 + d + 16), b2);
            b3 = _mm256_fmadd_ps(q3, _mm256_loadu_ps(cc1 + d + 24), b3);
        }
        float dot0 = HsumPs(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        float dot1 = HsumPs(_mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3)));
        for (; d + 8 <= dim; d += 8) {
            dot0 += HsumPs(
                _mm256_mul_ps(_mm256_loadu_ps(q + d), _mm256_loadu_ps(cc0 + d))
            );
            dot1 += HsumPs(
                _mm256_mul_ps(_mm256_loadu_ps(q + d), _mm256_loadu_ps(cc1 + d))
            );
        }
        for (; d < dim; ++d) {
            dot0 += q[d] * cc0[d];
            dot1 += q[d] * cc1[d];
        }
        // See RouteGemvGeneric: cn2, dist and keys are indexed by the global
        // cluster id `base + c`, not the block-local `c`.
        PackRouteKey(base + c, qn2 + cn2[base + c] - 2.0F * dot0, dist, keys);
        PackRouteKey(base + c + 1, qn2 + cn2[base + c + 1] - 2.0F * dot1, dist, keys);
    }
    for (; c < nlist; ++c) {
        const float* cc = centroids + c * pitch;
        _mm_prefetch(reinterpret_cast<const char*>(cc + 2 * dim), _MM_HINT_T0);
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();
        size_t d = 0;
        for (; d + 32 <= dim; d += 32) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d), _mm256_loadu_ps(cc + d), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 8), _mm256_loadu_ps(cc + d + 8), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 16), _mm256_loadu_ps(cc + d + 16), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(q + d + 24), _mm256_loadu_ps(cc + d + 24), a3);
        }
        float dot = HsumPs(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        for (; d + 8 <= dim; d += 8) {
            const __m256 p = _mm256_mul_ps(_mm256_loadu_ps(q + d), _mm256_loadu_ps(cc + d));
            dot += HsumPs(p);
        }
        for (; d < dim; ++d) {
            dot += q[d] * cc[d];
        }
        PackRouteKey(base + c, qn2 + cn2[base + c] - 2.0F * dot, dist, keys);
    }
}

void RouteL2DirectAvx2(
    const float* q, const float* centroids, size_t nlist, size_t dim, float* dist, uint64_t* keys
) {
    for (size_t c = 0; c < nlist; ++c) {
        const float* cen = centroids + c * dim;
        __m256 racc = _mm256_setzero_ps();
        __m256 racc1 = _mm256_setzero_ps();
        __m256 racc2 = _mm256_setzero_ps();
        __m256 racc3 = _mm256_setzero_ps();
        size_t j = 0;
        for (; j + 32 <= dim; j += 32) {
            const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(q + j), _mm256_loadu_ps(cen + j));
            const __m256 d1 =
                _mm256_sub_ps(_mm256_loadu_ps(q + j + 8), _mm256_loadu_ps(cen + j + 8));
            const __m256 d2 =
                _mm256_sub_ps(_mm256_loadu_ps(q + j + 16), _mm256_loadu_ps(cen + j + 16));
            const __m256 d3 =
                _mm256_sub_ps(_mm256_loadu_ps(q + j + 24), _mm256_loadu_ps(cen + j + 24));
            racc = _mm256_fmadd_ps(d0, d0, racc);
            racc1 = _mm256_fmadd_ps(d1, d1, racc1);
            racc2 = _mm256_fmadd_ps(d2, d2, racc2);
            racc3 = _mm256_fmadd_ps(d3, d3, racc3);
        }
        racc = _mm256_add_ps(_mm256_add_ps(racc, racc1), _mm256_add_ps(racc2, racc3));
        float sum = HsumPs(racc);
        for (; j + 8 <= dim; j += 8) {
            const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(q + j), _mm256_loadu_ps(cen + j));
            sum += HsumPs(_mm256_mul_ps(d, d));
        }
        for (; j < dim; ++j) {
            const float d = q[j] - cen[j];
            sum += d * d;
        }
        PackRouteKey(c, sum, dist, keys);
    }
}

void FlipSignsAvx2(const uint8_t* signs, float* data, size_t len) {
    size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        uint32_t mask_bits;
        std::memcpy(&mask_bits, &signs[i / 8], sizeof(mask_bits));
        for (int b = 0; b < 4; ++b) {
            const uint8_t m = static_cast<uint8_t>((mask_bits >> (b * 8)) & 0xFF);
            __m256i bits = _mm256_set1_epi32(m);
            const __m256i sel = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
            __m256i test = _mm256_and_si256(bits, sel);
            __m256i cmp = _mm256_cmpeq_epi32(test, sel);
            __m256 flips = _mm256_and_ps(
                _mm256_castsi256_ps(cmp),
                _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(0x80000000)))
            );
            __m256 v = _mm256_loadu_ps(&data[i + static_cast<size_t>(b) * 8]);
            _mm256_storeu_ps(&data[i + static_cast<size_t>(b) * 8], _mm256_xor_ps(v, flips));
        }
    }
    for (; i < len; ++i) {
        if ((signs[i / 8] >> (i % 8)) & 1) {
            data[i] = -data[i];
        }
    }
}

void KacsWalkAvx2(float* data, size_t len) {
    const size_t half = len / 2;
    size_t i = 0;
    for (; i + 8 <= half; i += 8) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        __m256 y = _mm256_loadu_ps(&data[i + half]);
        _mm256_storeu_ps(&data[i], _mm256_add_ps(x, y));
        _mm256_storeu_ps(&data[i + half], _mm256_sub_ps(x, y));
    }
    for (; i < half; ++i) {
        const float x = data[i];
        const float y = data[i + half];
        data[i] = x + y;
        data[i + half] = x - y;
    }
}

void FhtFloatAvx2(float* buf, size_t len) {
    for (size_t h = 1; h < len; h <<= 1) {
        if (h >= 8) {
            for (size_t i = 0; i < len; i += h << 1) {
                float* lo = buf + i;
                float* hi = buf + i + h;
                size_t j = 0;
                for (; j + 8 <= h; j += 8) {
                    __m256 x = _mm256_loadu_ps(lo + j);
                    __m256 y = _mm256_loadu_ps(hi + j);
                    _mm256_storeu_ps(lo + j, _mm256_add_ps(x, y));
                    _mm256_storeu_ps(hi + j, _mm256_sub_ps(x, y));
                }
                for (; j < h; ++j) {
                    const float x = lo[j];
                    const float y = hi[j];
                    lo[j] = x + y;
                    hi[j] = x - y;
                }
            }
        } else {
            for (size_t i = 0; i < len; i += h << 1) {
                for (size_t j = 0; j < h; ++j) {
                    const float x = buf[i + j];
                    const float y = buf[i + j + h];
                    buf[i + j] = x + y;
                    buf[i + j + h] = x - y;
                }
            }
        }
    }
}

}  // namespace e8qlib::simd
