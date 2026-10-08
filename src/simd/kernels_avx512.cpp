#include "simd/kernels.hpp"

#include <cstring>
#include <immintrin.h>

namespace e8qlib::simd {

namespace {

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi")))
#endif
inline float Hsum512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi")))
#endif
inline __m512 MaskzLoad(const float* p, size_t n) {
    const __mmask16 k = static_cast<__mmask16>((1U << n) - 1U);
    return _mm512_maskz_loadu_ps(k, p);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi")))
#endif
inline void StoreRoute512(
    size_t ci, __m512 x, __m512 y, float qn2, const float* cn2, float* dist, uint64_t* keys
) {
    PackRouteKey(ci, qn2 + cn2[ci] - 2.0F * Hsum512(_mm512_add_ps(x, y)), dist, keys);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi")))
#endif
inline void StoreL2Route512(size_t ci, __m512 x, __m512 y, float* dist, uint64_t* keys) {
    PackRouteKey(ci, Hsum512(_mm512_add_ps(x, y)), dist, keys);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi")))
#endif
inline void GemvTail(
    const float* q, const float* cc, size_t d, size_t dim, __m512* a, __m512* b
) {
    if (d + 16 <= dim) {
        *a = _mm512_fmadd_ps(_mm512_loadu_ps(q + d), _mm512_loadu_ps(cc + d), *a);
        d += 16;
    }
    if (d < dim) {
        const __m512 qv = MaskzLoad(q + d, dim - d);
        const __m512 cv = MaskzLoad(cc + d, dim - d);
        *b = _mm512_fmadd_ps(qv, cv, *b);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi")))
#endif
inline void L2Tail(
    const float* q, const float* cc, size_t d, size_t dim, __m512* a, __m512* b
) {
    if (d + 16 <= dim) {
        const __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(q + d), _mm512_loadu_ps(cc + d));
        *a = _mm512_fmadd_ps(diff, diff, *a);
        d += 16;
    }
    if (d < dim) {
        const __m512 diff = _mm512_sub_ps(MaskzLoad(q + d, dim - d), MaskzLoad(cc + d, dim - d));
        *b = _mm512_fmadd_ps(diff, diff, *b);
    }
}

}  // namespace


#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi"), noinline))
#endif
float
QueryNorm2Avx512(const float* q, size_t dim) {
    __m512 a = _mm512_setzero_ps();
    __m512 b = _mm512_setzero_ps();
    size_t d = 0;
    for (; d + 32 <= dim; d += 32) {
        const __m512 v0 = _mm512_loadu_ps(q + d);
        const __m512 v1 = _mm512_loadu_ps(q + d + 16);
        a = _mm512_fmadd_ps(v0, v0, a);
        b = _mm512_fmadd_ps(v1, v1, b);
    }
    if (d + 16 <= dim) {
        const __m512 v0 = _mm512_loadu_ps(q + d);
        a = _mm512_fmadd_ps(v0, v0, a);
        d += 16;
    }
    if (d < dim) {
        const __m512 v = MaskzLoad(q + d, dim - d);
        b = _mm512_fmadd_ps(v, v, b);
    }
    return Hsum512(_mm512_add_ps(a, b));
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi"), noinline))
#endif
void
RouteGemv512(
    const float* __restrict__ rot_q,
    const float* __restrict__ centroids,
    const float* __restrict__ cn2,
    size_t kc,
    size_t pd,
    float qn2,
    float* __restrict__ dist,
    uint64_t* __restrict__ keys,
    size_t base,
    size_t stride
) {
    const size_t pitch = stride != 0 ? stride : pd;
    size_t c = 0;
    for (; c + 4 <= kc; c += 4) {
        const float* cc0 = centroids + c * pitch;
        const float* cc1 = cc0 + pitch;
        const float* cc2 = cc1 + pitch;
        const float* cc3 = cc2 + pitch;
        _mm_prefetch(reinterpret_cast<const char*>(cc3 + pitch), _MM_HINT_T0);
        __m512 a0 = _mm512_setzero_ps();
        __m512 b0 = _mm512_setzero_ps();
        __m512 a1 = _mm512_setzero_ps();
        __m512 b1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps();
        __m512 b2 = _mm512_setzero_ps();
        __m512 a3 = _mm512_setzero_ps();
        __m512 b3 = _mm512_setzero_ps();
        size_t d = 0;
        for (; d + 32 <= pd; d += 32) {
            const __m512 ql = _mm512_loadu_ps(rot_q + d);
            const __m512 qh = _mm512_loadu_ps(rot_q + d + 16);
            a0 = _mm512_fmadd_ps(ql, _mm512_loadu_ps(cc0 + d), a0);
            b0 = _mm512_fmadd_ps(qh, _mm512_loadu_ps(cc0 + d + 16), b0);
            a1 = _mm512_fmadd_ps(ql, _mm512_loadu_ps(cc1 + d), a1);
            b1 = _mm512_fmadd_ps(qh, _mm512_loadu_ps(cc1 + d + 16), b1);
            a2 = _mm512_fmadd_ps(ql, _mm512_loadu_ps(cc2 + d), a2);
            b2 = _mm512_fmadd_ps(qh, _mm512_loadu_ps(cc2 + d + 16), b2);
            a3 = _mm512_fmadd_ps(ql, _mm512_loadu_ps(cc3 + d), a3);
            b3 = _mm512_fmadd_ps(qh, _mm512_loadu_ps(cc3 + d + 16), b3);
        }
        GemvTail(rot_q, cc0, d, pd, &a0, &b0);
        GemvTail(rot_q, cc1, d, pd, &a1, &b1);
        GemvTail(rot_q, cc2, d, pd, &a2, &b2);
        GemvTail(rot_q, cc3, d, pd, &a3, &b3);
        StoreRoute512(base + c, a0, b0, qn2, cn2, dist, keys);
        StoreRoute512(base + c + 1, a1, b1, qn2, cn2, dist, keys);
        StoreRoute512(base + c + 2, a2, b2, qn2, cn2, dist, keys);
        StoreRoute512(base + c + 3, a3, b3, qn2, cn2, dist, keys);
    }
    for (; c < kc; ++c) {
        const float* cc = centroids + c * pitch;
        __m512 a = _mm512_setzero_ps();
        __m512 b = _mm512_setzero_ps();
        size_t d = 0;
        for (; d + 32 <= pd; d += 32) {
            a = _mm512_fmadd_ps(_mm512_loadu_ps(rot_q + d), _mm512_loadu_ps(cc + d), a);
            b = _mm512_fmadd_ps(
                _mm512_loadu_ps(rot_q + d + 16), _mm512_loadu_ps(cc + d + 16), b
            );
        }
        GemvTail(rot_q, cc, d, pd, &a, &b);
        StoreRoute512(base + c, a, b, qn2, cn2, dist, keys);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi"), noinline))
#endif
void
RouteL2Direct512(
    const float* q, const float* centroids, size_t nlist, size_t dim, float* dist, uint64_t* keys
) {
    size_t c = 0;
    for (; c + 4 <= nlist; c += 4) {
        const float* cc0 = centroids + c * dim;
        const float* cc1 = cc0 + dim;
        const float* cc2 = cc1 + dim;
        const float* cc3 = cc2 + dim;
        _mm_prefetch(reinterpret_cast<const char*>(cc3 + dim), _MM_HINT_T0);
        __m512 a0 = _mm512_setzero_ps();
        __m512 b0 = _mm512_setzero_ps();
        __m512 a1 = _mm512_setzero_ps();
        __m512 b1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps();
        __m512 b2 = _mm512_setzero_ps();
        __m512 a3 = _mm512_setzero_ps();
        __m512 b3 = _mm512_setzero_ps();
        size_t d = 0;
        for (; d + 32 <= dim; d += 32) {
            const __m512 ql = _mm512_loadu_ps(q + d);
            const __m512 qh = _mm512_loadu_ps(q + d + 16);
            const __m512 d0l = _mm512_sub_ps(ql, _mm512_loadu_ps(cc0 + d));
            const __m512 d0h = _mm512_sub_ps(qh, _mm512_loadu_ps(cc0 + d + 16));
            const __m512 d1l = _mm512_sub_ps(ql, _mm512_loadu_ps(cc1 + d));
            const __m512 d1h = _mm512_sub_ps(qh, _mm512_loadu_ps(cc1 + d + 16));
            const __m512 d2l = _mm512_sub_ps(ql, _mm512_loadu_ps(cc2 + d));
            const __m512 d2h = _mm512_sub_ps(qh, _mm512_loadu_ps(cc2 + d + 16));
            const __m512 d3l = _mm512_sub_ps(ql, _mm512_loadu_ps(cc3 + d));
            const __m512 d3h = _mm512_sub_ps(qh, _mm512_loadu_ps(cc3 + d + 16));
            a0 = _mm512_fmadd_ps(d0l, d0l, a0);
            b0 = _mm512_fmadd_ps(d0h, d0h, b0);
            a1 = _mm512_fmadd_ps(d1l, d1l, a1);
            b1 = _mm512_fmadd_ps(d1h, d1h, b1);
            a2 = _mm512_fmadd_ps(d2l, d2l, a2);
            b2 = _mm512_fmadd_ps(d2h, d2h, b2);
            a3 = _mm512_fmadd_ps(d3l, d3l, a3);
            b3 = _mm512_fmadd_ps(d3h, d3h, b3);
        }
        L2Tail(q, cc0, d, dim, &a0, &b0);
        L2Tail(q, cc1, d, dim, &a1, &b1);
        L2Tail(q, cc2, d, dim, &a2, &b2);
        L2Tail(q, cc3, d, dim, &a3, &b3);
        StoreL2Route512(c, a0, b0, dist, keys);
        StoreL2Route512(c + 1, a1, b1, dist, keys);
        StoreL2Route512(c + 2, a2, b2, dist, keys);
        StoreL2Route512(c + 3, a3, b3, dist, keys);
    }
    for (; c < nlist; ++c) {
        const float* cc = centroids + c * dim;
        __m512 a = _mm512_setzero_ps();
        __m512 b = _mm512_setzero_ps();
        size_t d = 0;
        for (; d + 32 <= dim; d += 32) {
            const __m512 dl = _mm512_sub_ps(_mm512_loadu_ps(q + d), _mm512_loadu_ps(cc + d));
            const __m512 dh =
                _mm512_sub_ps(_mm512_loadu_ps(q + d + 16), _mm512_loadu_ps(cc + d + 16));
            a = _mm512_fmadd_ps(dl, dl, a);
            b = _mm512_fmadd_ps(dh, dh, b);
        }
        L2Tail(q, cc, d, dim, &a, &b);
        StoreL2Route512(c, a, b, dist, keys);
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi")))
#endif
void
FlipSignsAvx512(const uint8_t* signs, float* data, size_t len) {
    const __m512i sb = _mm512_set1_epi32(static_cast<int>(0x80000000U));
    size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        uint32_t bits = 0;
        std::memcpy(&bits, &signs[i / 8], sizeof(bits));
        const __mmask16 k0 = static_cast<__mmask16>(bits);
        const __mmask16 k1 = static_cast<__mmask16>(bits >> 16);
        __m512i v0 = _mm512_castps_si512(_mm512_loadu_ps(data + i));
        __m512i v1 = _mm512_castps_si512(_mm512_loadu_ps(data + i + 16));
        v0 = _mm512_mask_xor_epi32(v0, k0, v0, sb);
        v1 = _mm512_mask_xor_epi32(v1, k1, v1, sb);
        _mm512_storeu_ps(data + i, _mm512_castsi512_ps(v0));
        _mm512_storeu_ps(data + i + 16, _mm512_castsi512_ps(v1));
    }
    if (i + 16 <= len) {
        uint16_t bits = 0;
        std::memcpy(&bits, &signs[i / 8], sizeof(bits));
        const __mmask16 k = static_cast<__mmask16>(bits);
        __m512i v = _mm512_castps_si512(_mm512_loadu_ps(data + i));
        v = _mm512_mask_xor_epi32(v, k, v, sb);
        _mm512_storeu_ps(data + i, _mm512_castsi512_ps(v));
        i += 16;
    }
    for (; i < len; ++i) {
        if ((signs[i / 8] >> (i % 8)) & 1) {
            data[i] = -data[i];
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi"), noinline))
#endif
void
KacsWalkAvx512(float* data, size_t len) {
    const size_t half = len / 2;
    size_t i = 0;
    for (; i + 32 <= half; i += 32) {
        const __m512 x0 = _mm512_loadu_ps(data + i);
        const __m512 y0 = _mm512_loadu_ps(data + i + half);
        const __m512 x1 = _mm512_loadu_ps(data + i + 16);
        const __m512 y1 = _mm512_loadu_ps(data + i + half + 16);
        _mm512_storeu_ps(data + i, _mm512_add_ps(x0, y0));
        _mm512_storeu_ps(data + i + half, _mm512_sub_ps(x0, y0));
        _mm512_storeu_ps(data + i + 16, _mm512_add_ps(x1, y1));
        _mm512_storeu_ps(data + i + half + 16, _mm512_sub_ps(x1, y1));
    }
    for (; i + 16 <= half; i += 16) {
        const __m512 x = _mm512_loadu_ps(data + i);
        const __m512 y = _mm512_loadu_ps(data + i + half);
        _mm512_storeu_ps(data + i, _mm512_add_ps(x, y));
        _mm512_storeu_ps(data + i + half, _mm512_sub_ps(x, y));
    }
    for (; i < half; ++i) {
        const float x = data[i];
        const float y = data[i + half];
        data[i] = x + y;
        data[i + half] = x - y;
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi"), noinline))
#endif
void
FhtFloatAvx512(float* buf, size_t len) {
    for (size_t h = 1; h < len; h <<= 1) {
        if (h >= 16) {
            for (size_t i = 0; i < len; i += h << 1) {
                float* lo = buf + i;
                float* hi = buf + i + h;
                size_t j = 0;
                for (; j + 32 <= h; j += 32) {
                    const __m512 x0 = _mm512_loadu_ps(lo + j);
                    const __m512 y0 = _mm512_loadu_ps(hi + j);
                    const __m512 x1 = _mm512_loadu_ps(lo + j + 16);
                    const __m512 y1 = _mm512_loadu_ps(hi + j + 16);
                    _mm512_storeu_ps(lo + j, _mm512_add_ps(x0, y0));
                    _mm512_storeu_ps(hi + j, _mm512_sub_ps(x0, y0));
                    _mm512_storeu_ps(lo + j + 16, _mm512_add_ps(x1, y1));
                    _mm512_storeu_ps(hi + j + 16, _mm512_sub_ps(x1, y1));
                }
                for (; j + 16 <= h; j += 16) {
                    const __m512 x = _mm512_loadu_ps(lo + j);
                    const __m512 y = _mm512_loadu_ps(hi + j);
                    _mm512_storeu_ps(lo + j, _mm512_add_ps(x, y));
                    _mm512_storeu_ps(hi + j, _mm512_sub_ps(x, y));
                }
                for (; j < h; ++j) {
                    const float x = lo[j];
                    const float y = hi[j];
                    lo[j] = x + y;
                    hi[j] = x - y;
                }
            }
        } else if (h >= 8) {
            for (size_t i = 0; i < len; i += h << 1) {
                float* lo = buf + i;
                float* hi = buf + i + h;
                const __m256 x = _mm256_loadu_ps(lo);
                const __m256 y = _mm256_loadu_ps(hi);
                _mm256_storeu_ps(lo, _mm256_add_ps(x, y));
                _mm256_storeu_ps(hi, _mm256_sub_ps(x, y));
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
