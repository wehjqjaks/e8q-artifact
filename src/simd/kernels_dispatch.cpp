#include "simd/kernels.hpp"

#include "e8qlib/simd/dispatch.hpp"

namespace e8qlib::simd {

float QueryNorm2(const float* q, size_t dim) {
#if defined(E8Q_HAVE_AVX512)
    if (Avx512Enabled()) {
        return QueryNorm2Avx512(q, dim);
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (Avx2Enabled()) {
        return QueryNorm2Avx2(q, dim);
    }
#endif
    return QueryNorm2Generic(q, dim);
}

void RouteGemv(
    const float* q, const float* centroids, const float* cn2, size_t nlist, size_t dim,
    float qn2, float* dist, uint64_t* keys, size_t base, size_t stride
) {
#if defined(E8Q_HAVE_AVX512)
    if (Avx512Enabled()) {
        RouteGemv512(q, centroids, cn2, nlist, dim, qn2, dist, keys, base, stride);
        return;
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (Avx2Enabled()) {
        RouteGemvAvx2(q, centroids, cn2, nlist, dim, qn2, dist, keys, base, stride);
        return;
    }
#endif
    RouteGemvGeneric(q, centroids, cn2, nlist, dim, qn2, dist, keys, base, stride);
}

void RouteL2Direct(
    const float* q, const float* centroids, size_t nlist, size_t dim, float* dist, uint64_t* keys
) {
#if defined(E8Q_HAVE_AVX512)
    if (Avx512Enabled()) {
        RouteL2Direct512(q, centroids, nlist, dim, dist, keys);
        return;
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (Avx2Enabled()) {
        RouteL2DirectAvx2(q, centroids, nlist, dim, dist, keys);
        return;
    }
#endif
    RouteL2DirectGeneric(q, centroids, nlist, dim, dist, keys);
}

void FhtFloat(float* buf, size_t len) {
#if defined(E8Q_HAVE_AVX512)
    if (Avx512Enabled()) {
        FhtFloatAvx512(buf, len);
        return;
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (Avx2Enabled()) {
        FhtFloatAvx2(buf, len);
        return;
    }
#endif
    FhtFloatGeneric(buf, len);
}

void FlipSigns(const uint8_t* signs, float* data, size_t len) {
#if defined(E8Q_HAVE_AVX512)
    if (Avx512Enabled()) {
        FlipSignsAvx512(signs, data, len);
        return;
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (Avx2Enabled()) {
        FlipSignsAvx2(signs, data, len);
        return;
    }
#endif
    FlipSignsGeneric(signs, data, len);
}

void KacsWalk(float* data, size_t len) {
#if defined(E8Q_HAVE_AVX512)
    if (Avx512Enabled()) {
        KacsWalkAvx512(data, len);
        return;
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (Avx2Enabled()) {
        KacsWalkAvx2(data, len);
        return;
    }
#endif
    KacsWalkGeneric(data, len);
}

}  // namespace e8qlib::simd
