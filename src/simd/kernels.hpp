#pragma once

// Private SIMD kernels. Not installed. Included by IVF search TUs and the
// ISA-specific translation units. No SIMD types: generic TUs include this
// header while E8Q_HAVE_* is a public compile definition.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace e8qlib::simd {

inline void PackRouteKey(size_t ci, float d2, float* dist, uint64_t* keys) {
    if (d2 < 0.0F) {
        d2 = 0.0F;
    }
    dist[ci] = d2;
    uint32_t db = 0;
    std::memcpy(&db, &d2, 4);
    keys[ci] = (static_cast<uint64_t>(db) << 32) | ci;
}

float QueryNorm2(const float* q, size_t dim);
void RouteGemv(
    const float* q, const float* centroids, const float* cn2, size_t nlist, size_t dim,
    float qn2, float* dist, uint64_t* keys, size_t base = 0, size_t stride = 0
);
void RouteL2Direct(
    const float* q, const float* centroids, size_t nlist, size_t dim, float* dist, uint64_t* keys
);
void FhtFloat(float* buf, size_t len);
void FlipSigns(const uint8_t* signs, float* data, size_t len);
void KacsWalk(float* data, size_t len);

float QueryNorm2Generic(const float* q, size_t dim);
// `base` is the cluster index of centroids[0]: keys and dist stay indexed by
// the caller's global cluster id, so a range sweep writes the same key array
// a whole-matrix sweep would (the batched-route block loop).
void RouteGemvGeneric(
    const float* q, const float* centroids, const float* cn2, size_t nlist, size_t dim,
    float qn2, float* dist, uint64_t* keys, size_t base = 0, size_t stride = 0
);
void RouteL2DirectGeneric(
    const float* q, const float* centroids, size_t nlist, size_t dim, float* dist, uint64_t* keys
);
void FhtFloatGeneric(float* buf, size_t len);
void FlipSignsGeneric(const uint8_t* signs, float* data, size_t len);
void KacsWalkGeneric(float* data, size_t len);

#if defined(E8Q_HAVE_AVX2)
float QueryNorm2Avx2(const float* q, size_t dim);
void RouteGemvAvx2(
    const float* q, const float* centroids, const float* cn2, size_t nlist, size_t dim,
    float qn2, float* dist, uint64_t* keys, size_t base = 0, size_t stride = 0
);
void RouteL2DirectAvx2(
    const float* q, const float* centroids, size_t nlist, size_t dim, float* dist, uint64_t* keys
);
void FhtFloatAvx2(float* buf, size_t len);
void FlipSignsAvx2(const uint8_t* signs, float* data, size_t len);
void KacsWalkAvx2(float* data, size_t len);
#endif

#if defined(E8Q_HAVE_AVX512)
float QueryNorm2Avx512(const float* q, size_t dim);
// `stride` is the centroid row pitch when it differs from the dot length
// (`pd`); 0 means the rows are contiguous at `pd`. ME8Q routes over its coded
// dimensions only: the plan's trailing zero-bit dims are exact zeros in both
// the query and the rotated centroids, so dropping them from the dot changes
// nothing while halving the sweep on padded datasets (deep1M: 128 of 256).
void RouteGemv512(
    const float* __restrict__ rot_q,
    const float* __restrict__ centroids,
    const float* __restrict__ cn2,
    size_t kc,
    size_t pd,
    float qn2,
    float* __restrict__ dist,
    uint64_t* __restrict__ keys,
    size_t base = 0,
    size_t stride = 0
);
void RouteL2Direct512(
    const float* q, const float* centroids, size_t nlist, size_t dim, float* dist, uint64_t* keys
);
void FhtFloatAvx512(float* buf, size_t len);
void FlipSignsAvx512(const uint8_t* signs, float* data, size_t len);
void KacsWalkAvx512(float* data, size_t len);
#endif

}  // namespace e8qlib::simd
