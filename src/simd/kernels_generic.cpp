#include "simd/kernels.hpp"

namespace e8qlib::simd {

float QueryNorm2Generic(const float* q, size_t dim) {
    double s = 0.0;
    for (size_t d = 0; d < dim; ++d) {
        s += static_cast<double>(q[d]) * q[d];
    }
    return static_cast<float>(s);
}

void RouteGemvGeneric(
    const float* q, const float* centroids, const float* cn2, size_t nlist, size_t dim,
    float qn2, float* dist, uint64_t* keys, size_t base, size_t stride
) {
    const size_t pitch = stride != 0 ? stride : dim;
    for (size_t c = 0; c < nlist; ++c) {
        const float* cc = centroids + c * pitch;
        double dot = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            dot += static_cast<double>(q[d]) * cc[d];
        }
        // cn2, dist and keys are indexed by the caller's global cluster id:
        // `base` is the id of centroids[0], so a block sweep of the centroid
        // matrix (RouteL2Group) reads the same norms a whole-matrix sweep
        // would. Indexing cn2 by the block-local `c` picks up the first
        // block's norms for every later block.
        PackRouteKey(
            base + c, qn2 + cn2[base + c] - 2.0F * static_cast<float>(dot), dist, keys
        );
    }
}

void RouteL2DirectGeneric(
    const float* q, const float* centroids, size_t nlist, size_t dim, float* dist, uint64_t* keys
) {
    for (size_t c = 0; c < nlist; ++c) {
        const float* cen = centroids + c * dim;
        double sum = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            const double d = static_cast<double>(q[j]) - cen[j];
            sum += d * d;
        }
        PackRouteKey(c, static_cast<float>(sum), dist, keys);
    }
}

void FlipSignsGeneric(const uint8_t* signs, float* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        if ((signs[i / 8] >> (i % 8)) & 1) {
            data[i] = -data[i];
        }
    }
}

void KacsWalkGeneric(float* data, size_t len) {
    const size_t half = len / 2;
    for (size_t i = 0; i < half; ++i) {
        const float x = data[i];
        const float y = data[i + half];
        data[i] = x + y;
        data[i + half] = x - y;
    }
}

void FhtFloatGeneric(float* buf, size_t len) {
    for (size_t h = 1; h < len; h <<= 1) {
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

}  // namespace e8qlib::simd
