#include "e8qlib/index/ivf_common.hpp"

#include "simd/kernels.hpp"

#include <algorithm>

namespace e8qlib::ivf {

void RouteL2(
    const float* q, const float* centroids, const float* cn2, size_t nlist, size_t dim,
    size_t nprobe, float* dist, uint32_t* top_ids, uint64_t* keys, size_t stride
) {
    const size_t take = std::min(nprobe, nlist);
    if (take == 0) {
        return;
    }
    if (cn2 != nullptr) {
        const float qn2 = simd::QueryNorm2(q, dim);
        simd::RouteGemv(q, centroids, cn2, nlist, dim, qn2, dist, keys, 0, stride);
    } else {
        simd::RouteL2Direct(q, centroids, nlist, dim, dist, keys);
    }
    std::partial_sort(keys, keys + take, keys + nlist);
    for (size_t i = 0; i < take; ++i) {
        top_ids[i] = static_cast<uint32_t>(keys[i]);
    }
}

void RouteL2Group(
    const float* const* qs, size_t nq, const float* centroids, const float* cn2,
    size_t nlist, size_t dim, size_t nprobe, float* dists, uint64_t* keys,
    uint32_t* top_ids, size_t stride
) {
    const size_t take = std::min(nprobe, nlist);
    if (take == 0 || nq == 0) {
        return;
    }
    if (cn2 == nullptr) {
        for (size_t i = 0; i < nq; ++i) {
            RouteL2(
                qs[i], centroids, cn2, nlist, dim, nprobe, dists + i * nlist,
                top_ids + i * take, keys + i * nlist
            );
        }
        return;
    }
    // Queries per sweep. Each extra query in the group costs another L2 pass
    // over the block, so the group stops where the block stops fitting L2.
    constexpr size_t kGroup = 8;
    // Centroid block: ~128 KiB, the point where the sweep stays L2-resident
    // between the group's passes (dim 128 -> 256 rows, dim 1024 -> 32).
    //
    // dim can be 0.  ME8Q routes over the coded prefix of its plan, and the
    // plan an unsatisfiable rate falls back to (me8q::Allocate's SAQ-parity
    // all-dropped run) has no coded prefix at all.  The sweep is then a
    // column of constants -- every cluster scores ||q||^2 + ||c||^2 and the
    // block size is arbitrary -- but the division below still has to be
    // defined, and it is the only place an empty plan reaches.
    const size_t block_rows =
        dim == 0 ? kGroup : std::max<size_t>(8, 131072 / (dim * sizeof(float)));
    float qn2[kGroup];
    for (size_t g0 = 0; g0 < nq; g0 += kGroup) {
        const size_t gn = std::min(kGroup, nq - g0);
        for (size_t i = 0; i < gn; ++i) {
            qn2[i] = simd::QueryNorm2(qs[g0 + i], dim);
        }
        for (size_t c0 = 0; c0 < nlist; c0 += block_rows) {
            const size_t cnt = std::min(block_rows, nlist - c0);
            for (size_t i = 0; i < gn; ++i) {
                // centroids are block-local (offset pointer), but the kernel
                // indexes cn2/dist/keys by the GLOBAL cluster id (`base`), so
                // those three are the full arrays, not block views.
                simd::RouteGemv(
                    qs[g0 + i], centroids + c0 * stride, cn2, cnt, dim, qn2[i],
                    dists + (g0 + i) * nlist, keys + (g0 + i) * nlist, c0, stride
                );
            }
        }
        for (size_t i = 0; i < gn; ++i) {
            uint64_t* k = keys + (g0 + i) * nlist;
            std::partial_sort(k, k + take, k + nlist);
            for (size_t p = 0; p < take; ++p) {
                top_ids[(g0 + i) * take + p] = static_cast<uint32_t>(k[p]);
            }
        }
    }
}

}  // namespace e8qlib::ivf
