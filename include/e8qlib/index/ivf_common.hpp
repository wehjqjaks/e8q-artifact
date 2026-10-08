// Shared IVF navigator used by E8Q (uniform rate) and ME8Q (mixed rate).
// Lists, L2 routing, and the size-k heap are codec-blind. Each index still
// owns its codes, estimator, and prune bound.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace e8qlib::ivf {

struct Lists {
    std::vector<uint64_t> begin;  // nlist + 1
    std::vector<uint32_t> ids;    // cluster-major global ids
};

inline Lists BuildLists(const uint32_t* cluster_ids, size_t n, size_t nlist) {
    if (cluster_ids == nullptr || n == 0 || nlist == 0) {
        throw std::invalid_argument("BuildLists: empty input");
    }
    Lists out;
    out.begin.assign(nlist + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        if (cluster_ids[i] >= nlist) {
            throw std::invalid_argument("cluster id out of range");
        }
        ++out.begin[cluster_ids[i] + 1];
    }
    for (size_t c = 0; c < nlist; ++c) {
        out.begin[c + 1] += out.begin[c];
    }
    out.ids.resize(n);
    std::vector<uint64_t> cursor(out.begin.begin(), out.begin.end() - 1);
    for (size_t i = 0; i < n; ++i) {
        out.ids[cursor[cluster_ids[i]]++] = static_cast<uint32_t>(i);
    }
    return out;
}

inline void CentroidNorm2(
    const float* centroids, size_t nlist, size_t dim, float* n2
) {
    for (size_t c = 0; c < nlist; ++c) {
        const float* row = centroids + c * dim;
        double s = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            s += static_cast<double>(row[d]) * row[d];
        }
        n2[c] = static_cast<float>(s);
    }
}

inline void CentroidNorm2(
    const float* centroids, size_t nlist, size_t dim, std::vector<float>* n2
) {
    n2->assign(nlist, 0.0F);
    CentroidNorm2(centroids, nlist, dim, n2->data());
}

// Full-dim fp32 L2 against every centroid, then top-nprobe.
// `centroids` is nlist * dim. `cn2` is optional ||c||^2 (GEMV form); when null
// the kernel uses ||q-c||^2. `dist` is nlist (indexed by cluster id).
// `top_ids` receives nprobe cluster ids in increasing d2. `keys` is nlist scratch.
// `stride` is the centroid row pitch when it differs from `dim` (ME8Q routes
// over a prefix of the padded rows); 0 means rows are contiguous at `dim`.
void RouteL2(
    const float* q, const float* centroids, const float* cn2, size_t nlist, size_t dim,
    size_t nprobe, float* dist, uint32_t* top_ids, uint64_t* keys, size_t stride = 0
);

// Batched twin of RouteL2. Every query gets the SAME distances and the SAME
// top-nprobe list it would get from RouteL2 (the per-cluster arithmetic and
// key packing are unchanged), but the centroid matrix is swept in L2-sized
// blocks shared by a group of queries, so the matrix leaves DRAM/L3 once per
// group instead of once per query. `dists` (nq * nlist) and `keys`
// (nq * nlist) are caller scratch; `top_ids` is nq * nprobe output.
// `cn2` must be non-null (the GEMV form) for the batched sweep to engage;
// with a null `cn2` every query falls back to its own RouteL2 call, which is
// correct but sweeps the centroid matrix once per query.
void RouteL2Group(
    const float* const* qs, size_t nq, const float* centroids, const float* cn2,
    size_t nlist, size_t dim, size_t nprobe, float* dists, uint64_t* keys,
    uint32_t* top_ids, size_t stride = 0
);

}  // namespace e8qlib::ivf
