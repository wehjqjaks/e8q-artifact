#include "e8qlib/index/me8q.hpp"

#include <chrono>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>

#include "e8qlib/fastscan/fastscan.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace e8qlib::me8q {
void Me8qIndex::EnsureIdMap() const {
    std::lock_guard<std::mutex> lock(id_map_mu_);
    if (id_map_ready_) {
        return;
    }
    id_to_row_.assign(n_, 0);
    id_to_list_.assign(n_, std::numeric_limits<uint32_t>::max());
    for (size_t c = 0; c < num_clusters_; ++c) {
        for (size_t i = list_begin_[c]; i < list_begin_[c + 1]; ++i) {
            id_to_row_[ids_[i]] = static_cast<uint32_t>(i - list_begin_[c]);
            id_to_list_[ids_[i]] = static_cast<uint32_t>(c);
        }
    }
    id_map_ready_ = true;
}

float Me8qIndex::Estimate(const float* query, uint32_t id) const {
    EnsureIdMap();
    if (id >= n_ || id_to_list_[id] == std::numeric_limits<uint32_t>::max()) {
        throw std::out_of_range("Me8qIndex::Estimate: unknown id");
    }
    const size_t c = id_to_list_[id];
    const size_t row_idx = id_to_row_[id];
    const ListLayout& lay = layouts_[c];
    const float* cen = centroids_.data() + c * padded_dim_;
    std::vector<float> qpad;
    PadQuery(query, &qpad);
    const float* q = qpad.data();
    double d2 = 0.0;
    for (size_t j = 0; j < padded_dim_; ++j) {
        const double d = static_cast<double>(q[j]) - cen[j];
        d2 += d * d;
    }
    const uint8_t* row = codes_.data() + byte_begin_[c] + row_idx * lay.stride;
    std::vector<float> rot_q(padded_dim_, 0.0F);
    PrepareRotatedQuery(q, rot_q.data());
    return static_cast<float>(
        EstimateRow(reinterpret_cast<const float*>(row), lay, d2, rot_q.data())
    );
}

// ---------------------------------------------------------------------------
}  // namespace e8qlib::me8q
