#include "e8qlib/index/me8q.hpp"
#include "e8qlib/index/ivf_common.hpp"

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
void Me8qIndex::PadQuery(const float* query, std::vector<float>* padded) const {
    padded->assign(padded_dim_, 0.0F);
    std::memcpy(padded->data(), query, dim_ * sizeof(float));
}

void Me8qIndex::PrepareRotatedQuery(const float* q, float* rot) const {
    // One rotation per coded run; the caller zero-initializes the dropped
    // tail (it never enters a score).
    size_t off = 0;
    size_t run = 0;
    for (const PlanSeg& seg : shared_plan_) {
        if (seg.bits != 0) {
            run_rot_[run].Rotate(q + off, rot + off);
            ++run;
        }
        off += seg.len;
    }
}

void Me8qIndex::PrepareCentroidsRotated() {
    cent_rot_.assign(num_clusters_ * padded_dim_, 0.0F);
    for (size_t c = 0; c < num_clusters_; ++c) {
        const float* src = centroids_.data() + c * padded_dim_;
        float* dst = cent_rot_.data() + c * padded_dim_;
        size_t off = 0;
        size_t run = 0;
        for (const PlanSeg& seg : shared_plan_) {
            if (seg.bits != 0) {
                run_rot_[run].Rotate(src + off, dst + off);
                ++run;
            }
            off += seg.len;
        }
    }
    centroid_n2_.assign(num_clusters_, 0.0F);
    ivf::CentroidNorm2(cent_rot_.data(), num_clusters_, padded_dim_, centroid_n2_.data());
    // Route only over the coded prefix: trailing zero-bit segments are exact
    // zeros on both sides, so the GEMV's zero tail contributes nothing (and
    // on padded datasets it is half the sweep).
    route_dim_ = padded_dim_;
    for (auto seg = shared_plan_.rbegin(); seg != shared_plan_.rend(); ++seg) {
        if (seg->bits != 0) {
            break;
        }
        route_dim_ -= seg->len;
    }
}

double Me8qIndex::EstimateRow(
    const float* row, const ListLayout& lay, double d2, const float* q_rot
) const {
    double acc = d2;
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(row);
    for (const SegSlot& slot : lay.segs) {
        const float* fac = reinterpret_cast<const float*>(bytes + slot.factor_off);
        const double coef = fac[0];
        if (coef == 0.0) {
            acc += fac[1];
            continue;
        }
        const double sc = static_cast<double>(codec_.ScoreFast(
            bytes + slot.code_off, static_cast<int>(slot.len), static_cast<int>(slot.bits),
            q_rot + slot.dim_off
        ));
        acc += static_cast<double>(fac[1]) - 2.0 * coef * sc;
    }
    return acc;
}

void Me8qIndex::EstimateRow4(
    const uint8_t* const rows[4], const ListLayout& lay, double d2, const float* q_rot,
    double out[4]
) const {
    double acc[4] = {d2, d2, d2, d2};
    for (const SegSlot& slot : lay.segs) {
        const uint8_t* in[4] = {
            rows[0] + slot.code_off, rows[1] + slot.code_off,
            rows[2] + slot.code_off, rows[3] + slot.code_off
        };
        float s[4];
        codec_.ScoreFast4(
            in, static_cast<int>(slot.len), static_cast<int>(slot.bits),
            q_rot + slot.dim_off, s
        );
        for (int j = 0; j < 4; ++j) {
            const float* fac = reinterpret_cast<const float*>(rows[j] + slot.factor_off);
            acc[j] += static_cast<double>(fac[1]) -
                      2.0 * static_cast<double>(fac[0]) * static_cast<double>(s[j]);
        }
    }
    for (int j = 0; j < 4; ++j) {
        out[j] = acc[j];
    }
}

void Me8qIndex::PushCandidate(
    float dist, uint32_t id, size_t k, std::vector<std::pair<float, uint32_t>>* out
) const {
    auto by_dist = [](const std::pair<float, uint32_t>& a,
                      const std::pair<float, uint32_t>& b) { return a.first < b.first; };
    if (out->size() >= k && !(dist < out->back().first)) {
        return;
    }
    if (out->size() >= k) {
        out->pop_back();
    }
    const std::pair<float, uint32_t> item{dist, id};
    out->insert(std::lower_bound(out->begin(), out->end(), item, by_dist), item);
}

void Me8qIndex::SearchIVF(
    const float* query, size_t k, size_t nprobe, uint32_t* out_ids, SearchStats* stats,
    float* out_dists
) const {
    if (k == 0 || nprobe == 0 || n_ == 0) {
        return;
    }
    const size_t take = std::min(nprobe, num_clusters_);
    // Phase timers, same four buckets the parent's ANN ledger reports
    // (rotate/route/scan/refine), so the two engines can be compared phase by
    // phase. ~4 clock calls per probed list, well under 1% of a query.
    std::vector<float> qpad;
    PadQuery(query, &qpad);
    const float* q = qpad.data();
    std::vector<float> rot_q(padded_dim_, 0.0F);
    PrepareRotatedQuery(q, rot_q.data());
    std::vector<float> probe_dist(num_clusters_);
    std::vector<uint32_t> probe_id(take);
    std::vector<uint64_t> keys(num_clusters_);
    ivf::RouteL2(
        rot_q.data(), cent_rot_.data(), centroid_n2_.data(), num_clusters_, route_dim_,
        take, probe_dist.data(), probe_id.data(), keys.data(), padded_dim_
    );

    std::vector<std::pair<float, uint32_t>> heap;
    heap.reserve(k + 1);
    size_t candidates = 0;
    size_t bits_scored = 0;
    size_t bits_total = 0;
    for (size_t p = 0; p < take; ++p) {
        const size_t c = probe_id[p];
        const ListLayout& lay = layouts_[c];
        const double d2 = probe_dist[c];
        const size_t i0 = list_begin_[c];
        const size_t i1 = list_begin_[c + 1];
        size_t bits_row = 0;
        for (const SegSlot& sl : lay.segs) {
            bits_row += static_cast<size_t>(sl.bits) * static_cast<size_t>(sl.len);
        }
        // Full-estimate oracle (4-wide in-flight, parent RefineBatch shape;
        // leftover 1–3 via EstimateRow). Serves the fast-path equivalence
        // tests; production search is SearchIVFFast.
        size_t i = i0;
        const uint8_t* base = codes_.data() + byte_begin_[c];
        for (; i + 4 <= i1; i += 4) {
            const uint8_t* rows[4] = {
                base + (i - i0) * lay.stride,
                base + (i - i0 + 1) * lay.stride,
                base + (i - i0 + 2) * lay.stride,
                base + (i - i0 + 3) * lay.stride
            };
            if (i + 8 <= i1) {
                __builtin_prefetch(base + (i - i0 + 4) * lay.stride, 0, 3);
                __builtin_prefetch(base + (i - i0 + 5) * lay.stride, 0, 3);
                __builtin_prefetch(base + (i - i0 + 6) * lay.stride, 0, 3);
                __builtin_prefetch(base + (i - i0 + 7) * lay.stride, 0, 3);
            }
            double dist[4];
            EstimateRow4(rows, lay, d2, rot_q.data(), dist);
            for (int j = 0; j < 4; ++j) {
                PushCandidate(static_cast<float>(dist[j]), ids_[i + static_cast<size_t>(j)], k, &heap);
            }
            bits_scored += 4 * bits_row;
            bits_total += 4 * bits_row;
            candidates += 4;
        }
        for (; i < i1; ++i) {
            const uint8_t* row = base + (i - i0) * lay.stride;
            const double dist = EstimateRow(
                reinterpret_cast<const float*>(row), lay, d2, rot_q.data()
            );
            bits_scored += bits_row;
            bits_total += bits_row;
            PushCandidate(static_cast<float>(dist), ids_[i], k, &heap);
            ++candidates;
        }
    }
    for (size_t i = 0; i < k; ++i) {
        if (i < heap.size()) {
            out_ids[i] = heap[i].second;
            if (out_dists != nullptr) {
                out_dists[i] = heap[i].first;
            }
        } else {
            out_ids[i] = std::numeric_limits<uint32_t>::max();
            if (out_dists != nullptr) {
                out_dists[i] = std::numeric_limits<float>::infinity();
            }
        }
    }
    if (stats != nullptr) {
        stats->candidates = candidates;
        stats->clusters = take;
        stats->bits_scored = bits_scored;
        stats->bits_total = bits_total;
        // The oracle path scores every row of every probed list, so nothing
        // is pruned: bits_scored == bits_total and every candidate survives.
        stats->survivors = candidates;
        stats->pruned = 0;
    }
}

void Me8qIndex::ExhaustiveTopK(
    const float* query, size_t k, std::vector<std::pair<float, uint32_t>>* out
) const {
    out->clear();
    out->reserve(k + 1);
    std::vector<float> qpad;
    PadQuery(query, &qpad);
    const float* q = qpad.data();
    std::vector<float> rot_q(padded_dim_, 0.0F);
    // Query-only: the rotation is the same for every cluster, and the zeroed
    // dropped tail (never written by the per-run rotations) stays zero.
    PrepareRotatedQuery(q, rot_q.data());
    for (size_t c = 0; c < num_clusters_; ++c) {
        const float* cen = centroids_.data() + c * padded_dim_;
        double d2 = 0.0;
        for (size_t j = 0; j < padded_dim_; ++j) {
            const double d = static_cast<double>(q[j]) - cen[j];
            d2 += d * d;
        }
        const ListLayout& lay = layouts_[c];
        for (size_t i = list_begin_[c]; i < list_begin_[c + 1]; ++i) {
            const uint8_t* row = codes_.data() + byte_begin_[c] + (i - list_begin_[c]) * lay.stride;
            const double dist = EstimateRow(
                reinterpret_cast<const float*>(row), lay, d2, rot_q.data()
            );
            PushCandidate(static_cast<float>(dist), ids_[i], k, out);
        }
    }
}

}  // namespace e8qlib::me8q
