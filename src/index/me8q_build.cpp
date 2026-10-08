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

#include "index/me8q_internal.hpp"

namespace e8qlib::me8q {

void Me8qIndex::BuildRunRotators() {
    run_rot_.clear();
    uint64_t s = opt_.rot_seed;
    for (const PlanSeg& seg : shared_plan_) {
        if (seg.bits == 0) {
            continue;
        }
        // golden-ratio step: decorrelate consecutive run rotators
        s += 0x9E3779B97F4A7C15ULL;
        run_rot_.emplace_back(seg.len, s);
    }
}

void Me8qIndex::Build(
    const float* base, size_t ld, const float* centroids, size_t ld_c,
    const uint32_t* cluster_ids, size_t n, size_t dim, size_t num_clusters,
    const BuildOptions& opt, const float* data_variance
) {
    ortho_alpha_ = opt.prune_alpha > 0.0F ? opt.prune_alpha : 1.0F;
    if (base == nullptr || centroids == nullptr || cluster_ids == nullptr || n == 0 ||
        dim == 0 || num_clusters == 0) {
        throw std::invalid_argument("Me8qIndex::Build: empty input");
    }
    id_map_ready_ = false;
    n_ = n;
    dim_ = dim;
    padded_dim_ = PaddedDim(dim);
    num_clusters_ = num_clusters;
    opt_ = opt;
    if (opt_.max_bits < 1 || opt_.max_bits > kMaxBits) {
        throw std::invalid_argument("Me8qIndex::Build: max_bits must be in [1, 13]");
    }

    centroids_.assign(num_clusters_ * padded_dim_, 0.0F);
    for (size_t c = 0; c < num_clusters_; ++c) {
        std::memcpy(
            centroids_.data() + c * padded_dim_, centroids + c * ld_c,
            static_cast<size_t>(dim_) * sizeof(float)
        );
    }

    const auto lists = ivf::BuildLists(cluster_ids, n_, num_clusters_);
    list_begin_ = lists.begin;
    ids_.assign(lists.ids.begin(), lists.ids.end());

    // ---- plans ---------------------------------------------------------
    if (!opt_.plan_spec.empty()) {
        shared_plan_ = detail::ParsePlanSpec(opt_.plan_spec, padded_dim_);
    } else {
        std::vector<float> var(padded_dim_, 0.0F);
        if (data_variance != nullptr) {
            std::memcpy(var.data(), data_variance, dim_ * sizeof(float));
        } else {
            detail::DatasetVariance(base, ld, n_, dim_, &var);
        }
        PlanInput in;
        in.var = var.data();
        in.dim = padded_dim_;
        in.avg_bits = opt_.avg_bits;
        in.max_bits = opt_.max_bits;
        in.extra_bits = opt_.extra_budget_bits;
        shared_plan_ = Allocate(in);
        {
            std::string s2;
            char buf[32];
            for (const PlanSeg& sg : shared_plan_) {
                std::snprintf(buf, sizeof(buf), "%zu:%d ", sg.len, sg.bits);
                s2 += buf;
            }
            std::fprintf(stderr, "me8q: plan=%s\n", s2.c_str());
        }
    }

    // ---- layouts: [factor pairs][segment code bytes] per row ------------
    layouts_.assign(num_clusters_, ListLayout{});
    size_t stride_sum = 0;
    for (size_t c = 0; c < num_clusters_; ++c) {
        const Plan& plan = PlanOf(c);
        ListLayout& lay = layouts_[c];
        size_t nseg = 0;
        for (const PlanSeg& seg : plan) {
            if (seg.bits > 0) {
                ++nseg;
            }
        }
        size_t code_off = nseg * 2 * sizeof(float);
        size_t factor_off = 0;
        size_t dim_off = 0;
        for (const PlanSeg& seg : plan) {
            if (seg.bits == 0) {
                dim_off += seg.len;
                continue;
            }
            SegSlot slot;
            slot.factor_off = static_cast<uint32_t>(factor_off);
            slot.code_off = static_cast<uint32_t>(code_off);
            slot.dim_off = static_cast<uint32_t>(dim_off);
            slot.len = static_cast<uint16_t>(seg.len);
            slot.bits = static_cast<uint8_t>(seg.bits);
            lay.segs.push_back(slot);
            factor_off += 2 * sizeof(float);
            code_off += static_cast<size_t>(SegmentBytes(seg.bits, static_cast<int>(seg.len)));
            dim_off += seg.len;
        }
        lay.stride = code_off;
        stride_sum += code_off;
    }
    bytes_per_vector_ = stride_sum / num_clusters_;
    byte_begin_.assign(num_clusters_ + 1, 0);
    for (size_t c = 0; c < num_clusters_; ++c) {
        byte_begin_[c + 1] =
            byte_begin_[c] + (list_begin_[c + 1] - list_begin_[c]) * layouts_[c].stride;
    }
    codes_.assign(byte_begin_[num_clusters_], 0);

    // Per-run FHT–Kac-4 rotators: the DP runs (bit-homogeneous spans) are
    // the quantization and factor-fitting units, so each run is rotated as a
    // whole. A uniform plan degenerates to E8Q's whole-vector rotation.
    BuildRunRotators();
    PrepareCentroidsRotated();

    // ---- encode --------------------------------------------------------
    // Rows are walked in FILE order, not in posting-list order.  A
    // cluster-major walk follows ids_ and so touches a corpus one scattered
    // 4 KiB row at a time; on a corpus that does not fit the page cache
    // (stella: 68 GiB against 58 GiB of host RAM) every one of those rows
    // faults, and the fault path then dominates the encoder -- measured at
    // 445 major faults/s with 8.9 cores of kernel time against 2 of user
    // time, which is what turned a 68 GiB stream the device serves at
    // 2.1 GB/s into a ~25 MB/s random walk.  Walking row i in order makes
    // the corpus a single sequential pass, and `list_rank` sends each row's
    // code to the byte it already had, so the index is bit-identical.
    std::vector<uint32_t> list_rank(n_, 0);
    for (size_t c = 0; c < num_clusters_; ++c) {
        const size_t off = static_cast<size_t>(list_begin_[c]);
        const size_t cnt = static_cast<size_t>(list_begin_[c + 1]) - off;
        for (size_t k = 0; k < cnt; ++k) {
            list_rank[ids_[off + k]] = static_cast<uint32_t>(k);
        }
    }
    // Dimension offset of every coded run (the rotation units).  Every
    // cluster shares one plan (PlanOf), so this is hoisted out of the loop
    // and the per-run rotation stays one call per coded run per row.
    std::vector<uint32_t> run_dim_off;
    {
        size_t off = 0;
        for (const PlanSeg& seg : shared_plan_) {
            if (seg.bits != 0) {
                run_dim_off.push_back(static_cast<uint32_t>(off));
            }
            off += seg.len;
        }
    }
    // Per-cluster layout, flattened.  The loop below is per row rather than
    // per cluster, so every lookup it makes has to be a flat array index:
    // `layouts_[c].segs` is a vector-of-vectors, and one heap hop per row
    // costs more than the sequential read it buys (measured 3.6 s -> 8.2 s
    // on the resident sift1M b=4 cell).  The SegSlot arrays are copied into
    // one pool and addressed by offset.
    std::vector<uint32_t> cl_seg_off(num_clusters_ + 1, 0);
    for (size_t c = 0; c < num_clusters_; ++c) {
        cl_seg_off[c + 1] =
            cl_seg_off[c] + static_cast<uint32_t>(layouts_[c].segs.size());
    }
    std::vector<SegSlot> seg_pool(cl_seg_off[num_clusters_]);
    std::vector<size_t> cl_stride(num_clusters_, 0);
    for (size_t c = 0; c < num_clusters_; ++c) {
        cl_stride[c] = layouts_[c].stride;
        std::copy(
            layouts_[c].segs.begin(), layouts_[c].segs.end(),
            seg_pool.begin() + cl_seg_off[c]
        );
    }
    const int nthreads = static_cast<int>(std::max<size_t>(1, opt_.threads));
#pragma omp parallel num_threads(nthreads)
    {
        std::vector<float> rresid(padded_dim_, 0.0F);
        std::vector<float> rrrot(padded_dim_, 0.0F);
        // One encoder buffer per thread: the per-segment Encode() this loop
        // used to call allocated a unit residual, a SegCode, and two vectors
        // inside it for every segment of every row.
        VectorCode vcode;
#pragma omp for schedule(dynamic, 256)
        for (int64_t ri = 0; ri < static_cast<int64_t>(n_); ++ri) {
            const size_t row_id = static_cast<size_t>(ri);
            const size_t c = cluster_ids[row_id];
            const SegSlot* const segs = seg_pool.data() + cl_seg_off[c];
            const size_t nsegs = cl_seg_off[c + 1] - cl_seg_off[c];
            const Plan& plan = PlanOf(c);
            const float* cen = centroids_.data() + c * padded_dim_;
            const float* cen_rot_c = cent_rot_.data() + c * padded_dim_;
            const float* row_in = base + row_id * ld;
            std::memset(rresid.data(), 0, padded_dim_ * sizeof(float));
            for (size_t j = 0; j < dim_; ++j) {
                rresid[j] = row_in[j] - cen[j];
            }
            {
                size_t run = 0;
                for (const PlanSeg& seg : plan) {
                    if (seg.bits == 0) {
                        continue;
                    }
                    run_rot_[run].Rotate(
                        rresid.data() + run_dim_off[run],
                        rrrot.data() + run_dim_off[run]
                    );
                    ++run;
                }
            }
          {
            const float* resid_all = rrrot.data();
            const float* resid_unrot = rresid.data();
            uint8_t* row =
                codes_.data() + byte_begin_[c] + list_rank[row_id] * cl_stride[c];
            // Residual energy of the dropped dimensions (all 0-bit runs are
            // trailing, so one pass suffices); folded into the last coded
            // run's h2 when drop_energy is on. Read from the unrotated block
            // (energy is rotation-invariant); the rotated buffer only covers
            // coded runs.
            double drop_n2 = 0.0;
            if (opt_.drop_energy) {
                size_t off = 0;
                for (const PlanSeg& seg : plan) {
                    if (seg.bits == 0) {
                        for (size_t j = 0; j < seg.len; ++j) {
                            const double v = static_cast<double>(resid_unrot[off + j]);
                            drop_n2 += v * v;
                        }
                    }
                    off += seg.len;
                }
            }
            size_t slot_idx = 0;
            size_t last_slot = nsegs == 0 ? 0 : nsegs - 1;
            size_t dim_off = 0;
            for (const PlanSeg& seg : plan) {
                if (seg.bits == 0) {
                    dim_off += seg.len;
                    continue;
                }
                const SegSlot& slot = segs[slot_idx];
                const int len = static_cast<int>(seg.len);
                const int bits = seg.bits;
                const float* res_rot = resid_all + dim_off;
                double norm2 = 0.0;
                for (int j = 0; j < len; ++j) {
                    norm2 += static_cast<double>(res_rot[j]) * static_cast<double>(res_rot[j]);
                }
                const double norm = std::sqrt(norm2);
                codec_.EncodeInto(res_rot, len, bits, norm, vcode);
                uint8_t* dst = row + slot.code_off;
                PackSegment(vcode, len, bits, dst);
                // Both fits (the amplitude denominator and the h2 cross term)
                // read the digits the encoder just produced, in one pass,
                // rather than unpacking `dst` twice.
                const SegmentCodec::Fits fit = codec_.ScoreVectorCode(
                    vcode, bits, res_rot, cen_rot_c + dim_off
                );
                const double dot_r = fit.query;
                const double dot_c = fit.centroid;
                // Per-run dot-normalized amplitude (the SAQ per-segment
                // `rescale` analogue): coef = ||r||^2 / <r, levels>. Guard is
                // strictly positive (parent e8q convention): a negative fit
                // would flip the ZPR weight sign in the packed banks.
                double coef = dot_r > 1e-12 ? norm2 / dot_r : 0.0;
                if (!std::isfinite(coef)) {
                    coef = 0.0;
                }
                double h2 = norm2 + 2.0 * coef * dot_c;
                if (slot_idx == last_slot) {
                    h2 += drop_n2;
                }
                float* fac = reinterpret_cast<float*>(row + slot.factor_off);
                fac[0] = static_cast<float>(coef);
                fac[1] = static_cast<float>(std::isfinite(h2) ? h2 : 0.0);
                dim_off += seg.len;
                ++slot_idx;
            }
          }
        }
    }
    BuildFastLayout();
}

}  // namespace e8qlib::me8q
