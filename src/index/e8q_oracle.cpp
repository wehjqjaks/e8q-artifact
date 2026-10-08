#include "e8qlib/index/e8q.hpp"

#include "index/e8q_internal.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

using e8qlib::detail::Better;

namespace e8qlib {

void E8qIndex::ExhaustiveTopK(
    const float* query, size_t k, std::vector<std::pair<float, uint32_t>>& out
) const {
    std::vector<float> rot_q(padded_dim_, 0.0F);
    rotator_.Rotate(query, rot_q.data());
    for (size_t d = 0; d < padded_dim_; ++d) {
        rot_q[d] -= mean_rot_[d];
    }
    float qn2 = 0.0F;
    for (size_t d = 0; d < padded_dim_; ++d) {
        qn2 += rot_q[d] * rot_q[d];
    }
    out.clear();
    out.reserve(n_);
    for (size_t c = 0; c < num_clusters_; ++c) {
        const float* cc = &centroid_rot_[c * padded_dim_];
        float dot = 0.0F;
        for (size_t d = 0; d < padded_dim_; ++d) {
            dot += rot_q[d] * cc[d];
        }
        float d2 = qn2 + centroid_n2_[c] - 2.0F * dot;
        if (d2 < 0.0F) {
            d2 = 0.0F;
        }
        const uint64_t vb = cluster_begin_[c];
        const uint64_t ve = cluster_begin_[c + 1];
        for (uint64_t v = vb; v < ve; ++v) {
            const SurvivorRow row = RowsAt(c, static_cast<size_t>(v - vb));
            const float score = ExactScore(row.refine, row.plane, row.lane, rot_q.data());
            out.emplace_back(d2 + row.h3 - 2.0F * row.g1 * score, row.id);
        }
    }
    const size_t take = std::min(k, out.size());
    std::partial_sort(
        out.begin(),
        out.begin() + static_cast<long>(take),
        out.end(),
        [](const auto& a, const auto& b) {
            return Better(a.first, a.second, b.first, b.second);
        }
    );
    out.resize(take);
}

float E8qIndex::Estimate(const float* query, uint32_t id) const {
    if (n_ == 0) {
        throw std::runtime_error("Estimate: empty index");
    }
    if (inv_ids_.size() != n_) {
        std::lock_guard<std::mutex> lock(inv_ids_mu_);
        if (inv_ids_.size() != n_) {
            inv_ids_.assign(n_, std::numeric_limits<uint32_t>::max());
            for (size_t i = 0; i < n_; ++i) {
                const uint32_t gid = ids_[i];
                if (gid < n_) {
                    inv_ids_[gid] = static_cast<uint32_t>(i);
                }
            }
        }
    }
    if (id >= n_ || inv_ids_[id] == std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Estimate: id out of range");
    }
    const size_t pos = inv_ids_[id];
    const auto it = std::upper_bound(cluster_begin_.begin() + 1, cluster_begin_.end(), pos);
    const size_t c = static_cast<size_t>(it - cluster_begin_.begin() - 1);
    const size_t local = pos - static_cast<size_t>(cluster_begin_[c]);

    std::vector<float> rot_q(padded_dim_, 0.0F);
    rotator_.Rotate(query, rot_q.data());
    for (size_t d = 0; d < padded_dim_; ++d) {
        rot_q[d] -= mean_rot_[d];
    }
    float qn2 = 0.0F;
    float dot = 0.0F;
    const float* cc = &centroid_rot_[c * padded_dim_];
    for (size_t d = 0; d < padded_dim_; ++d) {
        qn2 += rot_q[d] * rot_q[d];
        dot += rot_q[d] * cc[d];
    }
    float d2 = qn2 + centroid_n2_[c] - 2.0F * dot;
    if (d2 < 0.0F) {
        d2 = 0.0F;
    }
    const SurvivorRow row = RowsAt(c, local);
    const float score = ExactScore(row.refine, row.plane, row.lane, rot_q.data());
    return d2 + row.h3 - 2.0F * row.g1 * score;
}

double E8qIndex::ProbeBoundW(
    const float* query, const float* base_row, uint32_t id
) const {
    if (n_ == 0) {
        throw std::runtime_error("ProbeBoundW: empty index");
    }
    if (inv_ids_.size() != n_) {
        std::lock_guard<std::mutex> lock(inv_ids_mu_);
        if (inv_ids_.size() != n_) {
            inv_ids_.assign(n_, std::numeric_limits<uint32_t>::max());
            for (size_t i = 0; i < n_; ++i) {
                const uint32_t gid = ids_[i];
                if (gid < n_) {
                    inv_ids_[gid] = static_cast<uint32_t>(i);
                }
            }
        }
    }
    if (id >= n_ || inv_ids_[id] == std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("ProbeBoundW: id out of range");
    }
    const size_t pos = inv_ids_[id];
    const auto it = std::upper_bound(cluster_begin_.begin() + 1, cluster_begin_.end(), pos);
    const size_t c = static_cast<size_t>(it - cluster_begin_.begin() - 1);
    const size_t local = pos - static_cast<size_t>(cluster_begin_[c]);
    const size_t pdim = padded_dim_;
    const float* cc = &centroid_rot_[c * pdim];

    std::vector<float> rot_q(pdim, 0.0F);
    std::vector<float> rot_x(pdim, 0.0F);
    rotator_.Rotate(query, rot_q.data());
    rotator_.Rotate(base_row, rot_x.data());
    for (size_t d = 0; d < pdim; ++d) {
        rot_q[d] -= mean_rot_[d];
        rot_x[d] -= mean_rot_[d];
    }

    // Query radius r_q, data radius r_x, and the unit data direction, all in
    // the centered rotated frame of Build.
    double d2 = 0.0;
    double rn2 = 0.0;
    std::vector<double> unit(pdim, 0.0);
    for (size_t d = 0; d < pdim; ++d) {
        const double qr = static_cast<double>(rot_q[d]) - static_cast<double>(cc[d]);
        const double xr = static_cast<double>(rot_x[d]) - static_cast<double>(cc[d]);
        d2 += qr * qr;
        rn2 += xr * xr;
        unit[d] = xr;
    }
    const double rn = std::sqrt(rn2);
    if (!(rn > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    for (size_t d = 0; d < pdim; ++d) {
        unit[d] /= rn;
    }

    // Both reconstructions of the block, on the build's own level tables:
    // the exact one (vE, refinement bits) and the scan basis (w_scan, the
    // branch-averaged class means priced by the Coarse Plane).
    const std::vector<double> lev[2] = {
        BranchLevels(bits_, 0), BranchLevels(bits_, 1)
    };
    const double rd = std::sqrt(static_cast<double>(pdim));
    const SurvivorRow row = RowsAt(c, local);
    double dot_ue = 0.0;  // <unit, vE>
    double n2_e = 0.0;    // ||vE||^2
    double dot_ce = 0.0;  // <centroid', vE>
    double dot_ew = 0.0;  // <vE, w_scan>
    double dot_uw = 0.0;  // <unit, w_scan>
    double n2_w = 0.0;    // ||w_scan||^2
    double dot_cw = 0.0;  // <centroid', w_scan>
    double s_exact = 0.0;   // <query', vE>
    double s_coarse = 0.0;  // <query', w_scan>
    for (int blk = 0; blk < blocks_; ++blk) {
        uint8_t digits[8] = {0};
        int br = 0;
        DecodeSplitBlockLane(
            bits_, row.plane, row.refine, static_cast<uint32_t>(blk), row.lane, digits, &br
        );
        for (int k = 0; k < 8; ++k) {
            const size_t idx = static_cast<size_t>(blk) * 8 + static_cast<size_t>(k);
            const int dg = digits[k];
            const double ve = lev[br][static_cast<size_t>(dg)] / rd;
            const double wc = gate_mid_[static_cast<size_t>(2 * k + (dg >> (bits_ - 1)))];
            dot_ue += unit[idx] * ve;
            n2_e += ve * ve;
            dot_ce += static_cast<double>(cc[idx]) * ve;
            dot_ew += ve * wc;
            dot_uw += unit[idx] * wc;
            n2_w += wc * wc;
            dot_cw += static_cast<double>(cc[idx]) * wc;
            s_exact += static_cast<double>(rot_q[idx]) * ve;
            s_coarse += static_cast<double>(rot_q[idx]) * wc;
        }
    }
    if (!(std::abs(dot_ue) > 0.0) || !(std::abs(dot_uw) > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double g1 = rn / dot_ue;
    const double h3 = rn2 + 2.0 * g1 * dot_ce;
    const double d_full = d2 + h3 - 2.0 * g1 * s_exact;

    const double w_x = 2.0 * rn / dot_uw;
    const double d_coarse = d2 + rn2 + w_x * dot_cw - w_x * s_coarse;

    const double rx2 = 4.0 * g1 * g1 * n2_e + w_x * w_x * n2_w - 4.0 * g1 * w_x * dot_ew;
    const double sqrt_dm1 =
        std::sqrt(static_cast<double>(std::max<size_t>(dim_ - 1, 1)));
    const double omega = std::sqrt(std::max(0.0, rx2)) / sqrt_dm1;
    const double r_q = std::sqrt(std::max(0.0, d2));
    if (!(omega > 0.0) || !(r_q > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (d_coarse - d_full) / (r_q * omega);
}

}  // namespace e8qlib
