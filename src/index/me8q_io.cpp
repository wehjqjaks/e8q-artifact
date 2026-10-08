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
#include <vector>

#include "e8qlib/fastscan/fastscan.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "index/me8q_internal.hpp"

namespace e8qlib::me8q {

void Me8qIndex::Save(const char* path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error(std::string("Me8qIndex::Save: cannot write ") + path);
    }
    detail::WriteScalars(out, detail::kMagic, sizeof(detail::kMagic));
    const uint64_t hdr[5] = {
        static_cast<uint64_t>(n_), static_cast<uint64_t>(dim_),
        static_cast<uint64_t>(padded_dim_), static_cast<uint64_t>(num_clusters_),
        static_cast<uint64_t>(bytes_per_vector_)};
    detail::WriteScalars(out, hdr, sizeof(hdr));
    const float avg = opt_.avg_bits;
    int32_t flags[4] = {0, opt_.max_bits, opt_.drop_energy ? 1 : 0, 0};
    std::memcpy(&flags[3], &opt_.prune_alpha, sizeof(float));
    detail::WriteScalars(out, &avg, sizeof(avg));
    detail::WriteScalars(out, flags, sizeof(flags));
    detail::WritePlan(out, shared_plan_);
    detail::WriteScalars(out, centroids_.data(), centroids_.size() * sizeof(float));
    detail::WriteScalars(out, ids_.data(), ids_.size() * sizeof(uint32_t));
    detail::WriteScalars(out, list_begin_.data(), list_begin_.size() * sizeof(uint64_t));
    detail::WriteScalars(out, byte_begin_.data(), byte_begin_.size() * sizeof(uint64_t));
    detail::WriteScalars(out, codes_.data(), codes_.size());
    detail::WriteScalars(out, &opt_.rot_seed, sizeof(opt_.rot_seed));
    const uint64_t n_rot = run_rot_.size();
    detail::WriteScalars(out, &n_rot, sizeof(n_rot));
    for (const FhtKac4Rotator& r : run_rot_) {
        const uint64_t rot_bytes = r.dump_bytes();
        detail::WriteScalars(out, &rot_bytes, sizeof(rot_bytes));
        std::vector<char> rot(rot_bytes);
        r.Save(rot.data());
        detail::WriteScalars(out, rot.data(), rot.size());
    }
    if (!out) {
        throw std::runtime_error(std::string("Me8qIndex::Save: write failed ") + path);
    }
}

void Me8qIndex::Load(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::string("Me8qIndex::Load: cannot read ") + path);
    }
    char magic[8] = {};
    detail::ReadScalars(in, magic, sizeof(magic), "magic");
    if (std::memcmp(magic, detail::kMagic, sizeof(detail::kMagic)) != 0) {
        throw std::runtime_error(std::string("Me8qIndex::Load: bad magic ") + path);
    }
    id_map_ready_ = false;
    uint64_t hdr[5] = {};
    detail::ReadScalars(in, hdr, sizeof(hdr), "header");
    n_ = static_cast<size_t>(hdr[0]);
    dim_ = static_cast<size_t>(hdr[1]);
    padded_dim_ = static_cast<size_t>(hdr[2]);
    num_clusters_ = static_cast<size_t>(hdr[3]);
    bytes_per_vector_ = static_cast<size_t>(hdr[4]);
    float avg = 0.0F;
    int32_t flags[4] = {};
    detail::ReadScalars(in, &avg, sizeof(avg), "avg_bits");
    detail::ReadScalars(in, flags, sizeof(flags), "flags");
    opt_ = BuildOptions{};
    opt_.avg_bits = avg;
    opt_.max_bits = flags[1];
    opt_.drop_energy = flags[2] != 0;
    std::memcpy(&opt_.prune_alpha, &flags[3], sizeof(float));
    // Union-bound expected-loss budget: any positive finite value.
    // Corrupted or foreign (NaN, <= 0) falls back to the default.
    ortho_alpha_ = (opt_.prune_alpha > 0.0F && opt_.prune_alpha == opt_.prune_alpha)
                       ? opt_.prune_alpha
                       : 1.0F;
    shared_plan_ = detail::ReadPlan(in);
    centroids_.resize(num_clusters_ * padded_dim_);
    detail::ReadScalars(in, centroids_.data(), centroids_.size() * sizeof(float), "centroids");
    ids_.resize(n_);
    detail::ReadScalars(in, ids_.data(), ids_.size() * sizeof(uint32_t), "ids");
    list_begin_.resize(num_clusters_ + 1);
    detail::ReadScalars(in, list_begin_.data(), list_begin_.size() * sizeof(uint64_t), "list_begin");
    byte_begin_.resize(num_clusters_ + 1);
    detail::ReadScalars(in, byte_begin_.data(), byte_begin_.size() * sizeof(uint64_t), "byte_begin");
    const uint64_t payload = byte_begin_[num_clusters_];
    codes_.resize(static_cast<size_t>(payload));
    detail::ReadScalars(in, codes_.data(), codes_.size(), "codes");

    uint64_t rot_seed = 0;
    detail::ReadScalars(in, &rot_seed, sizeof(rot_seed), "rot_seed");
    opt_.rot_seed = rot_seed;
    // The rotators are a pure function of (plan, rot_seed); rebuild them from
    // the already-loaded plan, then overwrite with the stored state so the
    // exact construction-time matrices are restored.
    BuildRunRotators();
    uint64_t n_rot = 0;
    detail::ReadScalars(in, &n_rot, sizeof(n_rot), "rotator count");
    if (n_rot != run_rot_.size()) {
        // Also rejects files from the earlier single-tile-rotator layout,
        // whose leading rotator-size field never equals the run count.
        throw std::runtime_error(
            std::string("Me8qIndex::Load: rotator count mismatch ") + path
        );
    }
    for (FhtKac4Rotator& r : run_rot_) {
        uint64_t rot_bytes = 0;
        detail::ReadScalars(in, &rot_bytes, sizeof(rot_bytes), "rot_bytes");
        if (rot_bytes != r.dump_bytes()) {
            throw std::runtime_error(
                std::string("Me8qIndex::Load: rotator payload size mismatch ") + path
            );
        }
        std::vector<char> rot(rot_bytes);
        detail::ReadScalars(in, rot.data(), rot.size(), "rotator");
        r.Load(rot.data());
    }

    // Layouts are a pure function of the plans, so they are rebuilt, not
    // stored; a mismatch here means a corrupt file.
    layouts_.assign(num_clusters_, ListLayout{});
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
    }
    PrepareCentroidsRotated();
    BuildFastLayout();
}
double Me8qIndex::bits_per_dim() const {
    if (dim_ == 0) {
        return 0.0;
    }
    return 8.0 * static_cast<double>(bytes_per_vector_) / static_cast<double>(dim_);
}

size_t Me8qIndex::memory_bytes() const {
    size_t total = codes_.size() + ids_.size() * sizeof(uint32_t) +
                   centroids_.size() * sizeof(float) +
                   list_begin_.size() * sizeof(uint64_t) + byte_begin_.size() * sizeof(uint64_t);
    for (const ListLayout& lay : layouts_) {
        total += lay.segs.size() * sizeof(SegSlot);
    }
    size_t rot_bytes_total = 0;
    for (const FhtKac4Rotator& r : run_rot_) {
        rot_bytes_total += r.dump_bytes();
    }
    total += rot_bytes_total + cent_rot_.size() * sizeof(float);
    return total;
}

}  // namespace e8qlib::me8q
