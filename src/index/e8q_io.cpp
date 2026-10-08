#include "e8qlib/index/e8q.hpp"

#include "index/e8q_internal.hpp"

#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "e8qlib/quantization/e8q_codec.hpp"

using e8qlib::detail::kMagic;
using e8qlib::detail::kOptMagic;
using e8qlib::detail::kMaxK;
using e8qlib::detail::AdviseHugePages;
using e8qlib::detail::AllocHuge;

namespace e8qlib {

size_t E8qIndex::memory_bytes() const {
    return arena_.size() + ref_arena_.size() + ids_.size() * 4 +
           centroid_rot_.size() * 4 + centroid_n2_.size() * 4 +
           cluster_begin_.size() * 8 + arena_begin_.size() * 8 +
           ref_arena_begin_.size() * 8 + mean_rot_.size() * 4 +
           refine_tab_.size() * 4;
}

void E8qIndex::Save(const char* path) const {
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr) {
        throw std::runtime_error("cannot open index file for writing");
    }
    bool ok = true;
    auto wr = [&](const void* p, size_t bytes) {
        if (bytes != 0 && std::fwrite(p, 1, bytes, f) != bytes) {
            ok = false;
        }
    };
    const uint32_t magic = kMagic;
    const uint64_t n = n_;
    const uint64_t dim = dim_;
    const uint64_t num_clusters = num_clusters_;
    const int bits = bits_;
    const uint64_t rot_bytes = rotator_.dump_bytes();
    wr(&magic, 4);
    wr(&n, 8);
    wr(&dim, 8);
    wr(&num_clusters, 8);
    wr(&bits, 4);
    wr(&rot_bytes, 8);

    wr(ids_.data(), ids_.size() * 4);
    std::vector<char> rot(rot_bytes);
    rotator_.Save(rot.data());
    wr(rot.data(), rot.size());
    wr(mean_rot_.data(), mean_rot_.size() * 4);
    wr(centroid_rot_.data(), centroid_rot_.size() * 4);
    wr(centroid_n2_.data(), centroid_n2_.size() * 4);
    wr(cluster_begin_.data(), cluster_begin_.size() * 8);
    wr(arena_begin_.data(), arena_begin_.size() * 8);
    wr(arena_.data(), arena_.size());
    wr(ref_arena_begin_.data(), ref_arena_begin_.size() * 8);
    wr(ref_arena_.data(), ref_arena_.size());
    // Options trailer: effective build knobs so the artifact carries its
    // own reproducibility record. Trailing magic marks the block; loaders
    // treat its absence as the historical default-options file.
    const uint32_t opt_magic = kOptMagic;
    uint32_t lam_bits = 0;
    std::memcpy(&lam_bits, &ortho_alpha_, 4);
    wr(&opt_magic, 4);
    wr(&lam_bits, 4);
    wr(&opt_magic, 4);
    std::fclose(f);
    if (!ok) {
        throw std::runtime_error("short write while saving index");
    }
}

void E8qIndex::Load(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        throw std::runtime_error("cannot open index file for reading");
    }
    auto rd = [&](void* p, size_t bytes) {
        if (bytes != 0 && std::fread(p, 1, bytes, f) != bytes) {
            std::fclose(f);
            throw std::runtime_error("short read while loading index");
        }
    };
    uint32_t magic = 0;
    rd(&magic, 4);
    if (magic != kMagic) {
        std::fclose(f);
        throw std::runtime_error("not an E8Q index file");
    }
    uint64_t n = 0;
    uint64_t dim = 0;
    uint64_t num_clusters = 0;
    uint32_t bits = 0;
    uint64_t rot_bytes = 0;
    rd(&n, 8);
    rd(&dim, 8);
    rd(&num_clusters, 8);
    rd(&bits, 4);
    rd(&rot_bytes, 8);
    if (bits < 1 || bits > kMaxQuantizeBits || num_clusters == 0) {
        std::fclose(f);
        throw std::runtime_error("corrupt E8Q index header");
    }
    n_ = n;
    dim_ = dim;
    num_clusters_ = num_clusters;
    bits_ = bits;
    rotator_ = FhtKac4Rotator(dim, 0);
    if (rot_bytes != rotator_.dump_bytes()) {
        std::fclose(f);
        throw std::runtime_error("rotator payload size mismatch");
    }
    padded_dim_ = rotator_.padded_dim();
    blocks_ = static_cast<int>(padded_dim_ / 8);

    AllocHuge(ids_, n_, uint32_t{0});
    rd(ids_.data(), ids_.size() * 4);

    std::vector<char> rot(rot_bytes);
    rd(rot.data(), rot.size());
    mean_rot_.assign(padded_dim_, 0.0F);
    rd(mean_rot_.data(), mean_rot_.size() * 4);
    rotator_.Load(rot.data());
    const std::vector<float> tab = BuildRefineTables(bits, static_cast<int>(padded_dim_));
    refine_tab_.assign(tab.begin(), tab.end());
    InitBitTables();
    // Same reserve -> madvise -> assign ordering as Build: a plain assign
    // faults 4 KB pages and a freshly LOADED index then runs the scan/route
    // arenas at 2-6x the per-query cost of the just-Built one (measured:
    // gist b7 np1 route-dominated 86 -> 540 us/q with small pages).
    AllocHuge(centroid_rot_, num_clusters * padded_dim_, 0.0F);
    centroid_n2_.assign(num_clusters, 0.0F);
    cluster_begin_.assign(num_clusters + 1, 0);
    arena_begin_.assign(num_clusters + 1, 0);
    rd(centroid_rot_.data(), centroid_rot_.size() * 4);
    rd(centroid_n2_.data(), centroid_n2_.size() * 4);
    rd(cluster_begin_.data(), cluster_begin_.size() * 8);
    rd(arena_begin_.data(), arena_begin_.size() * 8);
    if (cluster_begin_.back() != n || cluster_begin_[0] != 0) {
        std::fclose(f);
        throw std::runtime_error("cluster offsets inconsistent");
    }
    // Scan record: [ortho header 128 B][rowA blocks*32]; refine record:
    // [low (b-1) bits per dim, vector-major][g1][h3].
    batch_bytes_ = P1Off() + static_cast<size_t>(blocks_) * 32;
    batch_bytes_ = (batch_bytes_ + 63) & ~static_cast<size_t>(63);
    ref_batch_bytes_ = (RefRecordBytes() + 63) & ~static_cast<size_t>(63);
    uint64_t arena_bytes = 0;
    for (size_t c = 0; c < num_clusters; ++c) {
        const uint64_t batches = (cluster_begin_[c + 1] - cluster_begin_[c] + 31) / 32;
        if (arena_begin_[c] != arena_bytes) {
            std::fclose(f);
            throw std::runtime_error("arena offsets inconsistent");
        }
        arena_bytes += batches * batch_bytes_;
    }
    if (arena_begin_[num_clusters] != arena_bytes) {
        std::fclose(f);
        throw std::runtime_error("arena offsets inconsistent");
    }
    AllocHuge(arena_, static_cast<size_t>(arena_bytes) + 64, uint8_t{0});
    rd(arena_.data(), arena_.size());
    ref_arena_begin_.assign(num_clusters + 1, 0);
    rd(ref_arena_begin_.data(), ref_arena_begin_.size() * 8);
    uint64_t ref_bytes = 0;
    for (size_t c = 0; c < num_clusters; ++c) {
        const uint64_t batches = (cluster_begin_[c + 1] - cluster_begin_[c] + 31) / 32;
        if (ref_arena_begin_[c] != ref_bytes) {
            std::fclose(f);
            throw std::runtime_error("refine arena offsets inconsistent");
        }
        ref_bytes += batches * ref_batch_bytes_;
    }
    if (ref_arena_begin_[num_clusters] != ref_bytes) {
        std::fclose(f);
        throw std::runtime_error("refine arena offsets inconsistent");
    }
    AllocHuge(ref_arena_, static_cast<size_t>(ref_bytes) + 64, uint8_t{0});
    rd(ref_arena_.data(), ref_arena_.size());
    // Optional build-options trailer (legacy files end here): trailing magic
    // pair marks the block; any mismatch keeps the historical defaults.
    uint32_t opt_lead = 0;
    if (std::fread(&opt_lead, 4, 1, f) == 1 && opt_lead == kOptMagic) {
        uint32_t lam_bits = 0;
        uint32_t opt_tail = 0;
        if (std::fread(&lam_bits, 4, 1, f) != 1 ||
            std::fread(&opt_tail, 4, 1, f) != 1 || opt_tail != kOptMagic) {
            std::fclose(f);
            throw std::runtime_error("build-options trailer corrupt");
        }
        float lam = 0.0F;
        std::memcpy(&lam, &lam_bits, 4);
        // Union-bound expected-loss budget: any positive finite value.
        // Corrupted or foreign (NaN, <= 0) falls back to the default.
        ortho_alpha_ = (lam > 0.0F && lam == lam) ? lam : 1.0F;
    }
    std::fclose(f);
}

void E8qIndex::InitBitTables() {
    // Scan class-mean basis (same formula Build fits W_x against).
    double w_cls[2] = {0.0, 0.0};
    ScanClassMeans(bits_, w_cls);
    for (int d = 0; d < 8; ++d) {
        // b == 1: dim 7 of the scan byte is the BRANCH, not the parity
        // sign the refine decoder uses. Pricing that bit as a coordinate
        // injects a coordinate-wise independent noise term into S and
        // inflates ||R_x||. Zero the dim-7 class means so the LUT and
        // the W_x fit (Build uses the same rule) both ignore it.
        const double w0 = (bits_ == 1 && d == 7) ? 0.0 : w_cls[0];
        const double w1 = (bits_ == 1 && d == 7) ? 0.0 : w_cls[1];
        gate_mid_[static_cast<size_t>(2 * d)] = w0;
        gate_mid_[static_cast<size_t>(2 * d + 1)] = w1;
    }
}

}  // namespace e8qlib
