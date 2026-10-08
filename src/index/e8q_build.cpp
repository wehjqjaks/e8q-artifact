#include "e8qlib/index/e8q.hpp"

#include "index/e8q_internal.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "e8qlib/quantization/e8q_codec.hpp"

using e8qlib::detail::AllocHuge;
using e8qlib::detail::ClusterBuffers;

namespace e8qlib {
namespace {

// Orthogonal-projection scan banks: base rounds down and perp rounds up
// (every rounding direction pushes dlo down, so the bound stays sound);
// weight rounds nearest; W-rounding and LUT error are covered per lane
// in OrthoBatchHeader. sc[6] stores |W|_max for the batch skip only.
void WriteOrthoBanks(const detail::ClusterBuffers& cb, size_t lo, size_t hi, uint8_t* bp) {
    float bmin = 0.0F, bmax = 0.0F, wmin = 0.0F, wmax = 0.0F;
    float pmin = 0.0F, pmax = 0.0F;
    if (hi > lo) {
        bmin = bmax = cb.ortho_base[lo];
        wmin = wmax = cb.ortho_weight[lo];
        pmin = pmax = cb.ortho_perp[lo];
        for (size_t i = lo + 1; i < hi; ++i) {
            bmin = std::min(bmin, cb.ortho_base[i]);
            bmax = std::max(bmax, cb.ortho_base[i]);
            wmin = std::min(wmin, cb.ortho_weight[i]);
            wmax = std::max(wmax, cb.ortho_weight[i]);
            pmin = std::min(pmin, cb.ortho_perp[i]);
            pmax = std::max(pmax, cb.ortho_perp[i]);
        }
    }
    const float wamax = std::max(std::abs(wmin), std::abs(wmax));
    const float bsc = bmax > bmin ? (bmax - bmin) / 255.0F : 1.0F;
    const float wsc = wmax > wmin ? (wmax - wmin) / 255.0F : 1.0F;
    const float psc = pmax > pmin ? (pmax - pmin) / 255.0F : 1.0F;
    uint8_t* bq = bp + E8qIndex::kH3QOff;
    uint8_t* wq = bp + E8qIndex::kCqOff;
    uint8_t* pq = bp + E8qIndex::kAqOff;
    std::memset(bq, 0, E8qIndex::kHdrQBytes);
    for (size_t i = lo; i < hi; ++i) {
        const size_t lane = i - lo;
        bq[lane] = static_cast<uint8_t>(std::clamp<int>(
            static_cast<int>((cb.ortho_base[i] - bmin) / bsc), 0, 255
        ));
        wq[lane] = static_cast<uint8_t>(std::clamp<int>(
            std::lround((cb.ortho_weight[i] - wmin) / wsc), 0, 255
        ));
        // Perp enters the slack as an upper bound (dlo -= z*sqrt(d2)*perp),
        // so the bank rounds UP. Nearest + half-quantum pad would have the
        // same error law ([0,1) quantum) — ceil is kept as the simpler form.
        pq[lane] = static_cast<uint8_t>(std::clamp<int>(
            static_cast<int>(std::ceil((cb.ortho_perp[i] - pmin) / psc)), 0, 255
        ));
    }
    float* sc = reinterpret_cast<float*>(bp + E8qIndex::kScalesOff);
    sc[0] = bmin;
    sc[1] = bsc;
    sc[2] = wmin;
    sc[3] = wsc;
    sc[4] = pmin;
    sc[5] = psc;
    sc[6] = wamax;
}

}  // namespace


// One full build pass: encode, fit, pack the scan/refine arenas + the
// orthogonal-projection banks.
void E8qIndex::Build(
    const float* base,
    const float* centroids,
    const uint32_t* cluster_ids,
    size_t n,
    size_t dim,
    size_t num_clusters,
    int bits,
    uint64_t seed,
    size_t threads,
    const E8qBuildOptions& options
) {
    // The orthogonal-projection bound applies to EVERY bit budget with the
    // disclosed per-query prune-loss budget (union bound, no calibration).
    ortho_alpha_ = options.prune_alpha > 0.0F ? options.prune_alpha : 1.0F;
    // Projection-geometry ablation knob (README.md):
    // E8Q_PROJECTION=hilbert forces the bounded Hilbert projection instead
    // of the default orthogonal-residual (ZPR) weight. Default = zpr.
    const bool proj_hilbert = [] {
        const char* p = std::getenv("E8Q_PROJECTION");
        return p != nullptr && std::string(p) == "hilbert";
    }();
    if (bits < 1 || bits > kMaxQuantizeBits) {
        throw std::invalid_argument("E8Q bits must be in [1, 13]");
    }
    n_ = n;
    dim_ = dim;
    num_clusters_ = num_clusters;
    bits_ = bits;
    rotator_ = FhtKac4Rotator(dim, seed);
    padded_dim_ = rotator_.padded_dim();
    blocks_ = static_cast<int>(padded_dim_ / 8);

    std::vector<uint64_t> counts(num_clusters, 0);
    for (size_t i = 0; i < n; ++i) {
        if (cluster_ids[i] >= num_clusters) {
            throw std::invalid_argument("cluster id out of range");
        }
        ++counts[cluster_ids[i]];
    }
    std::vector<std::vector<uint32_t>> id_lists(num_clusters);
    for (size_t c = 0; c < num_clusters; ++c) {
        id_lists[c].reserve(counts[c]);
    }
    for (size_t i = 0; i < n; ++i) {
        id_lists[cluster_ids[i]].push_back(static_cast<uint32_t>(i));
    }

    // Coarse payload: fixed depth 1 — one rowA byte per block at EVERY rate
    // (digit top bits of dims 0..6, dim 7's stored bit 0 = branch at b == 1 /
    // top bit at b >= 2). One LUT, one stream, for every b.
    InitBitTables();

    cluster_begin_.assign(num_clusters + 1, 0);
    arena_begin_.assign(num_clusters + 1, 0);
    ref_arena_begin_.assign(num_clusters + 1, 0);
    // Scan record: [ortho header][rowA blocks*32]; refine record: [low
    // (b-1) bits per dim][g1][h3] — b bytes per vector per block in total
    // (rabitq parity) at every rate.
    batch_bytes_ = P1Off() + static_cast<size_t>(blocks_) * 32;
    batch_bytes_ = (batch_bytes_ + 63) & ~static_cast<size_t>(63);
    ref_batch_bytes_ = (RefRecordBytes() + 63) & ~static_cast<size_t>(63);
    for (size_t c = 0; c < num_clusters; ++c) {
        cluster_begin_[c + 1] = cluster_begin_[c] + counts[c];
        const uint64_t batches = (counts[c] + 31) / 32;
        arena_begin_[c + 1] = arena_begin_[c] + batches * batch_bytes_;
        ref_arena_begin_[c + 1] = ref_arena_begin_[c] + batches * ref_batch_bytes_;
    }

    const std::vector<float> tab = BuildRefineTables(bits, static_cast<int>(padded_dim_));
    refine_tab_.assign(tab.begin(), tab.end());
    // Fault the multi-MB buffers in as transparent huge pages. The
    // reserve -> madvise -> assign ordering is load-bearing; AllocHuge
    // (above) pins it.
    AllocHuge(centroid_rot_, num_clusters * padded_dim_, 0.0F);
    centroid_n2_.assign(num_clusters, 0.0F);
    // +64: tail pad so the refine decoder's constant 8-byte row loads can
    // overread the last row safely (bytes above c*8 are never used).
    AllocHuge(arena_, arena_begin_.back() + 64, uint8_t{0});
    AllocHuge(ref_arena_, ref_arena_begin_.back() + 64, uint8_t{0});
    ids_.assign(n, 0);

    const std::vector<double> levels[2] = {BranchLevels(bits, 0), BranchLevels(bits, 1)};
    // Per-vector encoding scale.  e8qlib::FixedScale reads a lazily
    // initialised schedule table behind a thread-safe-init guard, so calling
    // it once per vector costs a __tls_get_addr in the encode loop; the
    // value is a constant of the bit budget, so it is taken once here.
    const double s_vec = FixedScale(bits);
    // Level-to-unit-length conversion (1/rd) for the exact-code basis; a
    // constant of (bits, padded_dim_), so it is taken once for the whole build.
    const double inv_level_scale = 1.0 / std::sqrt(static_cast<double>(padded_dim_));
    double w_cls[2] = {0.0, 0.0};
    ScanClassMeans(bits, w_cls);

    const int nthreads =
        static_cast<int>(std::max<size_t>(1, std::min<size_t>(threads, num_clusters)));

    // Mean-centering: compute rot(mu) once and subtract it from every
    // rotated base/centroid/query vector. The shift is distance-preserving
    // but shrinks ||c|| and the per-cluster residual dynamic range, which is
    // what the coarse banks price and what the LUT quantization sees; on
    // unnormalized corpora this is worth +75..90% QPS at msong b4-8 (plus
    // recall) and 1-9% elsewhere, recall-neutral or better everywhere.
    {
        std::vector<double> mu(static_cast<size_t>(dim_), 0.0);
        for (size_t i = 0; i < n; ++i) {
            const float* x = base + i * static_cast<size_t>(dim_);
            for (int d = 0; d < dim_; ++d) {
                mu[static_cast<size_t>(d)] += static_cast<double>(x[d]);
            }
        }
        std::vector<float> mu_pad(padded_dim_, 0.0F);
        for (int d = 0; d < dim_; ++d) {
            mu_pad[static_cast<size_t>(d)] = static_cast<float>(mu[static_cast<size_t>(d)] / static_cast<double>(n));
        }
        mean_rot_.assign(padded_dim_, 0.0F);
        rotator_.Rotate(mu_pad.data(), mean_rot_.data());
    }

    const bool dump_ortho = std::getenv("E8Q_DUMP_ORTHO") != nullptr;
    uint64_t dump_n = 0;
    uint64_t dump_n_fallback = 0;
    uint64_t dump_n_cos_lo = 0;
    double dump_sum_abs_w = 0.0;
    double dump_sum_perp = 0.0;
    double dump_sum_cos = 0.0;
    double dump_max_abs_w = 0.0;
    double dump_max_perp = 0.0;

    // ---- encode --------------------------------------------------------
    // Rows are walked in FILE order, not in posting-list order. Posting-list
    // order reads the corpus one 4 KB row at a time from a scattered offset
    // (`base + gid * dim`), which turns the encode pass into a random-access
    // sweep: on a corpus larger than DRAM it is a page-fault storm, and even
    // resident it defeats the prefetcher. `list_rank` maps a row to its slot
    // inside its cluster's posting list, so a file-order walk writes every
    // byte exactly where the in-order walk put it — the index is byte-identical
    // and no recall column moves.
    std::vector<uint32_t> list_rank(n, 0);
    for (size_t c = 0; c < num_clusters; ++c) {
        const std::vector<uint32_t>& ids = id_lists[c];
        for (size_t k = 0; k < ids.size(); ++k) {
            list_rank[ids[k]] = static_cast<uint32_t>(k);
        }
    }
    // Orthogonal-bank inputs. The banks are per 32-vector batch and the
    // batches are posting-list order, so these three arrays are kept per
    // cluster until the file-order encode has filled them, then reduced in a
    // third pass. 12 B/row, against the ~(b+1) B/row the codes themselves cost.
    std::vector<std::vector<float>> ortho_base(num_clusters);
    std::vector<std::vector<float>> ortho_weight(num_clusters);
    std::vector<std::vector<float>> ortho_perp(num_clusters);
    for (size_t c = 0; c < num_clusters; ++c) {
        const size_t cnt = counts[c];
        ortho_base[c].assign(cnt, 0.0F);
        ortho_weight[c].assign(cnt, 0.0F);
        ortho_perp[c].assign(cnt, 0.0F);
    }

    // Rotated centroids. Split out of the encode pass so the encode can walk
    // rows in file order: the per-row residual needs rot(c) of its own cluster
    // only, and that is now a lookup instead of a per-cluster FHT.
    {
#pragma omp parallel num_threads(nthreads)
        {
            std::vector<float> rot_c(padded_dim_);
#pragma omp for schedule(static)
            for (long long cl = 0; cl < static_cast<long long>(num_clusters); ++cl) {
                const size_t c = static_cast<size_t>(cl);
                rotator_.Rotate(centroids + c * dim_, rot_c.data());
                for (size_t d = 0; d < padded_dim_; ++d) {
                    rot_c[d] -= mean_rot_[d];
                }
                std::memcpy(
                    &centroid_rot_[c * padded_dim_], rot_c.data(),
                    padded_dim_ * sizeof(float)
                );
                double cn2 = 0.0;
                for (size_t d = 0; d < padded_dim_; ++d) {
                    const double cd = static_cast<double>(rot_c[d]);
                    cn2 += cd * cd;
                }
                centroid_n2_[c] = static_cast<float>(cn2);
            }
        }
    }

#pragma omp parallel num_threads(nthreads) reduction(+ : dump_n, dump_n_fallback, dump_n_cos_lo, dump_sum_abs_w, dump_sum_perp, dump_sum_cos) reduction(max : dump_max_abs_w, dump_max_perp)
    {
        std::vector<float> rot_x(padded_dim_);
        std::vector<float> unit(padded_dim_);
        // Per-row staging for the two packed payloads. The record they belong
        // to is addressed by (cluster, rank), which a file-order walk hits in
        // no particular order; both are packed contiguously here and then
        // scattered into place one block at a time.
        std::vector<uint8_t> p1_row(static_cast<size_t>(blocks_));
        std::vector<uint8_t> refine_row(
            static_cast<size_t>(blocks_) * static_cast<size_t>(RefineBlockBytes(bits))
        );
        // One encode buffer per thread, reused across rows: the returned
        // VectorCode otherwise costs two heap allocations per vector.
        VectorCode vc_buf;
#pragma omp for schedule(dynamic, 256)
        for (long long ri = 0; ri < static_cast<long long>(n); ++ri) {
            const size_t row_id = static_cast<size_t>(ri);
            const size_t c = cluster_ids[row_id];
            const size_t rank = static_cast<size_t>(list_rank[row_id]);
            const size_t lane = rank % 32;
            const float* rot_c = &centroid_rot_[c * padded_dim_];
            uint8_t* rec =
                arena_.data() + arena_begin_[c] + (rank / 32) * batch_bytes_;
            uint8_t* rrec =
                ref_arena_.data() + ref_arena_begin_[c] + (rank / 32) * ref_batch_bytes_;
            std::vector<float>& ob_c = ortho_base[c];
            std::vector<float>& ow_c = ortho_weight[c];
            std::vector<float>& op_c = ortho_perp[c];
            // The row's posting slot, independent of how it was encoded.
            ids_[cluster_begin_[c] + rank] = static_cast<uint32_t>(row_id);

            rotator_.Rotate(base + row_id * static_cast<size_t>(dim_), rot_x.data());
            for (size_t d = 0; d < padded_dim_; ++d) {
                rot_x[d] -= mean_rot_[d];
            }

            double rn2 = 0.0;
            for (size_t d = 0; d < padded_dim_; ++d) {
                const double rv = static_cast<double>(rot_x[d]) - rot_c[d];
                rn2 += rv * rv;
            }
            const double rn = std::sqrt(rn2);
            if (rn < 1e-20) {
                // Point coincides with its centroid: distance is exactly d2.
                // Everything else — the payloads, g1/h3, the bank inputs —
                // stays at its zero fill, matching the in-order build.
                continue;
            }
            const double inv_rn = 1.0 / rn;
            for (size_t d = 0; d < padded_dim_; ++d) {
                unit[d] = static_cast<float>(
                    (static_cast<double>(rot_x[d]) - rot_c[d]) * inv_rn
                );
            }
            // Per-vector scale (`s_vec`, hoisted above), absorbed by g1
            // (no storage): the codec constant shapes the quantization
            // grid only; the estimator coefficient g1 (fitted below)
            // absorbs it exactly, so the scale never reaches storage or
            // the decode.
            EncodeVectorAtScaleInto(
                unit.data(), static_cast<int>(padded_dim_), bits, s_vec, vc_buf
            );
            const VectorCode& vc = vc_buf;

            double dot_ue = 0.0;  // <unit, v_hat_exact_raw>
            double n2_e = 0.0;    // ||v_hat_exact_raw||^2
            double dot_ce = 0.0;  // <rotated centroid, v_hat_exact_raw>
            double dot_ew = 0.0;  // <vE, w_scan> (orthogonal projection)
            double dot_uw = 0.0;  // <unit, w_scan>
            double n2_w = 0.0;    // ||w_scan||^2
            double dot_cw = 0.0;  // <rotated centroid, w_scan>
            for (int blk = 0; blk < blocks_; ++blk) {
                const std::array<uint16_t, 8>& dg = vc.digits[static_cast<size_t>(blk)];
                const int br = vc.branch[static_cast<size_t>(blk)];
                // NO per-vector scale here: the estimator coefficient g1
                // (fitted below as rn/<unit, vE>) absorbs it exactly, so
                // the scale never reaches storage or the decode.
                const double inv_scale = inv_level_scale;
                for (int k = 0; k < 8; ++k) {
                    const int digit = dg[static_cast<size_t>(k)];
                    const double lev = levels[br][static_cast<size_t>(digit)];
                    const double ve = lev * inv_scale;
                    const size_t idx = static_cast<size_t>(blk) * 8 + k;
                    // The scan is branch-blind at every bit budget: it
                    // prices the digit's top-bit class at the
                    // branch-averaged class mean w_cls (the lut_c model).
                    // At b == 1 the byte's dim-7 slot carries the BRANCH
                    // (the parity sign is absorbed by v_perp), so the fit
                    // basis must be exactly what the scan prices.
                    // b == 1 dim 7 is the branch bit, not the parity
                    // sign: drop it from the scan basis (matches
                    // InitBitTables). Dims 0..6 still price the MSB.
                    const double wc =
                        bits == 1 && k == 7 ? 0.0 : w_cls[digit >> (bits - 1)];
                    dot_ue += static_cast<double>(unit[idx]) * ve;
                    dot_ce += static_cast<double>(rot_c[idx]) * ve;
                    n2_e += ve * ve;
                    dot_ew += ve * wc;
                    dot_uw += static_cast<double>(unit[idx]) * wc;
                    n2_w += wc * wc;
                    dot_cw += static_cast<double>(rot_c[idx]) * wc;
                }
                if (bits != 1) {
                    PackRefineBlock(
                        dg.data(), br, bits,
                        refine_row.data() + blk * RefineBlockBytes(bits)
                    );
                }
            }
            for (int blk = 0; blk < blocks_; ++blk) {
                PackCoarseRow(
                    vc.digits[static_cast<size_t>(blk)].data(),
                    vc.branch[static_cast<size_t>(blk)], bits, blk, p1_row.data()
                );
            }
            // Both packed payloads are contiguous in the staging buffers
            // and lane-interleaved in the records: the rowA byte goes to
            // slot `lane` of its block's 32-lane group, the refine bytes to
            // the lane's (b-1)-byte run starting at lane*blocks*(b-1).
            for (int blk = 0; blk < blocks_; ++blk) {
                rec[P1Off() + static_cast<size_t>(blk) * 32 + lane] =
                    p1_row[static_cast<size_t>(blk)];
            }
            if (bits != 1) {
                // The refine record's lane runs are contiguous and in the
                // same order PackRefineBlock emits them, so the whole row
                // is one copy: lane run = blocks*(b-1) bytes at lane*run.
                const size_t run = static_cast<size_t>(blocks_) *
                                   static_cast<size_t>(RefineBlockBytes(bits));
                std::memcpy(
                    rrec + RefRefineOff() + lane * run, refine_row.data(), run
                );
            }
            double g1_fit = 0.0;
            double h3_fit = 0.0;
            if (n2_e > 1e-30 && dot_ue > 1e-12) {
                // Dot-normalized ("spherical") fit: g1 = rn / <u, vE>.
                g1_fit = rn / dot_ue;
                h3_fit = rn2 + 2.0 * g1_fit * dot_ce;
            }
            float* const g1b = reinterpret_cast<float*>(rrec + RefG1Off());
            float* const h3b = reinterpret_cast<float*>(rrec + RefH3Off());
            g1b[lane] = static_cast<float>(g1_fit);
            h3b[lane] = static_cast<float>(h3_fit);
            // Zero-parallel residual: W_x = 2*rn/<u,w_scan> makes
            // <u, R> == 0, so the NN (eta→0) has zero radial bias.
            // Compensation rides on W (fires when S is large).
            // |<u,w>| below a numerical eps cannot invert W_x; then
            // fall back to W_opt = 2 g1 <vE,w>/||w||^2. A 0.05-cos
            // floor is a no-op on 1-bit codes (cos ~ sqrt(2/pi)).
            const double g1v = static_cast<double>(g1b[lane]);
            const double sqrt_dm1 =
                std::sqrt(static_cast<double>(std::max<size_t>(dim_ - 1, 1)));
            const double n2_w_safe = std::max(n2_w, 1e-30);
            const double uw_eps = 1e-12 * std::sqrt(n2_w_safe);
            double w_use;
            double perp_use;
            if (!proj_hilbert && std::abs(dot_uw) > uw_eps) {
                const double wx = 2.0 * rn / dot_uw;
                const double rx2 =
                    4.0 * g1v * g1v * n2_e + wx * wx * n2_w - 4.0 * g1v * wx * dot_ew;
                w_use = wx;
                perp_use = std::sqrt(std::max(0.0, rx2)) / sqrt_dm1;
            } else {
                const double gamma = dot_ew / n2_w_safe;
                const double vp2 = std::max(n2_e - gamma * gamma * n2_w, 0.0);
                w_use = 2.0 * g1v * gamma;
                perp_use = 2.0 * g1v * std::sqrt(vp2) / sqrt_dm1;
            }
            ow_c[rank] = static_cast<float>(w_use);
            ob_c[rank] = static_cast<float>(rn2 + w_use * dot_cw);
            op_c[rank] = static_cast<float>(perp_use);
            if (dump_ortho) {
                const double nw = std::sqrt(n2_w_safe);
                const double cos_uw = (nw > 0.0) ? (dot_uw / nw) : 0.0;
                ++dump_n;
                dump_sum_abs_w += std::abs(w_use);
                dump_sum_perp += perp_use;
                dump_sum_cos += cos_uw;
                dump_max_abs_w = std::max(dump_max_abs_w, std::abs(w_use));
                dump_max_perp = std::max(dump_max_perp, perp_use);
                if (std::abs(dot_uw) <= uw_eps) {
                    ++dump_n_fallback;
                }
                if (std::abs(cos_uw) < 0.05) {
                    ++dump_n_cos_lo;
                }
            }
        }
    }

    // ---- scan banks ----------------------------------------------------
    // Per 32-vector batch and in posting-list order, so they are reduced from
    // the per-cluster inputs the file-order encode pass filled. base rounds
    // down, weight nearest, perp up: every rounding direction pushes dlo down,
    // so the bound stays sound.
#pragma omp parallel num_threads(nthreads)
    {
#pragma omp for schedule(static)
        for (long long cl = 0; cl < static_cast<long long>(num_clusters); ++cl) {
            const size_t c = static_cast<size_t>(cl);
            const size_t cnt = counts[c];
            ClusterBuffers cb;
            cb.ortho_base = std::move(ortho_base[c]);
            cb.ortho_weight = std::move(ortho_weight[c]);
            cb.ortho_perp = std::move(ortho_perp[c]);
            uint8_t* arena_c = arena_.data() + arena_begin_[c];
            const size_t nbatches = (cnt + 31) / 32;
            for (size_t b = 0; b < nbatches; ++b) {
                uint8_t* bp = arena_c + b * batch_bytes_;
                const size_t lo = b * 32;
                const size_t hi = std::min(cnt, lo + 32);
                WriteOrthoBanks(cb, lo, hi, bp);
            }
        }
    }
    if (dump_ortho) {
        const double n = static_cast<double>(std::max<uint64_t>(dump_n, 1));
        std::fprintf(
            stderr,
            "E8Q_DUMP_ORTHO bits=%d n=%llu fallback=%llu cos<0.05=%llu "
            "mean|W|=%.4g max|W|=%.4g mean_Omega=%.4g max_Omega=%.4g mean_cos=%.4g\n",
            bits_, static_cast<unsigned long long>(dump_n),
            static_cast<unsigned long long>(dump_n_fallback),
            static_cast<unsigned long long>(dump_n_cos_lo), dump_sum_abs_w / n,
            dump_max_abs_w, dump_sum_perp / n, dump_max_perp, dump_sum_cos / n
        );
    }
}

}  // namespace e8qlib