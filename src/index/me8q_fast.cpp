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

#include "index/e8q_internal.hpp"
#include "simd/me8q_prune.hpp"
#include "e8qlib/utils/aligned_alloc.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace e8qlib::me8q {
// P5 v2: two-level scan (top-bit plane + refine), mixed-rate, with the
// parent E8Q ZPR bound per run:
//   W_x = 2*rn/<u,w_scan> so <u,R>==0; base = rn^2 + W_x <c',w> (never
//   subtracts b_i); Omega = ||R_x||/sqrt(L_r-1); slack =
//   lambda * r_horizon_r * Omega with r_horizon_r = min(sqrt(d2_r),
//   sqrt(distk0)) and lambda = PhiInv(1 - alpha/k_eff), k_eff = max(4k, 32).
//   At rate 1 dim 7 of the scan byte is the BRANCH: w_scan[7] = 0.
// ---------------------------------------------------------------------------

void Me8qIndex::BuildFastLayout() {
    fast_.assign(num_clusters_, FastLayout{});
    for (int b = 1; b <= kMaxBits; ++b) {
        fast_cls_[static_cast<size_t>(b)] = codec_.ClassModelOf(b);
    }
    const int nthreads = static_cast<int>(std::max<size_t>(1, opt_.threads));
#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int64_t ci = 0; ci < static_cast<int64_t>(num_clusters_); ++ci) {
        const size_t c = static_cast<size_t>(ci);
        FastLayout& f = fast_[c];
        const ListLayout& lay = layouts_[c];
        f.cnt = list_begin_[c + 1] - list_begin_[c];
        for (size_t s = 0; s < lay.segs.size(); ++s) {
            const SegSlot& slot = lay.segs[s];
            f.run_rate.push_back(slot.bits);
            f.run_blk0.push_back(static_cast<uint32_t>(f.blocks_total));
            f.run_nblk.push_back(slot.len / 8);
            f.run_dim_off.push_back(slot.dim_off);
            f.run_len.push_back(slot.len);
            f.block_run.insert(
                f.block_run.end(), slot.len / 8, static_cast<uint8_t>(s)
            );
            f.blocks_total += slot.len / 8;
        }
        f.nrun = lay.segs.size();
        f.plane_cols = f.blocks_total;
        f.batches = (f.cnt + 31) / 32;
        f.plane_bytes = f.batches * (f.plane_cols * 32 + f.nrun * f.bank_run_bytes);
        f.refine_run_off.assign(f.nrun, 0);
        size_t off = f.nrun * 2 * sizeof(float);
        for (size_t r = 0; r < f.nrun; ++r) {
            f.refine_run_off[r] = static_cast<uint32_t>(off);
            const int rr = static_cast<int>(f.run_rate[r]);
            if (rr >= 2) {
                off += static_cast<size_t>(rr - 1) * f.run_len[r] / 8;
            }
        }
        f.refine_stride = off;
        // Per-run centroid-slice energy (the per-query d2_r needs it).
        f.rc_n2.assign(f.nrun, 0.0);
        for (size_t r = 0; r < f.nrun; ++r) {
            const float* rc = cent_rot_.data() + c * padded_dim_ + f.run_dim_off[r];
            double a = 0.0;
            for (uint32_t t = 0; t < f.run_len[r]; ++t) {
                const double v = static_cast<double>(rc[t]);
                a += v * v;
            }
            f.rc_n2[r] = a;
        }

        // Split each stored run code into plane bytes + refine bytes, and fit
        // the orthogonal-projection banks per (vector, run).
        std::vector<uint8_t> prow(f.cnt * f.plane_cols, 0);
        f.refine.assign(f.cnt * f.refine_stride, 0);
        std::vector<float> fac_coef(f.nrun * f.cnt, 0.0F);
        std::vector<float> fac_h2(f.nrun * f.cnt, 0.0F);
        f.ortho_gain.assign(f.nrun * f.cnt, 0.0F);
        f.ortho_vpc.assign(f.nrun * f.cnt, 0.0F);
        f.ortho_perp.assign(f.nrun * f.cnt, 0.0F);
        std::vector<uint16_t> digits;
        std::vector<uint8_t> branch;
        for (size_t i = 0; i < f.cnt; ++i) {
            const uint8_t* row =
                codes_.data() + byte_begin_[c] + i * lay.stride;
            uint8_t* rrow = f.refine.data() + i * f.refine_stride;
            for (size_t r = 0; r < f.nrun; ++r) {
                std::memcpy(rrow + r * 2 * sizeof(float), row + lay.segs[r].factor_off, 8);
                const float* fac = reinterpret_cast<const float*>(row + lay.segs[r].factor_off);
                fac_coef[r * f.cnt + i] = fac[0];
                fac_h2[r * f.cnt + i] = fac[1];
            }
            for (size_t r = 0; r < f.nrun; ++r) {
                const int bits = static_cast<int>(f.run_rate[r]);
                const int len = static_cast<int>(f.run_len[r]);
                const int nblk = len / 8;
                digits.assign(static_cast<size_t>(len), 0);
                branch.assign(static_cast<size_t>(nblk), 0);
                UnpackSegment(
                    row + lay.segs[r].code_off, len, bits, digits.data(), branch.data()
                );
                SegCode code;
                code.digits = digits;
                code.branch = branch;
                if (bits >= 2) {
                    PackSegment(code, len, bits - 1, rrow + f.refine_run_off[r]);
                }
                uint8_t* pb = prow.data() + i * f.plane_cols + f.run_blk0[r];
                for (int blk = 0; blk < nblk; ++blk) {
                    uint8_t byte = 0;
                    for (int t = 0; t < 8; ++t) {
                        const int d = digits[static_cast<size_t>(blk) * 8 + t];
                        // r >= 2: digit top bit per dim. r = 1: the sign bits
                        // of dims 0..6; bit 0 = branch (dim 7's sign is the
                        // E8 parity, absorbed by v_perp — same byte layout as
                        // the parent's b == 1).
                        const int top =
                            bits == 1 && t == 7
                                ? (branch[static_cast<size_t>(blk)] & 1)
                                : ((d >> (bits - 1)) & 1);
                        if (top != 0) {
                            byte = static_cast<uint8_t>(byte | (1u << (7 - t)));
                        }
                    }
                    pb[blk] = byte;
                }
                // ZPR fit per (vector, run) on the scan's own basis: w =
                // pbar[top-bit class]. Rate-1 dim 7 is the BRANCH, not a
                // coordinate — zero it (parent InitBitTables / Build).
                const int rate = bits;
                const SegmentCodec::ClassModel& cm = fast_cls_[static_cast<size_t>(rate)];
                double dot_ew = 0.0;   // <vE, w>
                double n2_w = 0.0;     // ||w||^2
                double n2_e = 0.0;     // ||vE||^2
                double dot_ce = 0.0;   // <rc_r, vE>
                double dot_cw = 0.0;   // <rc_r, w>
                const float* rc = cent_rot_.data() + c * padded_dim_ + f.run_dim_off[r];
                for (int blk = 0; blk < nblk; ++blk) {
                    const int br = branch[static_cast<size_t>(blk)];
                    const float* lev = codec_.Levels(rate, br);
                    for (int t = 0; t < 8; ++t) {
                        const int d = digits[static_cast<size_t>(blk) * 8 + t];
                        const double ve = static_cast<double>(lev[d]);
                        const double wc =
                            rate == 1 && t == 7
                                ? 0.0
                                : static_cast<double>(cm.pbar[d >> (rate - 1)]);
                        dot_ew += ve * wc;
                        n2_w += wc * wc;
                        n2_e += ve * ve;
                        dot_ce += static_cast<double>(rc[static_cast<size_t>(blk) * 8 + t]) * ve;
                        dot_cw += static_cast<double>(rc[static_cast<size_t>(blk) * 8 + t]) * wc;
                    }
                }
                const double coef = static_cast<double>(fac_coef[r * f.cnt + i]);
                const double h2 = static_cast<double>(fac_h2[r * f.cnt + i]);
                const double rn2 = std::max(h2 - 2.0 * coef * dot_ce, 0.0);
                const double rn = std::sqrt(rn2);
                const double sqrt_lm1 = std::sqrt(static_cast<double>(
                    std::max<uint32_t>(f.run_len[r] > 0 ? f.run_len[r] - 1 : 1, 1)
                ));
                const double abs_c = std::fabs(coef);
                if (n2_e > 1e-30 && abs_c > 1e-30) {
                    // Fill-pass has packed codes only: u := vE/||vE|| stands
                    // in for the residual unit (they coincide up to the
                    // per-run spherical-fit scale). Invertibility eps only
                    // — a 0.05-cos floor explodes W_x and is a no-op on
                    // 1-bit codes (cos ~ sqrt(2/pi)).
                    const double ve_norm = std::sqrt(n2_e);
                    const double n2_w_safe = std::max(n2_w, 1e-30);
                    const double dot_uw = dot_ew / ve_norm;
                    const double uw_eps = 1e-12 * std::sqrt(n2_w_safe);
                    double wx;
                    double norm_rx;
                    if (std::abs(dot_uw) > uw_eps) {
                        wx = 2.0 * rn / dot_uw;
                        const double rx2 = 4.0 * coef * coef * n2_e + wx * wx * n2_w -
                                           4.0 * coef * wx * dot_ew;
                        norm_rx = std::sqrt(std::max(0.0, rx2));
                    } else {
                        const double gamma = dot_ew / n2_w_safe;
                        wx = 2.0 * coef * gamma;
                        const double vp2 = std::max(n2_e - gamma * gamma * n2_w, 0.0);
                        norm_rx = 2.0 * abs_c * std::sqrt(vp2);
                    }
                    // Pack as (gain, vpc) so the cascade
                    //   h2 - 2*coef*(vpc + gain*S)
                    // is exact. For coef > 0 this is W_x and
                    // rn^2 + W_x <c',w> (NO b_i).
                    f.ortho_gain[r * f.cnt + i] =
                        static_cast<float>(wx / (2.0 * abs_c));
                    f.ortho_vpc[r * f.cnt + i] = static_cast<float>(
                        (h2 - rn2 - wx * dot_cw) / (2.0 * abs_c)
                    );
                    f.ortho_perp[r * f.cnt + i] = static_cast<float>(
                        (norm_rx / sqrt_lm1) / (2.0 * abs_c)
                    );
                } else if (n2_w > 1e-30) {
                    const double gamma = dot_ew / n2_w;
                    const double vp2 = std::max(n2_e - gamma * gamma * n2_w, 0.0);
                    const double vp_c = dot_ce - gamma * dot_cw;
                    f.ortho_gain[r * f.cnt + i] = static_cast<float>(gamma);
                    f.ortho_vpc[r * f.cnt + i] = static_cast<float>(vp_c);
                    f.ortho_perp[r * f.cnt + i] =
                        static_cast<float>(std::sqrt(vp2) / sqrt_lm1);
                }
            }
        }
        // Byte-per-lane batch rows (col*32 + lane), the parent E8Q rowA
        // layout: the scan then shares the parent's AccumulateCoarseHi
        // kernels and the cascade reads its row bytes directly (no
        // PlaneByteAt de-interleave). Trailing each batch: per-run ortho
        // u8 banks (base floor / weight nearest / perp ceil; 4th 32 B
        // unused, kept so bank_run_bytes stays 160).
        f.plane.assign(f.plane_bytes, 0);
        const size_t rec = f.plane_cols * 32 + f.nrun * f.bank_run_bytes;
        for (size_t b = 0; b < f.batches; ++b) {
            const size_t row0 = b * 32;
            uint8_t* rec_base = f.plane.data() + b * rec;
            for (size_t col = 0; col < f.plane_cols; ++col) {
                uint8_t* dst = rec_base + col * 32;
                for (size_t lane = 0; lane < 32; ++lane) {
                    const size_t row = row0 + lane;
                    dst[lane] = row < f.cnt
                                    ? prow[row * f.plane_cols + col]
                                    : 0;
                }
            }
            uint8_t* banks = rec_base + f.plane_cols * 32;
            const size_t hi = std::min<size_t>(f.cnt, row0 + 32);
            for (size_t r = 0; r < f.nrun; ++r) {
                uint8_t* bp = banks + r * f.bank_run_bytes;
                float vmin[3], vmax[3];
                bool any = false;
                float vals[32][3];
                for (size_t lane = row0; lane < hi; ++lane) {
                    const size_t i = lane - row0;
                    const float coef = fac_coef[r * f.cnt + lane];
                    const float h2 = fac_h2[r * f.cnt + lane];
                    vals[i][0] = h2 - 2.0F * coef * f.ortho_vpc[r * f.cnt + lane];
                    vals[i][1] = 2.0F * coef * f.ortho_gain[r * f.cnt + lane];
                    vals[i][2] = f.ortho_perp[r * f.cnt + lane] * 2.0F * std::fabs(coef);
                    if (!any) {
                        vmin[0] = vmax[0] = vals[i][0];
                        vmin[1] = vmax[1] = vals[i][1];
                        vmin[2] = vmax[2] = vals[i][2];
                        any = true;
                    } else {
                        for (int k = 0; k < 3; ++k) {
                            vmin[k] = std::min(vmin[k], vals[i][k]);
                            vmax[k] = std::max(vmax[k], vals[i][k]);
                        }
                    }
                }
                if (!any) {
                    vmin[0] = vmax[0] = vmin[1] = vmax[1] = vmin[2] = vmax[2] = 0.0F;
                }
                const float sc[3] = {
                    vmax[0] > vmin[0] ? (vmax[0] - vmin[0]) / 255.0F : 1.0F,
                    vmax[1] > vmin[1] ? (vmax[1] - vmin[1]) / 255.0F : 1.0F,
                    vmax[2] > vmin[2] ? (vmax[2] - vmin[2]) / 255.0F : 1.0F,
                };
                std::memset(bp, 0, f.bank_run_bytes);
                for (size_t lane = row0; lane < hi; ++lane) {
                    const size_t i = lane - row0;
                    bp[i] = static_cast<uint8_t>(std::clamp<int>(
                        static_cast<int>((vals[i][0] - vmin[0]) / sc[0]), 0, 255
                    ));
                    bp[32 + i] = static_cast<uint8_t>(std::clamp<int>(
                        std::lround((vals[i][1] - vmin[1]) / sc[1]), 0, 255
                    ));
                    bp[64 + i] = static_cast<uint8_t>(std::clamp<int>(
                        static_cast<int>(std::ceil((vals[i][2] - vmin[2]) / sc[2])), 0, 255
                    ));
                    // bp[96 + i] stays 0: 4th u8 bank is layout padding (unused by prune).
                }
                float* scl = reinterpret_cast<float*>(bp + 128);
                scl[0] = vmin[0];
                scl[1] = sc[0];
                scl[2] = vmin[1];
                scl[3] = sc[1];
                scl[4] = vmin[2];
                scl[5] = sc[2];
                scl[6] = 0.0F;
                scl[7] = 0.0F;
            }
        }
        // Perp is packed into the u8 bank; search cascade only rereads
        // ortho_gain / ortho_vpc (exact-score undo of the coarse S).
        f.ortho_perp.clear();
        f.ortho_perp.shrink_to_fit();
    }
    fast_ready_ = true;
}

// ---------------------------------------------------------------------------
// Routing: exact fp32 L2 over the plan's coded prefix. The single-query path
// below sweeps the centroid matrix once per query; SearchIVFFastBatch shares
// one sweep across a group of queries (ivf::RouteL2Group) whenever the caller
// hands it more than one.

void Me8qIndex::RouteQuery(
    const float* q, size_t nprobe, std::vector<std::pair<float, uint32_t>>* cd
) const {
    const size_t take = std::min(nprobe, num_clusters_);
    // Per-query scratch reused across queries (the per-query allocation of
    // these four vectors dominated short-query routing).
    thread_local std::vector<float> dist;
    thread_local std::vector<uint32_t> top;
    thread_local std::vector<uint64_t> keys;
    dist.resize(num_clusters_);
    top.resize(take);
    keys.resize(num_clusters_);
    // `q` is already padded+rotated (SearchIVFFast rotates once).
    ivf::RouteL2(
        q, cent_rot_.data(), centroid_n2_.data(), num_clusters_, route_dim_,
        take, dist.data(), top.data(), keys.data(), padded_dim_
    );
    // Sorted probe list only (the parent E8Q route shape: the keys array is
    // already partial-sorted by RouteL2, so fill cd from it instead of
    // re-sorting the full pair vector).
    cd->resize(take);
    for (size_t p = 0; p < take; ++p) {
        (*cd)[p] = {dist[top[p]], top[p]};
    }
}
void Me8qIndex::SearchIVFFast(
    const float* query, size_t k, size_t nprobe, uint32_t* out_ids, SearchStats* stats,
    float* out_dists
) const {
    if (k == 0 || nprobe == 0 || n_ == 0 || !fast_ready_) {
        if (out_ids != nullptr) {
            for (size_t i = 0; i < k; ++i) {
                out_ids[i] = std::numeric_limits<uint32_t>::max();
            }
        }
        if (out_dists != nullptr) {
            for (size_t i = 0; i < k; ++i) {
                out_dists[i] = std::numeric_limits<float>::infinity();
            }
        }
        return;
    }
    // Phase timers, same four buckets the parent's ANN ledger reports
    // (rotate/route/scan/refine), so the two engines can be compared phase by
    // phase. Phase timing is opt-in (ME8Q_PHASES=1); ungated it cost 19% of a
    // deep1M b1 query, and reported QPS must not pay for the instrumentation.
    static const bool phases_on = std::getenv("ME8Q_PHASES") != nullptr;
    const auto clock_now = [] {
        return phases_on ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point{};
    };
    const auto since_us = [](auto t0, auto t1) {
        return phases_on
                   ? std::chrono::duration<double, std::micro>(t1 - t0).count()
                   : 0.0;
    };
    double t_rot = 0.0;
    const auto t_route0 = clock_now();
    // Per-query scratch, reused across queries (see RouteQuery): the
    // per-query allocations below were ~29% of a short query.
    thread_local std::vector<float> qpad;
    thread_local std::vector<std::pair<float, uint32_t>> cd;
    // 64B-aligned: the route GEMV streams rot_q with line-sized loads (the
    // e8q A1 alignment fix).
    thread_local AlignedVector<float> rot_q;
    PadQuery(query, &qpad);
    rot_q.assign(padded_dim_, 0.0F);
    {
        const auto t_prep = clock_now();
        PrepareRotatedQuery(qpad.data(), rot_q.data());
        t_rot += since_us(t_prep, clock_now());
    }
    cd.clear();
    RouteQuery(rot_q.data(), nprobe, &cd);  // already partial_sorts the first `take`
    const double route_us = since_us(t_route0, clock_now());
    ScanPreRouted(
        rot_q.data(), cd.data(), cd.size(), k, out_ids, stats, out_dists, route_us, t_rot
    );
}

// Batched FastScan: same routing arithmetic, same scan, same results query
// for query as SearchIVFFast, but the route sweeps the centroid matrix once
// per group of queries (ivf::RouteL2Group) instead of once per query and the
// three phases run as separate parallel regions.
//
// The re-sweep per query is what made routing cost ~28% of a deep1M b4 query:
// the scan streams the index through L3, so the per-query sweep of the 2 MiB
// centroid matrix comes from DRAM. Grouping 8 queries per sweep divides that
// traffic by 8.
namespace {
constexpr size_t kRouteGroup = 8;  // queries per centroid-matrix sweep
}  // namespace

void Me8qIndex::SearchIVFFastBatch(
    const float* queries, size_t nq, size_t k, size_t nprobe, uint32_t* out_ids,
    SearchStats* stats, float* out_dists, int num_threads
) const {
    if (nq == 0) {
        return;
    }
    if (k == 0 || nprobe == 0 || n_ == 0 || !fast_ready_) {
        for (size_t q = 0; q < nq; ++q) {
            for (size_t i = 0; i < k; ++i) {
                out_ids[q * k + i] = std::numeric_limits<uint32_t>::max();
                if (out_dists != nullptr) {
                    out_dists[q * k + i] = std::numeric_limits<float>::infinity();
                }
            }
        }
        return;
    }
    const size_t take = std::min(nprobe, num_clusters_);
    const int threads = std::max(1, num_threads);
    const auto t0 = std::chrono::steady_clock::now();
    const auto wall_us = [](auto a, auto b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };
    // Rotated queries, one 64B-aligned row per query (the A1 route fix).
    const size_t rot_stride = (padded_dim_ + 15) & ~size_t{15};
    AlignedVector<float> rot_scratch(nq * rot_stride, 0.0F);
    // Probe lists, filled by the route phase and consumed by the scan:
    // (distance, cluster) exactly as RouteQuery builds them.
    std::vector<std::pair<float, uint32_t>> cd_all(nq * take);
    std::vector<uint32_t> tops(nq * take);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
    for (int64_t qi = 0; qi < static_cast<int64_t>(nq); ++qi) {
        thread_local std::vector<float> qpad;
        PadQuery(queries + static_cast<size_t>(qi) * dim_, &qpad);
        PrepareRotatedQuery(
            qpad.data(), rot_scratch.data() + static_cast<size_t>(qi) * rot_stride
        );
    }
    const double rot_us_total = wall_us(t0, std::chrono::steady_clock::now());
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
        thread_local std::vector<float> dist_b;
        thread_local std::vector<uint64_t> keys_b;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (size_t g0 = 0; g0 < nq; g0 += kRouteGroup) {
            const size_t gn = std::min(kRouteGroup, nq - g0);
            dist_b.resize(gn * num_clusters_);
            keys_b.resize(gn * num_clusters_);
            // Sized by the constant it must match: RouteL2Group reads exactly
            // gn pointers, so growing kRouteGroup cannot outrun the array.
            std::array<const float*, kRouteGroup> qs{};
            for (size_t i = 0; i < gn; ++i) {
                qs[i] = rot_scratch.data() + (g0 + i) * rot_stride;
            }
            ivf::RouteL2Group(
                qs.data(), gn, cent_rot_.data(), centroid_n2_.data(), num_clusters_,
                route_dim_, take, dist_b.data(), keys_b.data(), tops.data() + g0 * take,
                padded_dim_
            );
            // cd straight off the route output, exactly as RouteQuery fills
            // it (`{dist[top[p]], top[p]}`).
            for (size_t i = 0; i < gn; ++i) {
                const float* di = dist_b.data() + i * num_clusters_;
                const uint32_t* ti = tops.data() + (g0 + i) * take;
                for (size_t p = 0; p < take; ++p) {
                    cd_all[(g0 + i) * take + p] = {di[ti[p]], ti[p]};
                }
            }
        }
    }
    const double route_us_share = wall_us(t0, std::chrono::steady_clock::now()) / static_cast<double>(nq);
    SearchStats acc{};
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
        SearchStats local{};
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (int64_t qi = 0; qi < static_cast<int64_t>(nq); ++qi) {
            const size_t q = static_cast<size_t>(qi);
            ScanPreRouted(
                rot_scratch.data() + q * rot_stride, cd_all.data() + q * take, take, k,
                out_ids + q * k, &local,
                out_dists != nullptr ? out_dists + q * k : nullptr, route_us_share,
                rot_us_total / static_cast<double>(nq)
            );
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            acc.candidates += local.candidates;
            acc.clusters += local.clusters;
            acc.survivors += local.survivors;
            acc.bits_scored += local.bits_scored;
            acc.bits_total += local.bits_total;
            acc.pruned += local.pruned;
            acc.rotate_us += local.rotate_us;
            acc.route_us += local.route_us;
            acc.scan_us += local.scan_us;
            acc.refine_us += local.refine_us;
            acc.lut_us += local.lut_us;
            acc.filter_us += local.filter_us;
            acc.lane_us += local.lane_us;
            acc.rest_us += local.rest_us;
        }
    }
    if (stats != nullptr) {
        stats->candidates += acc.candidates;
        stats->clusters += acc.clusters;
        stats->survivors += acc.survivors;
        stats->bits_scored += acc.bits_scored;
        stats->bits_total += acc.bits_total;
        stats->pruned += acc.pruned;
        stats->rotate_us += acc.rotate_us;
        stats->route_us += acc.route_us;
        stats->scan_us += acc.scan_us;
        stats->refine_us += acc.refine_us;
        stats->lut_us += acc.lut_us;
        stats->filter_us += acc.filter_us;
        stats->lane_us += acc.lane_us;
        stats->rest_us += acc.rest_us;
    }
}

// Scan phase of SearchIVFFast: everything after routing, for one query.
// `rot_qv` is the padded, rotated query, `cd`/`take` its probe list, and
// `route_us`/`rot_us` the phase times measured by the caller (the batch
// entry point divides its phase totals by nq).
void Me8qIndex::ScanPreRouted(
    const float* rot_qv, const std::pair<float, uint32_t>* cd, size_t take, size_t k,
    uint32_t* out_ids, SearchStats* stats, float* out_dists, double route_us, double rot_us
) const {
    double t_lut = 0.0;
    double t_filter = 0.0;
    double t_lane = 0.0;
    double t_scan = 0.0;
    double t_refine = 0.0;
    double t_rest = 0.0;
    // Phase timers, same buckets the parent's ANN ledger reports; opt-in
    // (ME8Q_PHASES=1) because the ungated clocks cost 19% of a short query.
    static const bool phases_on = std::getenv("ME8Q_PHASES") != nullptr;
    const auto clock_now = [] {
        return phases_on ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point{};
    };
    const auto since_us = [](auto t0, auto t1) {
        return phases_on
                   ? std::chrono::duration<double, std::micro>(t1 - t0).count()
                   : 0.0;
    };
    thread_local std::vector<std::pair<float, uint32_t>> heap;
    thread_local std::vector<float> lut_f;
    thread_local std::vector<uint8_t> lut_u8;
    // Pre-permuted pair LUT (the e8q A4 fix): built once per query in
    // prepare_list so the scan's hot loop does plain loads instead of
    // rebuilding PairLut512Lo/Hi registers every block pair.
    thread_local AlignedVector<uint8_t> lut_pair;
    // 64B-aligned: the coarse accumulator is written by vmovdqa32 in
    // UnpackAcc256Add; a plain std::vector only guarantees 16B and faults
    // on the aligned store (this was the deep1M b4/b8 SIGSEGV).
    thread_local AlignedVector<int32_t> acc;
    thread_local std::vector<double> run_qn2;
    thread_local std::vector<uint32_t> run_bits;
    thread_local std::vector<double> s_c;
    thread_local std::vector<double> t_r;
    thread_local std::vector<double> score_max_r;
    thread_local std::vector<double> lut_min_x;
    thread_local std::vector<double> qerr_r;
    thread_local std::vector<double> sqrt_d2_r;
    // FWER over the top-k, not over every scanned point (parent E8Q ZPR).
    // k_eff = max(4k, 32) leaves a kurtosis cushion; lambda stays ~3.3-3.5
    // at alpha=0.01 instead of growing with nprobe.
    const size_t effective_k = std::max<size_t>(k * 4, 32);
    const float lambda = e8qlib::detail::OrthoQuantile(
        static_cast<double>(ortho_alpha_), effective_k
    );

    heap.clear();
    heap.reserve(k + 1);
    size_t candidates = 0;
    size_t survivors_total = 0;
    size_t bits_scored = 0;
    size_t bits_total = 0;
    thread_local std::vector<size_t> order;
    // Coarse-score cache filled by the vectorized filter and consumed by the
    // scalar path for the survivors it keeps: recomputing the same score in
    // fp64 for survivors was pure duplication (the filter computes dc, the
    // pooled slack energy and the per-run terms for every lane anyway).
    thread_local std::vector<float> lane_dc;
    thread_local std::vector<float> lane_pool;
    thread_local std::vector<float> lane_qa;
    thread_local std::vector<float> lane_sc;  // [run][lane]
    thread_local std::vector<float> lane_tr;  // [run][lane]
    // Lanes the filter could not prune, compacted: the scalar loop then walks
    // only survivors instead of branching over all 32 lanes of every batch.
    thread_local std::vector<uint32_t> keep_lanes;
    // Global plan => every list shares the plan, the per-run-length rotation
    // matrices, hence rot_q and the ENTIRE LUT: build once per query and
    // reuse across all nprobe lists (per-cluster plans rebuild per list).
    const bool uniform_layout = true;
    bool lut_ready = false;
    float lut_min = 0.0F;
    float lut_delta = 1.0F;
    size_t bits_row = 0;
    // Per-run constants hoisted out of the lane loop (bit-identical values:
    // same expressions, evaluated once per list instead of once per lane).

    auto prepare_list = [&](const FastLayout& f, size_t /*cidx*/) {
        const auto t_lut0 = clock_now();
        // Parent E8Q lut_c layout, one 64-byte line per block: the low
        // nibble half (dims 4..7) at [0,32), the high nibble half (dims
        // 0..3) at [32,64), each 16-entry quarter duplicated.
        lut_f.assign(f.plane_cols * 64, 0.0F);
        for (size_t r = 0; r < f.nrun; ++r) {
            const int rate = static_cast<int>(f.run_rate[r]);
            const SegmentCodec::ClassModel& cm = fast_cls_[rate];
            const int nblk = static_cast<int>(f.run_nblk[r]);
            const float* qr = rot_qv + f.run_dim_off[r];
            for (int blk = 0; blk < nblk; ++blk) {
                const size_t col = f.run_blk0[r] + static_cast<size_t>(blk);
                const float* qb = qr + static_cast<size_t>(blk) * 8;
                float* gl = lut_f.data() + col * 64;
                float* gh = lut_f.data() + col * 64 + 32;
                // Rate-1 dim 7 is the BRANCH bit: price it as 0 so the LUT
                // matches the ZPR fit (w_scan[7] = 0).
                const float p0 = cm.pbar[0];
                const float p1 = cm.pbar[1];
                const float p7_0 = (rate == 1) ? 0.0F : p0;
                const float p7_1 = (rate == 1) ? 0.0F : p1;
                for (int v = 0; v < 16; ++v) {
                    gl[v] = qb[4] * ((v >> 3) & 1 ? p1 : p0) +
                            qb[5] * ((v >> 2) & 1 ? p1 : p0) +
                            qb[6] * ((v >> 1) & 1 ? p1 : p0) +
                            qb[7] * (v & 1 ? p7_1 : p7_0);
                    gh[v] = qb[0] * ((v >> 3) & 1 ? p1 : p0) +
                            qb[1] * ((v >> 2) & 1 ? p1 : p0) +
                            qb[2] * ((v >> 1) & 1 ? p1 : p0) +
                            qb[3] * (v & 1 ? p1 : p0);
                }
                for (int c = 0; c < 16; ++c) {
                    gl[16 + c] = gl[c];
                    gh[16 + c] = gh[c];
                }
            }
        }
        lut_u8.assign(f.plane_cols * 64, 0);
        e8qlib::fastscan::QuantizeLut(
            lut_u8.data(), lut_f.data(), f.plane_cols * 64, &lut_min, &lut_delta
        );
        // A4: pre-permute the pair-layout LUT per run so the scan kernels
        // take the plain-load path (same total bytes, run-relative pairs).
        lut_pair.assign(f.plane_cols * 64, 0);
        for (size_t r = 0; r < f.nrun; ++r) {
            FillPairLut(
                lut_u8.data() + static_cast<size_t>(f.run_blk0[r]) * 64,
                f.run_nblk[r],
                lut_pair.data() + static_cast<size_t>(f.run_blk0[r]) * 64
            );
        }
        run_qn2.assign(f.nrun, 0.0);
        run_bits.assign(f.nrun, 0);
        bits_row = 0;
        for (size_t r = 0; r < f.nrun; ++r) {
            double a = 0.0;
            const float* qr = rot_qv + f.run_dim_off[r];
            for (uint32_t t = 0; t < f.run_len[r]; ++t) {
                a += static_cast<double>(qr[t]) * static_cast<double>(qr[t]);
            }
            run_qn2[r] = a;
            run_bits[r] = static_cast<size_t>(f.run_rate[r]) * f.run_len[r];
            bits_row += run_bits[r];
        }
        lut_min_x.assign(f.nrun, 0.0);
        qerr_r.assign(f.nrun, 0.0);
        sqrt_d2_r.assign(f.nrun, 0.0);
        for (size_t r = 0; r < f.nrun; ++r) {
            const double n2b = 2.0 * static_cast<double>(f.run_nblk[r]);
            lut_min_x[r] = static_cast<double>(lut_min) * n2b;
            // CLT LUT pad: per-block rounding is U[-delta, delta]; the sum
            // of independent uniforms concentrates as sqrt(M). Two-sigma
            // cover is 2 sqrt(nblk) delta (parent E8Q; linear M*delta ate
            // recall slack).
            const double sqrt_nblk =
                std::sqrt(static_cast<double>(std::max<uint32_t>(f.run_nblk[r], 1)));
            qerr_r[r] = static_cast<double>(
                detail::LutErrClt(static_cast<float>(sqrt_nblk), lut_delta)
            );
        }
        s_c.assign(f.nrun, 0.0);
        t_r.assign(f.nrun, 0.0);
        // Per-run |S| bound for the weight-rounding pad: s_c ranges over
        // [lut_min_x, lut_min_x + 2*nblk*255*delta].
        score_max_r.assign(f.nrun, 0.0);
        for (size_t r = 0; r < f.nrun; ++r) {
            const double hi = lut_min_x[r] +
                              2.0 * static_cast<double>(f.run_nblk[r]) * 255.0 *
                                  static_cast<double>(lut_delta);
            score_max_r[r] = std::max(std::abs(lut_min_x[r]), std::abs(hi));
        }
        acc.assign(32 * f.nrun * 2, 0);
        lane_dc.assign(32, 0.0F);
        lane_pool.assign(32, 0.0F);
        lane_qa.assign(32, 0.0F);
        lane_sc.assign(32 * f.nrun, 0.0F);
        lane_tr.assign(32 * f.nrun, 0.0F);
        t_lut += since_us(t_lut0, clock_now());
        lut_ready = true;
    };
    for (size_t p = 0; p < take; ++p) {
        const size_t c = cd[p].second;
        const FastLayout& f = fast_[c];
        const double d2 = cd[p].first;
        if (f.nrun == 0) {
            // A list with no coded run: the plan dropped every dimension
            // (me8q::Allocate's all-dropped fallback for an unsatisfiable
            // rate), so the layout holds no segment and the row's estimate is
            // exactly the routing distance -- there is no code to read and no
            // refinement that could move it.  The rows are still the index's
            // candidates, and skipping the list made the whole index answer
            // nothing, so they enter the heap at d2 like any other.
            const uint32_t* row_ids = ids_.data() + list_begin_[c];
            for (size_t i = 0; i < f.cnt; ++i) {
                PushCandidate(static_cast<float>(d2), row_ids[i], k, &heap);
            }
            // Every row reaches the heap unpruned, matching the `survivors`
            // accounting that the refine path does one batch at a time.
            candidates += f.cnt;
            survivors_total += f.cnt;
            continue;
        }
        if (f.cnt == 0) {
            continue;
        }
        if (!uniform_layout || !lut_ready) {
            prepare_list(f, c);
        }
        {
            // d2_r = ||q_r - rc_r||^2: the per-run noise scale of the
            // perpendicular term (the centroid-projected part is banked at
            // build). Per-list, since rc_r is the list's centroid slice.
            for (size_t r = 0; r < f.nrun; ++r) {
                const float* qr = rot_qv + f.run_dim_off[r];
                const float* rc = cent_rot_.data() + c * padded_dim_ + f.run_dim_off[r];
                double dot = 0.0;
                for (uint32_t t = 0; t < f.run_len[r]; ++t) {
                    dot += static_cast<double>(qr[t]) * static_cast<double>(rc[t]);
                }
                const double d2r = run_qn2[r] + f.rc_n2[r] - 2.0 * dot;
                sqrt_d2_r[r] = std::sqrt(std::max(d2r, 0.0));
            }
        }
        CoarsePruneArgs prune_args;
        {
            prune_args.nrun = f.nrun;
            prune_args.d2 = static_cast<float>(d2);
            prune_args.lut_delta = lut_delta;
            prune_args.lambda = lambda;
            prune_args.lut_min_x = lut_min_x.data();
            prune_args.sqrt_d2_r = sqrt_d2_r.data();
            prune_args.qerr_r = qerr_r.data();
            prune_args.lane_dc = lane_dc.data();
            prune_args.lane_pool = lane_pool.data();
            prune_args.lane_qa = lane_qa.data();
            prune_args.lane_sc = lane_sc.data();
            prune_args.lane_tr = lane_tr.data();
            prune_args.keep_lanes = &keep_lanes;
            prune_args.candidates = &candidates;
            prune_args.bits_total = &bits_total;
            prune_args.bits_row = bits_row;
        }
        const size_t rec = f.plane_cols * 32 + f.nrun * f.bank_run_bytes;
        size_t b = 0;
        for (; b < f.batches; ++b) {
            const auto t_scan0 = clock_now();
            const uint8_t* batch = f.plane.data() + b * rec;
            const bool pair = (b + 1 < f.batches) &&
                              (f.cnt - b * 32 >= 64);
            const uint8_t* batch2 = pair
                                        ? f.plane.data() + (b + 1) * rec
                                        : nullptr;
            for (size_t r = 0; r < f.nrun; ++r) {
                const uint8_t* p1 = batch + static_cast<size_t>(f.run_blk0[r]) * 32;
                const uint8_t* lc =
                    lut_u8.data() + static_cast<size_t>(f.run_blk0[r]) * 64;
                if (pair) {
                    AccumulateCoarseHiPair(
                        p1, batch2 + static_cast<size_t>(f.run_blk0[r]) * 32, lc,
                        lut_pair.data() + static_cast<size_t>(f.run_blk0[r]) * 64,
                        f.run_nblk[r], acc.data() + r * 32,
                        acc.data() + f.nrun * 32 + r * 32
                    );
                } else {
                    AccumulateCoarseHi(
                        p1, lc,
                        lut_pair.data() + static_cast<size_t>(f.run_blk0[r]) * 64,
                        f.run_nblk[r], acc.data() + r * 32
                    );
                }
            }
            t_scan += since_us(t_scan0, clock_now());
            const auto t_refine0 = clock_now();
            const size_t lane_cnt = std::min<size_t>(32, f.cnt - b * 32);
            // Vectorized coarse prune, 8 lanes per AVX2 iteration. The filter
            // applies the same bound in fp32, so it cannot cost recall on its
            // own: it tests against distk0, the k-th distance captured before
            // this batch, which is >= the value the scalar path tests per lane.
            auto process_batch = [&](const uint8_t* pbatch, size_t pcnt,
                                    const int32_t* pacc, size_t row0) {
            keep_lanes.clear();
            const auto t_filter0 = clock_now();
            prune_args.lane_cnt = pcnt;
            prune_args.distk = heap.size() < k ? std::numeric_limits<float>::infinity()
                                               : heap.back().first;
            prune_args.acc = pacc;
            prune_args.banks = pbatch + f.plane_cols * 32;
            prune_args.score_max_r = score_max_r.data();
            const size_t filter_lanes = CoarsePrune(prune_args);
            const double fw = since_us(t_filter0, clock_now());
            t_filter += fw;
            for (size_t lane = filter_lanes; lane < pcnt; ++lane) {
                keep_lanes.push_back(static_cast<uint32_t>(lane));
            }
            const auto t_lane0 = clock_now();
            // 4-wide refine, nrun >= 1. Flush every 4 with a live heap so distk
            // tightens (collect-all froze distk0 and lost QPS). Cascade order is
            // the 4-group mean slack/nblk; death is valid in any order.
            uint32_t b_lane[4];
            size_t b_row[4];
            size_t b_id[4];
            const uint8_t* b_rrow[4];
            const float* b_fac[4];
            double b_pool[4];
            double b_qadd[4];
            double b_sc[4 * 32];
            double b_tr[4 * 32];
            int bn = 0;
            auto flush4 = [&]() {
                if (bn == 0) {
                    return;
                }
                const int n = bn;
                if (order.size() < f.nrun) {
                    order.resize(f.nrun);
                }
                for (size_t r = 0; r < f.nrun; ++r) {
                    order[r] = r;
                }
                if (f.nrun > 1) {
                    double mean[32];
                    for (size_t r = 0; r < f.nrun; ++r) {
                        mean[r] = 0.0;
                    }
                    for (int j = 0; j < n; ++j) {
                        for (size_t r = 0; r < f.nrun; ++r) {
                            const double t = b_tr[static_cast<size_t>(j) * f.nrun + r];
                            mean[r] += t * t;
                        }
                    }
                    for (size_t r = 0; r < f.nrun; ++r) {
                        mean[r] /= static_cast<double>(std::max<size_t>(1, f.run_nblk[r]));
                    }
                    std::sort(
                        order.begin(),
                        order.begin() + static_cast<std::ptrdiff_t>(f.nrun),
                        [&](size_t a, size_t b) { return mean[a] > mean[b]; }
                    );
                }
                double dist[4];
                double remain2[4];
                bool died[4] = {};
                int live_idx[4];
                int nlive = n;
                for (int j = 0; j < n; ++j) {
                    live_idx[j] = j;
                    remain2[j] = b_pool[j];
                    if (f.nrun == 1) {
                        dist[j] = d2 + static_cast<double>(b_fac[j][1]);
                    } else {
                        double dc = d2;
                        for (size_t r = 0; r < f.nrun; ++r) {
                            const double coef = b_fac[j][2 * r];
                            const double gain = f.ortho_gain[r * f.cnt + b_row[j]];
                            const double vpc = f.ortho_vpc[r * f.cnt + b_row[j]];
                            dc += static_cast<double>(b_fac[j][2 * r + 1]) -
                                  2.0 * coef *
                                      (vpc + gain * b_sc[static_cast<size_t>(j) * f.nrun + r]);
                        }
                        dist[j] = dc;
                    }
                }
                const uint8_t* refs[4];
                const uint8_t* pbs[4];
                for (size_t oi = 0; oi < f.nrun && nlive > 0; ++oi) {
                    const size_t r = order[oi];
                    const float* qr = rot_qv + f.run_dim_off[r];
                    // A3: one run's refine slices ahead of the cascade —
                    // flush4's own scan just touched the plane bytes, the
                    // refine codes of the NEXT run are still cold.
                    if (oi + 1 < f.nrun) {
                        const size_t rn = order[oi + 1];
                        const size_t off_n = f.refine_run_off[rn];
                        const size_t len_n = f.run_rate[rn] >= 2
                            ? static_cast<size_t>(f.run_rate[rn] - 1) *
                                  f.run_len[rn] / 8
                            : 0;
                        for (int j = 0; j < nlive; ++j) {
                            const uint8_t* pn = b_rrow[live_idx[j]] + off_n;
                            for (size_t o = 0; o < len_n; o += 64) {
                                __builtin_prefetch(pn + o, 0, 3);
                            }
                        }
                    }
                    for (int j = 0; j < nlive; ++j) {
                        const int jj = live_idx[j];
                        refs[j] = b_rrow[jj] + f.refine_run_off[r];
                        pbs[j] = pbatch + b_lane[jj];
                    }
                    float s_out[4];
                    codec_.ScoreRunSplit4(
                        refs, pbs, f.run_blk0[r], static_cast<int>(f.run_nblk[r]),
                        static_cast<int>(f.run_rate[r]), qr, nlive, s_out
                    );
                    const double distk = heap.size() < k
                                             ? std::numeric_limits<double>::infinity()
                                             : static_cast<double>(heap.back().first);
                    int w = 0;
                    for (int j = 0; j < nlive; ++j) {
                        const int jj = live_idx[j];
                        const double coef = b_fac[jj][2 * r];
                        const double s_exact = static_cast<double>(s_out[j]);
                        if (f.nrun == 1) {
                            dist[jj] -= 2.0 * coef * s_exact;
                        } else {
                            const double gain = f.ortho_gain[r * f.cnt + b_row[jj]];
                            const double vpc = f.ortho_vpc[r * f.cnt + b_row[jj]];
                            const double s_cr = b_sc[static_cast<size_t>(jj) * f.nrun + r];
                            dist[jj] += 2.0 * coef * (vpc + gain * s_cr - s_exact);
                        }
                        const double t = b_tr[static_cast<size_t>(jj) * f.nrun + r];
                        remain2[jj] -= t * t;
                        bits_scored += run_bits[r];
                        if (dist[jj] - lambda * std::sqrt(std::max(remain2[jj], 0.0)) -
                                b_qadd[jj] >=
                            distk) {
                            died[jj] = true;
                        } else {
                            live_idx[w++] = jj;
                        }
                    }
                    nlive = w;
                }
                for (int j = 0; j < n; ++j) {
                    if (died[j]) {
                        continue;
                    }
                    ++survivors_total;
                    PushCandidate(static_cast<float>(dist[j]), b_id[j], k, &heap);
                }
                bn = 0;
            };
            for (const uint32_t lane : keep_lanes) {
                const size_t row = row0 + lane;
                const uint8_t* rrow = f.refine.data() + row * f.refine_stride;
                // A8: staging is the earliest point this row's refine
                // address is known — pull the first 192 B in while the
                // remaining batches of this list are still scanning.
                for (size_t o = 0; o < f.refine_stride && o < 192; o += 64) {
                    __builtin_prefetch(rrow + o, 0, 3);
                }
                const float* fac = reinterpret_cast<const float*>(rrow);
                const size_t id = ids_[list_begin_[c] + row];
                const double distk = heap.size() < k
                                         ? std::numeric_limits<double>::infinity()
                                         : static_cast<double>(heap.back().first);
                double dc_q = d2;
                ++candidates;
                bits_total += bits_row;
                // Coarse score, pooled slack units, additive quant error. The
                // vectorized filter already produced all of these for the lanes
                // it covered; only the tail lanes (lane_cnt % 8) recompute.
                double pool2 = 0.0;
                double qadd = 0.0;
                if (lane < filter_lanes) {
                    // Quantized-bound prune decision straight from the filter.
                    dc_q = static_cast<double>(lane_dc[lane]);
                    pool2 = static_cast<double>(lane_pool[lane]);
                    qadd = static_cast<double>(lane_qa[lane]);
                    for (size_t r = 0; r < f.nrun; ++r) {
                        s_c[r] = static_cast<double>(lane_sc[32 * r + lane]);
                        t_r[r] = static_cast<double>(lane_tr[32 * r + lane]);
                    }
                }
                for (size_t r = 0; r < f.nrun && lane >= filter_lanes; ++r) {
                    // Scalar twin of the filter's u8-bank dequant (tail lanes).
                    const uint8_t* bp = pbatch + f.plane_cols * 32 +
                                        r * f.bank_run_bytes;
                    const float* scl = reinterpret_cast<const float*>(bp + 128);
                    const double s_cr =
                        lut_min_x[r] +
                        static_cast<double>(lut_delta) *
                            static_cast<double>(pacc[r * 32 + lane]);
                    s_c[r] = s_cr;
                    const double base_q =
                        static_cast<double>(scl[0]) +
                        static_cast<double>(bp[lane]) * static_cast<double>(scl[1]);
                    const double weight_q =
                        static_cast<double>(scl[2]) +
                        static_cast<double>(bp[32 + lane]) * static_cast<double>(scl[3]);
                    const double perp_q =
                        static_cast<double>(scl[4]) +
                        static_cast<double>(bp[64 + lane]) * static_cast<double>(scl[5]);
                    dc_q += base_q - weight_q * s_cr;
                    const double rh = std::min(
                        sqrt_d2_r[r],
                        distk < std::numeric_limits<double>::infinity()
                            ? std::sqrt(std::max(distk, 0.0))
                            : std::numeric_limits<double>::infinity()
                    );
                    t_r[r] = perp_q * rh;
                    pool2 += t_r[r] * t_r[r];
                    qadd += std::abs(weight_q) * qerr_r[r];
                    qadd += 0.5 * static_cast<double>(scl[3]) *
                            (score_max_r[r] + qerr_r[r]);
                }
                if (dc_q - (lambda * std::sqrt(pool2) + qadd) >= distk) {
                    continue;  // out before any refine
                }
                b_lane[bn] = lane;
                b_row[bn] = row;
                b_id[bn] = id;
                b_rrow[bn] = rrow;
                b_fac[bn] = fac;
                b_pool[bn] = pool2;
                b_qadd[bn] = qadd;
                for (size_t r = 0; r < f.nrun; ++r) {
                    b_sc[static_cast<size_t>(bn) * f.nrun + r] = s_c[r];
                    b_tr[static_cast<size_t>(bn) * f.nrun + r] = t_r[r];
                }
                ++bn;
                if (bn == 4) {
                    flush4();
                }
            }
            flush4();
            const double lw = since_us(t_lane0, clock_now());
            t_lane += lw;
            t_rest += since_us(t_refine0, clock_now()) - fw - lw;
            t_refine += since_us(t_refine0, clock_now());
            };
            process_batch(batch, lane_cnt, acc.data(), b * 32);
            if (pair) {
                const size_t lane_cnt2 = std::min<size_t>(32, f.cnt - (b + 1) * 32);
                process_batch(batch2, lane_cnt2, acc.data() + f.nrun * 32, (b + 1) * 32);
                ++b;
            }
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
        stats->route_us += route_us;
        stats->scan_us += t_scan;
        stats->rotate_us += rot_us;
        stats->lut_us += t_lut;
        stats->filter_us += t_filter;
        stats->lane_us += t_lane;
        stats->refine_us += t_refine;
        stats->rest_us += t_rest;
        stats->candidates += candidates;
        stats->clusters += take;
        stats->survivors += survivors_total;
        stats->bits_scored += bits_scored;
        stats->bits_total += bits_total;
        stats->pruned += candidates - survivors_total;
    }
}

}  // namespace e8qlib::me8q
