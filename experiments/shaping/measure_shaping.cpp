// Quantization-distortion / shaping-gain measurement for the SIGMOD appendix.
//
// Reproduces the production encode path of src/index/e8q_build.cpp on a
// stride sample of a corpus:
//
//   rotate(x) -> subtract mean_rot -> subtract the assigned rotated IVF
//   centroid -> unit residual u (||u|| = 1)
//   -> EncodeVectorAtScale(u, b, s)
//
// and measures, per rate b in 1..8, the normalized mean squared error
//
//   NMSE = E ||u - g v_hat||^2 / E ||u||^2        (||u|| = 1),
//
// with g the per-vector least-squares scale of the reconstruction, for four
// quantizers that share one compander:
//
//   * E8Q    -- the parity-coupled two-coset codebook of the paper
//               (2^(8b) codewords per 8-dim block);
//   * SCALAR -- the rate-matched product quantizer: every coordinate of the
//               block independently nearest-rounded on branch 0's 2^b-level
//               table (also 2^(8b) codewords per block);
//   * COSET  -- the same product quantizer allowed to pick, per block, the
//               better of the two interleaved tables (2^(8b+1) codewords, so
//               one extra bit per block);
//   * UNION  -- the product quantizer on both tables at once, i.e. 2^(b+1)
//               levels per coordinate (2^(8b+8) codewords).
//
// COSET and UNION are references, not rate-matched baselines: with SCALAR and
// E8Q they let the appendix attribute the measured gap between the two
// codebooks instead of asserting a mechanism.
//
// E8Q and SCALAR are evaluated over a grid of grid-alignment scales s (the
// constant that FixedScale(bits) supplies in production), so each is reported
// at its deployed scale and at its own best scale; the shaping gain
// that the appendix quotes uses the SCALAR's best scale against E8Q's
// deployed scale, i.e. it is a lower bound on the lattice's advantage.  The
// two references are swept on the same grid and reported at their own best
// scale as well.
//
// Output: one CSV row per (corpus, bits).
//
//   g++ -O2 -std=c++17 -Iinclude -Isrc measure_shaping.cpp \
//       ../../src/quantization/e8q_codec.cpp ../../src/utils/rotator.cpp \
//       ../../src/simd/kernels_generic.cpp ../../src/simd/kernels_dispatch.cpp \
//       ../../src/simd/dispatch.cpp -o measure_shaping

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "e8qlib/quantization/e8q_codec.hpp"
#include "e8qlib/utils/rotator.hpp"

namespace {

// The reciprocal continuum reference 1/sigma_c = sqrt(4/5), retained as a
// diagnostic candidate for using one fixed scale across the rate spectrum.
constexpr double kSReciprocal = 0.89442719099991586;

struct FvecFile {
    FILE* f = nullptr;
    int dim = 0;
    long long n = 0;

    bool Open(const std::string& path, int expect_dim) {
        f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            std::fprintf(stderr, "cannot open %s\n", path.c_str());
            return false;
        }
        int32_t d = 0;
        if (std::fread(&d, sizeof(d), 1, f) != 1) {
            return false;
        }
        dim = d;
        if (expect_dim > 0 && dim != expect_dim) {
            std::fprintf(stderr, "%s: dim %d != %d\n", path.c_str(), dim, expect_dim);
            return false;
        }
        std::fseek(f, 0, SEEK_END);
        n = std::ftell(f) / (4LL * (1 + dim));
        return true;
    }

    bool Row(long long i, std::vector<float>& dst) const {
        const long long off = i * (4LL * (1 + dim)) + 4LL;
        if (std::fseek(f, static_cast<long>(off), SEEK_SET) != 0) {
            return false;
        }
        return std::fread(dst.data(), sizeof(float), static_cast<size_t>(dim), f) ==
               static_cast<size_t>(dim);
    }
};

struct IvecFile {
    FILE* f = nullptr;
    int dim = 0;
    long long n = 0;

    bool Open(const std::string& path) {
        f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            std::fprintf(stderr, "cannot open %s\n", path.c_str());
            return false;
        }
        int32_t d = 0;
        if (std::fread(&d, sizeof(d), 1, f) != 1) {
            return false;
        }
        dim = d;
        std::fseek(f, 0, SEEK_END);
        n = std::ftell(f) / (4LL * (1 + dim));
        return true;
    }

    int Row(long long i) const {
        const long long off = i * (4LL * (1 + dim)) + 4LL;
        if (std::fseek(f, static_cast<long>(off), SEEK_SET) != 0) {
            return -1;
        }
        int32_t v = -1;
        if (std::fread(&v, sizeof(v), 1, f) != 1) {
            return -1;
        }
        return v;
    }
};

// Ascending view of branch 0's level table, for the product control.
struct SortedLevels {
    std::vector<double> lev;

    SortedLevels() = default;
    explicit SortedLevels(const std::vector<double>& table) : lev(table) {
        std::sort(lev.begin(), lev.end());
    }

    double Nearest(double v) const {
        auto it = std::lower_bound(lev.begin(), lev.end(), v);
        if (it == lev.begin()) {
            return lev.front();
        }
        if (it == lev.end()) {
            return lev.back();
        }
        const double hi = *it;
        const double lo = *(it - 1);
        return (v - lo) <= (hi - v) ? lo : hi;
    }
};

struct Err {
    double sum = 0.0;
    long long n = 0;

    void Add(double e) {
        sum += e;
        ++n;
    }
    [[nodiscard]] double Mean() const {
        return n == 0 ? std::numeric_limits<double>::quiet_NaN() : sum / static_cast<double>(n);
    }
};

// Grid-alignment scales probed around the deployed constant.
std::vector<double> ScaleGrid(double s0) {    const double f[] = {0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.1, 1.25, 1.45};
    std::vector<double> out;
    for (double v : f) {
        out.push_back(s0 * v);
    }
    return out;
}

// Window half-widths probed for the hypercubic (uncompanded Z8) control, in
// units of the coordinate standard deviation 1/sqrt(padded_dim).  The grid has
// exactly q = 2^b levels per coordinate and clamps outside the window, so the
// control is rate-matched to E8Q at every probe.
std::vector<double> WindowGrid(double rd) {
    std::vector<double> out;
    for (double f : {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0}) {
        out.push_back(f / rd);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_root = "datasets";
    std::string dataset;
    std::string out_path;
    long long sample = 100000;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--data-root" && i + 1 < argc) {
            data_root = argv[++i];
        } else if (a == "--dataset" && i + 1 < argc) {
            dataset = argv[++i];
        } else if (a == "--sample" && i + 1 < argc) {
            sample = std::atoll(argv[++i]);
        } else if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else {
            std::fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (dataset.empty() || out_path.empty()) {
        std::fprintf(stderr, "usage: measure_shaping --dataset NAME --out CSV "
                             "[--data-root DIR] [--sample N]\n");
        return 2;
    }

    const std::string dir = data_root + "/" + dataset + "/";
    FvecFile base;
    if (!base.Open(dir + dataset + "_base.fvecs", 0)) {
        return 1;
    }
    const int dim = base.dim;
    FvecFile cent;
    if (!cent.Open(dir + dataset + "_centroid_4096.fvecs", dim)) {
        return 1;
    }
    IvecFile cids;
    if (!cids.Open(dir + dataset + "_cluster_id_4096.ivecs")) {
        return 1;
    }
    const long long n = std::min(base.n, cids.n);
    const long long want = sample > 0 ? std::min(sample, n) : n;
    const long long stride = std::max<long long>(1, n / want);
    std::fprintf(stderr, "%s: n=%lld dim=%d stride=%lld\n", dataset.c_str(), n, dim, stride);

    const e8qlib::FhtKac4Rotator rot(dim, /*seed=*/0);
    const int pdim = static_cast<int>(rot.padded_dim());
    const double rd = std::sqrt(static_cast<double>(pdim));

    std::vector<float> row(dim);
    std::vector<double> mu_acc(dim, 0.0);
    long long used = 0;
    for (long long i = 0; i < n; i += stride) {
        if (!base.Row(i, row)) {
            return 1;
        }
        for (int d = 0; d < dim; ++d) {
            mu_acc[static_cast<size_t>(d)] += static_cast<double>(row[d]);
        }
        ++used;
    }
    std::vector<float> mu_pad(pdim, 0.0F), mean_rot(pdim, 0.0F);
    for (int d = 0; d < dim; ++d) {
        mu_pad[static_cast<size_t>(d)] = static_cast<float>(mu_acc[static_cast<size_t>(d)] /
                                                            static_cast<double>(used));
    }
    rot.Rotate(mu_pad.data(), mean_rot.data());

    std::vector<float> cent_rot(static_cast<size_t>(cent.n) * static_cast<size_t>(pdim));
    {
        std::vector<float> c(dim), c_pad(pdim, 0.0F);
        for (long long ci = 0; ci < cent.n; ++ci) {
            if (!cent.Row(ci, c)) {
                return 1;
            }
            std::fill(c_pad.begin(), c_pad.end(), 0.0F);
            std::copy(c.begin(), c.end(), c_pad.begin());
            float* dst = &cent_rot[static_cast<size_t>(ci) * static_cast<size_t>(pdim)];
            rot.Rotate(c_pad.data(), dst);
            for (int d = 0; d < pdim; ++d) {
                dst[d] -= mean_rot[static_cast<size_t>(d)];
            }
        }
    }

    std::vector<double> lev_tab[9][2];
    std::vector<SortedLevels> sorted0(9), sorted_union(9);
    SortedLevels sorted_tab[9][2];
    for (int bits = 1; bits <= 8; ++bits) {
        lev_tab[bits][0] = e8qlib::BranchLevels(bits, 0);
        lev_tab[bits][1] = e8qlib::BranchLevels(bits, 1);
        sorted0[static_cast<size_t>(bits)] = SortedLevels(lev_tab[bits][0]);
        sorted_tab[bits][0] = SortedLevels(lev_tab[bits][0]);
        sorted_tab[bits][1] = SortedLevels(lev_tab[bits][1]);
        // The union of the two branch tables: the interleaved alphabet that
        // the two cosets jointly address, at 2^(b+1) levels per coordinate.
        std::vector<double> merged = lev_tab[bits][0];
        merged.insert(merged.end(), lev_tab[bits][1].begin(), lev_tab[bits][1].end());
        sorted_union[static_cast<size_t>(bits)] = SortedLevels(merged);
    }

    // Per (bits, scale) accumulators for the two table-based quantizers and
    // per (bits, step) accumulators for the hypercubic control.
    std::vector<std::vector<Err>> e8q_e(9), sc_e(9), gr_e(9), un_e(9), ch_e(9);
    std::vector<std::vector<Err>> fx_e(9);
    std::vector<std::vector<double>> scales(9), steps(9);
    for (int bits = 1; bits <= 8; ++bits) {
        scales[static_cast<size_t>(bits)] = ScaleGrid(e8qlib::FixedScale(bits));
        steps[static_cast<size_t>(bits)] = WindowGrid(rd);
        e8q_e[static_cast<size_t>(bits)].assign(scales[static_cast<size_t>(bits)].size(), Err{});
        sc_e[static_cast<size_t>(bits)].assign(scales[static_cast<size_t>(bits)].size(), Err{});
        un_e[static_cast<size_t>(bits)].assign(scales[static_cast<size_t>(bits)].size(), Err{});
        ch_e[static_cast<size_t>(bits)].assign(scales[static_cast<size_t>(bits)].size(), Err{});
        gr_e[static_cast<size_t>(bits)].assign(steps[static_cast<size_t>(bits)].size(), Err{});
        fx_e[static_cast<size_t>(bits)].assign(2, Err{});
    }

    std::vector<float> rot_x(pdim), unit(pdim), x_pad(pdim, 0.0F);
    for (long long i = 0; i < n; i += stride) {
        if (!base.Row(i, row)) {
            return 1;
        }
        std::fill(x_pad.begin(), x_pad.end(), 0.0F);
        std::copy(row.begin(), row.end(), x_pad.begin());
        rot.Rotate(x_pad.data(), rot_x.data());
        const int cid = cids.Row(i);
        if (cid < 0 || cid >= cent.n) {
            continue;
        }
        const float* rc = &cent_rot[static_cast<size_t>(cid) * static_cast<size_t>(pdim)];
        double rn2 = 0.0;
        for (int d = 0; d < pdim; ++d) {
            const double rv = static_cast<double>(rot_x[d]) - mean_rot[static_cast<size_t>(d)] -
                              static_cast<double>(rc[d]);
            unit[static_cast<size_t>(d)] = static_cast<float>(rv);
            rn2 += rv * rv;
        }
        const double rn = std::sqrt(rn2);
        if (rn < 1e-20) {
            continue;
        }
        const double inv = 1.0 / rn;
        double n2_u = 0.0;
        for (int d = 0; d < pdim; ++d) {
            const double t = static_cast<double>(unit[static_cast<size_t>(d)]) * inv;
            unit[static_cast<size_t>(d)] = static_cast<float>(t);
            n2_u += t * t;
        }
        for (int bits = 1; bits <= 8; ++bits) {
            const std::vector<double>& l0 = lev_tab[bits][0];
            const std::vector<double>& l1 = lev_tab[bits][1];
            // E8Q error at one grid scale, as (n2 of the reconstruction, the
            // reconstruction's inner product with the unit residual, n2 of the
            // unit residual).  Shared by the swept grid and by the two
            // single-constant arms below.
            const auto e8q_terms = [&](double s, double* dot, double* n2) {
                const e8qlib::VectorCode vc =
                    e8qlib::EncodeVectorAtScale(unit.data(), pdim, bits, s);
                *dot = 0.0;
                *n2 = 0.0;
                for (int blk = 0; blk < pdim / 8; ++blk) {
                    const std::array<uint16_t, 8>& dg = vc.digits[static_cast<size_t>(blk)];
                    const std::vector<double>& tab =
                        vc.branch[static_cast<size_t>(blk)] == 1 ? l1 : l0;
                    for (int k = 0; k < 8; ++k) {
                        const double u =
                            static_cast<double>(unit[static_cast<size_t>(blk * 8 + k)]);
                        const double ve =
                            tab[static_cast<size_t>(dg[static_cast<size_t>(k)])] / rd;
                        *dot += u * ve;
                        *n2 += ve * ve;
                    }
                }
            };
            for (size_t si = 0; si < scales[static_cast<size_t>(bits)].size(); ++si) {
                const double s = scales[static_cast<size_t>(bits)][si];
                const double srd = s * rd;
                const e8qlib::VectorCode vc =
                    e8qlib::EncodeVectorAtScale(unit.data(), pdim, bits, s);
                double dot_ue = 0.0, n2_e = 0.0, dot_uw = 0.0, n2_w = 0.0;
                double dot_uu = 0.0, n2_uu = 0.0, dot_uc = 0.0, n2_c = 0.0;
                for (int blk = 0; blk < pdim / 8; ++blk) {
                    const std::array<uint16_t, 8>& dg = vc.digits[static_cast<size_t>(blk)];
                    const int br = vc.branch[static_cast<size_t>(blk)];
                    const std::vector<double>& tab = br == 1 ? l1 : l0;
                    // Per-block coset choice without the parity constraint:
                    // the product quantizer keeps 2^b levels per coordinate
                    // but may pick, for the whole block, whichever of the two
                    // interleaved tables fits that block better.
                    double err[2] = {0.0, 0.0};
                    for (int j = 0; j < 2; ++j) {
                        const SortedLevels& tj =
                            sorted_tab[static_cast<size_t>(bits)][static_cast<size_t>(j)];
                        for (int k = 0; k < 8; ++k) {
                            const double u =
                                static_cast<double>(unit[static_cast<size_t>(blk * 8 + k)]);
                            const double v = tj.Nearest(u * srd) / rd;
                            err[j] += (u - v) * (u - v);
                        }
                    }
                    const SortedLevels& tc =
                        sorted_tab[static_cast<size_t>(bits)][err[1] < err[0] ? 1 : 0];
                    for (int k = 0; k < 8; ++k) {
                        const int idx = blk * 8 + k;
                        const double u = static_cast<double>(unit[static_cast<size_t>(idx)]);
                        const double ve =
                            tab[static_cast<size_t>(dg[static_cast<size_t>(k)])] / rd;
                        const double w =
                            sorted0[static_cast<size_t>(bits)].Nearest(u * srd) / rd;
                        const double wu =
                            sorted_union[static_cast<size_t>(bits)].Nearest(u * srd) / rd;
                        const double wc = tc.Nearest(u * srd) / rd;
                        dot_ue += u * ve;
                        n2_e += ve * ve;
                        dot_uw += u * w;
                        n2_w += w * w;
                        dot_uu += u * wu;
                        n2_uu += wu * wu;
                        dot_uc += u * wc;
                        n2_c += wc * wc;
                    }
                }
                if (n2_e > 1e-30 && dot_ue > 1e-12) {
                    const double g = dot_ue / n2_e;
                    e8q_e[static_cast<size_t>(bits)][si].Add(n2_u - 2.0 * g * dot_ue +
                                                             g * g * n2_e);
                }
                if (n2_w > 1e-30 && dot_uw > 1e-12) {
                    const double g = dot_uw / n2_w;
                    sc_e[static_cast<size_t>(bits)][si].Add(n2_u - 2.0 * g * dot_uw +
                                                             g * g * n2_w);
                }
                if (n2_uu > 1e-30 && dot_uu > 1e-12) {
                    const double g = dot_uu / n2_uu;
                    un_e[static_cast<size_t>(bits)][si].Add(n2_u - 2.0 * g * dot_uu +
                                                             g * g * n2_uu);
                }
                if (n2_c > 1e-30 && dot_uc > 1e-12) {
                    const double g = dot_uc / n2_c;
                    ch_e[static_cast<size_t>(bits)][si].Add(n2_u - 2.0 * g * dot_uc +
                                                             g * g * n2_c);
                }
            }
            // Two single-constant arms: E8Q held at one scale across the whole
            // rate spectrum, the counterfactual the appendix contrasts with the
            // tabulated schedule.  The two candidates are the scale used by the
            // two-level sign code (sigma_c, the schedule's b=1 entry) and its
            // reciprocal 1/sigma_c, retained as a diagnostic fixed-scale arm.
            for (size_t fi = 0; fi < 2; ++fi) {
                const double s = fi == 0 ? e8qlib::kSigmaC : kSReciprocal;
                double dot = 0.0, n2 = 0.0;
                e8q_terms(s, &dot, &n2);
                if (n2 > 1e-30 && dot > 1e-12) {
                    const double g = dot / n2;
                    fx_e[static_cast<size_t>(bits)][fi].Add(n2_u - 2.0 * g * dot + g * g * n2);
                }
            }
            // Hypercubic control: the uncompanded Z8 quantizer at the same
            // rate -- q = 2^b uniform levels spanning [-L, L] per coordinate,
            // clamped outside it.  Only the window half-width L is swept.
            const std::vector<double>& st = steps[static_cast<size_t>(bits)];
            const int q = 1 << bits;
            for (size_t gi = 0; gi < st.size(); ++gi) {
                const double L = st[gi];
                const double h = 2.0 * L / static_cast<double>(q);
                double dot_ug = 0.0, n2_g = 0.0;
                for (int d = 0; d < pdim; ++d) {
                    const double u = static_cast<double>(unit[static_cast<size_t>(d)]);
                    double idx = std::nearbyint((u + L) / h - 0.5);
                    idx = std::min(std::max(idx, 0.0), static_cast<double>(q - 1));
                    const double w = -L + (idx + 0.5) * h;
                    dot_ug += u * w;
                    n2_g += w * w;
                }
                if (n2_g > 1e-30 && dot_ug > 1e-12) {
                    const double g = dot_ug / n2_g;
                    gr_e[static_cast<size_t>(bits)][gi].Add(n2_u - 2.0 * g * dot_ug +
                                                             g * g * n2_g);
                }
            }
        }
    }

    FILE* out = std::fopen(out_path.c_str(), "w");
    if (out == nullptr) {
        std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
        return 1;
    }
    std::fprintf(
        out,
        "corpus,bits,n,nmse_e8q_deployed,nmse_e8q_best,s_e8q_best,nmse_scalar_deployed,"
        "nmse_scalar_best,s_scalar_best,nmse_grid_best,window_grid_best,gain_db_lower,"
        "gain_db_opt,gain_db_grid,nmse_union_best,s_union_best,nmse_coset_best,"
        "s_coset_best,nmse_e8q_single_sigmac,nmse_e8q_single_reciprocal,"
        "gain_db_single_sigmac,gain_db_single_reciprocal\n"
    );
    for (int bits = 1; bits <= 8; ++bits) {
        const std::vector<double>& sg = scales[static_cast<size_t>(bits)];
        const std::vector<double>& st = steps[static_cast<size_t>(bits)];
        const std::vector<Err>& ee = e8q_e[static_cast<size_t>(bits)];
        const std::vector<Err>& se = sc_e[static_cast<size_t>(bits)];
        const std::vector<Err>& ge = gr_e[static_cast<size_t>(bits)];
        const std::vector<Err>& ue = un_e[static_cast<size_t>(bits)];
        const std::vector<Err>& ce = ch_e[static_cast<size_t>(bits)];
        size_t dep = 0;
        for (size_t si = 0; si < sg.size(); ++si) {
            if (std::abs(sg[si] - e8qlib::FixedScale(bits)) < 1e-12) {
                dep = si;
            }
        }
        double e_dep = ee[dep].Mean(), s_dep = se[dep].Mean();
        double e_best = e_dep, s_best = s_dep, s_e = sg[dep], s_s = sg[dep];
        for (size_t si = 0; si < sg.size(); ++si) {
            if (ee[si].Mean() < e_best) {
                e_best = ee[si].Mean();
                s_e = sg[si];
            }
            if (se[si].Mean() < s_best) {
                s_best = se[si].Mean();
                s_s = sg[si];
            }
        }
        // Union control: the 2^(b+1)-level interleaved alphabet the two
        // cosets jointly address (a strictly higher rate), and the per-block
        // coset-choice control (same rate, no parity constraint).  Both let
        // the appendix attribute the measured gap instead of asserting it.
        double u_best = ue[0].Mean(), s_u = sg[0];
        double c_best = ce[0].Mean(), s_c = sg[0];
        for (size_t si = 0; si < sg.size(); ++si) {
            if (ue[si].Mean() < u_best) {
                u_best = ue[si].Mean();
                s_u = sg[si];
            }
            if (ce[si].Mean() < c_best) {
                c_best = ce[si].Mean();
                s_c = sg[si];
            }
        }
        double g_best = ge[0].Mean(), h_best = st[0];
        for (size_t gi = 0; gi < st.size(); ++gi) {
            if (ge[gi].Mean() < g_best) {
                g_best = ge[gi].Mean();
                h_best = st[gi];
            }
        }
        const long long m = ee[dep].n;
        const std::vector<Err>& fe = fx_e[static_cast<size_t>(bits)];
        const double f_sc = fe[0].Mean(), f_cont = fe[1].Mean();
        std::fprintf(
            out,
            "%s,%d,%lld,%.6e,%.6e,%.4f,%.6e,%.6e,%.4f,%.6e,%.6e,%.4f,%.4f,%.4f,"
            "%.6e,%.4f,%.6e,%.4f,%.6e,%.6e,%.4f,%.4f\n",
            dataset.c_str(), bits, m, e_dep, e_best, s_e, s_dep, s_best, s_s, g_best,
            h_best, 10.0 * std::log10(s_best / e_dep), 10.0 * std::log10(s_best / e_best),
            10.0 * std::log10(g_best / e_dep), u_best, s_u, c_best, s_c,
            f_sc, f_cont, 10.0 * std::log10(s_best / f_sc), 10.0 * std::log10(s_best / f_cont)
        );
        std::fprintf(
            stderr,
            "  b=%d  E8Q %.4e (best %.4e @s=%.3f)  scalar %.4e (best %.4e @s=%.3f)  "
            "grid %.4e  gain %.3f dB lower / %.3f dB opt / %.3f dB vs grid | "
            "union %.4e  coset %.4e\n",
            bits, e_dep, e_best, s_e, s_dep, s_best, s_s, g_best,
            10.0 * std::log10(s_best / e_dep), 10.0 * std::log10(s_best / e_best),
            10.0 * std::log10(g_best / e_dep), u_best, c_best
        );
        (void)e_dep;
        (void)s_dep;
    }
    std::fclose(out);
    return 0;
}
