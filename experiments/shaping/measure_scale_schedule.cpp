// Per-rate grid-alignment scale s(b): offline derivation and corpus check.
//
// The production encoder aligns a block to the level tables by scaling it,
//     v = s_b * sqrt(D) * u_block,
// before the parity-constrained nearest-codeword rule of
// e8qlib::EncodeBlock runs.  The estimator coefficient g1 of
// src/index/e8q_build.cpp is a dot-normalized (spherical) fit,
// g1 = r_n / <u, v_hat>, so the reconstruction amplitude is absorbed exactly
// and the only quantity the scale controls is the ANGLE between the decoded
// block v_hat and the true block u:
//
//     |e_x|^2 = r_x^2 * tan^2(theta),   theta = angle(u, v_hat).
//
// The scale that a fixed-rate codec should ship is therefore the one that
// minimizes the expected squared angle, i.e. maximizes the expected squared
// directional cosine E[cos^2(u, v_hat)], of the STANDARD GAUSSIAN block of the
// paper's uniform-measure premise: a rotated unit-sphere coordinate vector has
// asymptotically independent N(0, 1/D) marginals, so
// s*sqrt(D)*u_block ~ N(0, s^2 I_8).
//
// Two sources are measured on the same scale grid:
//
//   gauss    isotropic Gaussian blocks (the offline derivation; no corpus
//            enters, which is what makes the resulting table training-free);
//   corpus   the production residual blocks of one corpus (rotate, subtract
//            mean and assigned rotated centroid, normalize) -- a robustness
//            check that the Gaussian optimum is also the corpus optimum.
//
// Both report, per (source, bits, s):
//   cos2  E[cos^2(u, v_hat)]        (the objective, maximized)
//   cos   E[cos(u, v_hat)]
//   nmse  E||u - g v_hat||^2 / E||u||^2 with g = <u, v_hat>/||v_hat||^2
//   mse   the same at g = 1 (unnormalized; the raw distortion of the code)
//
// Output: CSV of every probed (source, bits, s) plus, on stderr, the argmax
// table for each objective.
//
//   g++ -O2 -std=c++17 -Iinclude -Isrc measure_scale_schedule.cpp \
//       ../../src/quantization/e8q_codec.cpp ../../src/utils/rotator.cpp \
//       ../../src/simd/kernels_generic.cpp ../../src/simd/kernels_dispatch.cpp \
//       ../../src/simd/dispatch.cpp -o measure_scale_schedule

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "e8qlib/quantization/e8q_codec.hpp"
#include "e8qlib/utils/rotator.hpp"

namespace {

// Rates the schedule table must cover (matches kMaxQuantizeBits).
constexpr int kMaxBitsMax = 13;

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

struct Acc {
    double cos2 = 0.0;
    double cos = 0.0;
    double nmse = 0.0;   // least-squares g
    double mse = 0.0;    // g = 1
    double energy = 0.0;
    long long n = 0;

    void Add(double dot, double n2_u, double n2_v) {
        if (!(n2_u > 0.0) || !(n2_v > 1e-30)) {
            return;
        }
        const double c = dot / std::sqrt(n2_u * n2_v);
        cos2 += c * c;
        cos += c;
        const double g = dot / n2_v;
        nmse += (n2_u - 2.0 * g * dot + g * g * n2_v) / n2_u;
        mse += (n2_u - 2.0 * dot + n2_v) / n2_u;
        energy += n2_u;
        ++n;
    }

    [[nodiscard]] double MeanCos2() const {
        return n == 0 ? std::numeric_limits<double>::quiet_NaN() : cos2 / n;
    }
    [[nodiscard]] double MeanCos() const {
        return n == 0 ? std::numeric_limits<double>::quiet_NaN() : cos / n;
    }
    [[nodiscard]] double MeanNmse() const {
        return n == 0 ? std::numeric_limits<double>::quiet_NaN() : nmse / n;
    }
    [[nodiscard]] double MeanMse() const {
        return n == 0 ? std::numeric_limits<double>::quiet_NaN() : mse / n;
    }
};

// Encode one 8-dim block at scale `s` and return <u, v_hat>, ||u||^2, ||v_hat||^2.
// Mirrors the production loop of src/index/e8q_build.cpp (both cosets probed,
// cheaper one kept; the parity repair lives inside EncodeBlock).
void BlockStats(
    const double* u, int bits, double s, double rd,
    const std::vector<double> lev[2], double* dot, double* n2_u, double* n2_v
) {
    double v[8];
    for (int i = 0; i < 8; ++i) {
        v[i] = u[i] * (s * rd);
    }
    double cost0 = 0.0;
    double cost1 = 0.0;
    uint16_t dg[2][8];
    for (int br = 0; br < 2; ++br) {
        const e8qlib::BlockCode e = e8qlib::EncodeBlock(v, bits, lev[br], br);
        for (int i = 0; i < 8; ++i) {
            dg[br][i] = e.digits[static_cast<size_t>(i)];
        }
        (br == 0 ? cost0 : cost1) = e.cost;
    }
    const int take = cost1 < cost0 ? 1 : 0;
    double du = 0.0;
    double nv = 0.0;
    double nu = 0.0;
    for (int i = 0; i < 8; ++i) {
        const double lv = lev[take][dg[take][i]];
        du += u[i] * lv;
        nv += lv * lv;
        nu += u[i] * u[i];
    }
    *dot = du;
    *n2_u = nu;
    *n2_v = nv;
}

// Scale grid: default 0.60 .. 1.30 in steps of 0.0025, covering the deployed
// schedule and the reciprocal fixed-scale diagnostic 1/sigma_c = 0.89443.
// --grid LO,HI,STEP overrides it (the b = 1 optimum sits above 1.3 and needs
// the wider range).
std::vector<double> ScaleGrid(double lo, double hi, double step) {
    std::vector<double> out;
    const long long n = static_cast<long long>(std::llround((hi - lo) / step));
    for (long long k = 0; k <= n; ++k) {
        out.push_back(lo + step * static_cast<double>(k));
    }
    return out;
}

void WriteCsv(
    FILE* fh, const char* source, const char* level, int bits, double s, const Acc& a
) {
    std::fprintf(
        fh, "%s,%s,%d,%.4f,%.10f,%.10f,%.10f,%.10f,%lld\n", source, level, bits, s,
        a.MeanCos2(), a.MeanCos(), a.MeanNmse(), a.MeanMse(), a.n
    );
}

void Report(const char* source, const char* level, int bits,
            const std::vector<double>& scales, const std::vector<Acc>& acc) {
    size_t best_cos2 = 0;
    size_t best_nmse = 0;
    for (size_t i = 1; i < acc.size(); ++i) {
        if (acc[i].MeanCos2() > acc[best_cos2].MeanCos2()) {
            best_cos2 = i;
        }
        if (acc[i].MeanNmse() < acc[best_nmse].MeanNmse()) {
            best_nmse = i;
        }
    }
    std::fprintf(
        stderr, "  %-8s %-9s b=%-2d argmax cos2: s=%.4f (cos2=%.8f, nmse=%.8f)   "
                "argmin nmse: s=%.4f (nmse=%.8f)\n",
        source, level, bits, scales[best_cos2], acc[best_cos2].MeanCos2(),
        acc[best_cos2].MeanNmse(), scales[best_nmse], acc[best_nmse].MeanNmse()
    );
}

}  // namespace

int main(int argc, char** argv) {
    std::string data_root = "datasets";
    std::string dataset;
    std::string out_path;
    long long sample = 200000;      // gaussian mode: blocks
    long long vec_sample = 1500;    // corpus mode: vectors
    std::string sources = "gauss";
    long long gauss_dim = 1024;  // width of the synthetic vectors
    double grid_lo = 0.60;
    double grid_hi = 1.30;
    double grid_step = 0.0025;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--data-root" && i + 1 < argc) {
            data_root = argv[++i];
        } else if (a == "--dataset" && i + 1 < argc) {
            dataset = argv[++i];
        } else if (a == "--sample" && i + 1 < argc) {
            sample = std::atoll(argv[++i]);
        } else if (a == "--vec-sample" && i + 1 < argc) {
            vec_sample = std::atoll(argv[++i]);
        } else if (a == "--gauss-dim" && i + 1 < argc) {
            gauss_dim = std::atoll(argv[++i]);
            if (gauss_dim < 8 || gauss_dim % 8 != 0) {
                std::fprintf(stderr, "--gauss-dim must be a multiple of 8\n");
                return 2;
            }
        } else if (a == "--sources" && i + 1 < argc) {
            sources = argv[++i];
        } else if (a == "--grid" && i + 1 < argc) {
            const std::string g = argv[++i];
            if (std::sscanf(g.c_str(), "%lf,%lf,%lf", &grid_lo, &grid_hi, &grid_step) != 3) {
                std::fprintf(stderr, "--grid wants LO,HI,STEP\n");
                return 2;
            }
        } else if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else {
            std::fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (out_path.empty()) {
        std::fprintf(stderr, "usage: measure_scale_schedule --out CSV "
                             "[--dataset NAME] [--data-root DIR] "
                             "[--sources gauss,corpus] [--sample N] [--vec-sample N]\n");
        return 2;
    }
    const bool want_gauss = sources.find("gauss") != std::string::npos;
    const bool want_corpus = sources.find("corpus") != std::string::npos;
    if (want_corpus && dataset.empty()) {
        std::fprintf(stderr, "corpus source needs --dataset\n");
        return 2;
    }

    const std::vector<double> scales = ScaleGrid(grid_lo, grid_hi, grid_step);
    std::vector<std::vector<double>> lev_pair[kMaxBitsMax + 1];
    for (int b = 1; b <= kMaxBitsMax; ++b) {
        lev_pair[b].resize(2);
        lev_pair[b][0] = e8qlib::BranchLevels(b, 0);
        lev_pair[b][1] = e8qlib::BranchLevels(b, 1);
    }

    FILE* fh = std::fopen(out_path.c_str(), "w");
    if (fh == nullptr) {
        std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
        return 1;
    }
    std::fprintf(
        fh, "source,level,bits,scale,cos2,cos,nmse,mse,n\n"
    );

    // ---------------------------------------------------------------- gauss
    if (want_gauss) {
        std::mt19937_64 rng(20260930ULL);
        std::normal_distribution<double> nd(0.0, 1.0);
        // `sample` counts BLOCKS; each synthetic vector is `gauss_dim` wide, so
        // the per-vector objective below sees the block count a real dimension
        // produces (gauss_dim / 8).
        const long long vecs = std::max<long long>(
            1, std::max<long long>(1000, sample) / (gauss_dim / 8)
        );
        std::vector<double> z(static_cast<size_t>(vecs) * static_cast<size_t>(gauss_dim));
        for (double& v : z) {
            v = nd(rng);
        }
        const long long blocks_per_vec = gauss_dim / 8;
        // Encode at unit block energy: s*sqrt(D)*u_block ~ N(0, s^2), so the
        // block fed to the encoder is z * s; pass rd = 1.
        for (int b = 1; b <= kMaxBitsMax; ++b) {
            std::vector<Acc> acc(scales.size());
            std::vector<Acc> vacc(scales.size());
            for (size_t si = 0; si < scales.size(); ++si) {
                const double s = scales[si];
                Acc a;
                Acc va;
                const double* zb = z.data();
                for (long long vi = 0; vi < vecs; ++vi) {
                    double vdot = 0.0;
                    double vnu = 0.0;
                    double vnv = 0.0;
                    for (long long blk = 0; blk < blocks_per_vec; ++blk) {
                        double dot = 0.0;
                        double nu = 0.0;
                        double nv = 0.0;
                        BlockStats(zb + (vi * blocks_per_vec + blk) * 8, b, s, 1.0,
                                   lev_pair[b].data(), &dot, &nu, &nv);
                        a.Add(dot, nu, nv);
                        vdot += dot;
                        vnu += nu;
                        vnv += nv;
                    }
                    va.Add(vdot, vnu, vnv);
                }
                acc[si] = a;
                vacc[si] = va;
                WriteCsv(fh, "gauss", "block", b, s, a);
                WriteCsv(fh, "gauss", "vector", b, s, va);
            }
            Report("gauss", "block", b, scales, acc);
            Report("gauss", "vector", b, scales, vacc);
        }
        std::fflush(fh);
    }

    // --------------------------------------------------------------- corpus
    if (want_corpus) {
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
        const long long want = std::min(vec_sample, n);
        const long long stride = std::max<long long>(1, n / want);

        const e8qlib::FhtKac4Rotator rot(dim, /*seed=*/0);
        const int pdim = static_cast<int>(rot.padded_dim());
        const double rd = std::sqrt(static_cast<double>(pdim));
        const int blocks_per_vec = pdim / 8;

        // Mean over the sampled rows, then its rotated image.
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
            mu_pad[static_cast<size_t>(d)] =
                static_cast<float>(mu_acc[static_cast<size_t>(d)] /
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

        // One pass over the sampled vectors; for each vector every (rate,
        // scale) cell is updated from the same residual blocks.
        std::vector<std::vector<Acc>> acc(static_cast<size_t>(kMaxBitsMax + 1));
        std::vector<std::vector<Acc>> vacc(static_cast<size_t>(kMaxBitsMax + 1));
        for (int b = 1; b <= kMaxBitsMax; ++b) {
            acc[static_cast<size_t>(b)].assign(scales.size(), Acc{});
            vacc[static_cast<size_t>(b)].assign(scales.size(), Acc{});
        }
        std::vector<float> rot_x(pdim), x_pad(pdim, 0.0F);
        std::vector<double> ublk(8);
        double vdot = 0.0;
        double vnu = 0.0;
        double vnv = 0.0;
        long long seen = 0;
        for (long long i = 0; i < n; i += stride) {
            if (!base.Row(i, row)) {
                return 1;
            }
            const int cid = cids.Row(i);
            if (cid < 0 || cid >= cent.n) {
                continue;
            }
            std::fill(x_pad.begin(), x_pad.end(), 0.0F);
            std::copy(row.begin(), row.end(), x_pad.begin());
            rot.Rotate(x_pad.data(), rot_x.data());
            const float* rc = &cent_rot[static_cast<size_t>(cid) * static_cast<size_t>(pdim)];
            double rn2 = 0.0;
            std::vector<double> unit(static_cast<size_t>(pdim), 0.0);
            for (int d = 0; d < pdim; ++d) {
                const double rv = static_cast<double>(rot_x[static_cast<size_t>(d)]) -
                                  mean_rot[static_cast<size_t>(d)] -
                                  static_cast<double>(rc[d]);
                unit[static_cast<size_t>(d)] = rv;
                rn2 += rv * rv;
            }
            const double rn = std::sqrt(rn2);
            if (!(rn > 1e-20)) {
                continue;
            }
            for (int d = 0; d < pdim; ++d) {
                unit[static_cast<size_t>(d)] /= rn;
            }
            ++seen;
            for (int b = 1; b <= kMaxBitsMax; ++b) {
                const std::vector<double>* lp = lev_pair[b].data();
                std::vector<Acc>& row_acc = acc[static_cast<size_t>(b)];
                std::vector<Acc>& vec_acc = vacc[static_cast<size_t>(b)];
                for (size_t si = 0; si < scales.size(); ++si) {
                    const double s = scales[si];
                    Acc& a = row_acc[si];
                    vdot = 0.0;
                    vnu = 0.0;
                    vnv = 0.0;
                    for (int blk = 0; blk < blocks_per_vec; ++blk) {
                        for (int k = 0; k < 8; ++k) {
                            ublk[static_cast<size_t>(k)] =
                                unit[static_cast<size_t>(blk) * 8 + static_cast<size_t>(k)];
                        }
                        double dot = 0.0;
                        double nu = 0.0;
                        double nv = 0.0;
                        BlockStats(ublk.data(), b, s, rd, lp, &dot, &nu, &nv);
                        a.Add(dot, nu, nv);
                        vdot += dot;
                        vnu += nu;
                        vnv += nv;
                    }
                    vec_acc[si].Add(vdot, vnu, vnv);
                }
            }
        }
        std::fprintf(stderr, "%s: %lld vectors sampled (stride %lld)\n", dataset.c_str(),
                     seen, stride);
        for (int b = 1; b <= kMaxBitsMax; ++b) {
            const std::vector<Acc>& row_acc = acc[static_cast<size_t>(b)];
            const std::vector<Acc>& vec_acc = vacc[static_cast<size_t>(b)];
            for (size_t si = 0; si < scales.size(); ++si) {
                WriteCsv(fh, "corpus", "block", b, scales[si], row_acc[si]);
                WriteCsv(fh, "corpus", "vector", b, scales[si], vec_acc[si]);
            }
            Report("corpus", "block", b, scales, row_acc);
            Report("corpus", "vector", b, scales, vec_acc);
        }
    }

    std::fclose(fh);
    return 0;
}
