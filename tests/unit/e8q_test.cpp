// Unit checks for the E8Q codec and index invariants.
//
// 1. branch reflection identity: level[1][q-1-d] == -level[0][d]
// 2. scan-basis reflection identity: w_cls[1] == -w_cls[0]
// 3. refine pack/unpack roundtrip for b = 1..8 (parity-even digit tuples)
// 4. rowA + refine-field decode roundtrip for b = 2..13
// 5. encoder parity + digit range
// 6. synthetic end-to-end: search with nprobe == num_clusters reproduces the
//    exhaustive estimator top-k exactly (validates prune soundness + refine)
// 7. coarse kernel == branch-blind class-mean model
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "e8qlib/quantization/e8q_codec.hpp"
#include "e8qlib/index/e8q.hpp"

namespace {

int failures = 0;

void Check(bool ok, const char* what) {
    if (ok) {
        std::printf("ok   %s\n", what);
    } else {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

void TestIdentities() {
    for (int bits = 1; bits <= e8qlib::kMaxQuantizeBits; ++bits) {
        const int q = 1 << bits;
        const auto l0 = e8qlib::BranchLevels(bits, 0);
        const auto l1 = e8qlib::BranchLevels(bits, 1);
        bool ok = true;
        for (int d = 0; d < q; ++d) {
            if (std::abs(l1[static_cast<size_t>(q - 1 - d)] + l0[static_cast<size_t>(d)]) >
                1e-9) {
                ok = false;
            }
        }
        Check(ok, "reflection identity levels");
        // Scan-basis reflection: the branch-averaged class mean of the digit
        // top-bit class (w_cls, the lut_c model) must flip sign under the
        // class complement, so the scan LUT is symmetric around zero.
        const int cvals = 1 << (bits - 1);
        double wc[2] = {0.0, 0.0};
        for (int t2 = 0; t2 < 2; ++t2) {
            double s0 = 0.0;
            double s1 = 0.0;
            for (int lo = 0; lo < cvals; ++lo) {
                s0 += l0[static_cast<size_t>((t2 << (bits - 1)) | lo)];
                s1 += l1[static_cast<size_t>((t2 << (bits - 1)) | lo)];
            }
            wc[t2] = 0.5 * (s0 / cvals + s1 / cvals);
        }
        if (std::abs(wc[1] + wc[0]) > 1e-9) {
            std::printf("FAIL scan-basis reflection b=%d\n", bits);
            ++failures;
        }
    }
    std::printf("ok   scan-basis reflection b=1..13\n");
}

void TestPacking() {
    std::mt19937 rng(7);
    bool all_ok = true;
    for (int bits = 1; bits <= 8; ++bits) {
        const int q = 1 << bits;
        for (int trial = 0; trial < 200; ++trial) {
            uint8_t digits[8];
            int par = 0;
            for (int i = 0; i < 8; ++i) {
                digits[i] = static_cast<uint8_t>(rng() % static_cast<unsigned>(q));
            }
            // Even-parity shell: digit 7's LSB is the XOR of digits 0..6.
            for (int i = 0; i < 7; ++i) {
                par ^= digits[i] & 1;
            }
            digits[7] = static_cast<uint8_t>((digits[7] & ~1u) | static_cast<unsigned>(par));
            // RowA byte, digit top bit of dim i at bit 7-i (b == 1: the sign
            // bits of dims 0..6; bit 0 = branch).
            uint8_t pb = 0;
            for (int i = 0; i < 8; ++i) {
                const int top = bits == 1 ? (digits[i] & 1) : ((digits[i] >> (bits - 1)) & 1);
                pb = static_cast<uint8_t>(pb | static_cast<uint8_t>(top << (7 - i)));
            }
            const int branch = static_cast<int>(rng() & 1u);
            uint8_t buf[16] = {0};
            e8qlib::PackRefineBlock(digits, branch, bits, buf);
            uint8_t back[8];
            int br_back = -1;
            e8qlib::UnpackRefineBlock(buf, bits, pb, back, &br_back);
            for (int i = 0; i < 8; ++i) {
                if (back[i] != digits[i]) {
                    all_ok = false;
                }
            }
            if (br_back != branch) {
                all_ok = false;
            }
        }
    }
    Check(all_ok, "refine pack/unpack + rowA roundtrip b=1..8");
    // b = 9..13 wide arms: digits exceed 255, low fields span up to 96 bits.
    for (int bits = 9; bits <= e8qlib::kMaxQuantizeBits; ++bits) {
        const int q = 1 << bits;
        for (int trial = 0; trial < 200; ++trial) {
            uint16_t digits[8];
            int par = 0;
            for (int i = 0; i < 8; ++i) {
                digits[i] = static_cast<uint16_t>(rng() % static_cast<unsigned>(q));
            }
            for (int i = 0; i < 7; ++i) {
                par ^= digits[i] & 1;
            }
            digits[7] = static_cast<uint16_t>((digits[7] & ~1u) | static_cast<unsigned>(par));
            uint8_t pb = 0;
            for (int i = 0; i < 8; ++i) {
                pb = static_cast<uint8_t>(
                    pb | static_cast<uint8_t>(((digits[i] >> (bits - 1)) & 1) << (7 - i))
                );
            }
            const int branch = static_cast<int>(rng() & 1u);
            uint8_t buf[16] = {0};
            e8qlib::PackRefineBlock(digits, branch, bits, buf);
            uint16_t back[8];
            int br_back = -1;
            e8qlib::UnpackRefineBlock(buf, bits, pb, back, &br_back);
            for (int i = 0; i < 8; ++i) {
                if (back[i] != digits[i]) {
                    all_ok = false;
                }
            }
            if (br_back != branch) {
                all_ok = false;
            }
        }
    }
    Check(all_ok, "refine pack/unpack + rowA roundtrip b=9..13 (wide)");
}

void TestSplitPlanes() {
    // b = 2..13: pack random even-parity digits through the same staging rows
    // Build uses (PackCoarseRow rowA bytes + PackRefineBlock low fields),
    // then decode via DecodeSplitBlockLane and compare against the encoder's
    // true digits (validates the rowA + (b-1)-bit refine layout and the
    // parity-shell reconstruction, incl. the branch in digit 7's slot).
    std::mt19937 rng(31);
    bool ok = true;
    for (int bits = 2; bits <= e8qlib::kMaxQuantizeBits; ++bits) {
        const int q = 1 << bits;
        const int blocks = 2;
        std::vector<uint8_t> rows(32 * static_cast<size_t>(blocks), 0);
        std::vector<uint8_t> rf(
            32 * static_cast<size_t>(blocks) * static_cast<size_t>(bits - 1), 0
        );
        uint16_t ref_dig[32][8];
        int ref_br[32];
        for (int v = 0; v < 32; ++v) {
            for (int blk = 0; blk < blocks; ++blk) {
                uint16_t dg[8];
                int par = 0;
                for (int i = 0; i < 7; ++i) {
                    dg[i] = static_cast<uint16_t>(rng() % static_cast<unsigned>(q));
                    par ^= dg[i] & 1;
                }
                dg[7] = static_cast<uint16_t>(
                    ((rng() % static_cast<unsigned>(q / 2)) << 1) | static_cast<unsigned>(par)
                );
                const int br = static_cast<int>(rng() & 1u);
                if (blk == 0) {
                    std::memcpy(ref_dig[v], dg, sizeof(dg));
                    ref_br[v] = br;
                }
                e8qlib::PackCoarseRow(dg, br, bits, blk, &rows[v * static_cast<size_t>(blocks)]);
                e8qlib::PackRefineBlock(
                    dg, br, bits,
                    &rf[(static_cast<size_t>(v) * blocks + static_cast<size_t>(blk)) *
                        static_cast<size_t>(bits - 1)]
                );
            }
        }
        // Transpose to the byte-per-lane batch layout (Build's pack step).
        std::vector<uint8_t> rowa(static_cast<size_t>(blocks) * 32, 0);
        for (int blk = 0; blk < blocks; ++blk) {
            for (int lane = 0; lane < 32; ++lane) {
                rowa[static_cast<size_t>(blk) * 32 + lane] =
                    rows[static_cast<size_t>(lane) * static_cast<size_t>(blocks) +
                         static_cast<size_t>(blk)];
            }
        }
        for (int v = 0; v < 32; ++v) {
            if (bits <= 8) {
                uint8_t dg[8];
                int br = -1;
                e8qlib::DecodeSplitBlockLane(
                    bits, rowa.data(),
                    rf.data() + static_cast<size_t>(v) * static_cast<size_t>(blocks) *
                                    static_cast<size_t>(bits - 1),
                    0, static_cast<uint32_t>(v), dg, &br
                );
                for (int i = 0; i < 8; ++i) {
                    if (dg[i] != ref_dig[v][i]) {
                        if (ok) {
                            std::printf(
                                "  split mismatch bits=%d v=%d dim=%d got=%d want=%d br=%d/%d\n",
                                bits, v, i, dg[i], ref_dig[v][i], br, ref_br[v]
                            );
                        }
                        ok = false;
                    }
                }
                if (br != ref_br[v]) {
                    ok = false;
                }
            } else {
                uint16_t dg[8];
                int br = -1;
                e8qlib::DecodeSplitBlockLane(
                    bits, rowa.data(),
                    rf.data() + static_cast<size_t>(v) * static_cast<size_t>(blocks) *
                                    static_cast<size_t>(bits - 1),
                    0, static_cast<uint32_t>(v), dg, &br
                );
                for (int i = 0; i < 8; ++i) {
                    if (dg[i] != ref_dig[v][i]) {
                        ok = false;
                    }
                }
                if (br != ref_br[v]) {
                    ok = false;
                }
            }
        }
    }
    Check(ok, "split pack/decode roundtrip b=2..13");
}

void TestEncoder() {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    bool ok = true;
    for (int bits = 1; bits <= e8qlib::kMaxQuantizeBits; ++bits) {
        const int q = 1 << bits;
        for (int trial = 0; trial < 100; ++trial) {
            std::vector<float> u(64);
            double n2 = 0.0;
            for (auto& v : u) {
                v = nd(rng);
                n2 += static_cast<double>(v) * v;
            }
            const double inv = 1.0 / std::sqrt(n2);
            for (auto& v : u) {
                v = static_cast<float>(v * inv);
            }
            const e8qlib::VectorCode vc = e8qlib::EncodeVector(u.data(), 64, bits, nullptr);
            for (int blk = 0; blk < 8; ++blk) {
                int par = 0;
                for (int i = 0; i < 8; ++i) {
                    const int d = vc.digits[static_cast<size_t>(blk)][static_cast<size_t>(i)];
                    if (d < 0 || d >= q) {
                        ok = false;
                    }
                    par ^= d & 1;
                }
                if (par != 0) {
                    ok = false;
                }
            }
        }
    }
    Check(ok, "encoder parity + digit range b=1..13");
}

void TestEndToEnd(int bits) {
    // Synthetic cluster-friendly data: 16 gaussian clusters.
    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    const size_t dim = [] {
        const char* e = std::getenv("E8Q_TEST_DIM");
        return e != nullptr ? static_cast<size_t>(std::atol(e)) : 64UL;
    }();
    const size_t num_clusters = 16;
    const size_t per_cluster = 128;
    const size_t n = num_clusters * per_cluster;
    std::vector<float> centroids(num_clusters * dim);
    for (auto& v : centroids) {
        v = nd(rng) * 4.0F;
    }
    std::vector<float> base(n * dim);
    std::vector<uint32_t> cluster_ids(n);
    for (size_t c = 0; c < num_clusters; ++c) {
        for (size_t i = 0; i < per_cluster; ++i) {
            const size_t row = c * per_cluster + i;
            cluster_ids[row] = static_cast<uint32_t>(c);
            for (size_t d = 0; d < dim; ++d) {
                base[row * dim + d] = centroids[c * dim + d] + nd(rng);
            }
        }
    }
    e8qlib::E8qIndex index;
    // Conservative per-query prune-loss budget: the tests assert exact
    // agreement with the exhaustive estimator on small random data, where
    // the concentration tail is fatter than on large corpora.
    e8qlib::E8qBuildOptions test_opt;
    test_opt.prune_alpha = 1e-6F;
    index.Build(
        base.data(), centroids.data(), cluster_ids.data(), n, dim, num_clusters, bits, 99, 4,
        test_opt
    );
    // Round-trip through save/load.
    const std::string path = "/tmp/e8q_e8q_test.idx";
    index.Save(path.c_str());
    e8qlib::E8qIndex loaded;
    loaded.Load(path.c_str());

    e8qlib::QueryScratch scratch;
    const size_t k = 10;
    std::vector<uint32_t> got(k);
    std::vector<std::pair<float, uint32_t>> oracle;
    bool ok = true;
    for (int qq = 0; qq < 20; ++qq) {
        std::vector<float> query(dim);
        for (auto& v : query) {
            v = nd(rng) * 4.0F;
        }
        for (e8qlib::E8qIndex* idx : {&index, &loaded}) {
            // Exactness check with the probabilistic prune disabled.
            scratch.disable_prune = true;
            idx->Search(query.data(), k, num_clusters, got.data(), scratch, nullptr);
            scratch.disable_prune = false;
            idx->ExhaustiveTopK(query.data(), k + 8, oracle);
            // Ranks must match exactly except for boundary micro-ties:
            // two candidates whose estimator distances agree to float
            // noise may swap at the k-th slot (the search's fp32 centroid
            // score and the oracle's loop can differ by ~1 ulp). Allow a
            // swap only if the received id's oracle distance is within
            // 1e-4 relative of the k-th oracle distance.
            for (size_t i = 0; i < k; ++i) {
                if (oracle[i].second == got[i]) {
                    continue;
                }
                float got_dist = std::numeric_limits<float>::infinity();
                for (const auto& o : oracle) {
                    if (o.second == got[i]) {
                        got_dist = o.first;
                        break;
                    }
                }
                const float limit = oracle[k - 1].first * 1.0001F + 1e-4F;
                if (got_dist <= limit) {
                    continue;  // tie-level swap at the boundary
                }
                if (std::getenv("E8Q_TEST_VERBOSE") != nullptr) {
                    std::printf(
                        "  mismatch bits=%d qq=%d i=%zu oracle=%u got=%u (got dist %.4f limit %.4f)\n",
                        bits, qq, i, oracle[i].second, got[i], got_dist, limit
                    );
                }
                ok = false;
            }
        }
    }
    char msg[96];
    std::snprintf(msg, sizeof(msg), "end-to-end search == exhaustive estimator (bits=%d)", bits);
    Check(ok, msg);
}

void TestCoarseKernel(int bits, int dim_in) {
    // 32 random unit vectors; verify the packed rowA payload + the
    // branch-blind class-mean LUT + accumulate reproduce the w_cls model
    // per lane (the scan's exact price up to the u8 LUT quantization).
    std::mt19937 rng(4242 + bits + dim_in);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    const int padded = dim_in;
    const int blocks = padded / 8;
    // The scan's class-mean basis (same formula as PrepareQuery/InitBitTables).
    const std::vector<double> l0 = e8qlib::BranchLevels(bits, 0);
    const std::vector<double> l1 = e8qlib::BranchLevels(bits, 1);
    const int cvals = 1 << (bits - 1);
    double w_cls[2] = {0.0, 0.0};
    for (int t2 = 0; t2 < 2; ++t2) {
        double s0 = 0.0;
        double s1 = 0.0;
        for (int lo = 0; lo < cvals; ++lo) {
            s0 += l0[static_cast<size_t>((t2 << (bits - 1)) | lo)];
            s1 += l1[static_cast<size_t>((t2 << (bits - 1)) | lo)];
        }
        w_cls[t2] = 0.5 * (s0 / cvals + s1 / cvals);
    }
    std::vector<float> unit(32 * padded);
    std::vector<std::vector<uint8_t>> row_rows(32);
    for (int v = 0; v < 32; ++v) {
        double n2 = 0.0;
        for (int i = 0; i < padded; ++i) {
            unit[v * padded + i] = nd(rng);
            n2 += static_cast<double>(unit[v * padded + i]) * unit[v * padded + i];
        }
        for (int i = 0; i < padded; ++i) {
            unit[v * padded + i] = static_cast<float>(unit[v * padded + i] / std::sqrt(n2));
        }
        const e8qlib::VectorCode vc =
            e8qlib::EncodeVector(unit.data() + v * padded, padded, bits, nullptr);
        row_rows[v].assign(blocks, 0);
        for (int blk = 0; blk < blocks; ++blk) {
            e8qlib::PackCoarseRow(
                vc.digits[static_cast<size_t>(blk)].data(),
                vc.branch[static_cast<size_t>(blk)], bits, blk, row_rows[v].data()
            );
        }
    }
    // Transposed byte-per-lane batch layout (Build's pack step).
    std::vector<uint8_t> payload(static_cast<size_t>(blocks) * 32, 0);
    for (int blk = 0; blk < blocks; ++blk) {
        for (int lane = 0; lane < 32; ++lane) {
            payload[static_cast<size_t>(blk) * 32 + lane] =
                row_rows[static_cast<size_t>(lane)][static_cast<size_t>(blk)];
        }
    }
    std::vector<float> q(padded);
    for (auto& v : q) {
        v = nd(rng);
    }
    // PrepareQuery's LUT: per block, the 16-entry half tables A (dims 4..7,
    // low nibble) and B (dims 0..3, high nibble), each duplicated, symmetric
    // u8-quantized around zero.
    std::vector<float> cf(static_cast<size_t>(blocks) * 64, 0.0F);
    for (int blk = 0; blk < blocks; ++blk) {
        const float* qb = q.data() + static_cast<size_t>(blk) * 8;
        float* A = cf.data() + static_cast<size_t>(blk) * 64;
        float* B = A + 32;
        for (int v = 0; v < 16; ++v) {
            double av = 0.0;
            double bv = 0.0;
            for (int h = 0; h < 4; ++h) {
                const int bit = (v >> (3 - h)) & 1;
                av += static_cast<double>(qb[4 + h]) * w_cls[bit];
                bv += static_cast<double>(qb[h]) * w_cls[bit];
            }
            A[v] = static_cast<float>(av);
            B[v] = static_cast<float>(bv);
        }
        for (int c = 0; c < 16; ++c) {
            A[16 + c] = A[c];
            B[16 + c] = B[c];
        }
    }
    float hc = 0.0F;
    for (float v : cf) {
        hc = std::max(hc, std::abs(v));
    }
    const float delta = hc > 0.0F ? 2.0F * hc / 255.0F : 1.0F;
    std::vector<uint8_t> lut_c(cf.size());
    for (size_t i = 0; i < cf.size(); ++i) {
        const int v = static_cast<int>(std::lround((cf[i] + hc) / delta));
        lut_c[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
    }
    const float base_c = -hc * static_cast<float>(2 * blocks);
    alignas(32) int32_t accu[32];
    e8qlib::AccumulateCoarseHi(
        payload.data(), lut_c.data(), nullptr, static_cast<size_t>(blocks), accu
    );
    bool ok = true;
    double max_rel = 0.0;
    for (int v = 0; v < 32; ++v) {
        const e8qlib::VectorCode vc =
            e8qlib::EncodeVector(unit.data() + v * padded, padded, bits, nullptr);
        double model = 0.0;
        for (int blk = 0; blk < blocks; ++blk) {
            for (int i = 0; i < 7; ++i) {
                model += q[blk * 8 + i] *
                         w_cls[vc.digits[static_cast<size_t>(blk)][static_cast<size_t>(i)] >>
                               (bits - 1)];
            }
            // Dim 7's byte slot carries the BRANCH at b == 1 (the parity sign
            // is absorbed by the orthogonal residual) and the top bit at
            // b >= 2 — price exactly what the scan prices.
            model += q[blk * 8 + 7] *
                     (bits == 1 ? w_cls[vc.branch[static_cast<size_t>(blk)] & 1]
                                : w_cls[vc.digits[static_cast<size_t>(blk)][7] >>
                                        (bits - 1)]);
        }
        const double kernel = static_cast<double>(delta) * accu[v] + base_c;
        const double abs_err = std::abs(kernel - model);
        const double tol = (delta / 2.0) * (2 * blocks) * 1.3;
        const double rel = abs_err / tol;
        max_rel = std::max(max_rel, rel);
        if (abs_err > tol) {
            ok = false;
        }
    }
    char msg[96];
    std::snprintf(
        msg, sizeof(msg), "coarse kernel == class-mean model (bits=%d, max_err/tol=%.3f)",
        bits, max_rel
    );
    Check(ok, msg);
}

}  // namespace

int main() {
    TestIdentities();
    TestPacking();
    TestSplitPlanes();
    TestEncoder();
    for (int bits = 1; bits <= 8; ++bits) {
        TestEndToEnd(bits);
    }
    // b = 9..13: the wide IVF arms (1-bit gate + survivor-only refine).
    for (int bits : {9, 13}) {
        TestEndToEnd(bits);
    }
    for (int dim : {64, 1536}) {
        for (int bits : {1, 2, 4}) {
            TestCoarseKernel(bits, dim);
        }
    }
    if (failures != 0) {
        std::printf("%d FAILURES\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
