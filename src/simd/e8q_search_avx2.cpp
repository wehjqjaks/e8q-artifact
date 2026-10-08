#include "index/e8q_scan_common.hpp"
#include "simd/e8q_search.hpp"
#include "simd/simd_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>

#include "quantization/me8q_bits.hpp"

namespace e8qlib::search_simd {
namespace {

using A = Access;
using simd::HsumPs;
using simd::HminPs;
using simd::HmaxEpi32;
using simd::HminEpi32;

}  // namespace

void QuantizeU8Avx2(
    const float* src, size_t n, float qinv, float qoff, float hc, float delta, uint8_t* dst
) {
    size_t i = 0;
    const __m256 vinv = _mm256_set1_ps(qinv);
    const __m256 voff = _mm256_set1_ps(qoff);
    const __m256i vzero = _mm256_setzero_si256();
    const __m256i vmax = _mm256_set1_epi32(255);
    for (; i + 8 <= n; i += 8) {
        const __m256 x = _mm256_fmadd_ps(_mm256_loadu_ps(src + i), vinv, voff);
        __m256i v = _mm256_cvtps_epi32(x);
        v = _mm256_max_epi32(v, vzero);
        v = _mm256_min_epi32(v, vmax);
        const __m256i p16 = _mm256_packus_epi32(v, v);
        const __m256i p8 = _mm256_packus_epi16(p16, p16);
        uint32_t a = 0;
        uint32_t b = 0;
        std::memcpy(&a, &p8, 4);
        const __m128i hi = _mm256_extracti128_si256(p8, 1);
        std::memcpy(&b, &hi, 4);
        std::memcpy(dst + i, &a, 4);
        std::memcpy(dst + i + 4, &b, 4);
    }
    QuantizeU8Tail(src, i, n, hc, delta, dst);
}

// b == 1 exact score: the rowA byte is the whole code — digits 0..6 are the
// sign bits (bit 7-i), digit 7 is their parity (the E8 shell), bit 0 is the
// branch. The two-level tables are affine, so one gather per block.
float ExactScoreB1Avx2(
    const E8qIndex& idx, const uint8_t* plane, uint32_t lane, const float* q
) {
    const int blocks = A::blocks(idx);
    const float* tab0 = A::refine_tab(idx);
    const float t00 = tab0[0];
    const float t01 = tab0[1];
    const float t10 = tab0[256];
    const float t11 = tab0[257];
    alignas(32) const float fold8[8] = {t00, t01, t10, t11, t00, t01, t10, t11};
    const __m256 fold8v = _mm256_load_ps(fold8);
    const uint64_t* pl = PlaneExpandLut();
    __m256 acc = _mm256_setzero_ps();
    for (int blk = 0; blk < blocks; ++blk) {
        const int xb = plane[static_cast<size_t>(blk) * 32 + lane];
        const int par = __builtin_popcount(static_cast<unsigned>(xb & 0xFE)) & 1;
        const int br_sel = xb & 1;
        uint64_t e = pl[static_cast<unsigned>(xb) & 255u];
        e = (e & 0x00FFFFFFFFFFFFFFULL) | (static_cast<uint64_t>(par) << 56);
        const __m256i idx = _mm256_or_si256(
            _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(e))),
            _mm256_set1_epi32(br_sel << 1)
        );
        const __m256 lv = _mm256_permutevar8x32_ps(fold8v, idx);
        acc = _mm256_fmadd_ps(lv, _mm256_loadu_ps(q + blk * 8), acc);
    }
    return HsumPs(acc);
}

template <int BITS>
inline void RefineCoreAvx2(
    const E8qIndex& idx, const SurvivorRow* rows, size_t n, const float* q, QueryScratch& s
) {
    // b = 2..8 refine core, templated on the bit budget: the level-table
    // registers and the digit-unpack shape resolve at compile time. The
    // runtime-branch version kept every b's tables and code live in one
    // function and the compiler spilled the hot loop to the stack.
    static_assert(BITS >= 2 && BITS <= 8, "RefineCoreAvx2 covers b = 2..8");
    const int blocks = A::blocks(idx);
    const float* refine_tab = A::refine_tab(idx);
    // Low (BITS-1) bits per dim live in the refine field; the rowA byte
    // carries the top bit. Digit 7's stored field bit 0 is the branch, its
    // true LSB is the parity of digits 0..6.
    constexpr int CBITS = BITS - 1;
    const __m256i fmask = _mm256_set1_epi32((1 << CBITS) - 1);
    const __m256i fsh = _mm256_setr_epi32(
        0, CBITS, 2 * CBITS, 3 * CBITS, CBITS <= 4 ? 4 * CBITS : 0,
        CBITS <= 4 ? 5 * CBITS : 0, CBITS <= 4 ? 6 * CBITS : 0,
        CBITS <= 4 ? 7 * CBITS : 0
    );
    const uint64_t par_mask = me8q::ParMaskOf(CBITS);
    const uint64_t* pl = PlaneExpandLut();
    [[maybe_unused]] __m256 fold8v{};
    [[maybe_unused]] __m256 tbr0_0{}, tbr1_0{}, tbr0_1{}, tbr1_1{};
    [[maybe_unused]] __m256 tbr0_2{}, tbr1_2{}, tbr0_3{}, tbr1_3{};
    if constexpr (BITS == 2) {
        alignas(32) float fold8[8];
        for (int i = 0; i < 8; ++i) {
            fold8[static_cast<unsigned>(i)] =
                refine_tab[static_cast<size_t>(i >> 2) * 256 + static_cast<size_t>(i & 3)];
        }
        fold8v = _mm256_loadu_ps(fold8);
    } else if constexpr (BITS == 3) {
        tbr0_0 = _mm256_loadu_ps(refine_tab);
        tbr1_0 = _mm256_loadu_ps(refine_tab + 256);
    } else if constexpr (BITS == 4) {
        tbr0_0 = _mm256_loadu_ps(refine_tab);
        tbr1_0 = _mm256_loadu_ps(refine_tab + 256);
        tbr0_1 = _mm256_loadu_ps(refine_tab + 8);
        tbr1_1 = _mm256_loadu_ps(refine_tab + 264);
    } else if constexpr (BITS == 5) {
        tbr0_0 = _mm256_loadu_ps(refine_tab);
        tbr1_0 = _mm256_loadu_ps(refine_tab + 256);
        tbr0_1 = _mm256_loadu_ps(refine_tab + 8);
        tbr1_1 = _mm256_loadu_ps(refine_tab + 264);
        tbr0_2 = _mm256_loadu_ps(refine_tab + 16);
        tbr1_2 = _mm256_loadu_ps(refine_tab + 272);
        tbr0_3 = _mm256_loadu_ps(refine_tab + 24);
        tbr1_3 = _mm256_loadu_ps(refine_tab + 280);
    }
    for (size_t j0 = 0; j0 < n; j0 += 4) {
        const size_t cnt = std::min<size_t>(4, n - j0);
        __m256 acc[4];
        __m256 acc2[4];
        const uint8_t* refp[4];
        const uint8_t* rowp[4];
        for (size_t jj = 0; jj < cnt; ++jj) {
            acc[jj] = _mm256_setzero_ps();
            acc2[jj] = _mm256_setzero_ps();
            const uint32_t lj = rows[j0 + jj].lane;
            refp[jj] = rows[j0 + jj].refine;
            rowp[jj] = rows[j0 + jj].plane + lj;
        }
        for (int blk = 0; blk < blocks; ++blk) {
            const __m256 qv = _mm256_loadu_ps(q + blk * 8);
            for (size_t jj = 0; jj < cnt; ++jj) {
                uint64_t x = 0;
                const uint8_t* rp =
                    refp[jj] + static_cast<size_t>(blk) * static_cast<size_t>(CBITS);
                std::memcpy(&x, rp, 8);
                const int br_sel = static_cast<int>((x >> (7 * CBITS)) & 1ull);
                const uint64_t parbit = __builtin_popcountll(x & par_mask) & 1ull;
                x = (x & ~(uint64_t{1} << (7 * CBITS))) | (parbit << (7 * CBITS));
                __m256i dig;
                if constexpr (CBITS <= 4) {
                    dig = _mm256_and_si256(
                        _mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(x)), fsh),
                        fmask
                    );
                } else {
                    const __m256i lo = _mm256_and_si256(
                        _mm256_srlv_epi32(
                            _mm256_set1_epi32(static_cast<int>(x & 0xFFFFFFFFu)), fsh
                        ),
                        fmask
                    );
                    const __m256i hi = _mm256_and_si256(
                        _mm256_srlv_epi32(
                            _mm256_set1_epi32(
                                static_cast<int>((x >> (4 * CBITS)) & 0xFFFFFFFFu)
                            ),
                            fsh
                        ),
                        fmask
                    );
                    dig = _mm256_permute2x128_si256(lo, hi, 0x20);
                }
                const uint8_t rbA = rowp[jj][static_cast<size_t>(blk) * 32];
                const __m256i tops = _mm256_slli_epi32(
                    _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(pl[rbA]))),
                    CBITS
                );
                dig = _mm256_or_si256(dig, tops);
                __m256 lv;
                if constexpr (BITS == 2) {
                    const __m256i dg2 = _mm256_or_si256(
                        dig, _mm256_slli_epi32(_mm256_set1_epi32(br_sel), 2)
                    );
                    lv = _mm256_permutevar8x32_ps(fold8v, dg2);
                } else if constexpr (BITS == 3) {
                    lv = _mm256_permutevar8x32_ps(br_sel != 0 ? tbr1_0 : tbr0_0, dig);
                } else if constexpr (BITS == 4) {
                    const __m256 msk = _mm256_castsi256_ps(_mm256_slli_epi32(dig, 28));
                    lv = _mm256_blendv_ps(
                        _mm256_permutevar8x32_ps(br_sel != 0 ? tbr1_0 : tbr0_0, dig),
                        _mm256_permutevar8x32_ps(br_sel != 0 ? tbr1_1 : tbr0_1, dig), msk
                    );
                } else if constexpr (BITS == 5) {
                    const __m256 msk3 = _mm256_castsi256_ps(_mm256_slli_epi32(dig, 28));
                    const __m256 msk4 = _mm256_castsi256_ps(_mm256_slli_epi32(dig, 27));
                    const __m256 p0 = _mm256_permutevar8x32_ps(
                        br_sel != 0 ? tbr1_0 : tbr0_0, dig
                    );
                    const __m256 p1 = _mm256_permutevar8x32_ps(
                        br_sel != 0 ? tbr1_1 : tbr0_1, dig
                    );
                    const __m256 p2 = _mm256_permutevar8x32_ps(
                        br_sel != 0 ? tbr1_2 : tbr0_2, dig
                    );
                    const __m256 p3 = _mm256_permutevar8x32_ps(
                        br_sel != 0 ? tbr1_3 : tbr0_3, dig
                    );
                    lv = _mm256_blendv_ps(
                        _mm256_blendv_ps(p0, p1, msk3), _mm256_blendv_ps(p2, p3, msk3), msk4
                    );
                } else {
                    const float* tab = refine_tab + static_cast<size_t>(br_sel) * 256;
                    lv = _mm256_i32gather_ps(tab, dig, 4);
                }
                __m256& dst = (blk & 1) != 0 ? acc2[jj] : acc[jj];
                dst = _mm256_fmadd_ps(lv, qv, dst);
            }
        }
        for (size_t jj = 0; jj < cnt; ++jj) {
            acc[jj] = _mm256_add_ps(acc[jj], acc2[jj]);
            const SurvivorRow& row = rows[j0 + jj];
            A::InsertHeap(idx, row.h3 - 2.0F * row.g1 * HsumPs(acc[jj]), row.id, s);
        }
    }
}

void RefineBatchAvx2(
    const E8qIndex& idx, const SurvivorRow* rows, size_t n, const float* q, QueryScratch& s
) {
    if (n == 0) {
        return;
    }
    const int bits = A::bits(idx);
    const int blocks = A::blocks(idx);
    const float* refine_tab = A::refine_tab(idx);
    if (bits >= 9) {
        // Wide twin of RefineCore: the refine field is c = bits-1 up to 12
        // bytes per block, so the staging integer is 128-bit and the digits
        // are 16-bit.
        const int c = bits - 1;
        const size_t stride = static_cast<size_t>(1) << bits;
        const uint64_t fmask64 = (uint64_t{1} << c) - 1;
        const __m256i fmask = _mm256_set1_epi64x(static_cast<int64_t>(fmask64));
        const __m256i sh_lo = _mm256_setr_epi64x(0, c, 2 * c, 3 * c);
        // Fields 4..7 sit at bits 4c..7c of the 128-bit row; 7c <= 63 (c <= 9)
        // keeps them in the low 64 bits, above that the (lo>>32)|(hi<<32)
        // window covers bits 32..95. The hi half supplies global bits
        // 64..(8c-1): keep exactly the bits the fields span.
        const bool cross = 7 * c >= 64;
        const int sh_base = cross ? 4 * c - 32 : 4 * c;
        const uint64_t hi_mask =
            cross ? ((uint64_t{1} << (8 * c - 64)) - 1) : 0;
        const __m256i sh_hi = _mm256_setr_epi64x(
            sh_base, sh_base + c, sh_base + 2 * c, sh_base + 3 * c
        );
        uint64_t pm_lo = 0;
        uint64_t pm_w2 = 0;
        for (int i = 0; i < 7; ++i) {
            if (c * i < 64) {
                pm_lo |= uint64_t{1} << (c * i);
            } else {
                pm_w2 |= uint64_t{1} << (c * i - 32);
            }
        }
        const __m256i pick_lo32 = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        const uint64_t* pl = PlaneExpandLut();
        for (size_t j0 = 0; j0 < n; j0 += 4) {
            const size_t cnt = std::min<size_t>(4, n - j0);
            __m256 acc[4];
            __m256 acc2[4];
            const uint8_t* refp[4];
            const uint8_t* rowp[4];
            for (size_t jj = 0; jj < cnt; ++jj) {
                acc[jj] = _mm256_setzero_ps();
                acc2[jj] = _mm256_setzero_ps();
                refp[jj] = rows[j0 + jj].refine;
                rowp[jj] = rows[j0 + jj].plane + rows[j0 + jj].lane;
            }
            for (int blk = 0; blk < blocks; ++blk) {
                const __m256 qv = _mm256_loadu_ps(q + static_cast<size_t>(blk) * 8);
                __m256& dst0 = (blk & 1) != 0 ? acc2[0] : acc[0];
                __m256& dst1 = cnt > 1 ? ((blk & 1) != 0 ? acc2[1] : acc[1]) : dst0;
                __m256& dst2 = cnt > 2 ? ((blk & 1) != 0 ? acc2[2] : acc[2]) : dst0;
                __m256& dst3 = cnt > 3 ? ((blk & 1) != 0 ? acc2[3] : acc[3]) : dst0;
                for (size_t jj = 0; jj < cnt; ++jj) {
                    const uint8_t* rp =
                        refp[jj] + static_cast<size_t>(blk) * static_cast<size_t>(c);
                    uint64_t lo = 0;
                    std::memcpy(&lo, rp, 8);
                    uint64_t hi = 0;
                    std::memcpy(&hi, rp + 8, 8);
                    hi &= hi_mask;
                    const uint64_t w2 = (lo >> 32) | (hi << 32);
                    const uint8_t rbA = rowp[jj][static_cast<size_t>(blk) * 32];
                    const uint64_t bpos = static_cast<uint64_t>(cross ? c * 7 - 32 : c * 7);
                    const uint64_t src = cross ? w2 : lo;
                    const int br = static_cast<int>((src >> bpos) & 1u);
                    const int par = (__builtin_popcountll(lo & pm_lo) ^
                                     __builtin_popcountll(w2 & pm_w2)) &
                                    1;
                    const uint64_t patch = static_cast<uint64_t>(par) << bpos;
                    const uint64_t base2 = (src & ~(uint64_t{1} << bpos)) | patch;
                    const __m256i fl = _mm256_and_si256(
                        _mm256_srlv_epi64(
                            _mm256_set1_epi64x(static_cast<int64_t>(lo)), sh_lo
                        ),
                        fmask
                    );
                    const __m256i fh = _mm256_and_si256(
                        _mm256_srlv_epi64(
                            _mm256_set1_epi64x(static_cast<int64_t>(base2)), sh_hi
                        ),
                        fmask
                    );
                    const __m256i f04 = _mm256_permutevar8x32_epi32(fl, pick_lo32);
                    const __m256i f47 = _mm256_permutevar8x32_epi32(fh, pick_lo32);
                    const __m256i tops = _mm256_slli_epi32(
                        _mm256_cvtepu8_epi32(
                            _mm_cvtsi64_si128(static_cast<long long>(pl[rbA]))
                        ),
                        c
                    );
                    const __m256i dig = _mm256_or_si256(
                        tops, _mm256_permute2x128_si256(f04, f47, 0x20)
                    );
                    const float* tab = refine_tab + static_cast<size_t>(br) * stride;
                    const __m256 lv = _mm256_i32gather_ps(tab, dig, 4);
                    __m256& dst = jj == 0 ? dst0 : jj == 1 ? dst1 : jj == 2 ? dst2 : dst3;
                    dst = _mm256_fmadd_ps(lv, qv, dst);
                }
            }
            for (size_t jj = 0; jj < cnt; ++jj) {
                const SurvivorRow& row = rows[j0 + jj];
                const float dist =
                    row.h3 - 2.0F * row.g1 * HsumPs(_mm256_add_ps(acc[jj], acc2[jj]));
                A::InsertHeap(idx, dist, row.id, s);
            }
        }
        return;
    }
    if (bits >= 2) {
        switch (bits) {
            case 2: RefineCoreAvx2<2>(idx, rows, n, q, s); return;
            case 3: RefineCoreAvx2<3>(idx, rows, n, q, s); return;
            case 4: RefineCoreAvx2<4>(idx, rows, n, q, s); return;
            case 5: RefineCoreAvx2<5>(idx, rows, n, q, s); return;
            case 6: RefineCoreAvx2<6>(idx, rows, n, q, s); return;
            case 7: RefineCoreAvx2<7>(idx, rows, n, q, s); return;
            case 8: RefineCoreAvx2<8>(idx, rows, n, q, s); return;
            default: break;
        }
    }
    // b == 1: the rowA byte is the whole code (digits 0..6 = signs, digit 7
    // = parity, bit 0 = branch). Four affine levels pack into fold8v; one
    // permutevar replaces the 3-FMA (t0, dd, lv) plus gather/cvtepi32 path.
    const float t00 = refine_tab[0];
    const float t01 = refine_tab[1];
    const float t10 = refine_tab[256];
    const float t11 = refine_tab[257];
    alignas(32) const float fold8[8] = {t00, t01, t10, t11, t00, t01, t10, t11};
    const __m256 fold8v = _mm256_load_ps(fold8);
    const uint64_t* pl = PlaneExpandLut();
    for (size_t j0 = 0; j0 < n; j0 += 4) {
        const size_t cnt = std::min<size_t>(4, n - j0);
        __m256 acc[4];
        __m256 acc2[4];
        const uint8_t* rowp[4];
        for (size_t jj = 0; jj < cnt; ++jj) {
            acc[jj] = _mm256_setzero_ps();
            acc2[jj] = _mm256_setzero_ps();
            rowp[jj] = rows[j0 + jj].plane + rows[j0 + jj].lane;
        }
        for (int blk = 0; blk < blocks; ++blk) {
            const __m256 qv = _mm256_loadu_ps(q + blk * 8);
            for (size_t jj = 0; jj < cnt; ++jj) {
                const int xb = rowp[jj][static_cast<size_t>(blk) * 32];
                const int par = __builtin_popcount(static_cast<unsigned>(xb & 0xFE)) & 1;
                const int br_sel = xb & 1;
                uint64_t e = pl[static_cast<unsigned>(xb) & 255u];
                e = (e & 0x00FFFFFFFFFFFFFFULL) | (static_cast<uint64_t>(par) << 56);
                const __m256i idx = _mm256_or_si256(
                    _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<long long>(e))),
                    _mm256_set1_epi32(br_sel << 1)
                );
                const __m256 lv = _mm256_permutevar8x32_ps(fold8v, idx);
                __m256& dst = (blk & 1) != 0 ? acc2[jj] : acc[jj];
                dst = _mm256_fmadd_ps(lv, qv, dst);
            }
        }
        for (size_t jj = 0; jj < cnt; ++jj) {
            acc[jj] = _mm256_add_ps(acc[jj], acc2[jj]);
            const SurvivorRow& row = rows[j0 + jj];
            A::InsertHeap(idx, row.h3 - 2.0F * row.g1 * HsumPs(acc[jj]), row.id, s);
        }
    }
}

void ScanClusterAvx2(
    const E8qIndex& idx, size_t cid, float d2, const float* q, QueryScratch& s,
    SearchStats* stats
) {
    const uint64_t vb = A::cluster_begin(idx, cid);
    const size_t cnt = static_cast<size_t>(A::cluster_begin(idx, cid + 1) - vb);
    if (cnt == 0) {
        return;
    }
    const uint8_t* batchp = A::arena(idx) + A::arena_begin(idx)[cid];
    const size_t batch_bytes = A::batch_bytes(idx);
    const size_t ref_batch_bytes = A::ref_batch_bytes(idx);
    const int bits = A::bits(idx);
    const int blocks = A::blocks(idx);
    const uint8_t* refbase = A::ref_arena(idx) + A::ref_arena_begin(idx)[cid];
    const uint32_t* ids = A::ids(idx);
    auto flush = [&]() { A::FlushSurvivors(idx, s, q, stats); };

    const __m256 vdelta_c = _mm256_set1_ps(s.prep_delta_c);
    const __m256 vbase_c = _mm256_set1_ps(s.prep_base_c);
    const float sqrt_d2 = std::sqrt(std::max(0.0F, d2));

    size_t off = 0;
    {
        auto finish_c = [&](const int32_t* accC, const uint8_t* bp, size_t off_cur,
                            size_t n_logic) {
            if (stats != nullptr) {
                stats->candidates += n_logic;
            }
            const uint8_t* refrec = refbase + (off_cur / 32) * ref_batch_bytes;
            const float* g1b = reinterpret_cast<const float*>(refrec + A::ref_g1_off(idx));
            const float* h3b = reinterpret_cast<const float*>(refrec + A::ref_h3_off(idx));
            const float distk0 = HeapLimit(s);
            const float r_horizon =
                (distk0 < d2) ? std::sqrt(std::max(0.0F, distk0)) : sqrt_d2;
            alignas(32) float dlo_lanes[32];
            __m256 dlo_r[4];
            {
                // Orthogonal-projection bound, three banks per point:
                //   dlo = d2 + h3~ - W*S - |W|*lut_err - W_round - lam*r_horizon*Omega
                // LUT pad is per-lane |W|; W u8-rounding is the batch pad_w_round.
                // Early-reject still uses Wmax (conservative skip). r_horizon =
                // min(sqrt(d2), sqrt(distk0)) once the heap is live.
                float base_min = 0.0F, base_scale_f = 0.0F, weight_min_f = 0.0F;
                float weight_scale_f = 0.0F, perp_min = 0.0F, perp_scale_f = 0.0F;
                float pad_w_lut = 0.0F, pad_w_round = 0.0F;
                OrthoBatchHeader(
                    bp, s.ortho_lut_err, s.ortho_score_max, base_min, base_scale_f,
                    weight_min_f, weight_scale_f, perp_min, perp_scale_f, pad_w_lut,
                    pad_w_round
                );
                if (!s.disable_prune && distk0 < std::numeric_limits<float>::infinity()) {
                    const float w_amax =
                        reinterpret_cast<const float*>(bp + E8qIndex::kScalesOff)[6];
                    const float pad_add = w_amax * pad_w_lut + pad_w_round;
                    const float perp_amax = perp_min + 255.0F * perp_scale_f;
                    __m256i m01 = _mm256_max_epi32(
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(accC)),
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(accC + 8))
                    );
                    __m256i m23 = _mm256_max_epi32(
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(accC + 16)),
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(accC + 24))
                    );
                    const __m256i m_all = _mm256_max_epi32(m01, m23);
                    const __m256i l01 = _mm256_min_epi32(
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(accC)),
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(accC + 8))
                    );
                    const __m256i l23 = _mm256_min_epi32(
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(accC + 16)),
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(accC + 24))
                    );
                    const __m256i l_all = _mm256_min_epi32(l01, l23);
                    // W is signed (ZPR): the upper bound of W*S over the batch
                    // needs max |S| in affine units, not the signed max.
                    const float s_hi = static_cast<float>(HmaxEpi32(m_all)) * s.prep_delta_c +
                                       s.prep_base_c;
                    const float s_lo = static_cast<float>(HminEpi32(l_all)) * s.prep_delta_c +
                                       s.prep_base_c;
                    const float s_absmax = std::max(std::abs(s_hi), std::abs(s_lo));
                    const float batch_dlo_lb = (d2 + base_min - pad_add) -
                                               w_amax * s_absmax -
                                               (s.ortho_z * r_horizon) * perp_amax;
                    if (batch_dlo_lb >= distk0) {
                        return;
                    }
                }
                const __m256 v_base_min = _mm256_set1_ps(base_min + d2 - pad_w_round);
                const __m256 v_base_scale = _mm256_set1_ps(base_scale_f);
                const __m256 v_weight_min = _mm256_set1_ps(weight_min_f);
                const __m256 v_weight_scale = _mm256_set1_ps(weight_scale_f);
                const __m256 v_perp_min = _mm256_set1_ps(perp_min);
                const __m256 v_perp_scale = _mm256_set1_ps(perp_scale_f);
                const __m256 v_slack_scale = _mm256_set1_ps(s.ortho_z * r_horizon);
                const __m256 v_pad_w_lut = _mm256_set1_ps(pad_w_lut);
                const uint8_t* base_bank = bp + E8qIndex::kH3QOff;
                const uint8_t* weight_bank = bp + E8qIndex::kCqOff;
                const uint8_t* perp_bank = bp + E8qIndex::kAqOff;
                for (int grp = 0; grp < 4; ++grp) {
                    const __m256 cf = _mm256_cvtepi32_ps(
                        _mm256_load_si256(reinterpret_cast<const __m256i*>(accC + grp * 8))
                    );
                    const __m256 scv = _mm256_fmadd_ps(cf, vdelta_c, vbase_c);
                    const __m256 base_term = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(base_bank + grp * 8)
                        ))),
                        v_base_scale, v_base_min
                    );
                    const __m256 weight = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(weight_bank + grp * 8)
                        ))),
                        v_weight_scale, v_weight_min
                    );
                    const __m256 perp = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(perp_bank + grp * 8)
                        ))),
                        v_perp_scale, v_perp_min
                    );
                    const __m256 dc = _mm256_fnmadd_ps(weight, scv, base_term);
                    const __m256 abs_w = _mm256_andnot_ps(_mm256_set1_ps(-0.0F), weight);
                    const __m256 dlo = _mm256_fnmadd_ps(
                        v_slack_scale, perp, _mm256_fnmadd_ps(abs_w, v_pad_w_lut, dc)
                    );
                    dlo_r[grp] = dlo;
                }
            }
            uint32_t mask = 0;
            if (s.disable_prune) {
                mask = 0xFFFFFFFFu;
            } else {
                const __m256 vlimit = _mm256_set1_ps(distk0);
                for (int grp = 0; grp < 4; ++grp) {
                    mask |= static_cast<uint32_t>(
                                _mm256_movemask_ps(_mm256_cmp_ps(dlo_r[grp], vlimit, _CMP_LT_OQ))
                            )
                            << (grp * 8);
                }
            }
            if (mask != 0) {
                for (int grp = 0; grp < 4; ++grp) {
                    _mm256_store_ps(dlo_lanes + grp * 8, dlo_r[grp]);
                }
            }
            CollectLanes(
                mask, n_logic, dlo_lanes, s, s.disable_prune, stats, flush,
                [&](int lane) {
                    return SurvivorRow{
                        bits >= 2
                            ? refbase + (off_cur / 32) * ref_batch_bytes +
                                  A::ref_refine_off(idx) +
                                  static_cast<size_t>(lane) *
                                      (static_cast<size_t>(blocks) *
                                       static_cast<size_t>(bits - 1))
                            : nullptr,
                        bp + A::p1_off(idx),
                        static_cast<uint32_t>(lane),
                        g1b[lane],
                        d2 + h3b[lane],
                        ids[static_cast<size_t>(vb) + off_cur + static_cast<size_t>(lane)]
                    };
                }
            );
        };
        while (off < cnt) {
            const size_t n_logic = std::min<size_t>(32, cnt - off);
            const bool pair = (n_logic == 32) && (cnt - off >= 64);
            alignas(32) int32_t accC[32];
            alignas(32) int32_t accB[32];
            if (pair) {
                AccumulateCoarseHiPair(batchp + A::p1_off(idx), batchp + batch_bytes + A::p1_off(idx),
                    s.lut_c.data(), s.lut_pair.data(), static_cast<size_t>(blocks), accC, accB
                );
            } else {
                AccumulateCoarseHi(batchp + A::p1_off(idx), s.lut_c.data(), s.lut_pair.data(), static_cast<size_t>(blocks),
                    accC
                );
            }
            finish_c(accC, batchp, off, n_logic);
            if (pair) {
                finish_c(accB, batchp + batch_bytes, off + 32, 32);
                batchp += 2 * batch_bytes;
                off += 64;
            } else {
                batchp += batch_bytes;
                off += 32;
            }
        }
        flush();
        return;
    }
}

}  // namespace e8qlib::search_simd
