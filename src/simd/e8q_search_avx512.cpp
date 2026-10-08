#include "index/e8q_scan_common.hpp"
#include "simd/e8q_search.hpp"
#include "simd/simd_helpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>

#include "quantization/me8q_bits.hpp"

namespace e8qlib::search_simd {
namespace {

using A = Access;

// AVX-512 top-bit unpack: bit (7-i) of the rowA byte is dim i's digit top
// bit, i.e. lane i of the tops vector. A bit-reversed index feeds
// vpbroadcastd-with-mask directly, replacing the u64 expand-table load, the
// GPR->XMM move, the byte->dword widening and the shift.
inline const uint8_t* BitRev8Lut() {
    static const std::array<uint8_t, 256> lut = [] {
        std::array<uint8_t, 256> t{};
        for (int v = 0; v < 256; ++v) {
            int r = 0;
            for (int i = 0; i < 8; ++i) {
                r |= ((v >> i) & 1) << (7 - i);
            }
            t[static_cast<size_t>(v)] = static_cast<uint8_t>(r);
        }
        return t;
    }();
    return lut.data();
}

// Level gather for b <= 8 digits (stride-256 tables): permute for b <= 5,
// 512-bit indexed gather above (the AVX2 kernel uses 4x256 gathers there).
inline __m256 LookupRefine(const float* tab, __m256i dig, int bits) {
    if (bits <= 4) {
        return _mm256_permutevar8x32_ps(_mm256_loadu_ps(tab), dig);
    }
    if (bits == 5) {
        const __m256 msk3 = _mm256_castsi256_ps(_mm256_slli_epi32(dig, 28));
        const __m256 msk4 = _mm256_castsi256_ps(_mm256_slli_epi32(dig, 27));
        const __m256 p0 = _mm256_permutevar8x32_ps(_mm256_loadu_ps(tab), dig);
        const __m256 p1 = _mm256_permutevar8x32_ps(_mm256_loadu_ps(tab + 8), dig);
        const __m256 p2 = _mm256_permutevar8x32_ps(_mm256_loadu_ps(tab + 16), dig);
        const __m256 p3 = _mm256_permutevar8x32_ps(_mm256_loadu_ps(tab + 24), dig);
        return _mm256_blendv_ps(
            _mm256_blendv_ps(p0, p1, msk3), _mm256_blendv_ps(p2, p3, msk3), msk4
        );
    }
    return simd::RefineLut512(tab, dig, 1 << bits);
}

}  // namespace

void QuantizeU8Avx512(
    const float* src, size_t n, float qinv, float qoff, float hc, float delta, uint8_t* dst
) {
    size_t i = 0;
    const __m512 vinv = _mm512_set1_ps(qinv);
    const __m512 voff = _mm512_set1_ps(qoff);
    for (; i + 16 <= n; i += 16) {
        const __m512 x = _mm512_fmadd_ps(_mm512_loadu_ps(src + i), vinv, voff);
        const __m512i v = _mm512_cvtps_epi32(x);
        const __m128i p8 = _mm512_cvtusepi32_epi8(v);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), p8);
    }
    QuantizeU8Tail(src, i, n, hc, delta, dst);
}

// b == 1 exact score: the rowA byte is the whole code — digits 0..6 are the
// sign bits (bit 7-i), digit 7 is their parity (the E8 shell), bit 0 is the
// branch. The two-level tables are affine, so one gather per block.
float ExactScoreB1Avx512(
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
    __m256 acc2 = _mm256_setzero_ps();
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
        __m256& dst = (blk & 1) != 0 ? acc2 : acc;
        dst = _mm256_fmadd_ps(lv, _mm256_loadu_ps(q + blk * 8), dst);
    }
    return simd::HsumPs(_mm256_add_ps(acc, acc2));
}

// Four-row group of the b = 2..8 refine core, templated on the bit budget so
// the level-table registers and the digit-unpack shape resolve at compile
// time. The row count is the compile-time constant kRefineRows: with a runtime
// `cnt` the accumulator arrays stay in memory and every FMA becomes a
// load-modify-store, which serialises the per-row digit chain (~21 cycles)
// instead of overlapping it across rows. `valid` may be < kRefineRows when the
// caller padded a short tail; only the first `valid` rows reach the heap.
constexpr int kRefineRows = 4;

template <int BITS>
inline void RefineRowsAvx512(
    const E8qIndex& idx, const SurvivorRow* rows, size_t valid, const float* q,
    QueryScratch& s
) {
    constexpr int ROWS = kRefineRows;
    static_assert(BITS >= 2 && BITS <= 8, "RefineRowsAvx512 covers b = 2..8");
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
    __m256 acc[ROWS];
    __m256 acc2[ROWS];
    const uint8_t* refp[ROWS];
    const uint8_t* rowp[ROWS];
    for (int jj = 0; jj < ROWS; ++jj) {
        acc[jj] = _mm256_setzero_ps();
        acc2[jj] = _mm256_setzero_ps();
        const uint32_t lj = rows[jj].lane;
        refp[jj] = rows[jj].refine;
        rowp[jj] = rows[jj].plane + lj;
    }
    if constexpr (BITS >= 7) {
        // 512-bit joint lookup: rows are processed in pairs, both 8-lane
        // halves of one zmm looking into the SAME branch-0 table. Branch-1
        // rows fold their digit through the reflection identity
        //   L1[d] == -L0[(1 << BITS) - 1 - d]
        // (verified bit-exact for every rate and every entry), the negation
        // re-applied to the looked-up floats, so the result is bit-identical
        // to the per-row dual-table lookup while touching one 1 KB table and
        // halving the vpermt2ps/blend count on the Port-5-bound tree. From
        // 2^b >= 128 entries up the tree is big enough to amortize the
        // pairing (combine + canonicalize + masked negate); at b = 6 the
        // 64-entry tree is one permute pair and the pairing loses.
        const __m512i qm1 = _mm512_set1_epi32((1 << BITS) - 1);
        const __m512 zero = _mm512_setzero_ps();
        for (int blk = 0; blk < blocks; ++blk) {
            const __m256 qv = _mm256_loadu_ps(q + blk * 8);
            for (int jj = 0; jj < ROWS; jj += 2) {
                int brv[2] = {0, 0};
                __m256i digs[2];
                for (int t = 0; t < 2; ++t) {
                    const int r = jj + t;
                    uint64_t x = 0;
                    const uint8_t* rp = refp[r] + static_cast<size_t>(blk) *
                                                     static_cast<size_t>(CBITS);
                    std::memcpy(&x, rp, 8);
                    brv[t] = static_cast<int>((x >> (7 * CBITS)) & 1ull);
                    const uint64_t parbit = __builtin_popcountll(x & par_mask) & 1ull;
                    x = (x & ~(uint64_t{1} << (7 * CBITS))) | (parbit << (7 * CBITS));
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
                    __m256i dig = _mm256_permute2x128_si256(lo, hi, 0x20);
                    const uint8_t rbA = rowp[r][static_cast<size_t>(blk) * 32];
                    const __m256i tops = _mm256_maskz_set1_epi32(
                        static_cast<__mmask8>(BitRev8Lut()[rbA]),
                        static_cast<int>(1u << CBITS)
                    );
                    digs[t] = _mm256_or_si256(dig, tops);
                }
                const __m512i both = _mm512_inserti64x4(
                    _mm512_castsi256_si512(digs[0]), digs[1], 1
                );
                const __mmask16 mk = static_cast<__mmask16>(
                    (brv[0] != 0 ? 0x00FFu : 0u) | (brv[1] != 0 ? 0xFF00u : 0u)
                );
                const __m512i canon = _mm512_mask_sub_epi32(both, mk, qm1, both);
                __m512 lv = simd::RefineLut512Full(refine_tab, canon, 1 << BITS);
                lv = _mm512_mask_sub_ps(lv, mk, zero, lv);
                __m256& dst_a = (blk & 1) != 0 ? acc2[jj] : acc[jj];
                __m256& dst_b = (blk & 1) != 0 ? acc2[jj + 1] : acc[jj + 1];
                dst_a = _mm256_fmadd_ps(_mm512_castps512_ps256(lv), qv, dst_a);
                dst_b = _mm256_fmadd_ps(_mm512_extractf32x8_ps(lv, 1), qv, dst_b);
            }
        }
    } else {
    for (int blk = 0; blk < blocks; ++blk) {
        const __m256 qv = _mm256_loadu_ps(q + blk * 8);
        for (int jj = 0; jj < ROWS; ++jj) {
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
                    _mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(x)), fsh), fmask
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
            const __m256i tops = _mm256_maskz_set1_epi32(
                static_cast<__mmask8>(BitRev8Lut()[rbA]),
                static_cast<int>(1u << CBITS)
            );
            dig = _mm256_or_si256(dig, tops);
            __m256 lv;
            // Branch select is a BLEND, never a branch: `br_sel` is an
            // encoded bit, so a conditional would mispredict on ~half the
            // (row, block) pairs and cost more than the whole lookup.
            const __m256 brmask =
                _mm256_castsi256_ps(_mm256_set1_epi32(-br_sel));
            if constexpr (BITS == 2) {
                const __m256i dg2 = _mm256_or_si256(
                    dig, _mm256_slli_epi32(_mm256_set1_epi32(br_sel), 2)
                );
                lv = _mm256_permutevar8x32_ps(fold8v, dg2);
            } else if constexpr (BITS == 3) {
                lv = _mm256_permutevar8x32_ps(
                    _mm256_blendv_ps(tbr0_0, tbr1_0, brmask), dig
                );
            } else if constexpr (BITS == 4) {
                const __m256 msk = _mm256_castsi256_ps(_mm256_slli_epi32(dig, 28));
                lv = _mm256_blendv_ps(
                    _mm256_permutevar8x32_ps(
                        _mm256_blendv_ps(tbr0_0, tbr1_0, brmask), dig
                    ),
                    _mm256_permutevar8x32_ps(
                        _mm256_blendv_ps(tbr0_1, tbr1_1, brmask), dig
                    ),
                    msk
                );
            } else if constexpr (BITS == 5) {
                const __m256 msk3 = _mm256_castsi256_ps(_mm256_slli_epi32(dig, 28));
                const __m256 msk4 = _mm256_castsi256_ps(_mm256_slli_epi32(dig, 27));
                const __m256 p0 = _mm256_permutevar8x32_ps(
                    _mm256_blendv_ps(tbr0_0, tbr1_0, brmask), dig
                );
                const __m256 p1 = _mm256_permutevar8x32_ps(
                    _mm256_blendv_ps(tbr0_1, tbr1_1, brmask), dig
                );
                const __m256 p2 = _mm256_permutevar8x32_ps(
                    _mm256_blendv_ps(tbr0_2, tbr1_2, brmask), dig
                );
                const __m256 p3 = _mm256_permutevar8x32_ps(
                    _mm256_blendv_ps(tbr0_3, tbr1_3, brmask), dig
                );
                lv = _mm256_blendv_ps(
                    _mm256_blendv_ps(p0, p1, msk3), _mm256_blendv_ps(p2, p3, msk3), msk4
                );
            } else {
                lv = LookupRefine(
                    refine_tab + static_cast<size_t>(br_sel) * 256, dig, BITS
                );
            }
            __m256& dst = (blk & 1) != 0 ? acc2[jj] : acc[jj];
            dst = _mm256_fmadd_ps(lv, qv, dst);
        }
    }
    }
    // Only the first `valid` rows reach the heap: a padded tail ran duplicate
    // rows through the block loop, and inserting those would put the same id
    // in the heap twice and crowd out real neighbours.
    for (size_t jj = 0; jj < valid; ++jj) {
        acc[jj] = _mm256_add_ps(acc[jj], acc2[jj]);
        const SurvivorRow& row = rows[jj];
        A::InsertHeap(
            idx, row.h3 - 2.0F * row.g1 * simd::HsumPs(acc[jj]), row.id, s
        );
    }
}

// Groups of kRefineRows, then one padded group for the tail: the short group
// re-runs duplicate rows so the kernel keeps its fixed shape, and only the
// real rows reach the heap.
template <int BITS>
inline void RefineCoreAvx512(
    const E8qIndex& idx, const SurvivorRow* rows, size_t n, const float* q, QueryScratch& s
) {
    constexpr size_t kRows = static_cast<size_t>(kRefineRows);
    const size_t ref_stride =
        static_cast<size_t>(A::blocks(idx)) *
        static_cast<size_t>(std::max(0, A::bits(idx) - 1));
    size_t j0 = 0;
    for (; j0 + kRows <= n; j0 += kRows) {
        // Prefetch the NEXT group's records. Both refine-arena streams are
        // read exactly once per row and the rows are scattered (survivor
        // order, ~300 B each, ~1000 per query), so a group otherwise starts
        // on cold L2 lines with only four rows of memory-level parallelism to
        // cover them. Same addresses, same order, no semantic effect.
        for (size_t t = j0 + kRows; t < j0 + 3 * kRows && t < n; ++t) {
            const uint8_t* r = rows[t].refine;
            const uint8_t* p = rows[t].plane + rows[t].lane;
            // Whole refine payload (blocks*(b-1) bytes, 448 at msong b8), two
            // groups ahead: one group of look-ahead only covers L2 latency,
            // and the payload is touched by nothing else before the refine.
            for (size_t o = 0; o < ref_stride; o += 64) {
                __builtin_prefetch(r + o, 0, 1);
            }
            __builtin_prefetch(p, 0, 1);
            __builtin_prefetch(p + 128, 0, 1);
        }
        RefineRowsAvx512<BITS>(idx, rows + j0, kRows, q, s);
    }
    if (j0 < n) {
        const size_t rest = n - j0;
        SurvivorRow tail[kRows];
        for (size_t t = 0; t < kRows; ++t) {
            tail[t] = rows[j0 + (t < rest ? t : rest - 1)];
        }
        RefineRowsAvx512<BITS>(idx, tail, rest, q, s);
    }
}

void RefineBatchAvx512(
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
                        _mm256_srlv_epi64(_mm256_set1_epi64x(static_cast<int64_t>(lo)), sh_lo),
                        fmask
                    );
                    const __m256i fh = _mm256_and_si256(
                        _mm256_srlv_epi64(_mm256_set1_epi64x(static_cast<int64_t>(base2)), sh_hi),
                        fmask
                    );
                    const __m256i f04 = _mm256_permutevar8x32_epi32(fl, pick_lo32);
                    const __m256i f47 = _mm256_permutevar8x32_epi32(fh, pick_lo32);
                    const __m256i tops = _mm256_maskz_set1_epi32(
                        static_cast<__mmask8>(BitRev8Lut()[rbA]),
                        static_cast<int>(1u << c)
                    );
                    const __m256i dig = _mm256_or_si256(
                        tops, _mm256_permute2x128_si256(f04, f47, 0x20)
                    );
                    const float* tab = refine_tab + static_cast<size_t>(br) * stride;
                    const __m256 lv = _mm256_i32gather_ps(tab, dig, 4);
                    __m256& dst = (blk & 1) != 0 ? acc2[jj] : acc[jj];
                    dst = _mm256_fmadd_ps(lv, qv, dst);
                }
            }
            for (size_t jj = 0; jj < cnt; ++jj) {
                const SurvivorRow& row = rows[j0 + jj];
                A::InsertHeap(
                    idx, row.h3 - 2.0F * row.g1 * simd::HsumPs(_mm256_add_ps(acc[jj], acc2[jj])),
                    row.id, s
                );
            }
        }
        return;
    }
    if (bits >= 2) {
        switch (bits) {
            case 2: RefineCoreAvx512<2>(idx, rows, n, q, s); return;
            case 3: RefineCoreAvx512<3>(idx, rows, n, q, s); return;
            case 4: RefineCoreAvx512<4>(idx, rows, n, q, s); return;
            case 5: RefineCoreAvx512<5>(idx, rows, n, q, s); return;
            case 6: RefineCoreAvx512<6>(idx, rows, n, q, s); return;
            case 7: RefineCoreAvx512<7>(idx, rows, n, q, s); return;
            case 8: RefineCoreAvx512<8>(idx, rows, n, q, s); return;
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
            A::InsertHeap(idx, row.h3 - 2.0F * row.g1 * simd::HsumPs(acc[jj]), row.id, s);
        }
    }
}

void ScanClusterAvx512(
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

    const __m512 vdelta_c = _mm512_set1_ps(s.prep_delta_c);
    const __m512 vbase_c = _mm512_set1_ps(s.prep_base_c);
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
            alignas(64) float dlo_lanes[32];
            __m512 dlo_r[2];
            {
                // Orthogonal-projection bound; AVX-512 mirror of the AVX2
                // branch (see OrthoBatchHeader for the additive pad).
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
                    const __m512i m_all = _mm512_max_epi32(
                        _mm512_load_si512(accC), _mm512_load_si512(accC + 16)
                    );
                    const __m512i l_all = _mm512_min_epi32(
                        _mm512_load_si512(accC), _mm512_load_si512(accC + 16)
                    );
                    // W is signed (ZPR): the upper bound of W*S over the batch
                    // needs max |S| in affine units, not the signed max.
                    const float s_hi =
                        static_cast<float>(_mm512_reduce_max_epi32(m_all)) * s.prep_delta_c +
                        s.prep_base_c;
                    const float s_lo =
                        static_cast<float>(_mm512_reduce_min_epi32(l_all)) * s.prep_delta_c +
                        s.prep_base_c;
                    const float s_absmax = std::max(std::abs(s_hi), std::abs(s_lo));
                    const float batch_dlo_lb = (d2 + base_min - pad_add) -
                                               w_amax * s_absmax -
                                               (s.ortho_z * r_horizon) * perp_amax;
                    if (batch_dlo_lb >= distk0) {
                        return;
                    }
                }
                const __m512 v_base_min = _mm512_set1_ps(base_min + d2 - pad_w_round);
                const __m512 v_base_scale = _mm512_set1_ps(base_scale_f);
                const __m512 v_weight_min = _mm512_set1_ps(weight_min_f);
                const __m512 v_weight_scale = _mm512_set1_ps(weight_scale_f);
                const __m512 v_perp_min = _mm512_set1_ps(perp_min);
                const __m512 v_perp_scale = _mm512_set1_ps(perp_scale_f);
                const __m512 v_slack_scale = _mm512_set1_ps(s.ortho_z * r_horizon);
                const __m512 v_pad_w_lut = _mm512_set1_ps(pad_w_lut);
                const uint8_t* base_bank = bp + E8qIndex::kH3QOff;
                const uint8_t* weight_bank = bp + E8qIndex::kCqOff;
                const uint8_t* perp_bank = bp + E8qIndex::kAqOff;
                for (int grp = 0; grp < 2; ++grp) {
                    const __m512 cf = _mm512_cvtepi32_ps(_mm512_load_si512(accC + grp * 16));
                    const __m512 scv = _mm512_fmadd_ps(cf, vdelta_c, vbase_c);
                    const __m512 base_term = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(base_bank + grp * 16)
                        ))),
                        v_base_scale, v_base_min
                    );
                    const __m512 weight = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(weight_bank + grp * 16)
                        ))),
                        v_weight_scale, v_weight_min
                    );
                    const __m512 perp = _mm512_fmadd_ps(
                        _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(perp_bank + grp * 16)
                        ))),
                        v_perp_scale, v_perp_min
                    );
                    const __m512 dc = _mm512_fnmadd_ps(weight, scv, base_term);
                    const __m512 abs_w = _mm512_abs_ps(weight);
                    const __m512 dlo = _mm512_fnmadd_ps(
                        v_slack_scale, perp, _mm512_fnmadd_ps(abs_w, v_pad_w_lut, dc)
                    );
                    dlo_r[grp] = dlo;
                }
            }
            uint32_t mask = 0;
            if (s.disable_prune) {
                mask = 0xFFFFFFFFu;
            } else {
                // The per-lane compares fully determine the mask: the old
                // reduce_min pre-check repeated exactly this test after a
                // 10-cycle horizontal reduction per batch.
                const __m512 vlimit = _mm512_set1_ps(distk0);
                const __mmask16 lo =
                    _mm512_cmp_ps_mask(dlo_r[0], vlimit, _CMP_LT_OQ);
                const __mmask16 hi =
                    _mm512_cmp_ps_mask(dlo_r[1], vlimit, _CMP_LT_OQ);
                mask = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
            }
            if (mask != 0) {
                for (int grp = 0; grp < 2; ++grp) {
                    _mm512_store_ps(dlo_lanes + grp * 16, dlo_r[grp]);
                }
            }
            CollectLanes(
                mask, n_logic, dlo_lanes, s, s.disable_prune, stats, flush,
                [&](int lane) {
                    const uint8_t* row_refine =
                        bits >= 2
                            ? refbase + (off_cur / 32) * ref_batch_bytes +
                                  A::ref_refine_off(idx) +
                                  static_cast<size_t>(lane) *
                                      (static_cast<size_t>(blocks) *
                                       static_cast<size_t>(bits - 1))
                            : nullptr;
                    if (row_refine != nullptr) {
                        // RefineBatch prefetches its rows 1-3 groups ahead, so
                        // every flush's first group starts cold. Staging is the
                        // earliest point the address is known: pull the payload
                        // in while the remaining batches of this cluster scan.
                        const size_t ref_stride =
                            static_cast<size_t>(blocks) * static_cast<size_t>(bits - 1);
                        for (size_t o = 0; o < ref_stride && o < 192; o += 64) {
                            __builtin_prefetch(row_refine + o, 0, 3);
                        }
                    }
                    return SurvivorRow{
                        row_refine,
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
            alignas(64) int32_t accC[32];
            alignas(64) int32_t accB[32];
            if (pair) {
                AccumulateCoarseHiPair(
                    batchp + A::p1_off(idx), batchp + batch_bytes + A::p1_off(idx),
                    s.lut_c.data(), s.lut_pair.data(), static_cast<size_t>(blocks), accC, accB
                );
            } else {
                AccumulateCoarseHi(
                    batchp + A::p1_off(idx), s.lut_c.data(), s.lut_pair.data(),
                    static_cast<size_t>(blocks), accC
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
