#include "simd/me8q_score.hpp"
#include "simd/simd_helpers.hpp"

#include <cstring>
#include <vector>

#include <immintrin.h>

#include "quantization/me8q_bits.hpp"

namespace e8qlib::me8q {
namespace {

using simd::HsumPs;

template <int BITS>
inline __m256 LookupLevelsT(const float* lev, __m256i dig) {
    if constexpr (BITS <= 3) {
        return _mm256_permutevar8x32_ps(_mm256_loadu_ps(lev), dig);
    } else if constexpr (BITS == 4) {
        const __m256 msk = _mm256_castsi256_ps(_mm256_slli_epi32(dig, 28));
        return _mm256_blendv_ps(
            _mm256_permutevar8x32_ps(_mm256_loadu_ps(lev), dig),
            _mm256_permutevar8x32_ps(_mm256_loadu_ps(lev + 8), dig), msk
        );
    } else if constexpr (BITS == 5) {
        const __m256 msk3 = _mm256_castsi256_ps(_mm256_slli_epi32(dig, 28));
        const __m256 msk4 = _mm256_castsi256_ps(_mm256_slli_epi32(dig, 27));
        const __m256 p0 = _mm256_permutevar8x32_ps(_mm256_loadu_ps(lev), dig);
        const __m256 p1 = _mm256_permutevar8x32_ps(_mm256_loadu_ps(lev + 8), dig);
        const __m256 p2 = _mm256_permutevar8x32_ps(_mm256_loadu_ps(lev + 16), dig);
        const __m256 p3 = _mm256_permutevar8x32_ps(_mm256_loadu_ps(lev + 24), dig);
        return _mm256_blendv_ps(
            _mm256_blendv_ps(p0, p1, msk3), _mm256_blendv_ps(p2, p3, msk3), msk4
        );
    } else {
        return _mm256_i32gather_ps(lev, dig, 4);
    }
}

template <int BITS>
inline __m256i UnpackDigitsT(const uint8_t* src, int* br) {
    if constexpr (BITS <= 4) {
        constexpr uint64_t kParMask = ParMaskOf(BITS);
        uint32_t w = 0;
        std::memcpy(&w, src, BITS);
        *br = static_cast<int>((w >> (8 * BITS - 1)) & 1U);
        const int parity =
            static_cast<int>(__builtin_popcount(w & static_cast<uint32_t>(kParMask)) & 1U);
        const __m256i sh = _mm256_setr_epi32(
            0, BITS, 2 * BITS, 3 * BITS, 4 * BITS, 5 * BITS, 6 * BITS, 0
        );
        __m256i dig = _mm256_and_si256(
            _mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(w)), sh),
            _mm256_set1_epi32((1 << BITS) - 1)
        );
        const int d7 = static_cast<int>(
            ((w >> (7 * BITS)) & ((uint32_t{1} << (BITS - 1)) - 1U)) << 1 |
            static_cast<uint32_t>(parity)
        );
        return _mm256_insert_epi32(dig, d7, 7);
    } else if constexpr (BITS <= 8) {
        constexpr uint64_t kParMask = ParMaskOf(BITS);
        uint64_t x = 0;
        std::memcpy(&x, src, BITS);
        *br = static_cast<int>((x >> (8 * BITS - 1)) & 1ULL);
        const uint64_t parbit =
            static_cast<uint64_t>(__builtin_popcountll(x & kParMask) & 1);
        const int d7 = static_cast<int>(
            (((x >> (7 * BITS)) & ((uint64_t{1} << (BITS - 1)) - 1)) << 1) | parbit
        );
        const __m256i sh4 = _mm256_setr_epi32(0, BITS, 2 * BITS, 3 * BITS, 0, 0, 0, 0);
        const __m256i lo = _mm256_srlv_epi32(
            _mm256_set1_epi32(static_cast<int>(x & 0xFFFFFFFFu)), sh4
        );
        const __m256i hi = _mm256_srlv_epi32(
            _mm256_set1_epi32(static_cast<int>((x >> (4 * BITS)) & 0xFFFFFFFFu)), sh4
        );
        __m256i dig = _mm256_and_si256(
            _mm256_permute2x128_si256(lo, hi, 0x20), _mm256_set1_epi32((1 << BITS) - 1)
        );
        return _mm256_insert_epi32(dig, d7, 7);
    } else {
        unsigned __int128 x = 0;
        std::memcpy(&x, src, BITS);
        *br = static_cast<int>(Extract128(x, 8 * BITS - 1, 1));
        const uint64_t mask = (uint64_t{1} << BITS) - 1;
        alignas(32) int32_t digs[8];
        int parity = 0;
        for (int i = 0; i < 7; ++i) {
            const uint64_t f = Extract128(x, BITS * i, mask);
            digs[i] = static_cast<int32_t>(f);
            parity ^= static_cast<int>(f & 1U);
        }
        digs[7] = static_cast<int32_t>(
            (Extract128(x, BITS * 7, (uint64_t{1} << (BITS - 1)) - 1) << 1) |
            static_cast<uint64_t>(parity)
        );
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(digs));
    }
}

template <int BITS>
float ScoreFastT(const SegmentCodec& c, const uint8_t* in, int d_s, const float* q) {
    const int blocks = d_s / 8;
    __m256 acc = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    for (int blk = 0; blk < blocks; ++blk) {
        int br = 0;
        const __m256i dig =
            UnpackDigitsT<BITS>(in + static_cast<size_t>(blk) * BITS, &br);
        const __m256 lv = LookupLevelsT<BITS>(c.Levels(BITS, br), dig);
        const __m256 qv = _mm256_loadu_ps(q + static_cast<size_t>(blk) * 8);
        if constexpr (BITS >= 2) {
            if ((blk & 1) != 0) {
                acc2 = _mm256_fmadd_ps(lv, qv, acc2);
            } else {
                acc = _mm256_fmadd_ps(lv, qv, acc);
            }
        } else {
            acc = _mm256_fmadd_ps(lv, qv, acc);
        }
    }
    if constexpr (BITS >= 2) {
        acc = _mm256_add_ps(acc, acc2);
    }
    return HsumPs(acc);
}

template <int BITS>
void ScoreFast4T(
    const SegmentCodec& c, const uint8_t* const in[4], int d_s, const float* q, float out[4]
) {
    const int blocks = d_s / 8;
    __m256 acc[4];
    __m256 acc2[4];
    for (int j = 0; j < 4; ++j) {
        acc[j] = _mm256_setzero_ps();
        acc2[j] = _mm256_setzero_ps();
    }
    for (int blk = 0; blk < blocks; ++blk) {
        const __m256 qv = _mm256_loadu_ps(q + static_cast<size_t>(blk) * 8);
        for (int j = 0; j < 4; ++j) {
            int br = 0;
            const __m256i dig =
                UnpackDigitsT<BITS>(in[j] + static_cast<size_t>(blk) * BITS, &br);
            const __m256 lv = LookupLevelsT<BITS>(c.Levels(BITS, br), dig);
            if constexpr (BITS >= 2) {
                if ((blk & 1) != 0) {
                    acc2[j] = _mm256_fmadd_ps(lv, qv, acc2[j]);
                } else {
                    acc[j] = _mm256_fmadd_ps(lv, qv, acc[j]);
                }
            } else {
                acc[j] = _mm256_fmadd_ps(lv, qv, acc[j]);
            }
        }
    }
    for (int j = 0; j < 4; ++j) {
        if constexpr (BITS >= 2) {
            acc[j] = _mm256_add_ps(acc[j], acc2[j]);
        }
        out[j] = HsumPs(acc[j]);
    }
}

inline float ScoreFastByBits(
    const SegmentCodec& c, const uint8_t* in, int d_s, int bits, const float* q
) {
    switch (bits) {
        case 1: return ScoreFastT<1>(c, in, d_s, q);
        case 2: return ScoreFastT<2>(c, in, d_s, q);
        case 3: return ScoreFastT<3>(c, in, d_s, q);
        case 4: return ScoreFastT<4>(c, in, d_s, q);
        case 5: return ScoreFastT<5>(c, in, d_s, q);
        case 6: return ScoreFastT<6>(c, in, d_s, q);
        case 7: return ScoreFastT<7>(c, in, d_s, q);
        case 8: return ScoreFastT<8>(c, in, d_s, q);
        case 9: return ScoreFastT<9>(c, in, d_s, q);
        case 10: return ScoreFastT<10>(c, in, d_s, q);
        case 11: return ScoreFastT<11>(c, in, d_s, q);
        case 12: return ScoreFastT<12>(c, in, d_s, q);
        default: return ScoreFastT<13>(c, in, d_s, q);
    }
}

inline void ScoreFast4ByBits(
    const SegmentCodec& c, const uint8_t* const in[4], int d_s, int bits, const float* q,
    float out[4]
) {
    switch (bits) {
        case 1: ScoreFast4T<1>(c, in, d_s, q, out); break;
        case 2: ScoreFast4T<2>(c, in, d_s, q, out); break;
        case 3: ScoreFast4T<3>(c, in, d_s, q, out); break;
        case 4: ScoreFast4T<4>(c, in, d_s, q, out); break;
        case 5: ScoreFast4T<5>(c, in, d_s, q, out); break;
        case 6: ScoreFast4T<6>(c, in, d_s, q, out); break;
        case 7: ScoreFast4T<7>(c, in, d_s, q, out); break;
        case 8: ScoreFast4T<8>(c, in, d_s, q, out); break;
        case 9: ScoreFast4T<9>(c, in, d_s, q, out); break;
        case 10: ScoreFast4T<10>(c, in, d_s, q, out); break;
        case 11: ScoreFast4T<11>(c, in, d_s, q, out); break;
        case 12: ScoreFast4T<12>(c, in, d_s, q, out); break;
        default: ScoreFast4T<13>(c, in, d_s, q, out); break;
    }
}

inline __m256i PlaneTopBits(uint8_t plane) {
    const __m256i bitmask = _mm256_setr_epi32(0x80, 0x40, 0x20, 0x10, 8, 4, 2, 1);
    return _mm256_and_si256(
        _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(plane), bitmask), bitmask),
        _mm256_set1_epi32(1)
    );
}

template <int BITS>
inline __m256 SplitLevels(
    const SegmentCodec& c, const uint8_t* ref, uint8_t plane, int /*br1*/
) {
    if constexpr (BITS == 1) {
        alignas(32) int32_t digs[8];
        int par = 0;
        for (int i = 0; i < 7; ++i) {
            const int d = (plane >> (7 - i)) & 1U;
            par ^= d;
            digs[i] = static_cast<int32_t>(d);
        }
        digs[7] = static_cast<int32_t>(par);
        return LookupLevelsT<1>(
            c.Levels(1, plane & 1U),
            _mm256_load_si256(reinterpret_cast<const __m256i*>(digs))
        );
    } else {
        int br = 0;
        __m256i dig = UnpackDigitsT<BITS - 1>(ref, &br);
        dig = _mm256_or_si256(dig, _mm256_slli_epi32(PlaneTopBits(plane), BITS - 1));
        return LookupLevelsT<BITS>(c.Levels(BITS, br), dig);
    }
}

// Byte i of lut[v] is bit 7-i of v: the 8 top bits of a split-plane byte as
// expanded lane digits. Same shape as the e8q scan's PlaneExpandLut, local so
// this TU carries no index dependency.
inline const uint64_t* PlaneExpandLut256() {
    static const std::vector<uint64_t> lut = [] {
        std::vector<uint64_t> t(256, 0);
        for (int v = 0; v < 256; ++v) {
            uint64_t e = 0;
            for (int i = 0; i < 8; ++i) {
                e |= static_cast<uint64_t>((v >> (7 - i)) & 1) << (8 * i);
            }
            t[static_cast<size_t>(v)] = e;
        }
        return t;
    }();
    return lut.data();
}

// Register-table twin of LookupLevelsT for the run scorers: the per-branch
// tables are hoisted into registers before the block loop (selected per
// block), so the loop itself never loads level entries from memory.
template <int BITS>
inline __m256 LookupLevelsR(const __m256* tb, __m256i dig) {
    if constexpr (BITS <= 3) {
        return _mm256_permutevar8x32_ps(tb[0], dig);
    } else if constexpr (BITS == 4) {
        const __m256 msk = _mm256_castsi256_ps(_mm256_slli_epi32(dig, 28));
        return _mm256_blendv_ps(
            _mm256_permutevar8x32_ps(tb[0], dig),
            _mm256_permutevar8x32_ps(tb[1], dig), msk
        );
    } else {
        const __m256 msk3 = _mm256_castsi256_ps(_mm256_slli_epi32(dig, 28));
        const __m256 msk4 = _mm256_castsi256_ps(_mm256_slli_epi32(dig, 27));
        const __m256 p0 = _mm256_permutevar8x32_ps(tb[0], dig);
        const __m256 p1 = _mm256_permutevar8x32_ps(tb[1], dig);
        const __m256 p2 = _mm256_permutevar8x32_ps(tb[2], dig);
        const __m256 p3 = _mm256_permutevar8x32_ps(tb[3], dig);
        return _mm256_blendv_ps(
            _mm256_blendv_ps(p0, p1, msk3), _mm256_blendv_ps(p2, p3, msk3), msk4
        );
    }
}

// Run scorer core in the e8q RefineCoreAvx2 shape: per-rate level tables
// hoisted into registers, digit 7's parity folded into the shift unpack (the
// branch bit is swapped out for the parity before srlv), and the plane's top
// bits expanded through PlaneExpandLut256. Covers BITS 1..5; wider rates keep
// the SplitLevels loop (the level gather dominates there). FMA alternation by
// block parity matches ScoreFastT, so S stays bit-identical to the per-block
// scorer.
template <int BITS>
inline void ScoreRunSplit4T(
    const SegmentCodec& c, const uint8_t* const ref[4], const uint8_t* const pb[4],
    size_t blk0, int nblk, const float* q, int n, float out[4]
) {
    const int m = n < 4 ? n : 4;
    __m256 acc[4];
    __m256 acc2[4];
    const uint8_t* rp[4];
    for (int j = 0; j < m; ++j) {
        acc[j] = _mm256_setzero_ps();
        acc2[j] = _mm256_setzero_ps();
        rp[j] = ref[j];
    }
    constexpr int T = BITS == 1 ? 0 : BITS - 1;
    if constexpr (BITS <= 5) {
        const uint64_t* pl = PlaneExpandLut256();
        constexpr int KTAB = BITS <= 3 ? 1 : (1 << (BITS - 3));
        __m256 tb0[KTAB];
        [[maybe_unused]] __m256 tb1[KTAB];
        if constexpr (BITS == 2) {
            // fold8: digits 0..3 of branch 0, 4..7 of branch 1; dg2 = dig |
            // (branch << 2) addresses one 8-entry table.
            const float* l0 = c.Levels(BITS, 0);
            const float* l1 = c.Levels(BITS, 1);
            alignas(32) float fold8[8];
            for (int i = 0; i < 8; ++i) {
                fold8[i] = (i >> 2 ? l1 : l0)[i & 3];
            }
            tb0[0] = _mm256_loadu_ps(fold8);
        } else {
            for (int k = 0; k < KTAB; ++k) {
                tb0[k] = _mm256_loadu_ps(c.Levels(BITS, 0) + 8 * k);
                tb1[k] = _mm256_loadu_ps(c.Levels(BITS, 1) + 8 * k);
            }
        }
        [[maybe_unused]] const __m256i fsh =
            _mm256_setr_epi32(0, T, 2 * T, 3 * T, 4 * T, 5 * T, 6 * T, 0);
        [[maybe_unused]] const __m256i fmask = _mm256_set1_epi32((1 << T) - 1);
        for (int blk = 0; blk < nblk; ++blk) {
            const __m256 qv = _mm256_loadu_ps(q + static_cast<size_t>(blk) * 8);
            for (int j = 0; j < m; ++j) {
                const uint8_t plane = pb[j][(blk0 + static_cast<size_t>(blk)) * 32];
                __m256 lv;
                if constexpr (BITS == 1) {
                    const int par = __builtin_popcount(plane >> 1) & 1;
                    __m256i dig = _mm256_cvtepu8_epi32(
                        _mm_cvtsi64_si128(static_cast<long long>(pl[plane]))
                    );
                    dig = _mm256_insert_epi32(dig, par, 7);
                    lv = _mm256_permutevar8x32_ps((plane & 1U) ? tb1[0] : tb0[0], dig);
                } else if constexpr (BITS == 2) {
                    // T = 1: digit 7 carries no stored bits — its value is the
                    // parity shell — and the branch rides at bit 7.
                    uint32_t w = 0;
                    std::memcpy(&w, rp[j], T);
                    const int br_sel = static_cast<int>((w >> (8 * T - 1)) & 1U);
                    const uint32_t parbit =
                        static_cast<uint32_t>(__builtin_popcount(w & ParMaskOf(T)) & 1U);
                    __m256i dig = _mm256_and_si256(
                        _mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(w)), fsh), fmask
                    );
                    dig = _mm256_insert_epi32(dig, static_cast<int>(parbit), 7);
                    dig = _mm256_or_si256(
                        dig,
                        _mm256_slli_epi32(
                            _mm256_cvtepu8_epi32(
                                _mm_cvtsi64_si128(static_cast<long long>(pl[plane]))
                            ),
                            T
                        )
                    );
                    const __m256i dg2 = _mm256_or_si256(
                        dig, _mm256_slli_epi32(_mm256_set1_epi32(br_sel), 2)
                    );
                    lv = _mm256_permutevar8x32_ps(tb0[0], dg2);
                } else {
                    // me8q pack: digits 0..6 low T bits at iT, digit 7's low
                    // T-1 bits at 7T..8T-2, branch at 8T-1. Digit 7's LSB is
                    // the parity shell, so lane 7 is rebuilt by insert (the
                    // e8q-style branch-at-7T parity swap does NOT apply here).
                    uint32_t w = 0;
                    std::memcpy(&w, rp[j], T);
                    const int br_sel = static_cast<int>((w >> (8 * T - 1)) & 1U);
                    const uint32_t parbit =
                        static_cast<uint32_t>(__builtin_popcount(w & ParMaskOf(T)) & 1U);
                    const int d7 = static_cast<int>(
                        (((w >> (7 * T)) & ((1U << (T - 1)) - 1U)) << 1) | parbit
                    );
                    __m256i dig = _mm256_and_si256(
                        _mm256_srlv_epi32(_mm256_set1_epi32(static_cast<int>(w)), fsh), fmask
                    );
                    dig = _mm256_insert_epi32(dig, d7, 7);
                    dig = _mm256_or_si256(
                        dig,
                        _mm256_slli_epi32(
                            _mm256_cvtepu8_epi32(
                                _mm_cvtsi64_si128(static_cast<long long>(pl[plane]))
                            ),
                            T
                        )
                    );
                    lv = LookupLevelsR<BITS>(br_sel != 0 ? tb1 : tb0, dig);
                }
                if ((blk & 1) != 0) {
                    acc2[j] = _mm256_fmadd_ps(lv, qv, acc2[j]);
                } else {
                    acc[j] = _mm256_fmadd_ps(lv, qv, acc[j]);
                }
                if constexpr (T > 0) {
                    rp[j] += T;
                }
            }
        }
    } else {
        for (int blk = 0; blk < nblk; ++blk) {
            const __m256 qv = _mm256_loadu_ps(q + static_cast<size_t>(blk) * 8);
            for (int j = 0; j < m; ++j) {
                const uint8_t plane = pb[j][(blk0 + static_cast<size_t>(blk)) * 32];
                const __m256 lv = SplitLevels<BITS>(c, rp[j], plane, 0);
                if ((blk & 1) != 0) {
                    acc2[j] = _mm256_fmadd_ps(lv, qv, acc2[j]);
                } else {
                    acc[j] = _mm256_fmadd_ps(lv, qv, acc[j]);
                }
                rp[j] += T;
            }
        }
    }
    for (int j = 0; j < m; ++j) {
        out[j] = HsumPs(_mm256_add_ps(acc[j], acc2[j]));
    }
}

}  // namespace

float ScoreFastAvx2(
    const SegmentCodec& c, const uint8_t* in, int d_s, int bits, const float* q
) {
    return ScoreFastByBits(c, in, d_s, bits, q);
}

void ScoreFast4Avx2(
    const SegmentCodec& c, const uint8_t* const in[4], int d_s, int bits, const float* q,
    float out[4]
) {
    ScoreFast4ByBits(c, in, d_s, bits, q, out);
}

float ScoreBlockSplitAvx2(
    const SegmentCodec& c, const uint8_t* ref, int bits, uint8_t plane, const float* q8,
    int br1
) {
    __m256 lv;
    switch (bits) {
        case 1: lv = SplitLevels<1>(c, ref, plane, br1); break;
        case 2: lv = SplitLevels<2>(c, ref, plane, br1); break;
        case 3: lv = SplitLevels<3>(c, ref, plane, br1); break;
        case 4: lv = SplitLevels<4>(c, ref, plane, br1); break;
        case 5: lv = SplitLevels<5>(c, ref, plane, br1); break;
        case 6: lv = SplitLevels<6>(c, ref, plane, br1); break;
        case 7: lv = SplitLevels<7>(c, ref, plane, br1); break;
        case 8: lv = SplitLevels<8>(c, ref, plane, br1); break;
        case 9: lv = SplitLevels<9>(c, ref, plane, br1); break;
        case 10: lv = SplitLevels<10>(c, ref, plane, br1); break;
        case 11: lv = SplitLevels<11>(c, ref, plane, br1); break;
        case 12: lv = SplitLevels<12>(c, ref, plane, br1); break;
        default: lv = SplitLevels<13>(c, ref, plane, br1); break;
    }
    return HsumPs(_mm256_mul_ps(lv, _mm256_loadu_ps(q8)));
}

float ScoreRunSplitAvx2(
    const SegmentCodec& c, const uint8_t* ref, const uint8_t* pb, size_t blk0, int nblk,
    int bits, const float* q
) {
    // Single-row form of the hoisted run core: same template, one row, so the
    // 1-wide and 4-wide paths share bit-identical FMA order by construction.
    float out[1] = {0.0F};
    const uint8_t* refs[1] = {ref};
    const uint8_t* pbs[1] = {pb};
    switch (bits) {
        case 1: ScoreRunSplit4T<1>(c, refs, pbs, blk0, nblk, q, 1, out); break;
        case 2: ScoreRunSplit4T<2>(c, refs, pbs, blk0, nblk, q, 1, out); break;
        case 3: ScoreRunSplit4T<3>(c, refs, pbs, blk0, nblk, q, 1, out); break;
        case 4: ScoreRunSplit4T<4>(c, refs, pbs, blk0, nblk, q, 1, out); break;
        case 5: ScoreRunSplit4T<5>(c, refs, pbs, blk0, nblk, q, 1, out); break;
        case 6: ScoreRunSplit4T<6>(c, refs, pbs, blk0, nblk, q, 1, out); break;
        case 7: ScoreRunSplit4T<7>(c, refs, pbs, blk0, nblk, q, 1, out); break;
        case 8: ScoreRunSplit4T<8>(c, refs, pbs, blk0, nblk, q, 1, out); break;
        case 9: ScoreRunSplit4T<9>(c, refs, pbs, blk0, nblk, q, 1, out); break;
        case 10: ScoreRunSplit4T<10>(c, refs, pbs, blk0, nblk, q, 1, out); break;
        case 11: ScoreRunSplit4T<11>(c, refs, pbs, blk0, nblk, q, 1, out); break;
        case 12: ScoreRunSplit4T<12>(c, refs, pbs, blk0, nblk, q, 1, out); break;
        default: ScoreRunSplit4T<13>(c, refs, pbs, blk0, nblk, q, 1, out); break;
    }
    return out[0];
}

void ScoreRunSplit4Avx2(
    const SegmentCodec& c, const uint8_t* const ref[4], const uint8_t* const pb[4],
    size_t blk0, int nblk, int bits, const float* q, int n, float out[4]
) {
    switch (bits) {
        case 1: ScoreRunSplit4T<1>(c, ref, pb, blk0, nblk, q, n, out); break;
        case 2: ScoreRunSplit4T<2>(c, ref, pb, blk0, nblk, q, n, out); break;
        case 3: ScoreRunSplit4T<3>(c, ref, pb, blk0, nblk, q, n, out); break;
        case 4: ScoreRunSplit4T<4>(c, ref, pb, blk0, nblk, q, n, out); break;
        case 5: ScoreRunSplit4T<5>(c, ref, pb, blk0, nblk, q, n, out); break;
        case 6: ScoreRunSplit4T<6>(c, ref, pb, blk0, nblk, q, n, out); break;
        case 7: ScoreRunSplit4T<7>(c, ref, pb, blk0, nblk, q, n, out); break;
        case 8: ScoreRunSplit4T<8>(c, ref, pb, blk0, nblk, q, n, out); break;
        case 9: ScoreRunSplit4T<9>(c, ref, pb, blk0, nblk, q, n, out); break;
        case 10: ScoreRunSplit4T<10>(c, ref, pb, blk0, nblk, q, n, out); break;
        case 11: ScoreRunSplit4T<11>(c, ref, pb, blk0, nblk, q, n, out); break;
        case 12: ScoreRunSplit4T<12>(c, ref, pb, blk0, nblk, q, n, out); break;
        default: ScoreRunSplit4T<13>(c, ref, pb, blk0, nblk, q, n, out); break;
    }
}

}  // namespace e8qlib::me8q
