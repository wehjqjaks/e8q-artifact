#include "simd/me8q_score.hpp"
#include "simd/simd_helpers.hpp"

#include <array>
#include <cstring>

#include <immintrin.h>

#include "quantization/me8q_bits.hpp"

namespace e8qlib::me8q {
namespace {

inline int BranchOfPacked(const uint8_t* src, int bits) {
    if (bits <= 8) {
        uint64_t x = 0;
        std::memcpy(&x, src, static_cast<size_t>(bits));
        return static_cast<int>((x >> (8 * bits - 1)) & 1ULL);
    }
    unsigned __int128 x = 0;
    std::memcpy(&x, src, static_cast<size_t>(bits));
    return static_cast<int>(Extract128(x, 8 * bits - 1, 1));
}

inline const uint8_t* BitRev8Lut();

template <int BITS>
inline void DecodeBlockDigits512(const uint8_t* src, int32_t* digs) {
    if constexpr (BITS <= 8) {
        uint64_t x = 0;
        std::memcpy(&x, src, BITS);
        for (int i = 0; i < 7; ++i) {
            digs[i] = static_cast<int32_t>(
                (x >> (static_cast<size_t>(BITS) * i)) & ((1 << BITS) - 1)
            );
        }
        const int par = static_cast<int>(__builtin_popcountll(x & ParMaskOf(BITS)) & 1);
        digs[7] = static_cast<int32_t>(
            (((x >> (7 * BITS)) & ((uint64_t{1} << (BITS - 1)) - 1)) << 1) | par
        );
    } else {
        unsigned __int128 x = 0;
        std::memcpy(&x, src, BITS);
        const uint64_t mask = (uint64_t{1} << BITS) - 1;
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
    }
}

// 16-lane table lookup. bits<=4: one vpermi2/permutexvar over 16 floats;
// bits==5: mask blend of two 16-entry halves; bits>=6: 16-lane gather.
template <int BITS>
inline __m512 LookupLevels512(const float* lev, __m512i dig) {
    if constexpr (BITS <= 4) {
        return _mm512_permutexvar_ps(
            _mm512_and_si512(dig, _mm512_set1_epi32(15)), _mm512_loadu_ps(lev)
        );
    } else if constexpr (BITS == 5) {
        const __m512i idx4 = _mm512_and_si512(dig, _mm512_set1_epi32(15));
        const __mmask16 m4 = _mm512_movepi32_mask(_mm512_slli_epi32(dig, 27));
        return _mm512_mask_blend_ps(
            m4, _mm512_permutexvar_ps(idx4, _mm512_loadu_ps(lev)),
            _mm512_permutexvar_ps(idx4, _mm512_loadu_ps(lev + 16))
        );
    } else {
        return _mm512_i32gather_ps(dig, lev, 4);
    }
}

// Block-pair lookup over the combined branch table (SegmentCodec::PairLevels):
// the branch rides in the digit as `br << BITS`, so ONE 16-lane gather covers
// both blocks even when they sit in different class branches. Values are
// element-identical to the two-table gather + blend it replaces.
template <int BITS>
inline __m512 LookupPairCombined512(const float* comb, __m512i dig, int br0, int br1) {
    dig = _mm512_mask_add_epi32(dig, 0x00FF, dig, _mm512_set1_epi32(br0 << BITS));
    dig = _mm512_mask_add_epi32(dig, 0xFF00, dig, _mm512_set1_epi32(br1 << BITS));
    return _mm512_i32gather_ps(dig, comb, 4);
}

// One 8-dim block widened into the low half of a zmm (high half zero).
template <int BITS>
inline __m512 LookupBlock512(const float* lev, const int32_t digs8[8]) {
    alignas(64) int32_t digs[16] = {};
    for (int i = 0; i < 8; ++i) {
        digs[i] = digs8[i];
    }
    return LookupLevels512<BITS>(lev, _mm512_load_si512(digs));
}

template <int BITS>
inline __m512 ScorePairLv(
    const SegmentCodec& c, const uint8_t* a, const uint8_t* b
) {
    alignas(64) int32_t digs[16];
    DecodeBlockDigits512<BITS>(a, digs);
    DecodeBlockDigits512<BITS>(b, digs + 8);
    const __m512i dig = _mm512_load_si512(digs);
    const __m512 l0 = LookupLevels512<BITS>(c.Levels(BITS, BranchOfPacked(a, BITS)), dig);
    const __m512 l1 = LookupLevels512<BITS>(c.Levels(BITS, BranchOfPacked(b, BITS)), dig);
    return _mm512_mask_blend_ps(static_cast<__mmask16>(0xFF00), l0, l1);
}

template <int BITS>
float ScoreFast512T(
    const SegmentCodec& c, const uint8_t* in, int d_s, const float* q
) {
    const int blocks = d_s / 8;
    __m512 acc = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    for (int blk = 0; blk + 1 < blocks; blk += 2) {
        const uint8_t* a = in + static_cast<size_t>(blk) * BITS;
        const __m512 lv = ScorePairLv<BITS>(c, a, a + BITS);
        const __m512 qv = _mm512_loadu_ps(q + static_cast<size_t>(blk) * 8);
        if constexpr (BITS >= 2) {
            if ((blk & 2) != 0) {
                acc2 = _mm512_fmadd_ps(lv, qv, acc2);
            } else {
                acc = _mm512_fmadd_ps(lv, qv, acc);
            }
        } else {
            acc = _mm512_fmadd_ps(lv, qv, acc);
        }
    }
    if ((blocks & 1) != 0) {
        const int blk = blocks - 1;
        alignas(32) int32_t d8[8];
        DecodeBlockDigits512<BITS>(in + static_cast<size_t>(blk) * BITS, d8);
        const __m512 lv = LookupBlock512<BITS>(
            c.Levels(BITS, BranchOfPacked(in + static_cast<size_t>(blk) * BITS, BITS)), d8
        );
        const __m512 qv =
            _mm512_maskz_loadu_ps(static_cast<__mmask16>(0x00FF), q + static_cast<size_t>(blk) * 8);
        acc = _mm512_fmadd_ps(lv, qv, acc);
    }
    if constexpr (BITS >= 2) {
        acc = _mm512_add_ps(acc, acc2);
    }
    return _mm512_reduce_add_ps(acc);
}

template <int BITS>
void ScoreFast4T512(
    const SegmentCodec& c, const uint8_t* const in[4], int d_s, const float* q, float out[4]
) {
    const int blocks = d_s / 8;
    __m512 acc[4];
    __m512 acc2[4];
    for (int j = 0; j < 4; ++j) {
        acc[j] = _mm512_setzero_ps();
        acc2[j] = _mm512_setzero_ps();
    }
    for (int blk = 0; blk + 1 < blocks; blk += 2) {
        const __m512 qv = _mm512_loadu_ps(q + static_cast<size_t>(blk) * 8);
        for (int j = 0; j < 4; ++j) {
            const uint8_t* a = in[j] + static_cast<size_t>(blk) * BITS;
            const __m512 lv = ScorePairLv<BITS>(c, a, a + BITS);
            if constexpr (BITS >= 2) {
                if ((blk & 2) != 0) {
                    acc2[j] = _mm512_fmadd_ps(lv, qv, acc2[j]);
                } else {
                    acc[j] = _mm512_fmadd_ps(lv, qv, acc[j]);
                }
            } else {
                acc[j] = _mm512_fmadd_ps(lv, qv, acc[j]);
            }
        }
    }
    if ((blocks & 1) != 0) {
        const int blk = blocks - 1;
        const __m512 qv =
            _mm512_maskz_loadu_ps(static_cast<__mmask16>(0x00FF), q + static_cast<size_t>(blk) * 8);
        for (int j = 0; j < 4; ++j) {
            alignas(32) int32_t d8[8];
            const uint8_t* src = in[j] + static_cast<size_t>(blk) * BITS;
            DecodeBlockDigits512<BITS>(src, d8);
            acc[j] = _mm512_fmadd_ps(
                LookupBlock512<BITS>(c.Levels(BITS, BranchOfPacked(src, BITS)), d8), qv, acc[j]
            );
        }
    }
    for (int j = 0; j < 4; ++j) {
        if constexpr (BITS >= 2) {
            acc[j] = _mm512_add_ps(acc[j], acc2[j]);
        }
        out[j] = _mm512_reduce_add_ps(acc[j]);
    }
}

inline float ScoreFast512Dispatch(
    const SegmentCodec& c, const uint8_t* in, int d_s, int bits, const float* q
) {
    switch (bits) {
        case 1: return ScoreFast512T<1>(c, in, d_s, q);
        case 2: return ScoreFast512T<2>(c, in, d_s, q);
        case 3: return ScoreFast512T<3>(c, in, d_s, q);
        case 4: return ScoreFast512T<4>(c, in, d_s, q);
        case 5: return ScoreFast512T<5>(c, in, d_s, q);
        case 6: return ScoreFast512T<6>(c, in, d_s, q);
        case 7: return ScoreFast512T<7>(c, in, d_s, q);
        case 8: return ScoreFast512T<8>(c, in, d_s, q);
        case 9: return ScoreFast512T<9>(c, in, d_s, q);
        case 10: return ScoreFast512T<10>(c, in, d_s, q);
        case 11: return ScoreFast512T<11>(c, in, d_s, q);
        case 12: return ScoreFast512T<12>(c, in, d_s, q);
        default: return ScoreFast512T<13>(c, in, d_s, q);
    }
}

inline void ScoreFast4Dispatch512(
    const SegmentCodec& c, const uint8_t* const in[4], int d_s, int bits, const float* q,
    float out[4]
) {
    switch (bits) {
        case 1: ScoreFast4T512<1>(c, in, d_s, q, out); break;
        case 2: ScoreFast4T512<2>(c, in, d_s, q, out); break;
        case 3: ScoreFast4T512<3>(c, in, d_s, q, out); break;
        case 4: ScoreFast4T512<4>(c, in, d_s, q, out); break;
        case 5: ScoreFast4T512<5>(c, in, d_s, q, out); break;
        case 6: ScoreFast4T512<6>(c, in, d_s, q, out); break;
        case 7: ScoreFast4T512<7>(c, in, d_s, q, out); break;
        case 8: ScoreFast4T512<8>(c, in, d_s, q, out); break;
        case 9: ScoreFast4T512<9>(c, in, d_s, q, out); break;
        case 10: ScoreFast4T512<10>(c, in, d_s, q, out); break;
        case 11: ScoreFast4T512<11>(c, in, d_s, q, out); break;
        case 12: ScoreFast4T512<12>(c, in, d_s, q, out); break;
        default: ScoreFast4T512<13>(c, in, d_s, q, out); break;
    }
}

// Split payload: refine field at rate t=bits-1, plane bit 7-i is digit i's top.
template <int BITS>
inline void DecodeSplitDigits(const uint8_t* ref, uint8_t plane, int32_t digs[8], int* br) {
    if constexpr (BITS == 1) {
        *br = plane & 1U;
        int par = 0;
        for (int i = 0; i < 7; ++i) {
            const int d = (plane >> (7 - i)) & 1U;
            par ^= d;
            digs[i] = static_cast<int32_t>(d);
        }
        digs[7] = static_cast<int32_t>(par);
    } else {
        constexpr int T = BITS - 1;
        DecodeBlockDigits512<T>(ref, digs);
        *br = BranchOfPacked(ref, T);
        for (int i = 0; i < 8; ++i) {
            digs[i] |= ((plane >> (7 - i)) & 1) << T;
        }
    }
}

template <int BITS>
inline float ScoreSplitBlock512(
    const SegmentCodec& c, const uint8_t* ref, uint8_t plane, const float* q8, int /*br1*/
) {
    alignas(32) int32_t digs[8];
    int br = 0;
    DecodeSplitDigits<BITS>(ref, plane, digs, &br);
    const __m512 lv = LookupBlock512<BITS>(c.Levels(BITS, br), digs);
    const __m512 qv = _mm512_maskz_loadu_ps(static_cast<__mmask16>(0x00FF), q8);
    return _mm512_reduce_add_ps(_mm512_mul_ps(lv, qv));
}

// Vectorized twin of DecodeBlockDigits512 + the plane-bit merge, one 8-dim
// block per call: the packed field is shifted at all eight digit offsets in
// one vpsrlvq (fields 0..3 out of the low word, 4..7 — and the branch at bit
// 8T-1 — out of the 32..95-bit window once T > 8) instead of eight scalar
// shifts into a stack array. Digit 7's LSB is the even-parity of the other
// digits' LSBs rather than a stored bit, so it is rebuilt from a popcount.
template <int T>
inline __m256i DecodeBlockDigitsVec(const uint8_t* ref, int* br) {
    uint64_t lo = 0;
    uint64_t w2 = 0;
    if constexpr (T <= 4) {
        uint32_t w = 0;
        if constexpr (T > 0) {
            std::memcpy(&w, ref, static_cast<size_t>(T));
        }
        lo = w;
    } else if constexpr (T <= 8) {
        // Two 4-byte register loads instead of one memcpy of T bytes: GCC
        // lowers a 6/7-byte memcpy to scalar moves assembled through a stack
        // slot (a store-forwarding round trip per block pair), which cost
        // b=7/8 cells ~2.7x in ScoreRunSplit. The two loads read exactly T
        // bytes: [ref, ref+4) and [ref+T-4, ref+T); bytes [4, T) are the top
        // T-4 bytes of hi4, realigned onto bit 32.
        uint32_t lo4 = 0;
        uint32_t hi4 = 0;
        std::memcpy(&lo4, ref, 4);
        std::memcpy(&hi4, ref + (T - 4), 4);
        lo = lo4 | (static_cast<uint64_t>(hi4 >> (32 - 8 * (T - 4))) << 32);
    } else {
        std::memcpy(&lo, ref, 8);
        uint64_t hi = 0;
        std::memcpy(&hi, ref + 8, static_cast<size_t>(T - 7));
        hi &= (uint64_t{1} << (8 * T - 64)) - 1;
        w2 = (lo >> 32) | (hi << 32);
    }
    // Digit LSBs live at bit i*T, i = 0..6; those at or past bit 64 are read
    // back out of w2. ParMaskOf itself overflows past T = 9, so the masks are
    // rebuilt here field by field.
    constexpr uint64_t kPmLo = [] {
        uint64_t m = 0;
        for (int i = 0; i < 7; ++i) {
            if (i * T < 64) {
                m |= uint64_t{1} << (i * T);
            }
        }
        return m;
    }();
    constexpr uint64_t kPmW2 = [] {
        uint64_t m = 0;
        for (int i = 0; i < 7; ++i) {
            if (i * T >= 64) {
                m |= uint64_t{1} << (i * T - 32);
            }
        }
        return m;
    }();
    const int par = static_cast<int>(
        (__builtin_popcountll(lo & kPmLo) ^ __builtin_popcountll(w2 & kPmW2)) & 1
    );
    *br = static_cast<int>((T <= 8 ? lo >> (8 * T - 1) : w2 >> (8 * T - 33)) & 1U);
    __m512i x;
    __m512i sh;
    if constexpr (T <= 8) {
        x = _mm512_set1_epi64(static_cast<int64_t>(lo));
        sh = _mm512_setr_epi64(0, T, 2 * T, 3 * T, 4 * T, 5 * T, 6 * T, 7 * T);
    } else {
        x = _mm512_set_epi64(
            static_cast<int64_t>(w2), static_cast<int64_t>(w2), static_cast<int64_t>(w2),
            static_cast<int64_t>(w2), static_cast<int64_t>(lo), static_cast<int64_t>(lo),
            static_cast<int64_t>(lo), static_cast<int64_t>(lo)
        );
        sh = _mm512_setr_epi64(
            0, T, 2 * T, 3 * T, 4 * T - 32, 5 * T - 32, 6 * T - 32, 7 * T - 32
        );
    }
    // Digit 7 keeps only its stored T-1 high bits here; the <<1|par below
    // puts them back and drops the branch bit that sits at 8T-1.
    const __m512i fm = _mm512_setr_epi64(
        (uint64_t{1} << T) - 1, (uint64_t{1} << T) - 1, (uint64_t{1} << T) - 1,
        (uint64_t{1} << T) - 1, (uint64_t{1} << T) - 1, (uint64_t{1} << T) - 1,
        (uint64_t{1} << T) - 1, (uint64_t{1} << (T - 1)) - 1
    );
    __m256i d8 = _mm512_cvtepi64_epi32(
        _mm512_and_si512(_mm512_srlv_epi64(x, sh), fm)
    );
    d8 = _mm256_mask_slli_epi32(d8, 0x80, d8, 1);
    d8 = _mm256_mask_add_epi32(d8, 0x80, d8, _mm256_set1_epi32(par));
    return d8;
}

// One block pair's 16 digits: the combined table index folds each block's
// branch in at bit T, so both blocks must come out as one 16-lane vector.
template <int BITS>
inline void DecodeSplitPair512(
    const uint8_t* ref, uint8_t plane0, uint8_t plane1, __m512i* dig, int* br0, int* br1
) {
    constexpr int T = BITS - 1;
    __m512i d = _mm512_inserti64x4(
        _mm512_castsi256_si512(DecodeBlockDigitsVec<T>(ref, br0)),
        DecodeBlockDigitsVec<T>(ref + T, br1), 1
    );
    // Digit i's top bit rides in the plane byte (bit 7-i).
    const int mask16 = BitRev8Lut()[plane0] | (BitRev8Lut()[plane1] << 8);
    *dig = _mm512_or_si512(
        d,
        _mm512_slli_epi32(_mm512_maskz_set1_epi32(static_cast<__mmask16>(mask16), 1), T)
    );
}

template <int BITS>
float ScoreRunSplit512T(
    const SegmentCodec& c, const uint8_t* ref, const uint8_t* pb, size_t blk0, int nblk,
    const float* q
) {
    static_assert(BITS >= 6 && BITS <= 13, "ScoreRunSplit512T covers b = 6..13");
    constexpr int T = BITS - 1;
    const float* const comb = c.PairLevels(BITS);
    __m512 acc = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    const uint8_t* rp = ref;
    int blk = 0;
    for (; blk + 1 < nblk; blk += 2) {
        __m512i dig;
        int br0 = 0;
        int br1 = 0;
        DecodeSplitPair512<BITS>(
            rp, pb[(blk0 + static_cast<size_t>(blk)) * 32],
            pb[(blk0 + static_cast<size_t>(blk) + 1) * 32], &dig, &br0, &br1
        );
        rp += 2 * T;
        const __m512 lv = LookupPairCombined512<BITS>(comb, dig, br0, br1);
        const __m512 qv = _mm512_loadu_ps(q + static_cast<size_t>(blk) * 8);
        if ((blk & 2) != 0) {
            acc2 = _mm512_fmadd_ps(lv, qv, acc2);
        } else {
            acc = _mm512_fmadd_ps(lv, qv, acc);
        }
    }
    if (blk < nblk) {
        alignas(64) int32_t d16[16] = {};
        int br = 0;
        DecodeSplitDigits<BITS>(rp, pb[(blk0 + static_cast<size_t>(blk)) * 32], d16, &br);
        for (int i = 0; i < 8; ++i) {
            d16[i] += br << BITS;
        }
        const __m512 lv =
            _mm512_i32gather_ps(_mm512_load_si512(d16), comb, 4);
        const __m512 qv =
            _mm512_maskz_loadu_ps(static_cast<__mmask16>(0x00FF), q + static_cast<size_t>(blk) * 8);
        acc = _mm512_fmadd_ps(lv, qv, acc);
    }
    acc = _mm512_add_ps(acc, acc2);
    return _mm512_reduce_add_ps(acc);
}

// Bit i of BitRev8Lut()[v] is bit 7-i of v, so a bit-reversed index feeds
// vpbroadcastd-with-mask directly: lane i of the broadcast holds the top
// bit of digit i, exactly the old PlaneExpandLut256 u64 word (the e8q A9
// fix — no u64 table load, no GPR->XMM move, no byte->dword widen).
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

// 4-wide run scorer in the AVX2 RefineCore shape (register-hoisted tables,
// permutevar, 4 in flight). 8-dim blocks are ymm-native; zmm pairing of two
// blocks does not hide the 4-row cascade. Covers BITS 1..5.
template <int BITS>
inline void ScoreRunSplit4T(
    const SegmentCodec& c, const uint8_t* const ref[4], const uint8_t* const pb[4],
    size_t blk0, int nblk, const float* q, int n, float out[4]
) {
    static_assert(BITS >= 1 && BITS <= 5, "ScoreRunSplit4T covers b = 1..5");
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
    constexpr int KTAB = BITS <= 3 ? 1 : (1 << (BITS - 3));
    __m256 tb0[KTAB];
    [[maybe_unused]] __m256 tb1[KTAB];
    if constexpr (BITS == 2) {
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
                __m256i dig = _mm256_maskz_set1_epi32(
                    static_cast<__mmask8>(BitRev8Lut()[plane]), 1
                );
                dig = _mm256_insert_epi32(dig, par, 7);
                lv = _mm256_permutevar8x32_ps((plane & 1U) ? tb1[0] : tb0[0], dig);
            } else if constexpr (BITS == 2) {
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
                        _mm256_maskz_set1_epi32(
                            static_cast<__mmask8>(BitRev8Lut()[plane]), 1
                        ),
                        T
                    )
                );
                const __m256i dg2 = _mm256_or_si256(
                    dig, _mm256_slli_epi32(_mm256_set1_epi32(br_sel), 2)
                );
                lv = _mm256_permutevar8x32_ps(tb0[0], dg2);
            } else {
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
                        _mm256_maskz_set1_epi32(
                            static_cast<__mmask8>(BitRev8Lut()[plane]), 1
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
    for (int j = 0; j < m; ++j) {
        out[j] = simd::HsumPs(_mm256_add_ps(acc[j], acc2[j]));
    }
}

}  // namespace

float ScoreFastAvx512(
    const SegmentCodec& c, const uint8_t* in, int d_s, int bits, const float* q
) {
    return ScoreFast512Dispatch(c, in, d_s, bits, q);
}

void ScoreFast4Avx512(
    const SegmentCodec& c, const uint8_t* const in[4], int d_s, int bits, const float* q,
    float out[4]
) {
    ScoreFast4Dispatch512(c, in, d_s, bits, q, out);
}

float ScoreBlockSplitAvx512(
    const SegmentCodec& c, const uint8_t* ref, int bits, uint8_t plane, const float* q8,
    int br1
) {
    switch (bits) {
        case 1: return ScoreSplitBlock512<1>(c, ref, plane, q8, br1);
        case 2: return ScoreSplitBlock512<2>(c, ref, plane, q8, br1);
        case 3: return ScoreSplitBlock512<3>(c, ref, plane, q8, br1);
        case 4: return ScoreSplitBlock512<4>(c, ref, plane, q8, br1);
        case 5: return ScoreSplitBlock512<5>(c, ref, plane, q8, br1);
        case 6: return ScoreSplitBlock512<6>(c, ref, plane, q8, br1);
        case 7: return ScoreSplitBlock512<7>(c, ref, plane, q8, br1);
        case 8: return ScoreSplitBlock512<8>(c, ref, plane, q8, br1);
        case 9: return ScoreSplitBlock512<9>(c, ref, plane, q8, br1);
        case 10: return ScoreSplitBlock512<10>(c, ref, plane, q8, br1);
        case 11: return ScoreSplitBlock512<11>(c, ref, plane, q8, br1);
        case 12: return ScoreSplitBlock512<12>(c, ref, plane, q8, br1);
        default: return ScoreSplitBlock512<13>(c, ref, plane, q8, br1);
    }
}

float ScoreRunSplitAvx512(
    const SegmentCodec& c, const uint8_t* ref, const uint8_t* pb, size_t blk0, int nblk,
    int bits, const float* q
) {
    // bits 1..5 share the 4-wide ymm core (n=1) so 1-wide and 4-wide stay
    // bit-identical, matching the AVX2 scorer.
    float out[1] = {0.0F};
    const uint8_t* refs[1] = {ref};
    const uint8_t* pbs[1] = {pb};
    switch (bits) {
        case 1:
            ScoreRunSplit4T<1>(c, refs, pbs, blk0, nblk, q, 1, out);
            return out[0];
        case 2:
            ScoreRunSplit4T<2>(c, refs, pbs, blk0, nblk, q, 1, out);
            return out[0];
        case 3:
            ScoreRunSplit4T<3>(c, refs, pbs, blk0, nblk, q, 1, out);
            return out[0];
        case 4:
            ScoreRunSplit4T<4>(c, refs, pbs, blk0, nblk, q, 1, out);
            return out[0];
        case 5:
            ScoreRunSplit4T<5>(c, refs, pbs, blk0, nblk, q, 1, out);
            return out[0];
        case 6:
            return ScoreRunSplit512T<6>(c, ref, pb, blk0, nblk, q);
        case 7:
            return ScoreRunSplit512T<7>(c, ref, pb, blk0, nblk, q);
        case 8:
            return ScoreRunSplit512T<8>(c, ref, pb, blk0, nblk, q);
        case 9:
            return ScoreRunSplit512T<9>(c, ref, pb, blk0, nblk, q);
        case 10:
            return ScoreRunSplit512T<10>(c, ref, pb, blk0, nblk, q);
        case 11:
            return ScoreRunSplit512T<11>(c, ref, pb, blk0, nblk, q);
        case 12:
            return ScoreRunSplit512T<12>(c, ref, pb, blk0, nblk, q);
        default:
            return ScoreRunSplit512T<13>(c, ref, pb, blk0, nblk, q);
    }
}

void ScoreRunSplit4Avx512(
    const SegmentCodec& c, const uint8_t* const ref[4], const uint8_t* const pb[4],
    size_t blk0, int nblk, int bits, const float* q, int n, float out[4]
) {
    switch (bits) {
        case 1: ScoreRunSplit4T<1>(c, ref, pb, blk0, nblk, q, n, out); return;
        case 2: ScoreRunSplit4T<2>(c, ref, pb, blk0, nblk, q, n, out); return;
        case 3: ScoreRunSplit4T<3>(c, ref, pb, blk0, nblk, q, n, out); return;
        case 4: ScoreRunSplit4T<4>(c, ref, pb, blk0, nblk, q, n, out); return;
        case 5: ScoreRunSplit4T<5>(c, ref, pb, blk0, nblk, q, n, out); return;
        default: break;
    }
    const int m = n < 4 ? n : 4;
    for (int j = 0; j < m; ++j) {
        out[j] = ScoreRunSplitAvx512(c, ref[j], pb[j], blk0, nblk, bits, q);
    }
}

}  // namespace e8qlib::me8q
