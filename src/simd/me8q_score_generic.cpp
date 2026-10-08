#include "simd/me8q_score.hpp"

#include <cstring>

#include "quantization/me8q_bits.hpp"

namespace e8qlib::me8q {
namespace {

inline float ScoreSplitBlockPortable(
    const SegmentCodec& c, const uint8_t* rp, int bits, uint8_t plane, const float* q8
) {
    const int t = bits - 1;
    unsigned __int128 x = 0;
    std::memcpy(&x, rp, static_cast<size_t>(t));
    const int br = static_cast<int>(Extract128(x, 8 * t - 1, 1));
    const float* lev = c.Levels(bits, br);
    const uint64_t mask = MaskBits(t);
    int parity = 0;
    float acc = 0.0F;
    for (int i = 0; i < 7; ++i) {
        const uint64_t f = Extract128(x, t * i, mask);
        parity ^= static_cast<int>(f & 1U);
        const int top = (plane >> (7 - i)) & 1;
        acc += q8[i] * lev[static_cast<int>(f) | (top << t)];
    }
    const uint64_t f7 = Extract128(x, t * 7, MaskBits(t > 0 ? t - 1 : 0));
    const int top7 = plane & 1;
    const int digit7 =
        static_cast<int>((f7 << 1) | static_cast<uint64_t>(parity)) | (top7 << t);
    acc += q8[7] * lev[digit7];
    return acc;
}

}  // namespace

float ScoreFastGeneric(
    const SegmentCodec& c, const uint8_t* in, int d_s, int bits, const float* q
) {
    const int blocks = d_s / 8;
    const uint64_t mask = MaskBits(bits);
    const uint64_t mask7 = MaskBits(bits - 1);
    float acc = 0.0F;
    for (int blk = 0; blk < blocks; ++blk) {
        unsigned __int128 x = 0;
        std::memcpy(&x, in + static_cast<size_t>(blk) * bits, static_cast<size_t>(bits));
        const int br = static_cast<int>(Extract128(x, 8 * bits - 1, 1));
        const float* lev = c.Levels(bits, br);
        const float* qb = q + static_cast<size_t>(blk) * 8;
        int parity = 0;
        uint32_t digs[8];
        for (int i = 0; i < 7; ++i) {
            const uint64_t f = Extract128(x, bits * i, mask);
            digs[i] = static_cast<uint32_t>(f);
            parity ^= static_cast<int>(f & 1U);
        }
        const uint64_t f7 = Extract128(x, bits * 7, mask7);
        digs[7] = static_cast<uint32_t>((f7 << 1) | static_cast<uint64_t>(parity));
        for (int i = 0; i < 8; ++i) {
            acc += qb[i] * lev[digs[i]];
        }
    }
    return acc;
}

void ScoreFast4Generic(
    const SegmentCodec& c, const uint8_t* const in[4], int d_s, int bits, const float* q,
    float out[4]
) {
    for (int j = 0; j < 4; ++j) {
        out[j] = ScoreFastGeneric(c, in[j], d_s, bits, q);
    }
}

float ScoreBlockSplitGeneric(
    const SegmentCodec& c, const uint8_t* ref, int bits, uint8_t plane, const float* q8,
    int /*br1*/
) {
    if (bits == 1) {
        const float* lev = c.Levels(1, plane & 1U);
        float acc = 0.0F;
        int par = 0;
        for (int i = 0; i < 7; ++i) {
            const int d = (plane >> (7 - i)) & 1U;
            par ^= d;
            acc += q8[i] * lev[static_cast<size_t>(d)];
        }
        acc += q8[7] * lev[static_cast<size_t>(par)];
        return acc;
    }
    return ScoreSplitBlockPortable(c, ref, bits, plane, q8);
}

float ScoreRunSplitGeneric(
    const SegmentCodec& c, const uint8_t* ref, const uint8_t* pb, size_t blk0, int nblk,
    int bits, const float* q
) {
    if (bits == 1) {
        float acc = 0.0F;
        for (int blk = 0; blk < nblk; ++blk) {
            const uint8_t plane = pb[(blk0 + static_cast<size_t>(blk)) * 32];
            const float* lev = c.Levels(1, plane & 1U);
            int par = 0;
            for (int i = 0; i < 7; ++i) {
                const int d = (plane >> (7 - i)) & 1U;
                par ^= d;
                acc += q[static_cast<size_t>(blk) * 8 + i] *
                       lev[static_cast<size_t>(d)];
            }
            acc += q[static_cast<size_t>(blk) * 8 + 7] *
                   lev[static_cast<size_t>(par)];
        }
        return acc;
    }
    const int t = bits - 1;
    const uint8_t* rp = ref;
    float acc = 0.0F;
    for (int blk = 0; blk < nblk; ++blk) {
        acc += ScoreSplitBlockPortable(
            c, rp, bits, pb[(blk0 + static_cast<size_t>(blk)) * 32], q + static_cast<size_t>(blk) * 8
        );
        rp += static_cast<size_t>(t);
    }
    return acc;
}

void ScoreRunSplit4Generic(
    const SegmentCodec& c, const uint8_t* const ref[4], const uint8_t* const pb[4],
    size_t blk0, int nblk, int bits, const float* q, int n, float out[4]
) {
    const int m = n < 4 ? n : 4;
    for (int j = 0; j < m; ++j) {
        out[j] = ScoreRunSplitGeneric(c, ref[j], pb[j], blk0, nblk, bits, q);
    }
}

}  // namespace e8qlib::me8q
