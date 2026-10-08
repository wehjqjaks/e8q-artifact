#include "simd/me8q_score.hpp"

#include "e8qlib/simd/dispatch.hpp"

namespace e8qlib::me8q {

float ScoreFastDispatch(
    const SegmentCodec& c, const uint8_t* in, int d_s, int bits, const float* q
) {
#if defined(E8Q_HAVE_AVX512)
    if (simd::Avx512Enabled()) {
        return ScoreFastAvx512(c, in, d_s, bits, q);
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (simd::Avx2Enabled()) {
        return ScoreFastAvx2(c, in, d_s, bits, q);
    }
#endif
    return ScoreFastGeneric(c, in, d_s, bits, q);
}

void ScoreFast4Dispatch(
    const SegmentCodec& c, const uint8_t* const in[4], int d_s, int bits, const float* q,
    float out[4]
) {
#if defined(E8Q_HAVE_AVX512)
    if (simd::Avx512Enabled()) {
        ScoreFast4Avx512(c, in, d_s, bits, q, out);
        return;
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (simd::Avx2Enabled()) {
        ScoreFast4Avx2(c, in, d_s, bits, q, out);
        return;
    }
#endif
    ScoreFast4Generic(c, in, d_s, bits, q, out);
}

float ScoreBlockSplitDispatch(
    const SegmentCodec& c, const uint8_t* ref, int bits, uint8_t plane, const float* q8,
    int br1
) {
#if defined(E8Q_HAVE_AVX512)
    if (simd::Avx512Enabled()) {
        return ScoreBlockSplitAvx512(c, ref, bits, plane, q8, br1);
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (simd::Avx2Enabled()) {
        return ScoreBlockSplitAvx2(c, ref, bits, plane, q8, br1);
    }
#endif
    return ScoreBlockSplitGeneric(c, ref, bits, plane, q8, br1);
}

float ScoreRunSplitDispatch(
    const SegmentCodec& c, const uint8_t* ref, const uint8_t* pb, size_t blk0, int nblk,
    int bits, const float* q
) {
#if defined(E8Q_HAVE_AVX512)
    if (simd::Avx512Enabled()) {
        return ScoreRunSplitAvx512(c, ref, pb, blk0, nblk, bits, q);
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (simd::Avx2Enabled()) {
        return ScoreRunSplitAvx2(c, ref, pb, blk0, nblk, bits, q);
    }
#endif
    return ScoreRunSplitGeneric(c, ref, pb, blk0, nblk, bits, q);
}

void ScoreRunSplit4Dispatch(
    const SegmentCodec& c, const uint8_t* const ref[4], const uint8_t* const pb[4],
    size_t blk0, int nblk, int bits, const float* q, int n, float out[4]
) {
#if defined(E8Q_HAVE_AVX512)
    if (simd::Avx512Enabled()) {
        ScoreRunSplit4Avx512(c, ref, pb, blk0, nblk, bits, q, n, out);
        return;
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (simd::Avx2Enabled()) {
        ScoreRunSplit4Avx2(c, ref, pb, blk0, nblk, bits, q, n, out);
        return;
    }
#endif
    ScoreRunSplit4Generic(c, ref, pb, blk0, nblk, bits, q, n, out);
}

}  // namespace e8qlib::me8q
