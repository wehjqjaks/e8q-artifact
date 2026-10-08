#pragma once

// ME8Q segment scorers. ISA TUs implement the Avx2/Avx512 variants; generic TUs
// include this header with no SIMD types. Dispatch never calls a backend that
// was not compiled.

#include <cstddef>
#include <cstdint>

#include "e8qlib/quantization/me8q_codec.hpp"

namespace e8qlib::me8q {

float ScoreFastGeneric(
    const SegmentCodec& c, const uint8_t* in, int d_s, int bits, const float* q
);
void ScoreFast4Generic(
    const SegmentCodec& c, const uint8_t* const in[4], int d_s, int bits, const float* q,
    float out[4]
);
float ScoreBlockSplitGeneric(
    const SegmentCodec& c, const uint8_t* ref, int bits, uint8_t plane, const float* q8,
    int br1
);
float ScoreRunSplitGeneric(
    const SegmentCodec& c, const uint8_t* ref, const uint8_t* pb, size_t blk0, int nblk,
    int bits, const float* q
);

#if defined(E8Q_HAVE_AVX2)
float ScoreFastAvx2(
    const SegmentCodec& c, const uint8_t* in, int d_s, int bits, const float* q
);
void ScoreFast4Avx2(
    const SegmentCodec& c, const uint8_t* const in[4], int d_s, int bits, const float* q,
    float out[4]
);
float ScoreBlockSplitAvx2(
    const SegmentCodec& c, const uint8_t* ref, int bits, uint8_t plane, const float* q8,
    int br1
);
float ScoreRunSplitAvx2(
    const SegmentCodec& c, const uint8_t* ref, const uint8_t* pb, size_t blk0, int nblk,
    int bits, const float* q
);
#endif

#if defined(E8Q_HAVE_AVX512)
float ScoreFastAvx512(
    const SegmentCodec& c, const uint8_t* in, int d_s, int bits, const float* q
);
void ScoreFast4Avx512(
    const SegmentCodec& c, const uint8_t* const in[4], int d_s, int bits, const float* q,
    float out[4]
);
float ScoreBlockSplitAvx512(
    const SegmentCodec& c, const uint8_t* ref, int bits, uint8_t plane, const float* q8,
    int br1
);
float ScoreRunSplitAvx512(
    const SegmentCodec& c, const uint8_t* ref, const uint8_t* pb, size_t blk0, int nblk,
    int bits, const float* q
);
#endif

float ScoreFastDispatch(
    const SegmentCodec& c, const uint8_t* in, int d_s, int bits, const float* q
);
void ScoreFast4Dispatch(
    const SegmentCodec& c, const uint8_t* const in[4], int d_s, int bits, const float* q,
    float out[4]
);
float ScoreBlockSplitDispatch(
    const SegmentCodec& c, const uint8_t* ref, int bits, uint8_t plane, const float* q8,
    int br1
);
float ScoreRunSplitDispatch(
    const SegmentCodec& c, const uint8_t* ref, const uint8_t* pb, size_t blk0, int nblk,
    int bits, const float* q
);
void ScoreRunSplit4Generic(
    const SegmentCodec& c, const uint8_t* const ref[4], const uint8_t* const pb[4],
    size_t blk0, int nblk, int bits, const float* q, int n, float out[4]
);
#if defined(E8Q_HAVE_AVX2)
void ScoreRunSplit4Avx2(
    const SegmentCodec& c, const uint8_t* const ref[4], const uint8_t* const pb[4],
    size_t blk0, int nblk, int bits, const float* q, int n, float out[4]
);
#endif
#if defined(E8Q_HAVE_AVX512)
void ScoreRunSplit4Avx512(
    const SegmentCodec& c, const uint8_t* const ref[4], const uint8_t* const pb[4],
    size_t blk0, int nblk, int bits, const float* q, int n, float out[4]
);
#endif
void ScoreRunSplit4Dispatch(
    const SegmentCodec& c, const uint8_t* const ref[4], const uint8_t* const pb[4],
    size_t blk0, int nblk, int bits, const float* q, int n, float out[4]
);

}  // namespace e8qlib::me8q
