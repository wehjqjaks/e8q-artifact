// me8q segment codec: the E8Q fixed-precision quantizer applied to ONE
// dimension segment of a residual vector.
//
// Per segment (length d_s, a multiple of 8) at rate b:
//   * the segment residual r is normalized to unit norm, and the encoder runs
//     on u = r / ||r|| scaled by s * sqrt(d_s) with s = e8qlib::FixedScale(b)
//     (identical to e8q's whole-vector path, with the segment's own rd);
//   * levels are e8qlib::BranchLevels(b, branch) — companded Phi^-1 quantiles,
//     even-parity digits, per-block branch selection, single-flip parity
//     repair: the SAME codebook as the fixed-precision e8q line;
//   * the per-vector amplitude is NOT stored in the code: the estimator
//     absorbs it into the per-segment coefficient, exactly as e8q's g1 does.
//
// Storage is bit-tight and identical to e8q's accounting: b bits/dim, i.e.
// 8b bits per 8-dim block, laid out as digits 0..6 at b bits + digit 7's high
// b-1 bits + the branch bit. Digit 7's LSB is redundant (the even-parity
// shell forces it to the XOR of the other digits' LSBs), so the branch rides
// in the freed slot for free — the same trick as e8q's refine row.
//
// Rate range: 1..13 (SAQ's KMaxQuantizeBits). EncodeVectorAtScale uses the
// same uint16 digits; me8q_test pins SegmentCodec to it digit-for-digit.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "e8qlib/quantization/e8q_codec.hpp"
#include "e8qlib/quantization/me8q_plan.hpp"  // kMaxBits

namespace e8qlib::me8q {

// Bytes of code payload for a segment of d_s dims at `bits` (d_s % 8 == 0).
inline int SegmentBytes(int bits, int d_s) { return bits * d_s / 8; }

struct SegCode {
    std::vector<uint16_t> digits;  // 8 per block, values in [0, 2^bits)
    std::vector<uint8_t> branch;   // one per block
};

// The segment codec: level geometry shared with e8q, plus the scorer. All
// tables are built in the constructor — Build and Search both call the codec
// from OpenMP regions, so no table may be created lazily. Encoding itself is
// e8qlib::EncodeVectorAtScale (one encoder for e8q and me8q).
class SegmentCodec {
   public:
    SegmentCodec();

    // Raw level values (e8qlib::BranchLevels as float) for query-time scoring;
    // 2^bits entries, bits 1..13, both branches.
    [[nodiscard]] const float* Levels(int bits, int branch) const;
    // Both branch tables concatenated: tab_[0][bits] ++ tab_[1][bits], the
    // combined table holding 2^(bits+1) floats (bits >= 6). A digit with the
    // branch folded in as `br << bits` indexes the combined table, so one
    // 16-lane gather covers a block pair whose blocks may sit in different
    // class branches (halves the gathers of the split scorer; identical
    // values, so results stay bit-identical).
    [[nodiscard]] const float* PairLevels(int bits) const;
    // max_b,br |level[b][d]| — certified |S| <= Lmax * Σ|q_i| for a run at this rate.
    [[nodiscard]] float LevelAbsMax(int bits) const;

    // Encodes the segment residual (d_s floats) at the segment's own rate.
    // `norm` = ||r|| (0 emits the zero codeword). Digit-identical to
    // EncodeVectorAtScale on the unit residual at FixedScale(bits).
    [[nodiscard]] SegCode Encode(const float* r, int d_s, int bits, double norm) const;

    // The same encode straight into a caller-owned buffer, which the builders
    // reuse across rows: `Encode` allocates a unit buffer and a SegCode per
    // call, and the build calls it once per segment of every row.
    void EncodeInto(const float* r, int d_s, int bits, double norm, VectorCode& vc) const;

    // The fp64 segment scores below, read from the code that `EncodeInto` just
    // wrote instead of re-unpacking the packed bytes the way Score /
    // ScoreAndNorm2 do. The unpacked digits are the encoder's digits (digit 7's
    // stored LSB is the parity it was repaired to), so the sums are the same
    // ones, coordinate for coordinate and in the same order.
    struct Fits {
        double query = 0.0;     // sum_i q0[i] * level[br_i][digit_i]
        double centroid = 0.0;  // sum_i q1[i] * level[br_i][digit_i]
    };
    // Both per-segment fits the amplitude needs -- the residual dot (the
    // denominator) and the centroid dot (the h2 cross term) -- against the
    // same code, in ONE pass over the digits. Each accumulator keeps its own
    // coordinate order, so this is bit-identical to scoring the two query rows
    // in two separate calls.
    [[nodiscard]] Fits ScoreVectorCode(
        const VectorCode& vc, int bits, const float* q0, const float* q1
    ) const;

    // S = sum_i q[i] * level[br_i][digit_i] over the segment.
    // fp64 accum — used at build for the amplitude fit (dot_c into h2).
    [[nodiscard]] double Score(
        const uint8_t* in, int d_s, int bits, const float* q
    ) const;
    // Search-path scorer: same S, AVX2 permute (bits<=5) / gather (bits>=6)
    // + FMA, fp32 accum. Matches Score to ~1e-4 relative; do not use for the fit.
    [[nodiscard]] float ScoreFast(
        const uint8_t* in, int d_s, int bits, const float* q
    ) const;
    // Split-storage scorer for one 8-dim block: `ref` holds the (rate-1)-bit
    // refine field, `plane` the 8 top bits (bit 7-i = digit i's top bit). At
    // rate 1 the plane byte is the whole code: signs of dims 0..6, branch in
    // bit 0, dim 7's sign = parity (`br1` is ignored).
    // Returns q . levels over the block.
    [[nodiscard]] float ScoreBlockSplit(
        const uint8_t* ref, int bits, uint8_t plane, const float* q8, int br1 = 0
    ) const;
    // Run-level form of ScoreBlockSplit: scores `nblk` consecutive blocks of
    // ONE run in a single templated pass — the rate dispatches once, the
    // level dot accumulates in a vector register, and the row's plane bytes
    // are read from a caller-built array `pb` (global block indexing, so the
    // run's blocks are pb[blk0 .. blk0+nblk)).
    // Bit-identical semantics to repeated ScoreBlockSplit calls.
    [[nodiscard]] float ScoreRunSplit(
        const uint8_t* ref, const uint8_t* pb, size_t blk0, int nblk, int bits,
        const float* q
    ) const;
    // Four independent rows of ScoreRunSplit (same run, same query). Bit-identical
    // to four ScoreRunSplit calls; AVX2 shares the query loads across the four.
    void ScoreRunSplit4(
        const uint8_t* const ref[4], const uint8_t* const pb[4], size_t blk0, int nblk,
        int bits, const float* q, int n, float out[4]
    ) const;
    // Branch-averaged top-bit class midpoint and the certified deviation of
    // any exact level from any class midpoint at this rate (the two-level
    // scan's slack constant — a codebook-geometry quantity, base-only).
    struct ClassModel {
        float pbar[2] = {0.0F, 0.0F};
        float gamma = 0.0F;
    };
    [[nodiscard]] ClassModel ClassModelOf(int bits) const;
    // Four independent rows of the same (d_s, bits, q). Hides gather latency
    // the way parent E8Q RefineBatch does (4 in flight). `in[j]` is the packed
    // segment start of row j, not the IVF row.
    void ScoreFast4(
        const uint8_t* const in[4], int d_s, int bits, const float* q, float out[4]
    ) const;
    // Same, plus N2 = sum_i level[br_i][digit_i]^2 (the amplitude fit's
    // denominator). fp64; build only.
    [[nodiscard]] double ScoreAndNorm2(
        const uint8_t* in, int d_s, int bits, const float* q, double* n2_out
    ) const;

   private:
    std::vector<float> tab_[2][kMaxBits + 1];       // raw levels, for scoring
    std::vector<float> comb_[kMaxBits + 1];         // tab_[0] ++ tab_[1] (bits >= 6)
    float lmax_[kMaxBits + 1] = {};                 // LevelAbsMax per rate
    // FixedScale per rate.  Encode runs once per segment per vector, and
    // e8qlib::FixedScale's lazily-initialised schedule table costs a
    // thread-safe-init guard -- a __tls_get_addr -- on every call.  Caching
    // the value here removes it from the build's inner loop; the numbers are
    // the same doubles, so the codes are unchanged.
    double scale_[kMaxBits + 1] = {};
};

// Bit-tight pack/unpack (b bytes per 8-dim block, see the header comment).
void PackSegment(const SegCode& code, int d_s, int bits, uint8_t* out);
// The same pack off a VectorCode, so a builder that encodes into a reused
// buffer never materializes the SegCode the encoder would otherwise return.
void PackSegment(const VectorCode& code, int d_s, int bits, uint8_t* out);
// `digits` receives d_s entries and `branch` d_s/8; digit 7's LSB is restored
// from the parity shell.
void UnpackSegment(
    const uint8_t* in, int d_s, int bits, uint16_t* digits, uint8_t* branch
);

// Reconstructed segment in level units (no amplitude): out[i] =
// level[br][digit]. Used by the shape-table measurement and the tests.
void ReconstructLevels(
    const uint8_t* in, int d_s, int bits, const SegmentCodec& codec, float* out
);

}  // namespace e8qlib::me8q
