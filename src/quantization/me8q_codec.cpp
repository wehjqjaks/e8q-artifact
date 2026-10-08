#include "e8qlib/quantization/me8q_codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "quantization/me8q_bits.hpp"
#include "simd/me8q_score.hpp"

namespace e8qlib::me8q {

SegmentCodec::SegmentCodec() {
    for (int bits = 1; bits <= kMaxBits; ++bits) {
        scale_[bits] = e8qlib::FixedScale(bits);
        for (int br = 0; br < 2; ++br) {
            const std::vector<double> lev = e8qlib::BranchLevels(bits, br);
            std::vector<float>& raw = tab_[br][bits];
            // Pad to 32 so bits<=5 permutevar8x32 / blendv loads of 8/16/32
            // floats never walk off the allocation. Unused tail is zero.
            raw.assign(std::max(static_cast<size_t>(32), lev.size()), 0.0F);
            for (size_t i = 0; i < lev.size(); ++i) {
                raw[i] = static_cast<float>(lev[i]);
            }
        }
        float mx = 0.0F;
        for (int br = 0; br < 2; ++br) {
            for (float v : tab_[br][bits]) {
                mx = std::max(mx, std::abs(v));
            }
        }
        lmax_[bits] = mx;
        // Combined branch table: branch 1 appended right after branch 0, at
        // the offset `1 << bits` the split scorer folds into the digit.
        if (bits >= 6) {
            const size_t half = tab_[0][bits].size();
            std::vector<float>& cb = comb_[bits];
            cb.assign(2 * half, 0.0F);
            std::copy(tab_[0][bits].begin(), tab_[0][bits].end(), cb.begin());
            std::copy(tab_[1][bits].begin(), tab_[1][bits].end(), cb.begin() + half);
        }
    }
}

const float* SegmentCodec::PairLevels(int bits) const {
    return comb_[bits].data();
}

const float* SegmentCodec::Levels(int bits, int branch) const {
    return tab_[branch != 0 ? 1 : 0][bits].data();
}

float SegmentCodec::LevelAbsMax(int bits) const {
    if (bits <= 0 || bits > kMaxBits) {
        return 0.0F;
    }
    return lmax_[bits];
}

SegCode SegmentCodec::Encode(const float* r, int d_s, int bits, double norm) const {
    VectorCode vc;
    EncodeInto(r, d_s, bits, norm, vc);
    SegCode out;
    out.digits.assign(static_cast<size_t>(d_s), 0);
    out.branch.assign(static_cast<size_t>(d_s) / 8, 0);
    for (size_t blk = 0; blk < vc.digits.size(); ++blk) {
        for (int i = 0; i < 8; ++i) {
            out.digits[blk * 8 + static_cast<size_t>(i)] = vc.digits[blk][static_cast<size_t>(i)];
        }
        out.branch[blk] = static_cast<uint8_t>(vc.branch[blk]);
    }
    return out;
}

void SegmentCodec::EncodeInto(
    const float* r, int d_s, int bits, double norm, VectorCode& vc
) const {
    if (bits < 1 || bits > kMaxBits) {
        throw std::invalid_argument("SegmentCodec::Encode: bits out of [1, 13]");
    }
    if (d_s % 8 != 0) {
        throw std::invalid_argument("SegmentCodec::Encode: segment length must be a multiple of 8");
    }
    const int blocks = d_s / 8;
    if (!(norm > 0.0)) {
        // Zero codeword: every digit and branch zero, which is even parity.
        vc.digits.assign(static_cast<size_t>(blocks), std::array<uint16_t, 8>{});
        vc.branch.assign(static_cast<size_t>(blocks), 0);
        return;
    }
    // One division buffer per thread, reused: the codec's outer loop cannot
    // share a compile unit with this caller's path (see the note above
    // EncodeVectorAtScaleInto), so the segment is divided here. The buffer
    // replaces the per-call allocation the builders used to pay, and the
    // rounding is the one the digits are defined against.
    static thread_local std::vector<float> unit;
    unit.resize(static_cast<size_t>(d_s));
    const double inv = 1.0 / norm;
    for (int i = 0; i < d_s; ++i) {
        unit[static_cast<size_t>(i)] =
            static_cast<float>(static_cast<double>(r[i]) * inv);
    }
    EncodeVectorAtScaleInto(unit.data(), d_s, bits, scale_[bits], vc);
}

namespace {

// One 8-dimension block into a `bits`-byte field: digits 0..6 at `bits` each,
// digit 7 without its parity bit at `bits-1`, and the branch in the TOP bit of
// the 8*bits field. NOTE: the e8q codec stores the branch in the field's bit 0
// (PackRefineBlock, e8q_codec.cpp) -- do not mix the two packings.
//
// Both code containers spell a block as 8 consecutive uint16_t (SegCode as a
// flat vector, VectorCode as a vector of std::array<uint16_t, 8>), so this is
// the one definition of the byte layout and each PackSegment only decides how
// to address it.
void PackBlock(const uint16_t* d, uint64_t branch, int bits, uint8_t* out) {
    const uint64_t mask = MaskBits(bits);
    const uint64_t mask7 = MaskBits(bits - 1);
    unsigned __int128 x = 0;
    for (int i = 0; i < 7; ++i) {
        x |= static_cast<unsigned __int128>(static_cast<uint64_t>(d[i]) & mask)
             << (bits * i);
    }
    x |= static_cast<unsigned __int128>((static_cast<uint64_t>(d[7]) >> 1) & mask7)
         << (bits * 7);
    x |= static_cast<unsigned __int128>(branch & 1U) << (8 * bits - 1);
    std::memcpy(out, &x, static_cast<size_t>(bits));
}

}  // namespace

void PackSegment(const SegCode& code, int d_s, int bits, uint8_t* out) {
    const int blocks = d_s / 8;
    for (int blk = 0; blk < blocks; ++blk) {
        PackBlock(
            &code.digits[static_cast<size_t>(blk) * 8], code.branch[static_cast<size_t>(blk)],
            bits, out + static_cast<size_t>(blk) * bits
        );
    }
}

void PackSegment(const VectorCode& code, int d_s, int bits, uint8_t* out) {
    const int blocks = d_s / 8;
    for (int blk = 0; blk < blocks; ++blk) {
        PackBlock(
            code.digits[static_cast<size_t>(blk)].data(),
            static_cast<uint64_t>(code.branch[static_cast<size_t>(blk)]), bits,
            out + static_cast<size_t>(blk) * bits
        );
    }
}

SegmentCodec::Fits SegmentCodec::ScoreVectorCode(
    const VectorCode& vc, int bits, const float* q0, const float* q1
) const {
    const float* const lev_b[2] = {Levels(bits, 0), Levels(bits, 1)};
    const size_t blocks = vc.digits.size();
    Fits f;
    for (size_t blk = 0; blk < blocks; ++blk) {
        const float* const lev = lev_b[vc.branch[blk] != 0 ? 1 : 0];
        const float* const b0 = q0 + blk * 8;
        const float* const b1 = q1 + blk * 8;
        const std::array<uint16_t, 8>& d = vc.digits[blk];
        for (int i = 0; i < 8; ++i) {
            const float lv = lev[d[static_cast<size_t>(i)]];
            f.query += static_cast<double>(b0[i]) * static_cast<double>(lv);
            f.centroid += static_cast<double>(b1[i]) * static_cast<double>(lv);
        }
    }
    return f;
}

void UnpackSegment(
    const uint8_t* in, int d_s, int bits, uint16_t* digits, uint8_t* branch
) {
    const int blocks = d_s / 8;
    const uint64_t mask = MaskBits(bits);
    const uint64_t mask7 = MaskBits(bits - 1);
    for (int blk = 0; blk < blocks; ++blk) {
        unsigned __int128 x = 0;
        std::memcpy(&x, in + static_cast<size_t>(blk) * bits, static_cast<size_t>(bits));
        uint16_t* d = &digits[static_cast<size_t>(blk) * 8];
        int parity = 0;
        for (int i = 0; i < 7; ++i) {
            const uint64_t f = Extract128(x, bits * i, mask);
            d[i] = static_cast<uint16_t>(f);
            parity ^= static_cast<int>(f & 1U);
        }
        const uint64_t f7 = Extract128(x, bits * 7, mask7);
        d[7] = static_cast<uint16_t>((f7 << 1) | static_cast<uint64_t>(parity));
        if (branch != nullptr) {
            branch[static_cast<size_t>(blk)] =
                static_cast<uint8_t>(Extract128(x, 8 * bits - 1, 1));
        }
    }
}

double SegmentCodec::Score(const uint8_t* in, int d_s, int bits, const float* q) const {
    return ScoreAndNorm2(in, d_s, bits, q, nullptr);
}

float SegmentCodec::ScoreFast(
    const uint8_t* in, int d_s, int bits, const float* q
) const {
    return ScoreFastDispatch(*this, in, d_s, bits, q);
}

void SegmentCodec::ScoreFast4(
    const uint8_t* const in[4], int d_s, int bits, const float* q, float out[4]
) const {
    ScoreFast4Dispatch(*this, in, d_s, bits, q, out);
}

float SegmentCodec::ScoreBlockSplit(
    const uint8_t* ref, int bits, uint8_t plane, const float* q8, int br1
) const {
    return ScoreBlockSplitDispatch(*this, ref, bits, plane, q8, br1);
}

float SegmentCodec::ScoreRunSplit(
    const uint8_t* ref, const uint8_t* pb, size_t blk0, int nblk, int bits,
    const float* q
) const {
    return ScoreRunSplitDispatch(*this, ref, pb, blk0, nblk, bits, q);
}

void SegmentCodec::ScoreRunSplit4(
    const uint8_t* const ref[4], const uint8_t* const pb[4], size_t blk0, int nblk,
    int bits, const float* q, int n, float out[4]
) const {
    ScoreRunSplit4Dispatch(*this, ref, pb, blk0, nblk, bits, q, n, out);
}

SegmentCodec::ClassModel SegmentCodec::ClassModelOf(int bits) const {
    ClassModel m;
    const int q = 1 << bits;
    const int half = q >> 1;
    for (int c = 0; c < 2; ++c) {
        double sum = 0.0;
        for (int br = 0; br < 2; ++br) {
            const float* lev = Levels(bits, br);
            for (int d = c * half; d < (c + 1) * half; ++d) {
                sum += static_cast<double>(lev[d]);
            }
        }
        m.pbar[c] = static_cast<float>(sum / static_cast<double>(2 * half));
    }
    double mx = 0.0;
    for (int br = 0; br < 2; ++br) {
        const float* lev = Levels(bits, br);
        for (int d = 0; d < q; ++d) {
            for (int c = 0; c < 2; ++c) {
                mx = std::max(
                    mx, std::abs(static_cast<double>(lev[d]) -
                                 static_cast<double>(m.pbar[c]))
                );
            }
        }
    }
    m.gamma = static_cast<float>(mx);
    return m;
}

double SegmentCodec::ScoreAndNorm2(
    const uint8_t* in, int d_s, int bits, const float* q, double* n2_out
) const {
    const int blocks = d_s / 8;
    const uint64_t mask = MaskBits(bits);
    const uint64_t mask7 = MaskBits(bits - 1);
    double sum = 0.0;
    double n2 = 0.0;
    for (int blk = 0; blk < blocks; ++blk) {
        unsigned __int128 x = 0;
        std::memcpy(&x, in + static_cast<size_t>(blk) * bits, static_cast<size_t>(bits));
        const int br = static_cast<int>(Extract128(x, 8 * bits - 1, 1));
        const float* lev = Levels(bits, br);
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
            const float lv = lev[digs[i]];
            sum += static_cast<double>(qb[i]) * static_cast<double>(lv);
            n2 += static_cast<double>(lv) * static_cast<double>(lv);
        }
    }
    if (n2_out != nullptr) {
        *n2_out = n2;
    }
    return sum;
}

void ReconstructLevels(
    const uint8_t* in, int d_s, int bits, const SegmentCodec& codec, float* out
) {
    std::vector<uint16_t> digits(static_cast<size_t>(d_s));
    std::vector<uint8_t> branch(static_cast<size_t>(d_s) / 8);
    UnpackSegment(in, d_s, bits, digits.data(), branch.data());
    for (int blk = 0; blk < d_s / 8; ++blk) {
        const float* lev = codec.Levels(bits, branch[static_cast<size_t>(blk)]);
        for (int i = 0; i < 8; ++i) {
            out[blk * 8 + i] = lev[digits[static_cast<size_t>(blk) * 8 + i]];
        }
    }
}

}  // namespace e8qlib::me8q
