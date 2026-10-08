// E8Q codec: companded E8-quotient block quantization.
//
// Per 8-dim block at rate b: 8 digits t_i in [0, q), q = 2^b, even parity of
// the digit sum at every rate b >= 1 (the E8 quotient shell: 8b-1 digit bits
// + 1 branch bit = 8b logical bits, capacity 2^(8b)), branch br in {0,1};
// levels are Phi^{-1}-companded quantiles at sigma_c^2 = 5/4. A whole vector
// is the per-block union over 3 anchor factors x 2 branches of greedy parity
// encodings, re-selected per block by one-step conditional gain.
//
// Two reconstruction scales appear below:
//   * raw level scale  : recon_i = level[br_i, d_i]                (v_hat_exact)
//   * coarse mid scale : recon_i = mid[br][top_bits(d_i)], and for
//                        digit 7 the branch mean level              (v_hat_coarse)
// The P1 kernel scores the coarse model; refine scores the exact model.
// err_v = || v_hat_exact - v_hat_coarse || is the certified prune slack.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace e8qlib {

constexpr double kSigmaC = 1.1180339887498949;  // sqrt(5/4)
// Codec rate range, matching SAQ CAQ (KMaxQuantizeBits). Uniform E8Q IVF uses
// the same range. Digits at b = 9..13 need a 16-bit container (q = 2^b > 256);
// the FastScan coarse gate stays the 1-bit digit-MSB plane at every rate, and
// the extra bits live in the survivor-only refine rows (the SAQ
// short-code/long-code split).
inline constexpr int kMaxQuantizeBits = 13;
// Size of the white-box fallback scale grid (DefaultScales). The production
// codec stores NO scales at all: the quantization grid uses the constant
// LevelRms(bits) and the estimator's g1 absorbs every per-vector amplitude.
inline constexpr int kNumScale = 16;
// Fallback scale grid (log-spaced) used by the white-box tests only.
std::vector<double> DefaultScales();

// The per-rate codec scale schedule s(b): the grid alignment of each level
// table, fitted once offline on the isotropic Gaussian block premise
// (a codec-geometry constant; no dataset, query set, or recall target
// enters).  It maximizes the expected squared directional cosine of the
// decoded VECTOR, which is the quantity the dot-normalized estimator
// coefficient g1 exposes: the amplitude is absorbed by g1, so the scale only
// shapes which codeword the encoder selects and nothing about it is stored.
// The whole-vector (not per-block) form is the one that predicts recall --
// one g1 is fitted per vector, so the angle that enters the error is the
// vector's.  See the definition in the .cpp for the full derivation pointer.
double FixedScale(int bits);

// levels[d] = sigma_c * Phi^{-1}(1/2 + (t + eta)/q), t from ceil(-q/2 - eta),
// digit d = t mod q; eta = 1/4 (branch 0) or 3/4 (branch 1).
std::vector<double> BranchLevels(int bits, int branch);

// Branch-averaged mean level of each digit top-bit class. Same scalar for
// every dim; callers zero dim 7 at b == 1 (that scan bit is the BRANCH).
void ScanClassMeans(int bits, double w_cls[2]);

struct BlockCode {
    std::array<uint16_t, 8> digits{};
    int branch = 0;
    double cost = 0.0;
};

// Greedy per-coordinate nearest even/odd-digit level with parity repair.
// `levels` is one branch's table; `v` are the 8 scaled values; `branch`
// selects the (cached) sorted parity tables for that branch.
BlockCode EncodeBlock(
    const double* v, int bits, const std::vector<double>& levels, int branch
);

// Whole-vector encode with data-adaptive per-block scales: for each block,
// estimate the closed-form optimal scale s* = (lev . lev) /
// ((unit . lev) * sqrt(dim)) from the better branch probe, snap it to the
// learned table, re-encode both branches at the snapped scale and keep the
// cheaper one (parity repair inside EncodeBlock). A per-VECTOR variant
// (median-snapped single scale) was measured at -0.0065 recall on gist b3
// and reverted. `scales` must hold kNumScale ascending positive values
// (EncodeVectorAtScale is the production path; this grid variant remains
// for the white-box codec tests, where nullptr selects DefaultScales()).
struct VectorCode {
    std::vector<std::array<uint16_t, 8>> digits;
    std::vector<int> branch;
};
// The production path: the quantization grid uses the caller's constant
// scale (LevelRms(bits) in Build; the rotated residual is isotropic, so the
// optimal grid alignment is a constant of the level tables) and the
// estimator's g1 absorbs every per-vector amplitude: score = sum
// q*lev/(s*rd) = (g1/s) * sum q*lev/rd with g1 fitted against the bare
// tables. The scale therefore costs ZERO storage.
VectorCode EncodeVectorAtScale(
    const float* unit_dir, int padded_dim, int bits, double s
);

// The same encode into a caller-owned buffer. The builders run this once per
// vector (me8q: once per segment), so reusing one buffer per thread removes
// two heap allocations and two frees per call. `vc` is resized in place and
// fully overwritten.
void EncodeVectorAtScaleInto(
    const float* unit_dir, int padded_dim, int bits, double s, VectorCode& vc
);

VectorCode EncodeVector(const float* unit_dir, int padded_dim, int bits, const float* scales);

// ---------------------------------------------------------------------------
// Packing.
//
// RowA byte per block (the scan's whole code at b == 1, the digit top bits at
// b >= 2 — packed by PackCoarseRow in the index, this documents the layout):
//   b == 1: bit (7-i) = digit i (the sign) for i in 0..6; bit 0 = branch
//           (dim 7's sign is the parity of digits 0..6).
//   b >= 2: bit (7-i) = top bit of digit i (dim 7's real top bit at bit 0).
//
// Refine payload: b >= 2 stores the LOW c = b-1 bits of every digit
// bit-tight little-endian, x = sum f_i << (c i) in c bytes per block; the
// digit top bits live in the scan rowA byte (one per dim, dim i at bit
// 7-i), so storage totals 1 + (b-1) = b bits/dim -- rabitq parity. Decode
// is one broadcast + srlv + and (c <= 4, the two-half extraction above c),
// plus one OR of the plane-injected top bits. b == 1 has no refine rows at
// all: the rowA byte IS the code.
inline int RefineBlockBytes(int bits) {
    // b == 1 has no refine rows at all: the digits ARE the scan plane byte.
    return bits == 1 ? 0 : bits - 1;
}
// b >= 2: the row carries the low c = b-1 bits of digits 0..6, and for
// digit 7 the bits 1..c-1 with bit 0 REPLACED BY THE BRANCH. Digit 7's true
// LSB is the parity of the other seven digits' LSBs (the E8 quotient shell
// constraint), so the branch costs zero extra storage: digits use 8b-1
// bits + 1 branch = 8b bits = rabitq parity.
// Pack is used at build; the Unpack overloads and the white-box EncodeVector
// grid exist for the codec round-trip tests (search decodes inline in the SIMD
// kernels, not through these).
void PackRefineBlock(const uint8_t digits[8], int branch, int bits, uint8_t* out);
void PackRefineBlock(const uint16_t digits[8], int branch, int bits, uint8_t* out);
// Test-only inverse of PackRefineBlock (digits + branch back out).
void UnpackRefineBlock(
    const uint8_t* in, int bits, uint8_t plane_byte, uint8_t digits[8], int* branch
);
void UnpackRefineBlock(
    const uint8_t* in, int bits, uint8_t plane_byte, uint16_t digits[8], int* branch
);

// Refinement level tables (built once per index, query-independent):
//   table[br * stride + d] = level[br][d] / sqrt(padded_dim)
// stride = max(256, 2^bits): b <= 8 keeps the 256-wide gather layout,
// b = 9..13 uses 2^b. Estimator's g1 carries the unstored scale.
std::vector<float> BuildRefineTables(int bits, int padded_dim);

}  // namespace e8qlib
