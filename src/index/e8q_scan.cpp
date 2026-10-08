#include "e8qlib/index/e8q.hpp"

#include "index/e8q_internal.hpp"

#include <cstdint>
#include <cstring>

namespace e8qlib {

// RowA byte for one (vector, block): digit top bits of dims 0..6 at bits
// 7..1; bit 0 is the BRANCH at b == 1 (dim 7's sign is the parity of digits
// 0..6, so the stored bit is free) and dim 7's top bit at b >= 2. One byte
// per block per lane at every rate; the scan prices it with one branch-blind
// LUT (per-dim class means of the top-bit class).
void PackCoarseRow(
    const uint16_t digits[8], int branch, int bits, int blk, uint8_t* row
) {
    uint8_t byte = 0;
    for (int i = 0; i < 8; ++i) {
        const int top = bits == 1 && i == 7 ? (branch & 1)
                                            : ((digits[i] >> (bits - 1)) & 1);
        if (top != 0) {
            byte = static_cast<uint8_t>(byte | static_cast<uint8_t>(1u << (7 - i)));
        }
    }
    row[blk] = byte;
}

// Decode of one (lane, block) from the rowA byte + the (b-1)-byte refine
// field: digits 0..6 = rowA top bit << (b-1) | field bits 0..b-2; digit 7 =
// rowA bit 7 << (b-1) | field bits 1..b-2 << 1 | parity, where the parity is
// the XOR of digits 0..6 (the E8 quotient shell) and the field's bit 0 is the
// branch. Storage is exactly b bits/dim.
void DecodeSplitBlockLane(
    int bits, const uint8_t* rowa_batch, const uint8_t* refine_lane, uint32_t blk,
    uint32_t lane, uint8_t digits[8], int* branch
) {
    const int c = bits - 1;
    const uint8_t rbA = rowa_batch[static_cast<size_t>(blk) * 32 + lane];
    uint64_t x = 0;
    std::memcpy(&x, refine_lane + static_cast<size_t>(blk) * static_cast<size_t>(c),
                static_cast<size_t>(c));
    const uint64_t mask = (uint64_t{1} << c) - 1;
    int par = 0;
    for (int i = 0; i < 7; ++i) {
        const uint64_t f = (x >> (c * i)) & mask;
        digits[i] = static_cast<uint8_t>(
            f | (static_cast<uint64_t>((rbA >> (7 - i)) & 1) << c)
        );
        par ^= static_cast<int>(f & 1u);
    }
    const uint64_t f7 = (x >> (c * 7)) & mask;
    digits[7] = static_cast<uint8_t>(
        ((f7 >> 1) << 1) | static_cast<uint64_t>(par) |
        (static_cast<uint64_t>((rbA >> 0) & 1) << c)
    );
    *branch = static_cast<int>(f7 & 1u);
}

// b = 9..13 twin of the decode above (same layout): the refine field is
// c = b-1 up to 12 bytes, so the staging integer is 128-bit and the digits
// are 16-bit.
void DecodeSplitBlockLane(
    int bits, const uint8_t* rowa_batch, const uint8_t* refine_lane, uint32_t blk,
    uint32_t lane, uint16_t digits[8], int* branch
) {
    const int c = bits - 1;
    const uint8_t rbA = rowa_batch[static_cast<size_t>(blk) * 32 + lane];
    unsigned __int128 x = 0;
    std::memcpy(&x, refine_lane + static_cast<size_t>(blk) * static_cast<size_t>(c),
                static_cast<size_t>(c));
    const unsigned __int128 mask = (static_cast<unsigned __int128>(1) << c) - 1;
    int par = 0;
    for (int i = 0; i < 7; ++i) {
        const unsigned __int128 f = (x >> (c * i)) & mask;
        digits[i] = static_cast<uint16_t>(
            f | (static_cast<unsigned __int128>((rbA >> (7 - i)) & 1) << c)
        );
        par ^= static_cast<int>(f & 1u);
    }
    const unsigned __int128 f7 = (x >> (c * 7)) & mask;
    digits[7] = static_cast<uint16_t>(
        ((f7 >> 1) << 1) | static_cast<unsigned __int128>(par) |
        (static_cast<unsigned __int128>((rbA >> 0) & 1) << c)
    );
    *branch = static_cast<int>(f7 & 1u);
}

}  // namespace e8qlib
