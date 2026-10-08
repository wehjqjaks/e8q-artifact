#pragma once

// Packed-block field helpers shared by the portable codec and the ISA scorers.
// Not installed.

#include <cstdint>

namespace e8qlib::me8q {

inline uint64_t MaskBits(int bits) {
    return bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1);
}

inline uint64_t Extract128(unsigned __int128 x, int shift, uint64_t mask) {
    return static_cast<uint64_t>((x >> shift) & static_cast<unsigned __int128>(mask));
}

constexpr uint64_t ParMaskOf(int bits) {
    uint64_t m = 0;
    for (int k = 0; k < 7; ++k) {
        m |= uint64_t{1} << (bits * k);
    }
    return m;
}

}  // namespace e8qlib::me8q
