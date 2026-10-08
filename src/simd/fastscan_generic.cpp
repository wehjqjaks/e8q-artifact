#include "simd/fastscan_kernels.hpp"

#include <algorithm>

namespace e8qlib::fastscan {
void QuantizeLutGeneric(
    uint8_t* dst, const float* src, size_t len, float* min_out, float* delta_out
) {
    float lo = src[0];
    float hi = src[0];
    for (size_t i = 1; i < len; ++i) {
        lo = std::min(lo, src[i]);
        hi = std::max(hi, src[i]);
    }
    const float delta = hi > lo ? (hi - lo) / 255.0F : 1.0F;
    for (size_t i = 0; i < len; ++i) {
        const float q = (src[i] - lo) / delta;
        const int v = static_cast<int>(q + 0.5F);
        dst[i] = static_cast<uint8_t>(std::clamp(v, 0, 255));
    }
    *min_out = lo;
    *delta_out = delta;
}

}  // namespace e8qlib::fastscan
