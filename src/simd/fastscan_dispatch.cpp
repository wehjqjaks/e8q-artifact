#include "e8qlib/fastscan/fastscan.hpp"
#include "e8qlib/simd/dispatch.hpp"
#include "simd/fastscan_kernels.hpp"

namespace e8qlib::fastscan {

void QuantizeLut(
    uint8_t* dst, const float* src, size_t len, float* min_out, float* delta_out
) {
#if defined(E8Q_HAVE_AVX512)
    if (simd::Avx512Enabled()) {
        QuantizeLutAvx512(dst, src, len, min_out, delta_out);
        return;
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (simd::Avx2Enabled()) {
        QuantizeLutAvx2(dst, src, len, min_out, delta_out);
        return;
    }
#endif
    QuantizeLutGeneric(dst, src, len, min_out, delta_out);
}

}  // namespace e8qlib::fastscan
