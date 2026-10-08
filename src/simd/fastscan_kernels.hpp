#pragma once

// ISA-specific LUT quantization kernels. Not installed. The public entry
// point in e8qlib/fastscan/fastscan.hpp dispatches to these at runtime.

#include <cstddef>
#include <cstdint>

namespace e8qlib::fastscan {

void QuantizeLutGeneric(
    uint8_t* dst, const float* src, size_t len, float* min_out, float* delta_out
);

#if defined(E8Q_HAVE_AVX2)
void QuantizeLutAvx2(
    uint8_t* dst, const float* src, size_t len, float* min_out, float* delta_out
);
#endif

#if defined(E8Q_HAVE_AVX512)
void QuantizeLutAvx512(
    uint8_t* dst, const float* src, size_t len, float* min_out, float* delta_out
);
#endif

}  // namespace e8qlib::fastscan
