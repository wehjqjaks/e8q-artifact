// Shared LUT quantization for the scan paths.
//
// The 32-vector FastScan batches themselves are laid out and scored by the
// engines that own them (the E8Q scan's `PlaneExpandLut` in
// src/index/e8q_scan_common.hpp, ME8Q's coarse plane in
// src/index/me8q_fast.cpp). What both need from here is the one kernel that
// turns a float LUT into the u8 form those batches accumulate: a single
// global (min, delta) affine map, so the scan can accumulate in u8 and undo
// the scale once per batch.
#pragma once

#include <cstddef>
#include <cstdint>

namespace e8qlib::fastscan {

// Quantize `len` floats to u8 under one global (min, delta) pair, returned in
// `min_out` / `delta_out`. Delta is 1 when the range is degenerate.
void QuantizeLut(
    uint8_t* dst, const float* src, size_t len, float* min_out, float* delta_out
);

}  // namespace e8qlib::fastscan
