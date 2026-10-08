#pragma once

// Runtime ISA dispatch. Public so callers can log which kernel ran.
// Detection never executes a high-ISA kernel; see cpu_features.hpp.
//
// Compile-time E8Q_ISA selects which kernel TUs are linked:
//   generic  → scalar only
//   avx2     → AVX2 kernel TUs
//   avx512   → AVX-512 kernel TUs (no AVX2 kernel TUs; missing 512 ops
//              fall back to generic)
// Runtime Avx2Enabled / Avx512Enabled never report a backend that was
// not compiled.

namespace e8qlib::simd {

bool Avx2Enabled();
bool Avx512Enabled();

}  // namespace e8qlib::simd
