#pragma once

namespace e8qlib::cpu {

// True when the running CPU (and OS XSAVE state) can execute AVX2+FMA.
bool has_avx2();

// True when AVX-512F + BW + DQ + VL + VBMI and the matching XSAVE state are usable.
bool has_avx512();

}  // namespace e8qlib::cpu
