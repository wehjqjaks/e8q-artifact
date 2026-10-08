#include "e8qlib/simd/dispatch.hpp"

#include "e8qlib/utils/cpu_features.hpp"

namespace e8qlib::simd {

bool Avx2Enabled() {
#if defined(E8Q_HAVE_AVX2)
    return cpu::has_avx2();
#else
    return false;
#endif
}

bool Avx512Enabled() {
#if defined(E8Q_HAVE_AVX512)
    return cpu::has_avx512();
#else
    return false;
#endif
}

}  // namespace e8qlib::simd
