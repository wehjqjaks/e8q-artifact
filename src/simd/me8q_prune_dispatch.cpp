#include "simd/me8q_prune.hpp"

#include "e8qlib/simd/dispatch.hpp"

namespace e8qlib::me8q {

size_t CoarsePrune(const CoarsePruneArgs& a) {
#if defined(E8Q_HAVE_AVX512)
    if (simd::Avx512Enabled()) {
        return CoarsePruneAvx512(a);
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (simd::Avx2Enabled()) {
        return CoarsePruneAvx2(a);
    }
#endif
    return CoarsePruneGeneric(a);
}

}  // namespace e8qlib::me8q
