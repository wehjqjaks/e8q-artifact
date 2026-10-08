#include "simd/e8q_search.hpp"

#include "e8qlib/simd/dispatch.hpp"

namespace e8qlib::search_simd {

void QuantizeU8(
    const float* src, size_t n, float qinv, float qoff, float hc, float delta, uint8_t* dst
) {
#if defined(E8Q_HAVE_AVX512)
    if (simd::Avx512Enabled()) {
        QuantizeU8Avx512(src, n, qinv, qoff, hc, delta, dst);
        return;
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (simd::Avx2Enabled()) {
        QuantizeU8Avx2(src, n, qinv, qoff, hc, delta, dst);
        return;
    }
#endif
    QuantizeU8Generic(src, n, qinv, qoff, hc, delta, dst);
}

float ExactScoreB1(const E8qIndex& idx, const uint8_t* plane, uint32_t lane, const float* q) {
#if defined(E8Q_HAVE_AVX512)
    if (simd::Avx512Enabled()) {
        return ExactScoreB1Avx512(idx, plane, lane, q);
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (simd::Avx2Enabled()) {
        return ExactScoreB1Avx2(idx, plane, lane, q);
    }
#endif
    return ExactScoreB1Generic(idx, plane, lane, q);
}

void RefineBatch(
    const E8qIndex& idx, const SurvivorRow* rows, size_t n, const float* q, QueryScratch& s
) {
#if defined(E8Q_HAVE_AVX512)
    if (simd::Avx512Enabled()) {
        RefineBatchAvx512(idx, rows, n, q, s);
        return;
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (simd::Avx2Enabled()) {
        RefineBatchAvx2(idx, rows, n, q, s);
        return;
    }
#endif
    RefineBatchGeneric(idx, rows, n, q, s);
}

void ScanCluster(
    const E8qIndex& idx, size_t cid, float d2, const float* q, QueryScratch& s,
    SearchStats* stats
) {
#if defined(E8Q_HAVE_AVX512)
    if (simd::Avx512Enabled()) {
        ScanClusterAvx512(idx, cid, d2, q, s, stats);
        return;
    }
#endif
#if defined(E8Q_HAVE_AVX2)
    if (simd::Avx2Enabled()) {
        ScanClusterAvx2(idx, cid, d2, q, s, stats);
        return;
    }
#endif
    ScanClusterGeneric(idx, cid, d2, q, s, stats);
}

}  // namespace e8qlib::search_simd
