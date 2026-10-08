#pragma once

// E8Q search hot path (quantize / b=1 exact / refine / scan). Generic TUs
// include this with no SIMD types. ISA TUs implement the Avx2/Avx512 variants.

#include <cstddef>
#include <cstdint>

#include "e8qlib/index/e8q.hpp"

namespace e8qlib::search_simd {

void QuantizeU8Generic(
    const float* src, size_t n, float qinv, float qoff, float hc, float delta, uint8_t* dst
);
float ExactScoreB1Generic(
    const E8qIndex& idx, const uint8_t* plane, uint32_t lane, const float* q
);
void RefineBatchGeneric(
    const E8qIndex& idx, const SurvivorRow* rows, size_t n, const float* q, QueryScratch& s
);
void ScanClusterGeneric(
    const E8qIndex& idx, size_t cid, float d2, const float* q, QueryScratch& s,
    SearchStats* stats
);

#if defined(E8Q_HAVE_AVX2)
void QuantizeU8Avx2(
    const float* src, size_t n, float qinv, float qoff, float hc, float delta, uint8_t* dst
);
float ExactScoreB1Avx2(
    const E8qIndex& idx, const uint8_t* plane, uint32_t lane, const float* q
);
void RefineBatchAvx2(
    const E8qIndex& idx, const SurvivorRow* rows, size_t n, const float* q, QueryScratch& s
);
void ScanClusterAvx2(
    const E8qIndex& idx, size_t cid, float d2, const float* q, QueryScratch& s,
    SearchStats* stats
);
#endif

#if defined(E8Q_HAVE_AVX512)
void QuantizeU8Avx512(
    const float* src, size_t n, float qinv, float qoff, float hc, float delta, uint8_t* dst
);
float ExactScoreB1Avx512(
    const E8qIndex& idx, const uint8_t* plane, uint32_t lane, const float* q
);
void RefineBatchAvx512(
    const E8qIndex& idx, const SurvivorRow* rows, size_t n, const float* q, QueryScratch& s
);
void ScanClusterAvx512(
    const E8qIndex& idx, size_t cid, float d2, const float* q, QueryScratch& s,
    SearchStats* stats
);
#endif

void QuantizeU8(
    const float* src, size_t n, float qinv, float qoff, float hc, float delta, uint8_t* dst
);
float ExactScoreB1(
    const E8qIndex& idx, const uint8_t* plane, uint32_t lane, const float* q
);
void RefineBatch(
    const E8qIndex& idx, const SurvivorRow* rows, size_t n, const float* q, QueryScratch& s
);
void ScanCluster(
    const E8qIndex& idx, size_t cid, float d2, const float* q, QueryScratch& s,
    SearchStats* stats
);

}  // namespace e8qlib::search_simd
