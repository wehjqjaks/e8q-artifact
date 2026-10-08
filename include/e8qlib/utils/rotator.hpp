// FHT-Kac-4 rotator (standalone port).
//
// Semantics mirror the RaBitQ reference rotator (rabitq-repository
// include/rabitqlib/utils/rotator.hpp rotate()): pad to a multiple of 64,
// 4 rounds of {random sign flip, unnormalized Hadamard transform of the
// power-of-two window, 1/sqrt(window) rescale}, an unnormalized (x+y, x-y)
// Kac walk between rounds when padded > window, and a final 0.25 rescale on
// the padded path. Signs are generated from an explicit seed so indexes are
// reproducible.
#pragma once

#include <cstdint>
#include <vector>

namespace e8qlib {

class FhtKac4Rotator {
   public:
    FhtKac4Rotator() = default;
    FhtKac4Rotator(size_t dim, uint64_t seed);

    [[nodiscard]] size_t padded_dim() const { return padded_dim_; }

    // src has the constructor's dim floats; dst has padded_dim() floats.
    void Rotate(const float* __restrict__ src, float* __restrict__ dst) const;

    // Independent FHT–Kac-4 on each 64-dim tile of a padded buffer. ME8Q uses
    // this so SAQ's 64-dim bit-allocation packing is not mixed across tiles.
    // `len` must be a multiple of 64. src and dst may alias.
    void RotateTiles(const float* __restrict__ src, float* __restrict__ dst, size_t len) const;

    void Save(char* out) const;
    void Load(const char* in);
    [[nodiscard]] size_t dump_bytes() const { return signs_.size(); }

   private:
    size_t dim_ = 0;
    size_t padded_dim_ = 0;
    size_t trunc_dim_ = 0;
    float fac_ = 0.0F;
    std::vector<uint8_t> signs_;  // 4 sign tables, one bit per float
};

}  // namespace e8qlib
