#include "e8qlib/utils/rotator.hpp"

#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "simd/kernels.hpp"

namespace e8qlib {

FhtKac4Rotator::FhtKac4Rotator(size_t dim, uint64_t seed) : dim_(dim) {
    padded_dim_ = (dim + 63) / 64 * 64;
    size_t bottom_log = 0;
    while ((size_t{1} << (bottom_log + 1)) <= dim) {
        ++bottom_log;
    }
    trunc_dim_ = size_t{1} << bottom_log;
    fac_ = 1.0F / std::sqrt(static_cast<float>(trunc_dim_));

    signs_.resize(4 * padded_dim_ / 8);
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& s : signs_) {
        s = static_cast<uint8_t>(dist(gen));
    }
}

void FhtKac4Rotator::RotateTiles(
    const float* __restrict__ src, float* __restrict__ dst, size_t len
) const {
    if (len == 0) {
        return;
    }
    if (len % 64 != 0) {
        throw std::invalid_argument("RotateTiles: length must be a multiple of 64");
    }
    // A dim-64 rotator (padded == trunc == 64) is four sign+FHT rounds with no
    // Kac walk. Reuse this object's first 64-bit sign table per round by
    // wrapping a 64-dim view: signs_ is 4 * padded_dim_/8 bytes, so a dim-64
    // object is required. Callers construct FhtKac4Rotator(64, seed).
    if (padded_dim_ != 64) {
        throw std::invalid_argument("RotateTiles: rotator padded_dim must be 64");
    }
    std::vector<float> tile(64);
    for (size_t off = 0; off < len; off += 64) {
        Rotate(src + off, tile.data());
        std::memcpy(dst + off, tile.data(), 64 * sizeof(float));
    }
}

void FhtKac4Rotator::Rotate(const float* __restrict__ src, float* __restrict__ dst) const {
    std::memcpy(dst, src, sizeof(float) * dim_);
    std::memset(dst + dim_, 0, sizeof(float) * (padded_dim_ - dim_));
    const size_t per_round = padded_dim_ / 8;
    if (trunc_dim_ == padded_dim_) {
        for (int r = 0; r < 4; ++r) {
            simd::FlipSigns(signs_.data() + r * per_round, dst, padded_dim_);
            simd::FhtFloat(dst, trunc_dim_);
            for (size_t i = 0; i < trunc_dim_; ++i) {
                dst[i] *= fac_;
            }
        }
        return;
    }
    const size_t start = padded_dim_ - trunc_dim_;
    for (int r = 0; r < 4; ++r) {
        simd::FlipSigns(signs_.data() + r * per_round, dst, padded_dim_);
        float* window = dst + ((r % 2 == 0) ? 0 : start);
        simd::FhtFloat(window, trunc_dim_);
        for (size_t i = 0; i < trunc_dim_; ++i) {
            window[i] *= fac_;
        }
        simd::KacsWalk(dst, padded_dim_);
    }
    for (size_t i = 0; i < padded_dim_; ++i) {
        dst[i] *= 0.25F;
    }
}

void FhtKac4Rotator::Save(char* out) const {
    std::memcpy(out, signs_.data(), signs_.size());
}

void FhtKac4Rotator::Load(const char* in) {
    std::memcpy(signs_.data(), in, signs_.size());
}

}  // namespace e8qlib
