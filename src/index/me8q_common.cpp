#include "index/me8q_internal.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace e8qlib::me8q {
namespace detail {

void DatasetVariance(
    const float* base, size_t ld, size_t n, size_t dim, std::vector<float>* var
) {
    std::vector<float> mean(dim, 0.0F);
    for (size_t i = 0; i < n; ++i) {
        const float* row = base + i * ld;
        for (size_t j = 0; j < dim; ++j) {
            mean[j] += row[j];
        }
    }
    for (size_t j = 0; j < dim; ++j) {
        mean[j] /= static_cast<float>(n);
    }
    std::vector<double> tmp(dim, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const float* row = base + i * ld;
        for (size_t j = 0; j < dim; ++j) {
            const double d = static_cast<double>(row[j]) - mean[j];
            tmp[j] += d * d;
        }
    }
    for (size_t j = 0; j < dim; ++j) {
        (*var)[j] = static_cast<float>(tmp[j] / static_cast<double>(n));
    }
}

Plan ParsePlanSpec(const std::string& spec, size_t padded_dim) {
    Plan plan;
    size_t pos = 0;
    size_t total = 0;
    while (pos <= spec.size()) {
        const size_t comma = spec.find(',', pos);
        const std::string tok =
            spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty()) {
            const size_t colon = tok.find(':');
            if (colon == std::string::npos) {
                throw std::invalid_argument("plan_spec: expected L:B, got " + tok);
            }
            const long len = std::atol(tok.substr(0, colon).c_str());
            const long bits = std::atol(tok.substr(colon + 1).c_str());
            if (len <= 0 || len % static_cast<long>(kSegDim) != 0 || bits < 0 ||
                bits > kMaxBits) {
                throw std::invalid_argument("plan_spec: bad run " + tok);
            }
            plan.push_back({static_cast<size_t>(len), static_cast<int>(bits)});
            total += static_cast<size_t>(len);
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    if (total != padded_dim) {
        throw std::invalid_argument(
            "plan_spec: runs cover " + std::to_string(total) + " of " +
            std::to_string(padded_dim) + " padded dims"
        );
    }
    return plan;
}


}  // namespace detail
}  // namespace e8qlib::me8q
