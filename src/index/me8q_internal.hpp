#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "e8qlib/index/me8q.hpp"

namespace e8qlib::me8q {
namespace detail {

constexpr char kMagic[8] = {'M', 'E', '8', 'Q', 'I', 'D', 'X', 'X'};

inline void WriteScalars(std::ofstream& out, const void* p, size_t bytes) {
    out.write(reinterpret_cast<const char*>(p), static_cast<long>(bytes));
}

inline void ReadScalars(std::ifstream& in, void* p, size_t bytes, const char* what) {
    in.read(reinterpret_cast<char*>(p), static_cast<long>(bytes));
    if (!in) {
        throw std::runtime_error(std::string("Me8qIndex::Load: truncated ") + what);
    }
}

inline void WritePlan(std::ofstream& out, const Plan& plan) {
    const uint64_t n = plan.size();
    WriteScalars(out, &n, sizeof(n));
    for (const PlanSeg& seg : plan) {
        const uint64_t len = seg.len;
        const int32_t bits = seg.bits;
        WriteScalars(out, &len, sizeof(len));
        WriteScalars(out, &bits, sizeof(bits));
    }
}

inline Plan ReadPlan(std::ifstream& in) {
    uint64_t n = 0;
    ReadScalars(in, &n, sizeof(n), "plan size");
    Plan plan;
    plan.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t len = 0;
        int32_t bits = 0;
        ReadScalars(in, &len, sizeof(len), "plan segment length");
        ReadScalars(in, &bits, sizeof(bits), "plan segment bits");
        plan.push_back({static_cast<size_t>(len), static_cast<int>(bits)});
    }
    return plan;
}

Plan ParsePlanSpec(const std::string& spec, size_t padded_dim);
void DatasetVariance(
    const float* base, size_t ld, size_t n, size_t dim, std::vector<float>* var
);

}  // namespace detail
}  // namespace e8qlib::me8q
