#include "e8qlib/utils/cpu_features.hpp"

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if (defined(__x86_64__) || defined(__i386__)) && !defined(_MSC_VER)
#include <cpuid.h>
#endif

namespace e8qlib::cpu {
namespace {

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
constexpr bool kIsX86 = true;
#else
constexpr bool kIsX86 = false;
#endif

struct Features {
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512bw = false;
    bool avx512dq = false;
    bool avx512vl = false;
    bool avx512vbmi = false;
};

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
void cpuid(
    uint32_t leaf, uint32_t subleaf, uint32_t* a, uint32_t* b, uint32_t* c, uint32_t* d
) {
    int info[4];
    __cpuidex(info, static_cast<int>(leaf), static_cast<int>(subleaf));
    *a = static_cast<uint32_t>(info[0]);
    *b = static_cast<uint32_t>(info[1]);
    *c = static_cast<uint32_t>(info[2]);
    *d = static_cast<uint32_t>(info[3]);
}
#elif defined(__x86_64__) || defined(__i386__)
void cpuid(
    uint32_t leaf, uint32_t subleaf, uint32_t* a, uint32_t* b, uint32_t* c, uint32_t* d
) {
    __cpuid_count(leaf, subleaf, *a, *b, *c, *d);
}
#else
void cpuid(uint32_t, uint32_t, uint32_t*, uint32_t*, uint32_t*, uint32_t*) {}
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
uint64_t xgetbv(uint32_t index) { return _xgetbv(index); }
#elif defined(__x86_64__) || defined(__i386__)
uint64_t xgetbv(uint32_t index) {
    uint32_t eax = 0;
    uint32_t edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return (static_cast<uint64_t>(edx) << 32) | eax;
}
#else
uint64_t xgetbv(uint32_t) { return 0; }
#endif

Features detect_features() {
    Features hardware{};
    if constexpr (!kIsX86) {
        return hardware;
    }

    uint32_t eax = 0;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;
    cpuid(0, 0, &eax, &ebx, &ecx, &edx);
    const uint32_t max_leaf = eax;
    if (max_leaf < 1) {
        return hardware;
    }

    cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    const bool fma = ((ecx >> 12) & 1U) != 0;
    const bool osxsave = ((ecx >> 27) & 1U) != 0;
    const bool avx = ((ecx >> 28) & 1U) != 0;

    if (max_leaf >= 7) {
        uint32_t l7_eax = 0;
        uint32_t l7_ebx = 0;
        uint32_t l7_ecx = 0;
        uint32_t l7_edx = 0;
        cpuid(7, 0, &l7_eax, &l7_ebx, &l7_ecx, &l7_edx);
        hardware.avx2 = ((l7_ebx >> 5) & 1U) != 0;
        hardware.avx512f = ((l7_ebx >> 16) & 1U) != 0;
        hardware.avx512dq = ((l7_ebx >> 17) & 1U) != 0;
        hardware.avx512bw = ((l7_ebx >> 30) & 1U) != 0;
        hardware.avx512vl = ((l7_ebx >> 31) & 1U) != 0;
        hardware.avx512vbmi = ((l7_ecx >> 11) & 1U) != 0;
    }

    constexpr uint64_t kAvxStateMask = (uint64_t{1} << 1) | (uint64_t{1} << 2);
    constexpr uint64_t kAvx512StateMask =
        kAvxStateMask | (uint64_t{1} << 5) | (uint64_t{1} << 6) | (uint64_t{1} << 7);
    const uint64_t xcr0 = avx && osxsave ? xgetbv(0) : 0;
    const bool avx_state = avx && osxsave && (xcr0 & kAvxStateMask) == kAvxStateMask;
    const bool avx512_state =
        avx_state && (xcr0 & kAvx512StateMask) == kAvx512StateMask;

    Features usable{};
    usable.fma = fma && avx_state;
    usable.avx2 = hardware.avx2 && avx_state;
    usable.avx512f = hardware.avx512f && avx512_state;
    usable.avx512bw = hardware.avx512bw && avx512_state;
    usable.avx512dq = hardware.avx512dq && avx512_state;
    usable.avx512vl = hardware.avx512vl && avx512_state;
    usable.avx512vbmi = hardware.avx512vbmi && avx512_state;
    return usable;
}

const Features& features() {
    static const Features detected = detect_features();
    return detected;
}

}  // namespace

bool has_avx2() {
    const Features& detected = features();
    return detected.avx2 && detected.fma;
}

bool has_avx512() {
    const Features& detected = features();
    return detected.fma && detected.avx512f && detected.avx512bw && detected.avx512dq &&
           detected.avx512vl && detected.avx512vbmi;
}

}  // namespace e8qlib::cpu
