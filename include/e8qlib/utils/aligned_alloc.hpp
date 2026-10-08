// Cache-line-aligned allocator for the hot index arrays.
//
// The AVX-512 scan/refine kernels read whole cache lines at a time: the level
// tables via 512-bit loads, the arena records via 256/512-bit loads. A
// 64-byte access whose base is not 64-byte aligned straddles two cache lines
// and costs two L1 accesses instead of one — which is why glibc's 16-byte
// malloc guarantee is not enough here (measured on the b >= 6 refine lookup:
// 3.0 vs 7.4 ns/call, 64-byte aligned vs 16-mod-64).
//
// Padding is one cache line per buffer. It is invisible to the memory ledger:
// `memory_bytes()` and the baseline file size both count the *used* bytes, and
// the arenas already carry a 64-byte tail slack for the same reason.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

namespace e8qlib {

template <class T, size_t Align = 64>
struct AlignedAlloc {
    using value_type = T;
    static constexpr size_t kAlign = Align;

    AlignedAlloc() noexcept = default;
    template <class U>
    AlignedAlloc(const AlignedAlloc<U, Align>&) noexcept {}
    template <class U>
    struct rebind {
        using other = AlignedAlloc<U, Align>;
    };

    [[nodiscard]] T* allocate(size_t n) {
        if (n == 0) {
            return nullptr;
        }
        void* p = nullptr;
        if (::posix_memalign(&p, Align, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }

    void deallocate(T* p, size_t) noexcept { std::free(p); }

    template <class U>
    bool operator==(const AlignedAlloc<U, Align>&) const noexcept {
        return true;
    }
    template <class U>
    bool operator!=(const AlignedAlloc<U, Align>&) const noexcept {
        return false;
    }
};

template <class T>
using AlignedVector = std::vector<T, AlignedAlloc<T, 64>>;

}  // namespace e8qlib
