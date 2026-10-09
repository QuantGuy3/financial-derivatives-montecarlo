#include "simd.hpp"

#include <algorithm>
#include <atomic>

#if defined(MC_X86_64) && defined(_MSC_VER) && !defined(__clang__)
#include <immintrin.h>
#include <intrin.h>
#endif

namespace mc::cpu {

namespace {

#ifdef MC_X86_64
bool detect_avx2() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#elif defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool osxsave = (r[2] & (1 << 27)) != 0, avx = (r[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;
    if ((_xgetbv(0) & 6) != 6) return false;     // el sistema operativo guarda el estado YMM
    __cpuidex(r, 7, 0);
    return (r[1] & (1 << 5)) != 0;
#else
    return false;
#endif
}
#endif

std::atomic<int> g_simd_level{-1};   // -1 = sin inicializar

} // namespace

SimdLevel simd_level_available() {
#ifdef MC_X86_64
    static const SimdLevel best = detect_avx2() ? SimdLevel::Avx2 : SimdLevel::Scalar;
    return best;
#else
    return SimdLevel::Scalar;
#endif
}

SimdLevel simd_level() {
    int v = g_simd_level.load(std::memory_order_relaxed);
    if (v < 0) {
        v = (int)simd_level_available();
        g_simd_level.store(v, std::memory_order_relaxed);
    }
    return (SimdLevel)v;
}

void set_simd_level(SimdLevel level) {
    g_simd_level.store(std::min((int)level, (int)simd_level_available()), std::memory_order_relaxed);
}

} // namespace mc::cpu
