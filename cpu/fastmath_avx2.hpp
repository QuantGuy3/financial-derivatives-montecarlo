#pragma once
// Piezas AVX2 compartidas por fastmath.cpp y normal.cpp. USO INTERNO: solo se incluye desde ficheros
// .cpp y todo lo que hay aquí debe llamarse desde funciones marcadas MC_TARGET_AVX2.
//
// Cada función hace las mismas operaciones, en el mismo orden, que su gemela escalar de fastmath.hpp.

#include "fastmath.hpp"
#include "simd.hpp"

#ifdef MC_X86_64
#include <immintrin.h>

namespace mc::cpu {

// Los cuatro pares {a, b} de una tabla de pares -> vector de las a y vector de las b.
MC_TARGET_AVX2 inline void gather_pairs_avx2(const double (*tab)[2], const int* idx, __m256d& first, __m256d& second) {
    const __m256d lo = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(tab[idx[0]])), _mm_loadu_pd(tab[idx[2]]), 1);
    const __m256d hi = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(tab[idx[1]])), _mm_loadu_pd(tab[idx[3]]), 1);
    first = _mm256_unpacklo_pd(lo, hi);
    second = _mm256_unpackhi_pd(lo, hi);
}

// fast_log de cuatro valores (finitos, normales y > 0; el llamante lo garantiza o descarta el resultado).
MC_TARGET_AVX2 inline __m256d fast_log4_avx2(__m256d vx) {
    using namespace fastmath_detail;
    const __m256i ix = _mm256_castpd_si256(vx);
    const __m256i tmp = _mm256_sub_epi64(ix, _mm256_set1_epi64x((long long)kLogOff));
    // dwords altos de tmp: de ahí salen k = tmp >> 52 (con signo) e i = (tmp >> 45) & 127
    const __m128i hi32 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(tmp, _mm256_setr_epi32(1, 3, 5, 7, 0, 0, 0, 0)));
    const __m256d kd = _mm256_cvtepi32_pd(_mm_srai_epi32(hi32, 20));
    alignas(16) int idx[4];
    _mm_store_si128((__m128i*)idx, _mm_and_si128(_mm_srli_epi32(hi32, 13), _mm_set1_epi32(127)));
    const __m256d z = _mm256_castsi256_pd(_mm256_sub_epi64(ix, _mm256_and_si256(tmp, _mm256_set1_epi64x((long long)0xFFF0000000000000ULL))));
    __m256d invc, logc;
    gather_pairs_avx2(kLogTab, idx, invc, logc);
    const __m256d r = _mm256_sub_pd(_mm256_mul_pd(z, invc), _mm256_set1_pd(1.0));
    const __m256d r2 = _mm256_mul_pd(r, r);
    //   p = r2 * ((kL2 + r*kL3) + r2 * ((kL4 + r*kL5) + r2*kL6))
    const __m256d a0 = _mm256_add_pd(_mm256_set1_pd(kL2), _mm256_mul_pd(r, _mm256_set1_pd(kL3)));
    const __m256d a1 = _mm256_add_pd(_mm256_add_pd(_mm256_set1_pd(kL4), _mm256_mul_pd(r, _mm256_set1_pd(kL5))),
                                     _mm256_mul_pd(r2, _mm256_set1_pd(kL6)));
    const __m256d p = _mm256_mul_pd(r2, _mm256_add_pd(a0, _mm256_mul_pd(r2, a1)));
    //   log = ((kd*kLn2Hi + logc) + r) + (kd*kLn2Lo + p)
    return _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(kd, _mm256_set1_pd(kLn2Hi)), logc), r),
                         _mm256_add_pd(_mm256_mul_pd(kd, _mm256_set1_pd(kLn2Lo)), p));
}

// fast_exp de cuatro valores con |y| < 700.
MC_TARGET_AVX2 inline __m256d fast_exp4_avx2(__m256d y) {
    using namespace fastmath_detail;
    const __m256d shift = _mm256_set1_pd(kShift);
    const __m256d ke = _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(y, _mm256_set1_pd(kInvLn2N)), shift), shift);
    const __m128i ki = _mm256_cvttpd_epi32(ke);
    const __m256d t = _mm256_sub_pd(_mm256_sub_pd(y, _mm256_mul_pd(ke, _mm256_set1_pd(kLn2NHi))),
                                    _mm256_mul_pd(ke, _mm256_set1_pd(kLn2NLo)));
    const __m256d t2 = _mm256_mul_pd(t, t);
    //   q = t + t2 * ((kE2 + t*kE3) + t2 * (kE4 + t*kE5))
    const __m256d b0 = _mm256_add_pd(_mm256_set1_pd(kE2), _mm256_mul_pd(t, _mm256_set1_pd(kE3)));
    const __m256d b1 = _mm256_add_pd(_mm256_set1_pd(kE4), _mm256_mul_pd(t, _mm256_set1_pd(kE5)));
    const __m256d q = _mm256_add_pd(t, _mm256_mul_pd(t2, _mm256_add_pd(b0, _mm256_mul_pd(t2, b1))));
    alignas(16) int j[4];
    _mm_store_si128((__m128i*)j, _mm_and_si128(ki, _mm_set1_epi32(127)));
    const __m256d s = _mm256_setr_pd(kExp2Tab[j[0]], kExp2Tab[j[1]], kExp2Tab[j[2]], kExp2Tab[j[3]]);
    const __m256d scale = _mm256_castsi256_pd(
        _mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm_add_epi32(_mm_srai_epi32(ki, 7), _mm_set1_epi32(1023))), 52));
    return _mm256_mul_pd(_mm256_add_pd(s, _mm256_mul_pd(s, q)), scale);
}

} // namespace mc::cpu

#endif // MC_X86_64
