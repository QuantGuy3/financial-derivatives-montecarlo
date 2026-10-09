#include "fastmath.hpp"

#include "fastmath_avx2.hpp"
#include "simd.hpp"

namespace mc::cpu {

namespace {

#ifdef MC_X86_64

// Cuatro valores. Hace las mismas operaciones, en el mismo orden, que fast_pow (fastmath.hpp).
// Devuelve false (sin escribir nada) si algún valor queda fuera del rango de la vía rápida.
MC_TARGET_AVX2 bool fast_pow4(const double* x, double b, double* out) {
    const __m256d vx = _mm256_loadu_pd(x);
    const __m256d in_range = _mm256_and_pd(_mm256_cmp_pd(vx, _mm256_set1_pd(2.2250738585072014e-308), _CMP_GE_OQ),
                                           _mm256_cmp_pd(vx, _mm256_set1_pd(1.7976931348623157e308), _CMP_LE_OQ));
    const __m256d y = _mm256_mul_pd(_mm256_set1_pd(b), fast_log4_avx2(vx));
    const __m256d absmask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
    const __m256d ok = _mm256_and_pd(in_range, _mm256_cmp_pd(_mm256_and_pd(y, absmask), _mm256_set1_pd(700.0), _CMP_LT_OQ));
    if (_mm256_movemask_pd(ok) != 15) return false;
    _mm256_storeu_pd(out, fast_exp4_avx2(y));
    return true;
}

MC_TARGET_AVX2 void fast_pow_n_avx2(const double* x, double b, double* out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        if (!fast_pow4(x + i, b, out + i)) {   // algún valor fuera de rango: los cuatro por la vía escalar
            const double x0 = x[i], x1 = x[i + 1], x2 = x[i + 2], x3 = x[i + 3];
            out[i] = fast_pow(x0, b); out[i + 1] = fast_pow(x1, b); out[i + 2] = fast_pow(x2, b); out[i + 3] = fast_pow(x3, b);
        }
    }
    for (; i < n; i++) out[i] = fast_pow(x[i], b);
    _mm256_zeroupper();
}

#endif // MC_X86_64

} // namespace

void fast_pow_n(const double* x, double b, double* out, int n) {
#ifdef MC_X86_64
    if (simd_level() == SimdLevel::Avx2) {
        fast_pow_n_avx2(x, b, out, n);
        return;
    }
#endif
    for (int i = 0; i < n; i++) out[i] = fast_pow(x[i], b);
}

} // namespace mc::cpu
