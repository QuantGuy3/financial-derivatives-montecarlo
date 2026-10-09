#pragma once
// Potencia x^b rápida para el motor CPU (volatilidad local de Dupire: (S/S0)^(beta-1) en cada paso).
//
// pow() de la biblioteca cuesta ~11 ns por llamada y era el 60 % del tiempo de los modelos Dupire.
// fast_pow calcula exp(b·log x) con dos funciones propias, sin llamadas:
//
//   fast_log:  x = 2^k · z, z en [0.748, 1.496); tabla de 128 intervalos {invc, logc} con
//              r = z·invc - 1 (|r| < 0.004) y log z = logc + log1p(r), log1p por serie de grado 6.
//              El intervalo que contiene al 1 tiene invc = 1 y logc = 0, así que log(1) = 0 exacto.
//   fast_exp:  y = n·ln2/128 + t (|t| < 0.0028); exp y = 2^(n/128) · exp(t), exp(t) por serie de grado 5.
//
// Los polinomios se evalúan en árbol (Estrin), no por Horner: sin FMA cada eslabón de Horner cuesta
// 6 ciclos de latencia y la función entera es una cadena de ~100; lo que limita la velocidad es esa
// cadena, no el nº de operaciones. Por la misma razón la versión de verdad rápida es fast_pow_n, que
// con AVX2 lleva 4 valores por cadena.
//
// Precisión: error relativo < 4e-16·(2 + |b·ln x|) (hasta 4 ulp en el rango de Dupire; std::pow de
// UCRT/glibc da ~0.5 ulp). Para una volatilidad local es irrelevante, pero NO es un sustituto general
// de std::pow. Con las tablas generadas (tools/gen_fastmath_tables.py) el resultado no depende de la
// libm de la plataforma, y la versión escalar y la AVX2 dan los mismos bits.
//
// Fuera del rango normal (x <= 0, subnormal, infinito, NaN, |b·ln x| >= 700) se delega en std::pow.

#include <bit>
#include <cmath>
#include <cstdint>

namespace mc::cpu {

namespace fastmath_detail {
#include "fastmath_tables.inc"
inline constexpr uint64_t kLogOff = 0x3FE8000000000000ULL - (1ULL << 44);
inline constexpr double kShift = 6755399441055744.0;   // 1.5·2^52: (v + kShift) - kShift redondea v al entero más cercano
// Coeficientes (los mismos valores en la versión escalar y en la vectorial)
inline constexpr double kL2 = -0.5, kL3 = 1.0 / 3.0, kL4 = -0.25, kL5 = 0.2, kL6 = -1.0 / 6.0;
inline constexpr double kE2 = 0.5, kE3 = 1.0 / 6.0, kE4 = 1.0 / 24.0, kE5 = 1.0 / 120.0;
} // namespace fastmath_detail

// log(x) para x finito, normal y > 0 (sin comprobaciones).
inline double fast_log(double x) {
    using namespace fastmath_detail;
    const uint64_t ix = std::bit_cast<uint64_t>(x);
    const uint64_t tmp = ix - kLogOff;
    const int i = (int)((tmp >> 45) & 127);
    const double kd = (double)(int)((int64_t)tmp >> 52);
    const double z = std::bit_cast<double>(ix - (tmp & 0xFFF0000000000000ULL));
    const double r = z * kLogTab[i][0] - 1.0;
    // log1p(r) - r = r²·(-1/2 + r/3 + r²·(-1/4 + r/5 + r²·(-1/6)))
    const double r2 = r * r;
    const double p = r2 * ((kL2 + r * kL3) + r2 * ((kL4 + r * kL5) + r2 * kL6));
    return ((kd * kLn2Hi + kLogTab[i][1]) + r) + (kd * kLn2Lo + p);
}

// exp(y) para |y| < 700 (sin comprobaciones).
inline double fast_exp(double y) {
    using namespace fastmath_detail;
    const double kd = (y * kInvLn2N + kShift) - kShift;
    const int ki = (int)kd;
    const double t = (y - kd * kLn2NHi) - kd * kLn2NLo;
    // exp(t) - 1 = t + t²·(1/2 + t/6 + t²·(1/24 + t/120))
    const double t2 = t * t;
    const double p = t + t2 * ((kE2 + t * kE3) + t2 * (kE4 + t * kE5));
    const double s = kExp2Tab[ki & 127];
    const double scale = std::bit_cast<double>((uint64_t)(int64_t)((ki >> 7) + 1023) << 52);   // 2^(ki div 128)
    return (s + s * p) * scale;
}

// x^b.
inline double fast_pow(double x, double b) {
    if (!(x >= 2.2250738585072014e-308 && x <= 1.7976931348623157e308)) return std::pow(x, b);
    const double y = b * fast_log(x);
    if (!(std::abs(y) < 700.0)) return std::pow(x, b);
    return fast_exp(y);
}

// out[i] = fast_pow(x[i], b), i en [0, n). Mismos bits que fast_pow; con AVX2 va de 4 en 4.
// x y out pueden ser el mismo array.
void fast_pow_n(const double* x, double b, double* out, int n);

} // namespace mc::cpu
