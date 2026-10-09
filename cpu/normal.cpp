#include "normal.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <numbers>
#include <vector>

#ifdef MC_X86_64
#include <immintrin.h>
#endif

namespace mc::cpu {

namespace {

// Polinomio de grado 7 por Horner con coeficientes en orden ascendente.
inline double poly7(const double (&c)[8], double x) {
    double r = c[7];
    for (int i = 6; i >= 0; --i) r = r * x + c[i];
    return r;
}

// Coeficientes de AS 241 (PPND16)
constexpr double A[8] = {3.3871328727963666080,    1.3314166789178437745e+2,
                         1.9715909503065514427e+3, 1.3731693765509461125e+4,
                         4.5921953931549871457e+4, 6.7265770927008700853e+4,
                         3.3430575583588128105e+4, 2.5090809287301226727e+3};
constexpr double B[8] = {1.0,                      4.2313330701600911252e+1,
                         6.8718700749205790830e+2, 5.3941960214247511077e+3,
                         2.1213794301586595867e+4, 3.9307895800092710610e+4,
                         2.8729085735721942674e+4, 5.2264952788528545610e+3};
constexpr double C[8] = {1.42343711074968357734,     4.63033784615654529590,
                         5.76949722146069140550,     3.64784832476320460504,
                         1.27045825245236838258,     2.41780725177450611770e-1,
                         2.27238449892691845833e-2,  7.74545014278341407640e-4};
constexpr double D[8] = {1.0,                        2.05319162663775882187,
                         1.67638483018380384940,     6.89767334985100004550e-1,
                         1.48103976427480074590e-1,  1.51986665636164571966e-2,
                         5.47593808499534494600e-4,  1.05075007164441684324e-9};
constexpr double E[8] = {6.65790464350110377720,     5.46378491116411436990,
                         1.78482653991729133580,     2.96560571828504891230e-1,
                         2.65321895265761230930e-2,  1.24266094738807843860e-3,
                         2.71155556874348757815e-5,  2.01033439929228813265e-7};
constexpr double F[8] = {1.0,                        5.99832206555887937690e-1,
                         1.36929880922735805310e-1,  1.48753612908506148525e-2,
                         7.86869131145613259100e-4,  1.84631831751005468180e-5,
                         1.42151175831644588870e-7,  2.04426310338993978564e-15};

} // namespace

double norm_inv_cdf(double p) {
    if (!(p > 0.0)) return p == 0.0 ? -std::numeric_limits<double>::infinity()
                                    :  std::numeric_limits<double>::quiet_NaN();
    if (!(p < 1.0)) return p == 1.0 ?  std::numeric_limits<double>::infinity()
                                    :  std::numeric_limits<double>::quiet_NaN();

    const double q = p - 0.5;
    if (std::abs(q) <= 0.425) {
        const double r = 0.180625 - q * q;
        return q * poly7(A, r) / poly7(B, r);
    }
    // Colas: se trabaja con la probabilidad más pequeña (simetría) para no perder precisión.
    double r = (q < 0.0) ? p : 1.0 - p;
    r = std::sqrt(-std::log(r));
    double val;
    if (r <= 5.0) {
        r -= 1.6;
        val = poly7(C, r) / poly7(D, r);
    } else {
        r -= 5.0;
        val = poly7(E, r) / poly7(F, r);
    }
    return (q < 0.0) ? -val : val;
}

// ---- Ziggurat secuencial (Marsaglia & Tsang 2000, 256 capas) ---------------------------------------------------
//
// Variante de un solo generador que usa fill_normals. El motor usa la versión por bloque de más abajo.
// El 98.8 % de las veces basta una multiplicación y una comparación; solo se evalúa exp en la cuña
// (~1.2 %) y log en la cola (~0.03 %). Constantes de la tabla de 256 capas:
//   R = 3.6541528853610088 (abscisa de la última capa), V = 0.00492867323399 (área de cada capa).

namespace {

struct ZigTables {
    static constexpr double R = 3.6541528853610088;
    static constexpr double V = 0.00492867323399;
    double x[257];
    double f[257];

    ZigTables() {
        auto pdf = [](double v) { return std::exp(-0.5 * v * v); };
        x[0] = V / pdf(R);
        x[1] = R;
        for (int i = 2; i < 256; i++) x[i] = std::sqrt(-2.0 * std::log(V / x[i - 1] + pdf(x[i - 1])));
        x[256] = 0.0;
        for (int i = 0; i <= 256; i++) f[i] = pdf(x[i]);
    }
};

const ZigTables& zig_tables() {
    static const ZigTables t;
    return t;
}

// Camino lento del Ziggurat (cuña y cola, ~1.2 % de las normales), fuera de línea para que el
// bucle rápido quede compacto. Recibe la capa i, el uniforme u y x = u*x[i] del intento que NO pasó
// la prueba rápida, y sigue consumiendo g hasta aceptar.
#if defined(__GNUC__)
__attribute__((noinline))
#endif
double zig_slow(Xoshiro256pp& g, const ZigTables& T, int i, double u, double x) {
    for (;;) {
        if (i == 0) {                                                  // cola: algoritmo de Marsaglia
            for (;;) {
                const double x1 = -std::log(g.uniform_open()) / ZigTables::R;
                const double y = -std::log(g.uniform_open());
                if (y + y >= x1 * x1) return u < 0.0 ? -(ZigTables::R + x1) : ZigTables::R + x1;
            }
        }
        // cuña: aceptación por comparación con la densidad
        if (T.f[i + 1] + (T.f[i] - T.f[i + 1]) * g.uniform() < std::exp(-0.5 * x * x)) return x;
        // rechazo: intento nuevo completo
        const uint64_t bits = g.next();
        i = (int)(bits & 0xFF);
        u = (double)((int64_t)bits >> 11) * (1.0 / 4503599627370496.0);
        x = u * T.x[i];
        if (std::abs(x) < T.x[i + 1]) return x;
    }
}

inline double zig_normal(Xoshiro256pp& g, const ZigTables& T) {
    const uint64_t bits = g.next();
    const int i = (int)(bits & 0xFF);
    // 53 bits con signo -> u en [-1, 1)
    const double u = (double)((int64_t)bits >> 11) * (1.0 / 4503599627370496.0);
    const double x = u * T.x[i];
    if (std::abs(x) < T.x[i + 1]) return x;                           // camino rápido
    return zig_slow(g, T, i, u, x);
}

// ---- Ziggurat por bloque -------------------------------------------------------------------------------------
//
// Es el generador que usa el motor (RngNoise). Definición, por carril (= camino) y dimensión d:
//
//   bits = palabra nº d del xoshiro256++ del camino           (siempre UNA palabra por normal)
//   i = bits & (N-1)                                           capa (N capas)
//   u = double(1.m) - 1.5, con m = bits 12..63                 uniforme en [-0.5, 0.5), 52 bits
//   x = u · (2·x[i])
//   si |x| < x[i+1]:  z = x                                    (>= 98.5 % de las veces)
//   si no:            z = zig_slow_aux(clave del camino, d, bits)
//
// Diferencias con el Ziggurat secuencial clásico (zig_normal, arriba), pensadas para vectorizar:
//   * La cuña y la cola no siguen consumiendo el generador principal: sus sorteos extra salen de un
//     SplitMix64 auxiliar sembrado con (clave del camino, d). Así el generador principal avanza
//     exactamente una palabra por normal y el bucle vectorial nunca se detiene: las excepciones se
//     apuntan en una lista (sin saltos) y se corrigen después con código escalar.
//   * El uniforme se forma poniendo 52 bits como mantisa de un double en [1, 2) y restando 1.5: son
//     tres operaciones vectoriales exactas, sin conversión entero -> double (que AVX2 no tiene).
//
// Dos implementaciones que dan los MISMOS bits (lo comprueba test_rng_normal): la escalar (referencia,
// única disponible fuera de x86-64) y la AVX2, de 4 carriles por vector, elegida en tiempo de ejecución.
// Una variante SSE2 de 2 carriles se midió y se descartó: no gana a la escalar (docs/perf).

#ifndef MC_ZIG_LAYERS
#define MC_ZIG_LAYERS 512
#endif

struct ZigBlock {
    static constexpr int N = MC_ZIG_LAYERS;
    // R = abscisa de la última capa, V = área de cada capa (densidad sin normalizar exp(-x²/2)).
    // Calculadas con 40 dígitos resolviendo la ecuación de cierre de la construcción.
#if MC_ZIG_LAYERS == 256
    static constexpr double R = 3.6541528853610088, V = 4.9286732339746553e-3;
#elif MC_ZIG_LAYERS == 512
    static constexpr double R = 3.8520461503683912, V = 2.4567663515413557e-3;
#elif MC_ZIG_LAYERS == 1024
    static constexpr double R = 4.0388498461095045, V = 1.2263246463530881e-3;
#else
#error "MC_ZIG_LAYERS debe ser 256, 512 o 1024"
#endif
    alignas(64) double pair[N][2];   // {2·x[i], x[i+1]}: una sola búsqueda por normal
    double f[N + 1];                 // exp(-x[i]²/2)

    ZigBlock() {
        auto pdf = [](double v) { return std::exp(-0.5 * v * v); };
        std::vector<double> x((size_t)N + 1);
        x[0] = V / pdf(R);
        x[1] = R;
        for (int i = 2; i < N; i++) x[(size_t)i] = std::sqrt(-2.0 * std::log(V / x[(size_t)i - 1] + pdf(x[(size_t)i - 1])));
        x[(size_t)N] = 0.0;
        for (int i = 0; i <= N; i++) f[i] = pdf(x[(size_t)i]);
        for (int i = 0; i < N; i++) {
            pair[i][0] = 2.0 * x[(size_t)i];
            pair[i][1] = x[(size_t)i + 1];
        }
    }
};

const ZigBlock& zig_block() {
    static const ZigBlock t;
    return t;
}

inline double zig_u(uint64_t bits) {
    return std::bit_cast<double>((bits >> 12) | 0x3FF0000000000000ULL) - 1.5;
}

// Cuña y cola con el generador auxiliar. `d` es el índice absoluto de la normal dentro del camino.
#if defined(__GNUC__)
__attribute__((noinline))
#endif
double zig_slow_aux(uint64_t key, uint64_t d, uint64_t bits, const ZigBlock& T) {
    uint64_t sm = key ^ (0x9FB21C651E98DF25ULL * (d + 1));
    for (;;) {
        const int i = (int)(bits & (uint64_t)(ZigBlock::N - 1));
        const double u = zig_u(bits);
        const double x = u * T.pair[i][0];
        if (std::abs(x) < T.pair[i][1]) return x;                      // solo en reintentos
        if (i == 0) {                                                  // cola: algoritmo de Marsaglia
            for (;;) {
                const double x1 = -std::log(Xoshiro256pp::to_uniform_open(splitmix64_next(sm))) / ZigBlock::R;
                const double y = -std::log(Xoshiro256pp::to_uniform_open(splitmix64_next(sm)));
                if (y + y >= x1 * x1) return u < 0.0 ? -(ZigBlock::R + x1) : ZigBlock::R + x1;
            }
        }
        // cuña: aceptación por comparación con la densidad
        const double uy = Xoshiro256pp::to_uniform(splitmix64_next(sm));
        if (T.f[i + 1] + (T.f[i] - T.f[i + 1]) * uy < std::exp(-0.5 * x * x)) return x;
        bits = splitmix64_next(sm);                                    // rechazo: intento nuevo completo
    }
}

void zig_fill_scalar(LaneRng& g, int l0, int l1, double* Z, int ld, int D, double scale, const ZigBlock& T) {
    for (int l = l0; l < l1; l++) {
        Xoshiro256pp gl = g.get(l);
        const uint64_t key = g.key[l];
        double* out = Z + l;
        for (int d = 0; d < D; d++, out += ld) {
            const uint64_t bits = gl.next();
            const int i = (int)(bits & (uint64_t)(ZigBlock::N - 1));
            const double x = zig_u(bits) * T.pair[i][0];
            if (std::abs(x) < T.pair[i][1]) *out = x * scale;
            else *out = zig_slow_aux(key, g.pos + (uint64_t)d, bits, T) * scale;
        }
        g.set(l, gl);
    }
}

#ifdef MC_X86_64

constexpr int kZigChunk = 64;   // normales por carril entre dos pasadas de corrección de excepciones

// Carriles l0 .. l0+3.
MC_TARGET_AVX2 void zig_fill_avx2(LaneRng& g, int l0, double* Z, int ld, int D, double scale, const ZigBlock& T) {
    __m256i s0 = _mm256_loadu_si256((const __m256i*)&g.s0[l0]);
    __m256i s1 = _mm256_loadu_si256((const __m256i*)&g.s1[l0]);
    __m256i s2 = _mm256_loadu_si256((const __m256i*)&g.s2[l0]);
    __m256i s3 = _mm256_loadu_si256((const __m256i*)&g.s3[l0]);
    const __m256i one_bits = _mm256_set1_epi64x(0x3FF0000000000000LL);
    const __m256d c15 = _mm256_set1_pd(1.5);
    const __m256d absmask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
    const __m256d vscale = _mm256_set1_pd(scale);
    constexpr uint64_t imask = (uint64_t)(ZigBlock::N - 1);
    alignas(32) uint64_t ebits[kZigChunk][4];   // palabras de las vueltas con algún carril rechazado
    alignas(32) uint64_t rb[4];                 // palabra de la vuelta actual (para sacar los índices)
    int ewho[kZigChunk];
    for (int d0 = 0; d0 < D; d0 += kZigChunk) {
        const int nd = std::min(kZigChunk, D - d0);
        double* row = Z + (size_t)d0 * ld + l0;
        int cnt = 0;
        for (int j = 0; j < nd; j++, row += ld) {
            // xoshiro256++
            const __m256i sum = _mm256_add_epi64(s0, s3);
            const __m256i r = _mm256_add_epi64(_mm256_or_si256(_mm256_slli_epi64(sum, 23), _mm256_srli_epi64(sum, 41)), s0);
            const __m256i t = _mm256_slli_epi64(s1, 17);
            s2 = _mm256_xor_si256(s2, s0);
            s3 = _mm256_xor_si256(s3, s1);
            s1 = _mm256_xor_si256(s1, s2);
            s0 = _mm256_xor_si256(s0, s3);
            s2 = _mm256_xor_si256(s2, t);
            s3 = _mm256_or_si256(_mm256_slli_epi64(s3, 45), _mm256_srli_epi64(s3, 19));
            // u en [-0.5, 0.5)
            const __m256d u = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(_mm256_srli_epi64(r, 12), one_bits)), c15);
            // tabla: {2·x[i], x[i+1]} de cada carril. Los índices salen de una copia en memoria (más
            // barato que extraerlos del registro). La lista de excepciones recibe otra copia: se
            // escribe siempre y el contador solo avanza si algún carril falla, así no hay salto.
            // (Son dos copias a propósito: leer los índices de ebits[cnt] haría depender la búsqueda
            // del contador y encadenaría todas las vueltas.)
            _mm256_store_si256((__m256i*)rb, r);
            _mm256_store_si256((__m256i*)ebits[cnt], r);
            const __m256d a = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_load_pd(T.pair[rb[0] & imask])),
                                                   _mm_load_pd(T.pair[rb[2] & imask]), 1);
            const __m256d b = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_load_pd(T.pair[rb[1] & imask])),
                                                   _mm_load_pd(T.pair[rb[3] & imask]), 1);
            const __m256d x = _mm256_mul_pd(u, _mm256_unpacklo_pd(a, b));
            const int ok = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_and_pd(x, absmask), _mm256_unpackhi_pd(a, b), _CMP_LT_OQ));
            _mm256_storeu_pd(row, _mm256_mul_pd(x, vscale));
            ewho[cnt] = (j << 4) | (ok ^ 15);
            cnt += (ok != 15);
        }
        for (int e = 0; e < cnt; e++) {
            const int j = ewho[e] >> 4;
            for (int k = 0; k < 4; k++)
                if ((ewho[e] >> k) & 1)
                    Z[(size_t)(d0 + j) * ld + l0 + k] =
                        zig_slow_aux(g.key[l0 + k], g.pos + (uint64_t)(d0 + j), ebits[e][k], T) * scale;
        }
    }
    _mm256_storeu_si256((__m256i*)&g.s0[l0], s0);
    _mm256_storeu_si256((__m256i*)&g.s1[l0], s1);
    _mm256_storeu_si256((__m256i*)&g.s2[l0], s2);
    _mm256_storeu_si256((__m256i*)&g.s3[l0], s3);
    _mm256_zeroupper();
}

#endif // MC_X86_64

} // namespace

void fill_normals(NormalMethod method, Xoshiro256pp& g, double* z, int n) {
    switch (method) {
    case NormalMethod::BoxMuller: {
        const double two_pi = 2.0 * std::numbers::pi;
        int i = 0;
        for (; i + 1 < n; i += 2) {
            const double u1 = g.uniform_open();
            const double u2 = g.uniform();
            const double r = std::sqrt(-2.0 * std::log(u1));
            const double th = two_pi * u2;
            z[i]     = r * std::cos(th);
            z[i + 1] = r * std::sin(th);
        }
        if (i < n) {   // n impar: se descarta la segunda normal del último par
            const double u1 = g.uniform_open();
            const double u2 = g.uniform();
            z[i] = std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
        }
        break;
    }
    case NormalMethod::InverseCdf:
        for (int i = 0; i < n; i++) z[i] = norm_inv_cdf(g.uniform_open());
        break;
    case NormalMethod::Ziggurat: {
        const ZigTables& T = zig_tables();
        for (int i = 0; i < n; i++) z[i] = zig_normal(g, T);
        break;
    }
    }
}

void fill_normals_lanes(NormalMethod method, LaneRng& g, int n, double* Z, int ld, int D, double scale) {
    constexpr int W = kLanes;
    uint64_t bits[W];
    switch (method) {
    case NormalMethod::Ziggurat: {
        const ZigBlock& T = zig_block();
        int l = 0;
#ifdef MC_X86_64
        if (simd_level() == SimdLevel::Avx2)
            for (; l + 4 <= n; l += 4) zig_fill_avx2(g, l, Z, ld, D, scale, T);
#endif
        if (l < n) zig_fill_scalar(g, l, n, Z, ld, D, scale, T);
        break;
    }
    case NormalMethod::BoxMuller: {
        const double two_pi = 2.0 * std::numbers::pi;
        uint64_t bits2[W];
        int d = 0;
        for (; d + 1 < D; d += 2) {
            g.next_all(bits);
            g.next_all(bits2);
            double* row0 = Z + (size_t)d * ld;
            double* row1 = row0 + ld;
            for (int l = 0; l < n; l++) {
                const double r = std::sqrt(-2.0 * std::log(Xoshiro256pp::to_uniform_open(bits[l])));
                const double th = two_pi * Xoshiro256pp::to_uniform(bits2[l]);
                row0[l] = (r * std::cos(th)) * scale;
                row1[l] = (r * std::sin(th)) * scale;
            }
        }
        if (d < D) {   // D impar: se descarta la segunda normal del último par
            g.next_all(bits);
            g.next_all(bits2);
            double* row = Z + (size_t)d * ld;
            for (int l = 0; l < n; l++)
                row[l] = (std::sqrt(-2.0 * std::log(Xoshiro256pp::to_uniform_open(bits[l]))) *
                          std::cos(two_pi * Xoshiro256pp::to_uniform(bits2[l]))) * scale;
        }
        break;
    }
    case NormalMethod::InverseCdf:
        for (int d = 0; d < D; d++) {
            g.next_all(bits);
            double* row = Z + (size_t)d * ld;
            for (int l = 0; l < n; l++) row[l] = norm_inv_cdf(Xoshiro256pp::to_uniform_open(bits[l])) * scale;
        }
        break;
    }
    g.pos += (uint64_t)D;
}

} // namespace mc::cpu
