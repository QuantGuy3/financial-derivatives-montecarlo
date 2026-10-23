#pragma once
// ---------------------------------------------------------------------------- //
// Utilidades __host__ __device__ compartidas por el motor GPU (nvcc) y por el   //
// motor CPU (g++): macros HD, funciones matemáticas con sobrecarga float/double //
// y generadores de números aleatorios contrarreloj (Philox4x32-10) y           //
// cuasialeatorios (Sobol + scrambling de Hong-Hickernell).                     //
//                                                                              //
// La idea es que la MISMA lógica (modelos, producto estructurado) se compile    //
// para CPU y para GPU: así se puede validar en CPU contra fórmulas cerradas lo  //
// que después ejecuta la GPU, y ambos backends generan exactamente las mismas   //
// trayectorias para una misma semilla (salvo redondeo float/double).           //
// ---------------------------------------------------------------------------- //

#include <math.h>
#include <stdint.h>

#if defined(__CUDACC__)
  #define HD        __host__ __device__
  #define HD_INLINE __host__ __device__ __forceinline__
#else
  #define HD
  #define HD_INLINE inline
#endif

// Valor "infinito" finito: con --use_fast_math el compilador puede suponer que no
// hay infinitos, así que los topes "sin límite" se representan con 1e30.
#define HD_INFINITO 1.0e30

namespace hd {

// ------------------------------------------------- //
// Funciones matemáticas con sobrecarga float/double //
// ------------------------------------------------- //

HD_INLINE float  exp_(float x)   { return expf(x); }
HD_INLINE double exp_(double x)  { return exp(x); }
HD_INLINE float  log_(float x)   { return logf(x); }
HD_INLINE double log_(double x)  { return log(x); }
HD_INLINE float  sqrt_(float x)  { return sqrtf(x); }
HD_INLINE double sqrt_(double x) { return sqrt(x); }
HD_INLINE float  abs_(float x)   { return fabsf(x); }
HD_INLINE double abs_(double x)  { return fabs(x); }
HD_INLINE float  erfc_(float x)  { return erfcf(x); }
HD_INLINE double erfc_(double x) { return erfc(x); }
HD_INLINE float  min_(float a, float b)   { return fminf(a, b); }
HD_INLINE double min_(double a, double b) { return fmin(a, b); }
HD_INLINE float  max_(float a, float b)   { return fmaxf(a, b); }
HD_INLINE double max_(double a, double b) { return fmax(a, b); }

template<typename Real>
HD_INLINE Real clamp_(Real x, Real lo, Real hi) { return min_(max_(x, lo), hi); }

// Función de distribución de la normal estándar
template<typename Real>
HD_INLINE Real norm_cdf(Real x) {
    return Real(0.5) * erfc_(-x * Real(0.70710678118654752440));
}

// Inversa de la normal estándar en CPU: algoritmo AS241 (Wichura 1988, PPND16),
// error relativo ~1e-16 en todo (0,1).
HD_INLINE double inv_norm_as241(double p) {
    double q = p - 0.5;
    if (fabs(q) <= 0.425) {
        double r = 0.180625 - q * q;
        return q * (((((((r * 2509.0809287301226727 +
                       33430.575583588128105) * r + 67265.770927008700853) * r +
                     45921.953931549871457) * r + 13731.693765509461125) * r +
                   1971.5909503065514427) * r + 133.14166789178437745) * r +
                 3.387132872796366608)
            / (((((((r * 5226.495278852545925 +
                     28729.085735721942674) * r + 39307.89580009271061) * r +
                   21213.794301586595867) * r + 5394.1960214247511077) * r +
                 687.1870074920579083) * r + 42.313330701600911252) * r + 1.0);
    }
    double r = (q < 0.0) ? p : 1.0 - p;
    r = sqrt(-log(r));
    double val;
    if (r <= 5.0) {
        r -= 1.6;
        val = (((((((r * 7.7454501427834140764e-4 +
                   0.0227238449892691845833) * r + 0.24178072517745061177) *
                 r + 1.27045825245236838258) * r +
                3.64784832476320460504) * r + 5.7694972214606914055) *
              r + 4.6303378461565452959) * r +
             1.42343711074968357734)
            / (((((((r *
                     1.05075007164441684324e-9 + 5.475938084995344946e-4) *
                    r + 0.0151986665636164571966) * r +
                   0.14810397642748007459) * r + 0.68976733498510000455) *
                 r + 1.6763848301838038494) * r +
                2.05319162663775882187) * r + 1.0);
    } else {
        r -= 5.0;
        val = (((((((r * 2.01033439929228813265e-7 +
                   2.71155556874348757815e-5) * r +
                  0.0012426609473880784386) * r + 0.026532189526576123093) *
                r + 0.29656057182850489123) * r +
               1.7848265399172913358) * r + 5.4637849111641143699) *
             r + 6.6579046435011037772)
            / (((((((r *
                     2.04426310338993978564e-15 + 1.4215117583164458887e-7) *
                    r + 1.8463183175100546818e-5) * r +
                   7.868691311456132591e-4) * r + 0.0148753612908506148525)
                 * r + 0.13692988092273580531) * r +
                0.59983220655588793769) * r + 1.0);
    }
    return (q < 0.0) ? -val : val;
}

// Inversa de la normal estándar: en GPU usa la intrínseca de CUDA, en CPU AS241.
HD_INLINE double inv_norm_cdf(double p) {
#if defined(__CUDA_ARCH__)
    return normcdfinv(p);
#else
    return inv_norm_as241(p);
#endif
}
HD_INLINE float inv_norm_cdf(float p) {
#if defined(__CUDA_ARCH__)
    return normcdfinvf(p);
#else
    return (float)inv_norm_as241((double)p);
#endif
}

// ----------------------------------------------- //
// Operaciones de enteros de 32 bits (CPU y GPU)   //
// ----------------------------------------------- //

HD_INLINE uint32_t mulhi32(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__)
    return __umulhi(a, b);
#else
    return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
#endif
}

HD_INLINE uint32_t popc32(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return (uint32_t)__popc(x);
#else
    return (uint32_t)__builtin_popcount(x);
#endif
}

// Mezclador "splitmix32" (mismo que usa methods_cuda.cu para el scrambling QMC)
HD_INLINE uint32_t splitmix32(uint32_t x) {
    x += 0x9E3779B9u;
    x = (x ^ (x >> 16)) * 0x21F0AAADu;
    x = (x ^ (x >> 15)) * 0x735A2D97u;
    x = x ^ (x >> 15);
    return x;
}

// ------------------------------------------------------------------------- //
// Philox4x32-10 (Salmon et al. 2011, "Parallel random numbers: as easy as   //
// 1, 2, 3"). Generador contrarreloj: la salida depende solo de (contador,   //
// clave), así que cada trayectoria y cada dimensión tienen su propio número //
// sin estado compartido. Es el mismo algoritmo que CURAND_RNG_PSEUDO_PHILOX. //
// ------------------------------------------------------------------------- //

struct Philox4 { uint32_t v[4]; };

HD_INLINE Philox4 philox4x32_10(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                                uint32_t k0, uint32_t k1) {
    const uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    const uint32_t W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
#if defined(__CUDA_ARCH__)
    #pragma unroll
#endif
    for (int r = 0; r < 10; ++r) {
        uint32_t hi0 = mulhi32(M0, c0), lo0 = M0 * c0;
        uint32_t hi1 = mulhi32(M1, c2), lo1 = M1 * c2;
        uint32_t n0 = hi1 ^ c1 ^ k0;
        uint32_t n2 = hi0 ^ c3 ^ k1;
        c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
        k0 += W0; k1 += W1;
    }
    Philox4 out;
    out.v[0] = c0; out.v[1] = c1; out.v[2] = c2; out.v[3] = c3;
    return out;
}

// Uniforme estrictamente en (0,1) a partir de 32 bits aleatorios. Se usan los
// 24 bits altos para que el valor sea EXACTAMENTE representable en float y en
// double (CPU y GPU ven la misma uniforme), y +0.5 evita 0 y 1 exactos.
template<typename Real>
HD_INLINE Real uniforme24(uint32_t bits) {
    return ((Real)(bits >> 8) + Real(0.5)) * Real(5.9604644775390625e-8); // 2^-24
}

// ---------------------------------------------------------------------------- //
// Sobol + scrambling de Hong-Hickernell (matriz triangular inferior aleatoria  //
// sobre F_2 más desplazamiento digital), idéntico al de methods_cuda.cu: la    //
// matriz de cada (réplica, dimensión) se deriva al vuelo con splitmix32.       //
// ---------------------------------------------------------------------------- //

// Punto de Sobol sin scrambling de índice `idx` en la dimensión cuyos 32
// vectores directores son `dv` (mismo convenio Gray que curand_init/skipahead).
HD_INLINE uint32_t sobol32(const uint32_t* dv, uint32_t idx) {
    uint32_t g = idx ^ (idx >> 1);
    uint32_t x = 0;
    int b = 0;
    while (g) {
        if (g & 1u) x ^= dv[b];
        g >>= 1;
        ++b;
    }
    return x;
}

HD_INLINE uint32_t hh_scramble(uint32_t raw, uint32_t base) {
    uint32_t scrambled = 0;
#if defined(__CUDA_ARCH__)
    #pragma unroll
#endif
    for (int k = 0; k < 32; k++) {
        uint32_t row_seed = splitmix32(base ^ (uint32_t)k);
        uint32_t diag_bit = 1u << (31 - k);
        uint32_t top_mask = (k == 0) ? 0u : (0xFFFFFFFFu << (32 - k));
        uint32_t row = diag_bit | (row_seed & top_mask);
        uint32_t bit = popc32(row & raw) & 1u;
        scrambled |= (bit << (31 - k));
    }
    uint32_t shift = splitmix32(base ^ 0xA5A5A5A5u);
    return scrambled ^ shift;
}

} // namespace hd
