#pragma once
// Secuencias de Sobol (Joe-Kuo, hasta 21201 dimensiones) con el scrambling de
// Hong-Hickernell de la GPU (methods_cuda.cu: splitmix32 / hh_scramble).
//
// Orden de los puntos: código Gray (0, 1/2, 3/4, 1/4, ...), igual que scipy.stats.qmc y que
// la versión de dispositivo de cuRAND (curand_init(dir, offset) + curand()).
//
// Truco de rendimiento (verificado bit a bit contra la versión ingenua en los tests):
// hh_scramble(raw) = L·raw XOR shift, donde L es una matriz triangular inferior sobre F2
// que depende solo de (réplica, dimensión). Como el punto de Sobol es un XOR de vectores
// directores, L·x_n = XOR_j g_j (L·V_j) con g = gray(n). Basta precalcular V'_j = L·V_j
// una vez por (réplica, dimensión): cada punto nuevo cuesta UN XOR por dimensión
// (x ^= V'[ctz(n)]) en vez de 32 hashes splitmix32 + 32 popcounts.

#include <cstdint>
#include <vector>

namespace mc::cpu {

inline constexpr int kSobolMaxDim = 21201;   // dimensiones 0-based en [0, 21201)

// splitmix32: idéntico al de methods_cuda.cu (es __host__ __device__ allí).
inline constexpr uint32_t splitmix32(uint32_t x) {
    x += 0x9E3779B9u;
    x = (x ^ (x >> 16)) * 0x21F0AAADu;
    x = (x ^ (x >> 15)) * 0x735A2D97u;
    x = x ^ (x >> 15);
    return x;
}

// ---- Sobol sin scrambling ------------------------------------------------------------------

// 32 vectores directores (alineados a la izquierda) de la dimensión dim (0-based).
// Tabla expandida a partir de los parámetros de Joe-Kuo la primera vez (thread-safe).
const uint32_t* sobol_direction_vectors(int dim);

// Punto n de la secuencia (palabra de 32 bits): XOR de V_j para los bits j de gray(n).
uint32_t sobol_raw(int dim, uint64_t n);

// ---- Hong-Hickernell ------------------------------------------------------------------------

// Semilla base del scrambling de (réplica, dimensión): igual que la GPU.
inline constexpr uint32_t hh_base(uint32_t replica_salt, int dim) {
    return splitmix32(replica_salt) ^ (uint32_t)dim;
}

// Versión ingenua, IDÉNTICA a hh_scramble de la GPU (32 splitmix32 + 32 popcounts por palabra).
uint32_t hh_scramble_naive(uint32_t raw, uint32_t base);

// Punto Sobol con scrambling, ingenuo (referencia para los tests y los benchmarks).
inline uint32_t scrambled_sobol_naive(uint32_t replica_salt, int dim, uint64_t n) {
    return hh_scramble_naive(sobol_raw(dim, n), hh_base(replica_salt, dim));
}

// Uniforme estrictamente en (0,1) a partir de la palabra scrambleada (como la GPU).
inline double sobol_word_to_u(uint32_t w) { return ((double)w + 0.5) * (1.0 / 4294967296.0); }

// Generador rápido por réplica: precalcula V' y genera por Gray-code incremental.
class ScrambledSobol {
public:
    // Dimensiones 0..n_dims-1 (n_dims <= kSobolMaxDim).
    ScrambledSobol(uint32_t replica_salt, int n_dims);

    int dims() const { return D_; }

    // Estado (una palabra por dimensión) del punto n. Lanza SobolLimitReached si n > 2^32-1.
    void state_at(uint64_t n, uint32_t* x) const;
    // Pasa del punto n-1 al n (n >= 1): x[d] ^= V'[ctz(n)][d].
    void advance(uint64_t n, uint32_t* x) const;
    // out[d] = Phi^{-1}(u(x[d])) para las dimensiones 0..D-1.
    void normals_from_state(const uint32_t* x, double* out) const;
    // Normales de los puntos n0..n0+count-1: out[p*D + d].
    void fill_normals(uint64_t n0, int count, double* out) const;

private:
    int D_;
    std::vector<uint32_t> vprime_;   // [32][D]: V'_j de cada dimensión
    std::vector<uint32_t> shift_;    // [D]
};

} // namespace mc::cpu
