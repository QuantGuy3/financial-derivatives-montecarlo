#include "sobol.hpp"

#include "normal.hpp"
#include "../mc_types.hpp"

#include <bit>
#include <climits>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace mc::cpu {

namespace detail {
// Definidos en joe_kuo_data.cpp (generado por tools/gen_joe_kuo.py)
extern const int      kJoeKuoDims;
extern const uint8_t  kJoeKuoS[];
extern const uint32_t kJoeKuoA[];
extern const uint32_t kJoeKuoMOff[];
extern const uint32_t kJoeKuoM[];
} // namespace detail

static_assert(kSobolMaxDim == 21201, "kSobolMaxDim debe coincidir con la tabla de Joe-Kuo");

// ---- Tabla de vectores directores ----------------------------------------------------------

namespace {

std::vector<uint32_t> g_dirvec;          // [kSobolMaxDim][32]
std::once_flag g_dirvec_once;

void build_direction_table() {
    g_dirvec.assign((size_t)kSobolMaxDim * 32, 0u);

    // Dimensión 0 (van der Corput): V_i = 2^(31-i)
    for (int i = 0; i < 32; i++) g_dirvec[i] = 1u << (31 - i);

    for (int d = 1; d < kSobolMaxDim; d++) {
        const unsigned s = detail::kJoeKuoS[d - 1];
        const uint32_t a = detail::kJoeKuoA[d - 1];
        const uint32_t* m = detail::kJoeKuoM + detail::kJoeKuoMOff[d - 1];
        uint32_t* V = g_dirvec.data() + (size_t)d * 32;

        // Recurrencia de Joe-Kuo con índices 1-based en la literatura; aquí V[i-1].
        for (unsigned i = 1; i <= s; i++) V[i - 1] = m[i - 1] << (32 - i);
        for (unsigned i = s + 1; i <= 32; i++) {
            uint32_t v = V[i - s - 1] ^ (V[i - s - 1] >> s);
            for (unsigned k = 1; k < s; k++)
                if ((a >> (s - 1 - k)) & 1u) v ^= V[i - k - 1];
            V[i - 1] = v;
        }
    }
}

} // namespace

const uint32_t* sobol_direction_vectors(int dim) {
    if (dim < 0 || dim >= kSobolMaxDim)
        throw std::out_of_range("sobol_direction_vectors: dimensión fuera de rango");
    std::call_once(g_dirvec_once, build_direction_table);
    return g_dirvec.data() + (size_t)dim * 32;
}

uint32_t sobol_raw(int dim, uint64_t n) {
    const uint32_t* V = sobol_direction_vectors(dim);
    uint64_t g = n ^ (n >> 1);
    uint32_t x = 0;
    while (g) {
        const int j = std::countr_zero(g);
        x ^= V[j];
        g &= g - 1;
    }
    return x;
}

// ---- Hong-Hickernell -----------------------------------------------------------------------------

uint32_t hh_scramble_naive(uint32_t raw, uint32_t base) {
    uint32_t scrambled = 0;
    for (int k = 0; k < 32; k++) {
        uint32_t row_seed = splitmix32(base ^ (uint32_t)k);
        uint32_t diag_bit = 1u << (31 - k);
        uint32_t top_mask = (k == 0) ? 0u : (0xFFFFFFFFu << (32 - k));
        uint32_t row = diag_bit | (row_seed & top_mask);
        uint32_t bit = (uint32_t)(std::popcount(row & raw) & 1);
        scrambled |= (bit << (31 - k));
    }
    uint32_t shift = splitmix32(base ^ 0xA5A5A5A5u);
    return scrambled ^ shift;
}

namespace {
// Filas de la matriz L de (réplica, dimensión): fila k produce el bit 31-k de la salida.
inline void hh_rows(uint32_t base, uint32_t (&rows)[32]) {
    for (int k = 0; k < 32; k++) {
        uint32_t row_seed = splitmix32(base ^ (uint32_t)k);
        uint32_t diag_bit = 1u << (31 - k);
        uint32_t top_mask = (k == 0) ? 0u : (0xFFFFFFFFu << (32 - k));
        rows[k] = diag_bit | (row_seed & top_mask);
    }
}
inline uint32_t hh_apply_rows(const uint32_t (&rows)[32], uint32_t raw) {
    uint32_t out = 0;
    for (int k = 0; k < 32; k++)
        out |= (uint32_t)(std::popcount(rows[k] & raw) & 1) << (31 - k);
    return out;
}
} // namespace

ScrambledSobol::ScrambledSobol(uint32_t replica_salt, int n_dims) : D_(n_dims) {
    if (n_dims < 1 || n_dims > kSobolMaxDim)
        throw std::out_of_range("ScrambledSobol: nº de dimensiones fuera de rango");
    vprime_.resize((size_t)32 * D_);
    shift_.resize(D_);
    for (int d = 0; d < D_; d++) {
        const uint32_t base = hh_base(replica_salt, d);
        uint32_t rows[32];
        hh_rows(base, rows);
        const uint32_t* V = sobol_direction_vectors(d);
        for (int j = 0; j < 32; j++) vprime_[(size_t)j * D_ + d] = hh_apply_rows(rows, V[j]);
        shift_[d] = splitmix32(base ^ 0xA5A5A5A5u);
    }
}

void ScrambledSobol::state_at(uint64_t n, uint32_t* x) const {
    if (n > (uint64_t)UINT_MAX) throw SobolLimitReached{};
    for (int d = 0; d < D_; d++) x[d] = shift_[d];
    uint64_t g = n ^ (n >> 1);
    while (g) {
        const int j = std::countr_zero(g);
        const uint32_t* row = vprime_.data() + (size_t)j * D_;
        for (int d = 0; d < D_; d++) x[d] ^= row[d];
        g &= g - 1;
    }
}

void ScrambledSobol::advance(uint64_t n, uint32_t* x) const {
    const int j = std::countr_zero(n);
    const uint32_t* row = vprime_.data() + (size_t)j * D_;
    for (int d = 0; d < D_; d++) x[d] ^= row[d];
}

void ScrambledSobol::normals_from_state(const uint32_t* x, double* out) const {
    for (int d = 0; d < D_; d++) out[d] = norm_inv_cdf(sobol_word_to_u(x[d]));
}

void ScrambledSobol::fill_normals(uint64_t n0, int count, double* out) const {
    if (count <= 0) return;
    if (n0 + (uint64_t)count - 1 > (uint64_t)UINT_MAX) throw SobolLimitReached{};
    std::vector<uint32_t> x(D_);
    state_at(n0, x.data());
    normals_from_state(x.data(), out);
    for (int p = 1; p < count; p++) {
        advance(n0 + (uint64_t)p, x.data());
        normals_from_state(x.data(), out + (size_t)p * D_);
    }
}

} // namespace mc::cpu
