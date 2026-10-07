// Sobol (Joe-Kuo) + scrambling Hong-Hickernell: valores de scipy, propiedad de red,
// equivalencia bit a bit del truco lineal con la versión ingenua de la GPU.
#include "doctest.h"
#include "cpu/normal.hpp"
#include "cpu/sobol.hpp"
#include "mc_types.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "golden/sobol_scipy.inc"

using namespace mc::cpu;

namespace {

uint64_t fnv1a_words(uint64_t h, uint32_t v) {
    for (int b = 0; b < 4; b++) {
        h ^= (v >> (8 * b)) & 0xFFu;
        h *= 0x100000001B3ULL;
    }
    return h;
}

// ¿Los primeros 2^m puntos de las dims (0,1) forman una red (0,m,2)? Cada caja elemental
// 2^-a x 2^-(m-a) debe contener exactamente un punto, para todo a en 0..m.
template <class Word0, class Word1>
bool is_net_0_m_2(int m, Word0&& w0, Word1&& w1) {
    const uint64_t n = 1ULL << m;
    for (int a = 0; a <= m; a++) {
        std::vector<int> count(n, 0);
        for (uint64_t i = 0; i < n; i++) {
            uint64_t cx = (a == 0) ? 0 : (uint64_t)(w0(i) >> (32 - a));   // a bits de la dim 0
            uint64_t cy = (m - a == 0) ? 0 : (uint64_t)(w1(i) >> (32 - (m - a)));
            uint64_t cell = (cx << (m - a)) | cy;
            if (++count[cell] != 1) return false;
        }
    }
    return true;
}

} // namespace

TEST_SUITE("fast") {

TEST_CASE("Sobol sin scrambling coincide con scipy (dimensiones muestreadas)") {
    for (int i = 0; i < kGoldenSobolNDimsSample; i++) {
        int dim = kGoldenSobolDims[i];
        for (int n = 0; n < kGoldenSobolN; n++)
            REQUIRE_MESSAGE(sobol_raw(dim, (uint64_t)n) == kGoldenSobolWords[i * kGoldenSobolN + n],
                            "dim " << dim << " n " << n);
    }
}

TEST_CASE("Sobol: hash de las 21201 dimensiones x 64 puntos coincide con scipy") {
    uint64_t h = 0xCBF29CE484222325ULL;
    for (int d = 0; d < kSobolMaxDim; d++)
        for (int n = 0; n < 64; n++) h = fnv1a_words(h, sobol_raw(d, (uint64_t)n));
    CHECK(h == kGoldenSobolFnv);
}

TEST_CASE("Sobol dim 0 es van der Corput en orden Gray") {
    CHECK(sobol_raw(0, 0) == 0u);
    CHECK(sobol_raw(0, 1) == 0x80000000u);   // 0.5
    CHECK(sobol_raw(0, 2) == 0xC0000000u);   // 0.75
    CHECK(sobol_raw(0, 3) == 0x40000000u);   // 0.25
}

TEST_CASE("Sobol sin scrambling: las dims 0 y 1 forman una red (0,m,2)") {
    for (int m = 1; m <= 10; m++)
        CHECK_MESSAGE(is_net_0_m_2(m, [](uint64_t i) { return sobol_raw(0, i); },
                                      [](uint64_t i) { return sobol_raw(1, i); }), "m=" << m);
}

TEST_CASE("Sobol con scrambling HH (ingenuo): sigue siendo red (0,m,2)") {
    for (uint32_t salt : {0u, 42u, 123456u}) {
        for (int m : {1, 4, 7, 10})
            CHECK_MESSAGE(is_net_0_m_2(m, [&](uint64_t i) { return scrambled_sobol_naive(salt, 0, i); },
                                          [&](uint64_t i) { return scrambled_sobol_naive(salt, 1, i); }),
                          "salt=" << salt << " m=" << m);
    }
}

TEST_CASE("Sobol con scrambling HH: cada dimension es una (0,m,1)-red (1 punto por celda)") {
    for (int dim : {0, 1, 2, 5, 17, 63, 200}) {
        const int m = 10; const uint32_t n = 1u << m;
        std::vector<int> cnt(n, 0);
        for (uint32_t i = 0; i < n; i++) cnt[scrambled_sobol_naive(99u, dim, i) >> (32 - m)]++;
        for (uint32_t c = 0; c < n; c++) REQUIRE_MESSAGE(cnt[c] == 1, "dim " << dim << " celda " << c);
    }
}

TEST_CASE("ScrambledSobol (Gray-code + V'=L*V) == versión ingenua de la GPU, bit a bit") {
    std::mt19937 rng(1234);
    for (uint32_t salt : {0u, 1u, 42u, 0xDEADBEEFu}) {
        const int D = 300;
        ScrambledSobol sob(salt, D);
        std::vector<uint32_t> x(D);

        // Secuencial desde 0
        sob.state_at(0, x.data());
        for (uint64_t n = 0; n < 2048; n++) {
            if (n > 0) sob.advance(n, x.data());
            if (n % 37 == 0 || n < 40 || n > 2040)
                for (int d : {0, 1, 2, 31, 32, 100, 299})
                    REQUIRE(x[d] == scrambled_sobol_naive(salt, d, n));
        }
        // Salto directo a posiciones arbitrarias (inicio de chunk)
        for (uint64_t n0 : {uint64_t(12345), uint64_t(1) << 20, (uint64_t(1) << 31) + 7, uint64_t(4294967295u)}) {
            sob.state_at(n0, x.data());
            for (int d : {0, 3, 150, 299}) REQUIRE(x[d] == scrambled_sobol_naive(salt, d, n0));
            if (n0 < 4294967295u) {
                sob.advance(n0 + 1, x.data());
                for (int d : {0, 3, 150, 299}) REQUIRE(x[d] == scrambled_sobol_naive(salt, d, n0 + 1));
            }
        }
    }
}

TEST_CASE("ScrambledSobol: dimensiones altas y límite de 2^32 puntos") {
    ScrambledSobol sob(5u, kSobolMaxDim);   // todas las dimensiones
    std::vector<uint32_t> x(kSobolMaxDim);
    sob.state_at(1000, x.data());
    for (int d : {0, 5000, 19999, 21200})
        CHECK(x[d] == scrambled_sobol_naive(5u, d, 1000));
    CHECK_THROWS_AS(sob.state_at(uint64_t(1) << 32, x.data()), SobolLimitReached);
}

TEST_CASE("fill_normals de Sobol scrambleado: puntos consecutivos == generados por separado") {
    ScrambledSobol sob(7u, 16);
    std::vector<double> block(16 * 40), single(16);
    sob.fill_normals(100, 40, block.data());
    for (int p = 0; p < 40; p++) {
        sob.fill_normals(100 + p, 1, single.data());
        for (int d = 0; d < 16; d++) REQUIRE(block[p * 16 + d] == single[d]);
    }
    // y coincide con Phi^-1 de la versión ingenua
    for (int d = 0; d < 16; d++)
        CHECK(block[3 * 16 + d] ==
              norm_inv_cdf(sobol_word_to_u(scrambled_sobol_naive(7u, d, 103))));
}

TEST_CASE("Sobol scrambleado: las normales tienen media 0 y varianza 1 con 2^12 puntos") {
    ScrambledSobol sob(3u, 8);
    const int n = 1 << 12;
    std::vector<double> z((size_t)n * 8);
    sob.fill_normals(0, n, z.data());
    for (int d = 0; d < 8; d++) {
        double m = 0, v = 0;
        for (int p = 0; p < n; p++) m += z[(size_t)p * 8 + d];
        m /= n;
        for (int p = 0; p < n; p++) { double dz = z[(size_t)p * 8 + d] - m; v += dz * dz; }
        v /= n;
        CHECK(std::abs(m) < 2e-3);        // QMC: error mucho menor que 1/sqrt(n)=0.0156
        CHECK(std::abs(v - 1.0) < 2e-2);
    }
}

} // TEST_SUITE
