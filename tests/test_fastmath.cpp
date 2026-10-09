// fast_pow (potencia de la volatilidad local de Dupire): precisión frente a std::pow, identidad
// bit a bit entre la versión escalar y la AVX2, y casos fuera de rango.
#include "doctest.h"
#include "cpu/fastmath.hpp"
#include "cpu/kernels.hpp"
#include "cpu/rng.hpp"
#include "cpu/simd.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

using namespace mc::cpu;

namespace {

// Error en ulp de `got` respecto a `ref`.
double ulps(double got, double ref) {
    if (got == ref) return 0.0;
    const double u = std::nextafter(std::abs(ref), std::numeric_limits<double>::infinity()) - std::abs(ref);
    return std::abs(got - ref) / u;
}

} // namespace

TEST_SUITE("fast") {

TEST_CASE("fast_pow: valores exactos y casos limite") {
    CHECK(fast_log(1.0) == 0.0);
    CHECK(fast_exp(0.0) == 1.0);
    for (double b : {-3.0, -0.3, 0.0, 0.5, 2.0}) CHECK(fast_pow(1.0, b) == 1.0);   // sigma_loc(S0) = sigma0 exacto
    CHECK(fast_pow(7.5, 0.0) == 1.0);
    CHECK(fast_pow(2.0, 10.0) == doctest::Approx(1024.0).epsilon(1e-15));
    CHECK(fast_pow(4.0, 0.5) == doctest::Approx(2.0).epsilon(1e-15));
    CHECK(fast_pow(10.0, -2.0) == doctest::Approx(0.01).epsilon(1e-15));
    // fuera del rango de la vía rápida: se comporta como std::pow
    CHECK(fast_pow(0.0, -0.3) == std::pow(0.0, -0.3));                       // +inf
    CHECK(fast_pow(0.0, 0.3) == 0.0);
    CHECK(std::isnan(fast_pow(-1.0, 0.3)));
    CHECK(std::isnan(fast_pow(std::numeric_limits<double>::quiet_NaN(), 2.0)));
    CHECK(fast_pow(std::numeric_limits<double>::infinity(), -1.0) == 0.0);
    CHECK(fast_pow(5e-324, 0.5) == std::pow(5e-324, 0.5));                   // subnormal
    CHECK(fast_pow(1e300, 3.0) == std::numeric_limits<double>::infinity()); // desbordamiento
    CHECK(fast_pow(1e-300, 3.0) == 0.0);
}

TEST_CASE("fast_pow: error frente a std::pow en el rango de Dupire y en un rango amplio") {
    Xoshiro256pp g = Xoshiro256pp::for_path(11, Stream::Test, 0, 0);
    double worst_dupire = 0.0, worst_wide = 0.0, worst_rel_wide = 0.0;
    for (int n = 0; n < 400000; n++) {
        // S/S0 en [0.2, 5], exponente beta-1 en [-0.95, 1] (beta en [0.05, 2], lo que admite la GUI)
        const double x = std::exp(std::log(0.2) + (std::log(5.0) - std::log(0.2)) * g.uniform());
        const double b = -0.95 + 1.95 * g.uniform();
        worst_dupire = std::max(worst_dupire, ulps(fast_pow(x, b), std::pow(x, b)));
        // rango amplio: el error crece con |b·ln x|
        const double xw = std::exp(std::log(1e-6) + (std::log(1e6) - std::log(1e-6)) * g.uniform());
        const double bw = -4.0 + 8.0 * g.uniform();
        const double ref = std::pow(xw, bw);
        worst_wide = std::max(worst_wide, ulps(fast_pow(xw, bw), ref));
        worst_rel_wide = std::max(worst_rel_wide, std::abs(fast_pow(xw, bw) - ref) / ref / (2.0 + std::abs(bw * std::log(xw))));
    }
    MESSAGE("error maximo de fast_pow: " << worst_dupire << " ulp (rango Dupire), " << worst_wide << " ulp (rango amplio)");
    CHECK(worst_dupire < 6.0);                // medido: 4 ulp
    CHECK(worst_rel_wide < 4e-16);            // cota documentada: 4e-16·(2 + |b·ln x|); medido: 3.0e-16
}

TEST_CASE("fast_pow_n: la version AVX2 da los mismos bits que la escalar") {
    if (simd_level_available() != SimdLevel::Avx2) { MESSAGE("sin AVX2: nada que comparar"); return; }
    Xoshiro256pp g = Xoshiro256pp::for_path(12, Stream::Test, 0, 0);
    const int N = 1 << 16;
    std::vector<double> x((size_t)N), ref((size_t)N), out((size_t)N);
    for (double b : {-0.3, -0.95, 0.5, 1.0, 2.5, -3.0, 0.0}) {
        for (int range = 0; range < 3; range++) {
            for (auto& v : x) {
                const double u = g.uniform();
                if (range == 0) v = 0.2 + 4.8 * u;                                   // Dupire
                else if (range == 1) v = std::exp(-300.0 + 600.0 * u);               // amplio: algunos |y| >= 700
                else v = (u < 0.1) ? 0.0 : (u < 0.15 ? -1.0 : (u < 0.2 ? 1e-320 : 1.0 + u));   // con valores fuera de rango
            }
            set_simd_level(SimdLevel::Scalar);
            fast_pow_n(x.data(), b, ref.data(), N);
            for (int i = 0; i < 64; i++) {      // la versión escalar por bloque es fast_pow
                const double want = fast_pow(x[(size_t)i], b);
                CHECK(std::memcmp(&ref[(size_t)i], &want, sizeof(double)) == 0);
            }
            set_simd_level(SimdLevel::Avx2);
            for (int n : {N, 1, 3, 4, 5, 7, 8, 13}) {      // incluye longitudes que no son múltiplo de 4
                fast_pow_n(x.data(), b, out.data(), n);
                CHECK_MESSAGE(std::memcmp(out.data(), ref.data(), sizeof(double) * (size_t)n) == 0,
                              "b=" << b << " rango=" << range << " n=" << n);
            }
            // en sitio (x y out son el mismo array)
            std::vector<double> inplace(x.begin(), x.begin() + 37);
            fast_pow_n(inplace.data(), b, inplace.data(), 37);
            CHECK(std::memcmp(inplace.data(), ref.data(), sizeof(double) * 37) == 0);
        }
    }
    set_simd_level(simd_level_available());
}

TEST_CASE("euler_dupire_block: mismos bits que euler_dupire carril a carril, con y sin AVX2") {
    CpuModel m;
    m.kind = ModelKind::Dupire;
    m.S0 = 100.0; m.mu = 0.05; m.sigma0 = 0.2; m.alpha = 0.5; m.beta_d = 0.7; m.T = 1.0;
    Xoshiro256pp g = Xoshiro256pp::for_path(13, Stream::Test, 0, 0);
    for (SimdLevel lv : {SimdLevel::Scalar, SimdLevel::Avx2}) {
        if (lv > simd_level_available()) continue;
        set_simd_level(lv);
        for (int rep = 0; rep < 2000; rep++) {
            double S[kLanes], dw[kLanes], want[kLanes];
            for (int l = 0; l < kLanes; l++) {
                S[l] = 100.0 * std::exp(-1.0 + 2.0 * g.uniform());
                if (rep % 7 == 0 && l == rep % kLanes) S[l] = 0.0;                    // carril absorbido
                dw[l] = (rep % 11 == 0 && l == 0) ? -30.0 : 0.2 * (g.uniform() - 0.5);  // un incremento que lleva S por debajo de 0
                want[l] = euler_dupire(m, S[l], dw[l], 1.0 / 64, 0.9, m.S0);
            }
            euler_dupire_block(m, S, dw, 1.0 / 64, 0.9, m.S0);
            CHECK(std::memcmp(S, want, sizeof(S)) == 0);
        }
    }
    set_simd_level(simd_level_available());
}

} // TEST_SUITE
