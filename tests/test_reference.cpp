// Precios de referencia analíticos.
#include "doctest.h"
#include "cpu/mc_cpu.hpp"
#include "reference_prices.hpp"
#include "utils.hpp"

#include <cmath>

TEST_SUITE("fast") {

TEST_CASE("gbm_call_drift con mu = r coincide con Black-Scholes; con mu != r cambia") {
    CHECK(gbm_call_drift(100, 100, 1.0, 0.05, 0.05, 0.2) == doctest::Approx(bs_call(100, 100, 1.0, 0.05, 0.2)).epsilon(1e-12));
    CHECK(gbm_call_drift(100, 100, 1.0, 0.10, 0.05, 0.2) > gbm_call_drift(100, 100, 1.0, 0.05, 0.05, 0.2));
}

TEST_CASE("heston_call: caso de referencia de la literatura y parámetros por defecto") {
    HestonParams lit;   // Schoutens-Simons-Tistaert / Fang-Oosterlee: 5.785155450
    lit.S0 = 100; lit.mu = 0.0; lit.kappa = 1.5768; lit.theta = 0.0398; lit.xi = 0.5751;
    lit.rho = -0.5711; lit.v0 = 0.0175; lit.T = 1.0;
    CHECK(heston_call(lit, 100.0, 0.0) == doctest::Approx(5.785155450).epsilon(1e-7));

    HestonParams def; def.compute_cholesky();     // valores por defecto del repositorio (mu = r = 0.05)
    CHECK(heston_call(def, 100.0, 0.05) == doctest::Approx(10.150275235).epsilon(1e-8));   // scipy.quad
    CHECK(heston_call(def, 130.0, 0.05) == doctest::Approx(0.115844820).epsilon(1e-7));
}

TEST_CASE("reference_price: qué combinaciones tienen referencia") {
    GBMParams g; HestonParams h; h.compute_cholesky(); DupireLocalParams d;
    CHECK(reference_price(g, European{100.0, 0.05, 1.0}, 0).has_value());
    CHECK(reference_price(g, GeomAsian{100.0}, 64).has_value());
    CHECK_FALSE(reference_price(g, GeomAsian{100.0}, 0).has_value());     // necesita n_steps
    CHECK(reference_price(h, European{100.0, 0.05, 1.0}, 0).has_value());
    CHECK_FALSE(reference_price(d, European{100.0, 0.05, 1.0}, 0).has_value());
    CHECK_FALSE(reference_price(g, Asian{100.0}, 64).has_value());
}

} // TEST_SUITE

TEST_SUITE("stat") {

TEST_CASE("el motor CPU converge a la referencia de Heston (esquema de Euler, sesgo O(h))") {
    HestonParams h; h.compute_cholesky();
    PayoffVariant pv = European{100.0, 0.05, 1.0};
    auto [m, v] = mc::cpu::run_mc_fixed(h, pv, 256, 6000000, 9u);
    const double ref = heston_call(h, 100.0, 0.05);
    CHECK(std::abs(m - ref) < 5.0 * std::sqrt(v) + 0.03);
}

TEST_CASE("el motor CPU converge a la Asian geométrica discreta (referencia lognormal exacta)") {
    GBMParams g;
    PayoffVariant pv = GeomAsian{100.0};
    const int n = 64;
    auto [m, v] = mc::cpu::run_mc_fixed(g, pv, n, 4000000, 4u);
    CHECK(std::abs(m - geom_asian_analytic(g.S0, 100.0, g.T, g.mu, g.sigma, n)) < 5.0 * std::sqrt(v) + 0.02);
}

} // TEST_SUITE
