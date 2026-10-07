// Muestreador de trayectorias de la GUI: usa los mismos pasos que el motor.
#include "doctest.h"
#include "cpu/mc_cpu.hpp"
#include "cpu/path_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

using namespace mc::cpu;

TEST_SUITE("fast") {

TEST_CASE("sample_paths: forma, reproducibilidad por semilla y distinción entre caminos") {
    GBMParams g;
    SampleOptions o; o.n_paths = 5; o.n_steps = 64; o.seed = 7;
    auto a = sample_paths(g, European{100.0, 0.05, 1.0}, o);
    auto b = sample_paths(g, European{100.0, 0.05, 1.0}, o);
    REQUIRE(a.paths.size() == 5);
    REQUIRE(a.t.size() == 65);
    CHECK(a.t.front() == 0.0);
    CHECK(a.t.back() == doctest::Approx(1.0));
    for (size_t i = 0; i < a.paths.size(); i++) {
        CHECK(a.paths[i].S.size() == 65);
        CHECK(a.paths[i].S.front() == 100.0);
        CHECK(a.paths[i].S == b.paths[i].S);                          // misma semilla => mismos caminos
    }
    CHECK(a.paths[0].S != a.paths[1].S);                              // caminos distintos
    o.seed = 8;
    CHECK(sample_paths(g, European{100.0, 0.05, 1.0}, o).paths[0].S != a.paths[0].S);
    // el payoff coincide con max(S_T-K,0)·descuento
    for (const auto& p : a.paths)
        CHECK(p.payoff == doctest::Approx(std::max(p.S.back() - 100.0, 0.0) * std::exp(-0.05)).epsilon(1e-12));
}

TEST_CASE("sample_paths: la media de muchos caminos es el payoff simulado por el motor (mismos pasos de Euler)") {
    GBMParams g; PayoffVariant pv = Asian{100.0};
    SampleOptions o; o.n_paths = 40000; o.n_steps = 16; o.seed = 3;
    auto s = sample_paths(g, pv, o);
    double m = 0, m2 = 0;
    for (const auto& p : s.paths) { m += p.payoff; m2 += p.payoff * p.payoff; }
    m /= s.paths.size(); m2 /= s.paths.size();
    const double se = std::sqrt((m2 - m * m) / s.paths.size());
    auto [em, ev] = run_mc_fixed(g, pv, 16, 2000000, 5u);
    CHECK(std::abs(m - em) < 5.0 * std::sqrt(se * se + ev));
}

TEST_CASE("sample_paths: Asian, Lookback y Barrera exponen su estadístico corriente") {
    GBMParams g; SampleOptions o; o.n_paths = 3; o.n_steps = 32; o.seed = 1;
    auto as = sample_paths(g, Asian{100.0}, o);
    double sum = 0;
    for (int k = 1; k <= 32; k++) sum += as.paths[0].S[(size_t)k];
    CHECK(as.paths[0].run.back() == doctest::Approx(sum / 32).epsilon(1e-12));
    auto lb = sample_paths(g, Lookback{0.2}, o);
    CHECK(lb.paths[0].run.back() == *std::min_element(lb.paths[0].S.begin() + 1, lb.paths[0].S.end()) );
    auto br = sample_paths(g, Barrier{100.0, 101.0, 0.2, 0.05, 1.0}, o);   // barrera pegada a S0: casi siempre se toca
    int knocked = 0;
    for (const auto& p : br.paths) knocked += p.knocked_out ? 1 : 0;
    CHECK(knocked >= 1);
    for (const auto& p : br.paths) if (p.knocked_out) CHECK(p.payoff == 0.0);
}

TEST_CASE("sample_paths: Heston devuelve la varianza; cestas devuelven los activos mostrados") {
    HestonParams h; h.compute_cholesky(); SampleOptions o; o.n_paths = 2; o.n_steps = 32;
    auto s = sample_paths(h, European{100.0, 0.05, 1.0}, o);
    REQUIRE(s.paths[0].V.size() == 33);
    CHECK(s.paths[0].V.front() == h.v0);

    MultiDupireParams b; b.n = 8; b.uncorrelated = true; b.S0.assign(8, 100.0);
    o.n_steps = 8;
    auto sb = sample_paths(b, Basket{100.0, 0.05, 1.0, 8}, o);
    REQUIRE(sb.paths[0].assets.size() == 5);
    CHECK(sb.paths[0].assets[0].size() == 9);
    CHECK(sb.paths[0].S.size() == 9);
}

TEST_CASE("sample_paths: Sobol y construcciones BB/PCA") {
    GBMParams g; SampleOptions o; o.n_paths = 4; o.n_steps = 16; o.seed = 5; o.sobol = true;
    for (auto c : {NoiseMode::Raw, NoiseMode::BrownianBridge, NoiseMode::PCA}) {
        o.construction = c;
        auto s = sample_paths(g, European{100.0, 0.05, 1.0}, o);
        REQUIRE(s.paths.size() == 4);
        for (const auto& p : s.paths) for (double v : p.S) CHECK(std::isfinite(v));
    }
    o.construction = NoiseMode::BrownianBridge; o.n_steps = 12;
    CHECK_THROWS_AS(sample_paths(g, European{100.0, 0.05, 1.0}, o), std::invalid_argument);
    HestonParams h; h.compute_cholesky(); o.n_steps = 16; o.construction = NoiseMode::PCA;
    CHECK_THROWS_AS(sample_paths(h, European{100.0, 0.05, 1.0}, o), std::invalid_argument);
}

TEST_CASE("sample_coupled: el grueso comparte el ruido del fino") {
    GBMParams g;
    auto cp = sample_coupled(g, European{100.0, 0.05, 1.0}, 3, 2, 11u);
    CHECK(cp.n_fine == 8);
    CHECK(cp.n_coarse == 4);
    CHECK(cp.S_fine.size() == 9);
    CHECK(cp.S_coarse.size() == 5);
    CHECK(cp.t_coarse.back() == doctest::Approx(1.0));
    // el payoff del par estará correlado: |fino - grueso| pequeño frente al nivel del payoff en promedio
    double sd = 0;
    for (int i = 0; i < 500; i++) {
        auto p = sample_coupled(g, European{100.0, 0.05, 1.0}, 6, 2, 11u, i);
        sd += std::abs(p.payoff_fine - p.payoff_coarse);
    }
    CHECK(sd / 500 < 1.0);    // sin acoplar serían ~ 8
}

TEST_CASE("sample_fan: percentiles ordenados y media cercana a S0 e^{mu t}") {
    GBMParams g;
    auto f = sample_fan(g, European{100.0, 0.05, 1.0}, 32, 4000, 3u);
    REQUIRE(f.t.size() == 33);
    for (size_t k = 1; k < f.t.size(); k++) {
        CHECK(f.p05[k] <= f.p25[k]);
        CHECK(f.p25[k] <= f.p50[k]);
        CHECK(f.p50[k] <= f.p75[k]);
        CHECK(f.p75[k] <= f.p95[k]);
    }
    CHECK(f.mean.back() == doctest::Approx(100.0 * std::exp(0.05)).epsilon(0.02));
    CHECK(f.terminal.size() == 4000);
}

} // TEST_SUITE
