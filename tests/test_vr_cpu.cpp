// Reducción de varianza en CPU: variables de control e importance sampling.
#include "doctest.h"
#include "cpu/vr_cpu.hpp"
#include "utils.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

using namespace mc::cpu;

namespace {

bool bits_equal(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

struct AsianCv {
    GBMParams g; double K = 100.0;
    PayoffVariant main = Asian{100.0}, ctrl = GeomAsian{100.0};
    ModelVariant mv() const { return g; }
    double E(int ns) const { return geom_asian_analytic(g.S0, K, g.T, g.mu, g.sigma, ns); }
};

} // namespace

TEST_SUITE("fast") {

TEST_CASE("cv_pilot: beta ~ 1 y la varianza cae >50x para Asian aritmética + geométrica") {
    AsianCv a; const int ns = 32;
    CVPilot p = cv_pilot(a.mv(), a.mv(), a.main, a.ctrl, a.E(ns), ns, 200000, 5u);
    CHECK(p.beta > 0.9);
    CHECK(p.beta < 1.2);
    CHECK(p.var_cv < p.var_plain / 50.0);
}

TEST_CASE("CV: tipos no soportados lanzan excepción") {
    GBMParams g; HestonParams h; h.compute_cholesky();
    CHECK_THROWS_AS(cv_pilot(g, g, European{100.0, 0.05, 1.0}, GeomAsian{100.0}, 0.0, 16, 1000), std::invalid_argument);
    CHECK_THROWS_AS(run_mc_cv(h, h, Asian{100.0}, GeomAsian{100.0}, 0.0, 1.0, 0.1, 16), std::invalid_argument);
    CHECK_THROWS_AS(run_mlmc_cv(DupireLocalParams{}, DupireLocalParams{}, European{100.0, 0.05, 1.0},
                                European{100.0, 0.05, 1.0}, 0.0, 1.0, 0.1), std::invalid_argument);
}

TEST_CASE("IS con z_star = 0 reproduce el MC simple bit a bit") {
    GBMParams g; European call{100.0, 0.05, 1.0};
    MCConfig cfg;
    const double eps = 0.05;
    MCResult a = run_is(g, call, 0.0, eps, cfg);
    MCResult b = run_mc(g, call, eps, is_n_steps(g.T, eps), cfg);
    CHECK(bits_equal(a.price, b.price));
    CHECK(bits_equal(a.std_error, b.std_error));
    CHECK(a.n_samples == b.n_samples);
}

TEST_CASE("is_n_steps como en la GPU") {
    CHECK(is_n_steps(1.0, 0.05) == 32);     // ceil(20) -> 32
    CHECK(is_n_steps(1.0, 1.0) == 4);
    CHECK(is_n_steps(1.0, 1e-6) == 2048);
}

TEST_CASE("IS: validaciones y límites") {
    GBMParams g;
    CHECK_THROWS_AS(run_mlmc_is(g, European{100.0, 0.05, 1.0}, 1.0, 0.0), std::invalid_argument);
}

} // TEST_SUITE

TEST_SUITE("stat") {

TEST_CASE("MC + CV: insesgado respecto al MC simple y necesita muchas menos muestras") {
    AsianCv a; const int ns = 32; const double eps = 0.01;
    CVPilot p = cv_pilot(a.mv(), a.mv(), a.main, a.ctrl, a.E(ns), ns, 100000, 1u);
    MCResult cv = run_mc_cv(a.mv(), a.mv(), a.main, a.ctrl, a.E(ns), p.beta, eps, ns);
    MCResult plain = run_mc(a.mv(), a.main, eps, ns);
    // E_ctrl es la fórmula del lognormal exacto, no del Euler discreto: sesgo O(h) del control
    CHECK(std::abs(cv.price - plain.price) < 5.0 * std::sqrt(cv.std_error * cv.std_error
                                                            + plain.std_error * plain.std_error) + 0.01);
    CHECK(cv.n_samples < plain.n_samples / 20);
    CHECK(cv.std_error < 1.2 * eps);
}

TEST_CASE("Dupire + CV (control GBM sigma0) concuerda con MC simple") {
    DupireLocalParams d; const int ns = 32; const double K = 100.0, r = 0.05;
    PayoffVariant main = European{K, r, d.T};
    const double Ec = bs_call(d.S0, K, d.T, r, d.sigma0);   // control GBM(sigma0)
    CVPilot p = cv_pilot(d, d, main, main, Ec, ns, 100000, 3u);
    MCResult cv = run_mc_cv(d, d, main, main, Ec, p.beta, 0.01, ns);
    MCResult plain = run_mc(d, main, 0.01, ns);
    CHECK(std::abs(cv.price - plain.price) < 5.0 * std::sqrt(cv.std_error * cv.std_error
                                                            + plain.std_error * plain.std_error) + 0.02);
    CHECK(p.var_cv < p.var_plain);
}

TEST_CASE("QMC + CV (Raw, BB, PCA) concuerda con MC + CV") {
    AsianCv a; const int ns = 32;
    CVPilot p = cv_pilot(a.mv(), a.mv(), a.main, a.ctrl, a.E(ns), ns, 100000, 1u);
    MCResult ref = run_mc_cv(a.mv(), a.mv(), a.main, a.ctrl, a.E(ns), p.beta, 0.005, ns);
    for (NoiseMode mode : {NoiseMode::Raw, NoiseMode::BrownianBridge, NoiseMode::PCA}) {
        MCResult q = run_qmc_cv(a.mv(), a.mv(), a.main, a.ctrl, a.E(ns), p.beta, 0.005, ns, QMCConfig{}, mode);
        CHECK_MESSAGE(std::abs(q.price - ref.price) < 5.0 * std::sqrt(q.std_error * q.std_error
                                                                     + ref.std_error * ref.std_error) + 1e-3,
                      "modo " << (int)mode << ": " << q.price << " vs " << ref.price);
    }
}

TEST_CASE("MLMC + CV y MLQMC + CV (GBM Asian) concuerdan con MLMC simple y necesitan menos muestras") {
    AsianCv a; const int ns = 256;
    // E_ctrl es la fórmula del lognormal exacto (n_steps = ns); el esquema de Euler de los niveles
    // difiere en O(h), de ahí la holgura de 0.05 en los casos con beta != 0.
    CVPilot p = cv_pilot(a.mv(), a.mv(), a.main, a.ctrl, a.E(ns), ns, 100000, 1u);
    MCResult plain = run_mlmc(a.mv(), a.main, 0.01);
    auto close = [&](const MCResult& x, const MCResult& y, double slack) {
        return std::abs(x.price - y.price) < 5.0 * std::sqrt(x.std_error * x.std_error + y.std_error * y.std_error) + slack;
    };
    // beta = 0: el estimador es exactamente el MLMC simple (valida el núcleo acoplado y la fontanería)
    CHECK(close(run_mlmc_cv(a.mv(), a.mv(), a.main, a.ctrl, a.E(ns), 0.0, 0.01), plain, 1e-3));
    CHECK(close(run_mlqmc_cv(a.mv(), a.mv(), a.main, a.ctrl, a.E(ns), 0.0, 0.01), plain, 1e-3));
    // beta de piloto: misma esperanza (salvo el sesgo del control) con muchas menos muestras
    MCResult cv = run_mlmc_cv(a.mv(), a.mv(), a.main, a.ctrl, a.E(ns), p.beta, 0.01);
    MCResult qcv = run_mlqmc_cv(a.mv(), a.mv(), a.main, a.ctrl, a.E(ns), p.beta, 0.01);
    CHECK(close(cv, plain, 0.05));
    CHECK(close(qcv, plain, 0.05));
    CHECK(cv.n_samples < plain.n_samples);
}

TEST_CASE("IS (MC, QMC, MLMC, MLQMC) es insesgado y reduce la varianza en una call muy OTM") {
    GBMParams g; European call{160.0, 0.05, 1.0};
    const double z_star = 2.3, eps = 0.01;
    const int ns = is_n_steps(g.T, eps);
    MCResult plain = run_mc(g, call, eps, ns);
    MCResult is = run_is(g, call, z_star, eps);
    MCResult qis = run_qmc_is(g, call, z_star, eps);
    MCResult mis = run_mlmc_is(g, call, z_star, eps);
    MCResult mqis = run_mlqmc_is(g, call, z_star, eps);
    auto close = [&](const MCResult& x, const MCResult& y) {
        return std::abs(x.price - y.price) < 5.0 * std::sqrt(x.std_error * x.std_error + y.std_error * y.std_error) + 0.01;
    };
    CHECK(close(is, plain));
    CHECK(close(qis, plain));
    CHECK(close(mis, plain));
    CHECK(close(mqis, plain));
    CHECK(is.n_samples < plain.n_samples / 5);
}

} // TEST_SUITE

TEST_SUITE("determinism") {

TEST_CASE("CV e IS idénticos bit a bit con 1, 3, 8 y 16 hilos") {
    AsianCv a; const int ns = 16;
    GBMParams g; European call{120.0, 0.05, 1.0};
    auto run_all = [&](int threads, MCResult out[6]) {
        CpuOptions o; o.threads = threads;
        out[0] = run_mc_cv(a.mv(), a.mv(), a.main, a.ctrl, a.E(ns), 1.0, 0.05, ns, MCConfig{}, o);
        QMCConfig q; q.R = 8; q.n0 = 512; q.max_doublings = 3;
        out[1] = run_qmc_cv(a.mv(), a.mv(), a.main, a.ctrl, a.E(ns), 1.0, 1e-9, ns, q, NoiseMode::BrownianBridge, o);
        MLMCConfig ml; ml.max_L = 5;
        out[2] = run_mlmc_cv(a.mv(), a.mv(), a.main, a.ctrl, a.E(ns), 1.0, 0.05, ml, o);
        out[3] = run_is(g, call, 1.0, 0.05, MCConfig{}, o);
        out[4] = run_mlmc_is(g, call, 1.0, 0.05, ml, o);
        out[5] = run_mlqmc_is(g, call, 1.0, 0.05, ml, q, NoiseMode::Raw, o);
    };
    MCResult ref[6], cur[6];
    run_all(1, ref);
    for (int t : {3, 8, 16}) {
        run_all(t, cur);
        for (int i = 0; i < 6; i++) {
            CHECK_MESSAGE(bits_equal(cur[i].price, ref[i].price), "caso " << i << " hilos " << t);
            CHECK(bits_equal(cur[i].std_error, ref[i].std_error));
            CHECK(cur[i].n_samples == ref[i].n_samples);
        }
    }
}

} // TEST_SUITE
