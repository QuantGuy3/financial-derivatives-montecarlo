// Fachada mc::run / mc::run_* : despacho CPU, planificación, progreso y errores del backend CUDA.
#include "doctest.h"
#include "engine/api.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {
bool bits_equal(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }
}

TEST_SUITE("fast") {

TEST_CASE("backend_from_string y nombres") {
    CHECK(mc::backend_from_string("cpu") == mc::Backend::Cpu);
    CHECK(mc::backend_from_string("gpu") == mc::Backend::Cuda);
    CHECK(mc::backend_from_string("cuda") == mc::Backend::Cuda);
    CHECK_THROWS_AS(mc::backend_from_string("tpu"), std::invalid_argument);
    CHECK(std::string(mc::backend_name(mc::Backend::Cpu)) == "cpu");
    CHECK(std::string(mc::family_name(mc::Family::MLQMC)) == "MLQMC");
    CHECK(mc::hardware_threads() >= 1);
}

TEST_CASE("sin CUDA: pedir el backend CUDA falla con un mensaje claro (y cuda_available() es false)") {
    CHECK_FALSE(mc::cuda_available());   // este build no incluye nvcc
    mc::RunOptions o; o.backend = mc::Backend::Cuda;
    GBMParams g;
    CHECK_THROWS_AS(mc::run_mc(g, European{100.0, 0.05, 1.0}, 0.1, 16, MCConfig{}, o), std::runtime_error);
    mc::RunSpec s;
    CHECK_THROWS_AS(mc::run(g, European{100.0, 0.05, 1.0}, s, o), std::runtime_error);
}

TEST_CASE("plan_n_steps: potencia de 2 acotada; cestas limitadas por Sobol") {
    GBMParams g;
    double c1 = 0;
    const int n = mc::plan_n_steps(g, European{100.0, 0.05, 1.0}, 0.01, mc::RunOptions{}, &c1);
    CHECK(c1 > 0.0);
    CHECK(n >= 1);
    CHECK(n <= 2048);
    CHECK((n & (n - 1)) == 0);
    // eps más fino => nunca menos pasos
    CHECK(mc::plan_n_steps(g, European{100.0, 0.05, 1.0}, 0.001) >= n);

    MultiDupireParams b; b.n = 1000; b.uncorrelated = true; b.S0.assign(1000, 100.0);
    CHECK(mc::plan_n_steps(b, Basket{100.0, 0.05, 1.0, 1000}, 1e-4) <= D_MAX_SOBOL / 1000);
}

TEST_CASE("run: las cuatro familias con la europea concuerdan con Black-Scholes") {
    GBMParams g; PayoffVariant pv = European{100.0, 0.05, 1.0};
    const double bs = bs_call(g.S0, 100.0, g.T, 0.05, g.sigma);
    for (auto fam : {mc::Family::MC, mc::Family::QMC, mc::Family::MLMC, mc::Family::MLQMC}) {
        mc::RunSpec s; s.family = fam; s.eps = 0.02; s.noise = NoiseMode::BrownianBridge;
        s.n_steps = 64;
        mc::RunReport r = mc::run(g, pv, s);
        CAPTURE(mc::family_name(fam));
        CHECK(r.result.n_samples > 0);
        CHECK(std::abs(r.result.price - bs) < 5.0 * r.result.std_error + 0.02);
        CHECK(r.backend == mc::Backend::Cpu);
        if (fam == mc::Family::MLMC || fam == mc::Family::MLQMC) CHECK(r.levels.size() >= 3);
        else CHECK(r.n_steps == 64);
    }
}

TEST_CASE("run: CV con E_ctrl y beta automáticos (GBM Asian y Dupire europea)") {
    {
        GBMParams g; mc::RunSpec s; s.family = mc::Family::MC; s.variance = mc::Variance::ControlVariate; s.eps = 0.02;
        mc::RunReport r = mc::run(g, Asian{100.0}, s);
        CHECK(r.beta > 0.8);
        CHECK(r.E_ctrl > 0.0);
        mc::RunSpec plain = s; plain.variance = mc::Variance::None;
        mc::RunReport p = mc::run(g, Asian{100.0}, plain);
        CHECK(r.result.n_samples < p.result.n_samples / 10);
    }
    {
        DupireLocalParams d; mc::RunSpec s; s.family = mc::Family::QMC; s.variance = mc::Variance::ControlVariate;
        s.eps = 0.02; s.noise = NoiseMode::Raw;
        mc::RunReport r = mc::run(d, European{100.0, 0.05, 1.0}, s);
        CHECK(r.beta > 0.0);
        CHECK(std::isfinite(r.result.price));
    }
    {
        HestonParams h; h.compute_cholesky(); mc::RunSpec s; s.variance = mc::Variance::ControlVariate;
        CHECK_THROWS_AS(mc::run(h, European{100.0, 0.05, 1.0}, s), std::invalid_argument);
    }
}

TEST_CASE("run: IS con z_star automático (OTM > 0, ITM = 0)") {
    GBMParams g; mc::RunSpec s; s.variance = mc::Variance::ImportanceSampling; s.eps = 0.02;
    mc::RunReport otm = mc::run(g, European{150.0, 0.05, 1.0}, s);
    CHECK(otm.z_star > 1.0);
    mc::RunReport itm = mc::run(g, European{80.0, 0.05, 1.0}, s);
    CHECK(itm.z_star == 0.0);
    CHECK(otm.n_steps >= 4);
    HestonParams h; h.compute_cholesky();
    CHECK_THROWS_AS(mc::run(h, European{100.0, 0.05, 1.0}, s), std::invalid_argument);
}

TEST_CASE("run: informe de hilos, progreso y cancelación") {
    struct Rec : ProgressSink {
        int n = 0; bool stop = false;
        void on_snapshot(const Snapshot&) override { ++n; }
        bool should_cancel() const override { return stop && n > 3; }
    } rec;
    GBMParams g; mc::RunSpec s; s.eps = 0.02; s.n_steps = 16;
    mc::RunOptions o; o.threads = 2; o.sink = &rec;
    mc::RunReport r = mc::run(g, European{100.0, 0.05, 1.0}, s, o);
    CHECK(r.threads == 2);
    CHECK(rec.n >= 3);

    rec.n = 0; rec.stop = true;
    s.eps = 0.001;
    mc::RunReport c = mc::run(g, European{100.0, 0.05, 1.0}, s, o);
    CHECK(c.cancelled);
}

TEST_CASE("default_options gobierna las llamadas de la capa 1") {
    mc::RunOptions saved = mc::default_options();
    mc::default_options().threads = 1;
    GBMParams g;
    auto a = mc::run_mc_fixed(g, European{100.0, 0.05, 1.0}, 16, 100000, 3u);
    mc::default_options().threads = 4;
    auto b = mc::run_mc_fixed(g, European{100.0, 0.05, 1.0}, 16, 100000, 3u);
    mc::default_options() = saved;
    CHECK(bits_equal(a.first, b.first));    // determinismo: mismos bits con 1 y 4 hilos
}

} // TEST_SUITE

// Contraste con la tabla guardada de la A100 (resultados/ejemplo01_barrido.txt, nivel eps=0.02,
// fila "MLQMC Raw": precio 10.4462, StdErr 0.0020, N=76800 con R=10 repeticiones de semilla
// seed+rep*977). El QMC de la CPU reproduce los puntos Sobol+HH de la GPU, así que las cifras
// coinciden a 4 decimales a pesar de que la CPU trabaja en double y la GPU en float.
TEST_SUITE("stat") {

TEST_CASE("paridad con la A100: MLQMC Raw de la europea a eps=0.02 (ejemplo01)") {
    GBMParams g; PayoffVariant pv = European{100.0, 0.05, 1.0};
    MLMCConfig ml; ml.M = 2; ml.max_L = 10;
    std::vector<double> prices; long long n_last = 0;
    for (int rep = 0; rep < 10; rep++) {
        QMCConfig q; q.seed += (unsigned)rep * 977u;
        MCResult r = mc::run_mlqmc(g, pv, 0.02, ml, q, NoiseMode::Raw);
        prices.push_back(r.price); n_last = r.n_samples;
    }
    double m = 0; for (double v : prices) m += v; m /= prices.size();
    double var = 0; for (double v : prices) var += (v - m) * (v - m); var /= prices.size() - 1;
    CHECK(m == doctest::Approx(10.4462).epsilon(1e-4));            // A100: 10.4462
    CHECK(std::sqrt(var / prices.size()) == doctest::Approx(0.0020).epsilon(0.1));
    CHECK(n_last == 76800);                                        // A100: 76800
}

} // TEST_SUITE
