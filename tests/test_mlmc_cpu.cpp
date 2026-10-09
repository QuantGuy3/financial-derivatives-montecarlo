// MLMC y MLQMC en CPU: acoplamiento, telescopía, decaimiento de la varianza, consistencia con MC.
#include "doctest.h"
#include "cpu/mc_cpu.hpp"
#include "cpu/mlmc_cpu.hpp"
#include "utils.hpp"

#include <cmath>
#include <cstring>
#include <numbers>
#include <stdexcept>
#include <vector>

using namespace mc::cpu;

namespace {

bool bits_equal(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }
double phi(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi); }
double mean_S_N(double S0, double mu, double T, int n) { return S0 * std::pow(1.0 + mu * T / n, n); }

struct MlRun { MCResult r; RunInfo info; };

MlRun ml(const ModelVariant& mv, const PayoffVariant& pv, double eps, MLMCConfig c = {}, int threads = 0) {
    MlRun out; CpuOptions o; o.threads = threads; o.info = &out.info;
    out.r = run_mlmc(mv, pv, eps, c, o);
    return out;
}

} // namespace

TEST_SUITE("fast") {

TEST_CASE("MLMC: nivel 0 = Euler de 1 paso (forma cerrada) y E_l exactas para payoff lineal") {
    GBMParams g;
    // payoff K=0, r=0: S_T -> E[Yf_l] = S0 (1+mu h_l)^{N_l} exacto; E_l = E[Yf_l] - E[Yf_{l-1}]
    auto run = ml(g, European{0.0, 0.0, 1.0}, 0.05);
    REQUIRE(run.info.levels.size() >= 3);
    for (const auto& lv : run.info.levels) {
        const int nf = 1 << lv.l;
        double exact = mean_S_N(g.S0, g.mu, g.T, nf);
        if (lv.l > 0) exact -= mean_S_N(g.S0, g.mu, g.T, nf / 2);
        const double se = std::sqrt(lv.V / (double)lv.N);
        CHECK_MESSAGE(std::abs(lv.E - exact) < 5.0 * se + 1e-12,
                      "nivel " << lv.l << " E=" << lv.E << " exacto=" << exact << " se=" << se);
    }
}

TEST_CASE("MLMC: nivel 0 de la call europea == forma cerrada Bachelier") {
    GBMParams g; const double K = 100, r = 0.05;
    const double a = g.S0 * (1.0 + g.mu * g.T), b = g.S0 * g.sigma * std::sqrt(g.T), d = (a - K) / b;
    const double exact = std::exp(-r * g.T) * ((a - K) * norm_cdf(d) + b * phi(d));
    auto run = ml(g, European{K, r, g.T}, 0.02);
    const auto& l0 = run.info.levels.at(0);
    CHECK(std::abs(l0.E - exact) < 5.0 * std::sqrt(l0.V / (double)l0.N));
}

TEST_CASE("MLMC: precio de la europea cerca de Black-Scholes") {
    GBMParams g;
    auto run = ml(g, European{100.0, 0.05, 1.0}, 0.01);
    const double bs = bs_call(g.S0, 100.0, g.T, 0.05, g.sigma);
    CHECK(run.r.std_error < 0.01);
    CHECK(std::abs(run.r.price - bs) < 5.0 * run.r.std_error + 0.01);
    CHECK(run.r.n_samples > 1000);
    CHECK(run.info.n_nonfinite == 0);
}

TEST_CASE("MLMC: validaciones de entrada") {
    GBMParams g; MultiDupireParams b; b.n = 2; b.S0.assign(2, 100.0);
    PayoffVariant pv = European{100.0, 0.05, 1.0};
    CHECK_THROWS_AS(run_mlmc(b, Basket{100.0, 0.05, 1.0, 2}, 0.1), std::invalid_argument);
    MLMCConfig c1; c1.M = 1;
    CHECK_THROWS_AS(run_mlmc(g, pv, 0.1, c1), std::invalid_argument);
    CHECK_THROWS_AS(run_mlmc(g, pv, 0.0), std::invalid_argument);
    HestonParams h; h.compute_cholesky();
    CHECK_THROWS_AS(run_mlqmc(h, pv, 0.1, MLMCConfig{}, QMCConfig{}, NoiseMode::PCA), std::invalid_argument);
}

TEST_CASE("MLMC: progreso con estadísticos por nivel, cancelación y max_seconds") {
    struct Rec : ProgressSink {
        std::vector<Snapshot> s; std::vector<std::vector<LevelStat>> lv;
        void on_snapshot(const Snapshot& x) override {
            s.push_back(x);
            lv.emplace_back(x.levels, x.levels + x.n_levels);
        }
    } rec;
    GBMParams g; CpuOptions o; o.sink = &rec;
    MCResult r = run_mlmc(g, European{100.0, 0.05, 1.0}, 0.02, MLMCConfig{}, o);
    REQUIRE(rec.s.size() >= 3);
    CHECK(rec.s.front().stage == Stage::Pilot);
    CHECK(rec.s.back().stage == Stage::Done);
    CHECK(rec.s.back().mean == r.price);
    CHECK(rec.lv.back().size() >= 3);
    for (size_t i = 1; i < rec.s.size(); i++) CHECK(rec.s[i].seq == rec.s[i - 1].seq + 1);

    {   // max_seconds
        CpuOptions o2; o2.max_seconds = 0.2; RunInfo info; o2.info = &info;
        MCResult rr = run_mlmc(g, European{100.0, 0.05, 1.0}, 1e-4, MLMCConfig{}, o2);
        CHECK(info.truncated);
        CHECK(rr.time_s < 20.0);
    }
    {   // cancelación
        struct C : ProgressSink {
            int n = 0;
            void on_snapshot(const Snapshot& s) override { if (s.stage != Stage::Done) n++; }
            bool should_cancel() const override { return n >= 1; }
        } c;
        CpuOptions o3; o3.sink = &c; RunInfo info; o3.info = &info;
        run_mlmc(g, European{100.0, 0.05, 1.0}, 1e-4, MLMCConfig{}, o3);
        CHECK(info.cancelled);
    }
}

} // TEST_SUITE

TEST_SUITE("stat") {

TEST_CASE("MLMC: telescopía — la suma de E_l reproduce el MC con el paso más fino") {
    GBMParams g; PayoffVariant pv = European{100.0, 0.05, 1.0};
    auto run = ml(g, pv, 0.01);
    const int L = (int)run.info.levels.size() - 1;
    auto [m, v] = run_mc_fixed(g, pv, 1 << L, 4000000, 17u);
    CHECK(std::abs(run.r.price - m) < 5.0 * std::sqrt(run.r.std_error * run.r.std_error + v));
}

TEST_CASE("MLMC: la varianza de las correcciones decae como 2^-l (europea Euler)") {
    GBMParams g;
    MLMCConfig c; c.pilot_n = 20000;
    auto run = ml(g, European{100.0, 0.05, 1.0}, 0.003, c);
    const auto& lv = run.info.levels;
    REQUIRE(lv.size() >= 5);
    // pendiente de log2 V_l entre los niveles 1 y L-1
    const int a = 1, b = (int)lv.size() - 1;
    const double slope = (std::log2(lv[b].V) - std::log2(lv[a].V)) / (b - a);
    CHECK(slope > -1.4);
    CHECK(slope < -0.6);
}

TEST_CASE("MLMC: Asian, Lookback, Barrera, Heston y Dupire concuerdan con MC") {
    GBMParams g; HestonParams h; h.compute_cholesky(); DupireLocalParams d;
    struct Case { ModelVariant mv; PayoffVariant pv; double eps; } cases[] = {
        {g, Asian{100.0}, 0.02},
        {g, Lookback{0.2}, 0.05},
        {g, Barrier{100.0, 130.0, 0.2, 0.05, 1.0}, 0.02},
        {h, European{100.0, 0.05, 1.0}, 0.02},
        {d, European{100.0, 0.05, 1.0}, 0.02},
    };
    for (const auto& c : cases) {
        auto run = ml(c.mv, c.pv, c.eps);
        const int L = (int)run.info.levels.size() - 1;
        auto [m, v] = run_mc_fixed(c.mv, c.pv, 1 << L, 1500000, 3u);
        CHECK_MESSAGE(std::abs(run.r.price - m) < 5.0 * std::sqrt(run.r.std_error * run.r.std_error + v),
                      "ml=" << run.r.price << " mc=" << m << " se=" << run.r.std_error);
    }
}

TEST_CASE("MLQMC (Raw, BB, PCA) concuerda con Black-Scholes y con MLMC") {
    GBMParams g; PayoffVariant pv = European{100.0, 0.05, 1.0};
    const double bs = bs_call(g.S0, 100.0, g.T, 0.05, g.sigma);
    for (NoiseMode mode : {NoiseMode::Raw, NoiseMode::BrownianBridge, NoiseMode::PCA}) {
        CpuOptions o; RunInfo info; o.info = &info;
        MCResult r = run_mlqmc(g, pv, 0.01, MLMCConfig{}, QMCConfig{}, mode, o);
        CHECK_MESSAGE(std::abs(r.price - bs) < 5.0 * r.std_error + 0.01,
                      "modo " << (int)mode << " precio " << r.price << " se " << r.std_error);
        CHECK(info.levels.size() >= 3);
    }
}

TEST_CASE("MLQMC Raw con Heston y Asian concuerda con MLMC") {
    HestonParams h; h.compute_cholesky(); GBMParams g;
    {
        PayoffVariant pv = European{100.0, 0.05, 1.0};
        MCResult q = run_mlqmc(h, pv, 0.02, MLMCConfig{}, QMCConfig{}, NoiseMode::Raw);
        MCResult m = run_mlmc(h, pv, 0.02);
        CHECK(std::abs(q.price - m.price) < 5.0 * std::sqrt(q.std_error * q.std_error + m.std_error * m.std_error) + 0.02);
    }
    {
        PayoffVariant pv = Asian{100.0};
        MCResult q = run_mlqmc(g, pv, 0.02, MLMCConfig{}, QMCConfig{}, NoiseMode::Raw);
        MCResult m = run_mlmc(g, pv, 0.02);
        CHECK(std::abs(q.price - m.price) < 5.0 * std::sqrt(q.std_error * q.std_error + m.std_error * m.std_error) + 0.02);
    }
}

} // TEST_SUITE

TEST_SUITE("determinism") {

TEST_CASE("MLMC y MLQMC idénticos bit a bit con 1, 3, 8 y 16 hilos") {
    GBMParams g; PayoffVariant pv = Asian{100.0};
    MLMCConfig mc; mc.max_L = 6;
    QMCConfig qc; qc.R = 8;
    CpuOptions o1; o1.threads = 1;
    MCResult ref_ml = run_mlmc(g, pv, 0.05, mc, o1);
    MCResult ref_q[3];
    const NoiseMode modes[3] = {NoiseMode::Raw, NoiseMode::BrownianBridge, NoiseMode::PCA};
    for (int i = 0; i < 3; i++) ref_q[i] = run_mlqmc(g, pv, 0.05, mc, qc, modes[i], o1);
    for (int t : {3, 8, 16}) {
        CpuOptions o; o.threads = t;
        MCResult r = run_mlmc(g, pv, 0.05, mc, o);
        CHECK(bits_equal(r.price, ref_ml.price));
        CHECK(bits_equal(r.std_error, ref_ml.std_error));
        CHECK(r.n_samples == ref_ml.n_samples);
        for (int i = 0; i < 3; i++) {
            MCResult q = run_mlqmc(g, pv, 0.05, mc, qc, modes[i], o);
            CHECK(bits_equal(q.price, ref_q[i].price));
            CHECK(bits_equal(q.std_error, ref_q[i].std_error));
            CHECK(q.n_samples == ref_q[i].n_samples);
        }
    }
}

} // TEST_SUITE
