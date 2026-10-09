// Monte Carlo estándar en CPU: exactitud (referencias discretas exactas y Black-Scholes),
// relaciones entre payoffs, cestas, determinismo, progreso, cancelación.
#include "doctest.h"
#include "cpu/mc_cpu.hpp"
#include "cpu/path_sim.hpp"
#include "utils.hpp"

#include <cmath>
#include <cstring>
#include <numbers>
#include <vector>

using namespace mc::cpu;

namespace {

bool bits_equal(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

double phi(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi); }

// Media exacta de S_N para Euler con deriva multiplicativa: E[S_{k+1}|S_k] = S_k (1+mu h)
double mean_S_N(double S0, double mu, double T, int n) { return S0 * std::pow(1.0 + mu * T / n, n); }

// Payoff con K=0 y r=0: max(S_T - 0, 0) * 1 = S_T  -> su media es E[S_T]
PayoffVariant identity_payoff() { return European{0.0, 0.0, 1.0}; }

struct Est { double mean, se; };
Est fixed(const ModelVariant& mv, const PayoffVariant& pv, int n_steps, long long N, unsigned seed,
          int threads = 0) {
    CpuOptions o; o.threads = threads;
    auto [m, v] = run_mc_fixed(mv, pv, n_steps, N, seed, o);
    return {m, std::sqrt(v)};
}

#define CHECK_WITHIN_SE(est, exact, k) \
    CHECK_MESSAGE(std::abs((est).mean - (exact)) < (k) * (est).se, \
                  "estimado " << (est).mean << " exacto " << (exact) << " se " << (est).se)

} // namespace

TEST_SUITE("fast") {

TEST_CASE("GBM Euler con 1 paso: la call tiene forma cerrada tipo Bachelier") {
    GBMParams g;   // S0=100, mu=0.05, sigma=0.2, T=1
    const double K = 100, r = 0.05;
    const double a = g.S0 * (1.0 + g.mu * g.T), b = g.S0 * g.sigma * std::sqrt(g.T);
    const double d = (a - K) / b;
    const double exact = std::exp(-r * g.T) * ((a - K) * norm_cdf(d) + b * phi(d));
    auto e = fixed(g, European{K, r, g.T}, 1, 4000000, 11u);
    CHECK_WITHIN_SE(e, exact, 5.0);
}

TEST_CASE("E[S_T] exacta: GBM, Heston y Dupire (Euler es martingala corregida por deriva)") {
    const int n = 16;
    {   GBMParams g; auto e = fixed(g, identity_payoff(), n, 2000000, 1u);
        CHECK_WITHIN_SE(e, mean_S_N(g.S0, g.mu, g.T, n), 5.0); }
    {   HestonParams h; h.compute_cholesky();
        auto e = fixed(h, identity_payoff(), n, 2000000, 2u);
        CHECK_WITHIN_SE(e, mean_S_N(h.S0, h.mu, h.T, n), 5.0); }
    {   DupireLocalParams d;
        auto e = fixed(d, identity_payoff(), n, 2000000, 3u);
        CHECK_WITHIN_SE(e, mean_S_N(d.S0, d.mu, d.T, n), 5.0); }
}

TEST_CASE("Asian aritmética con K=0: media exacta (1/N) sum S0 (1+mu h)^k") {
    GBMParams g; const int n = 12;
    double exact = 0; for (int k = 1; k <= n; k++) exact += std::pow(1.0 + g.mu * g.T / n, k);
    exact *= g.S0 / n;
    auto e = fixed(g, Asian{0.0}, n, 2000000, 5u);
    CHECK_WITHIN_SE(e, exact, 5.0);
}

TEST_CASE("Europea GBM con 64 pasos coincide con Black-Scholes salvo sesgo O(h)") {
    GBMParams g;
    auto e = fixed(g, European{100.0, 0.05, 1.0}, 64, 4000000, 21u);
    const double bs = bs_call(g.S0, 100.0, g.T, 0.05, g.sigma);
    CHECK(std::abs(e.mean - bs) < 5.0 * e.se + 0.01);
}

TEST_CASE("GeomAsian <= Asian para el mismo conjunto de caminos (AM-GM)") {
    GBMParams g;
    auto ar = fixed(g, Asian{0.0}, 32, 400000, 7u);
    auto ge = fixed(g, GeomAsian{0.0}, 32, 400000, 7u);   // mismos caminos: misma semilla
    CHECK(ge.mean < ar.mean);
}

TEST_CASE("Barrera: B enorme == Europea (bit a bit); B <= S0 => precio 0") {
    GBMParams g;
    auto eu = fixed(g, European{100.0, 0.05, 1.0}, 32, 200000, 9u);
    auto ba = fixed(g, Barrier{100.0, 1e12, 0.2, 0.05, 1.0}, 32, 200000, 9u);
    CHECK(bits_equal(eu.mean, ba.mean));
    auto z = fixed(g, Barrier{100.0, 90.0, 0.2, 0.05, 1.0}, 32, 100000, 9u);
    CHECK(z.mean == 0.0);
}

TEST_CASE("Lookback: S_T - min >= 0 y precio positivo; crece con sigma") {
    GBMParams lo, hi; lo.sigma = 0.1; hi.sigma = 0.4;
    auto a = fixed(lo, Lookback{0.1}, 64, 300000, 3u);
    auto b = fixed(hi, Lookback{0.4}, 64, 300000, 3u);
    CHECK(a.mean > 0.0);
    CHECK(b.mean > a.mean);
}

TEST_CASE("Dupire con alpha=0 y beta=1 reproduce GBM bit a bit") {
    GBMParams g;
    DupireLocalParams d; d.alpha = 0.0; d.beta_d = 1.0; d.sigma0 = g.sigma;
    auto a = fixed(g, European{100.0, 0.05, 1.0}, 16, 100000, 4u);
    auto b = fixed(d, European{100.0, 0.05, 1.0}, 16, 100000, 4u);
    CHECK(bits_equal(a.mean, b.mean));
    CHECK(bits_equal(a.se, b.se));
}

TEST_CASE("Heston: correlación negativa baja el precio de la call OTM frente a rho=0 (asimetría)") {
    HestonParams neg, zero; neg.rho = -0.9; zero.rho = 0.0;
    neg.compute_cholesky(); zero.compute_cholesky();
    PayoffVariant pv = European{130.0, 0.05, 1.0};   // OTM
    auto a = fixed(neg, pv, 64, 1000000, 17u);
    auto b = fixed(zero, pv, 64, 1000000, 17u);
    CHECK(a.mean < b.mean);
}

TEST_CASE("Cesta sin correlación: E[media de activos] exacta") {
    MultiDupireParams b; b.n = 10; b.uncorrelated = true; b.S0.assign(10, 100.0);
    const int n_steps = 8;
    auto e = fixed(b, Basket{0.0, 0.0, 1.0, 10}, n_steps, 400000, 6u);
    CHECK_WITHIN_SE(e, mean_S_N(100.0, b.mu, b.T, n_steps), 5.0);
}

TEST_CASE("Cesta correlacionada: más correlación => call de la cesta más cara") {
    const int n = 4;
    auto chol = [&](double rho) {
        std::vector<double> C(n * n), L(n * n, 0.0);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) C[i * n + j] = (i == j) ? 1.0 : rho;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double s = C[i * n + j];
                for (int k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
                L[i * n + j] = (i == j) ? std::sqrt(s) : s / L[j * n + j];
            }
        }
        return L;
    };
    MultiDupireParams hi, lo;
    hi.n = lo.n = n; hi.S0.assign(n, 100.0); lo.S0.assign(n, 100.0);
    hi.uncorrelated = false; hi.L = chol(0.95);
    lo.uncorrelated = true;
    PayoffVariant pv = Basket{100.0, 0.05, 1.0, n};
    auto a = fixed(hi, pv, 8, 300000, 31u);
    auto c = fixed(lo, pv, 8, 300000, 31u);
    CHECK(a.mean > c.mean + 3.0 * (a.se + c.se));
}

TEST_CASE("PathSim: el valor de un camino no depende de su posición en el bloque ni del chunk") {
    GBMParams g; CpuModel m = make_cpu_model(g); CpuPayoff p = make_cpu_payoff(European{100.0, 0.05, 1.0});
    const int n_steps = 8;
    RngNoise noise(5, Stream::Main, 0, n_steps, std::sqrt(g.T / n_steps));
    PathSim sim(m, p, n_steps, noise);
    Scratch s;
    // Y_i individual (count=1: la primera muestra se guarda en acc.c)
    std::vector<double> Y(41);
    for (int i = 0; i < 41; i++) { ChunkAcc a; sim.run_chunk(i, 1, s, a); Y[i] = a.acc.c; }
    // Media de 41 caminos en un solo chunk == media de las Y individuales
    ChunkAcc all; sim.run_chunk(0, 41, s, all);
    double mean = 0; for (double v : Y) mean += v; mean /= 41;
    CHECK(all.acc.to_moments().mean == doctest::Approx(mean).epsilon(1e-12));
    // y a partir de otro origen: los caminos 7..40
    ChunkAcc tail; sim.run_chunk(7, 34, s, tail);
    double mt = 0; for (int i = 7; i < 41; i++) mt += Y[i]; mt /= 34;
    CHECK(tail.acc.to_moments().mean == doctest::Approx(mt).epsilon(1e-12));
}

TEST_CASE("run_mc: alcanza la precisión pedida y el resultado es coherente") {
    GBMParams g; const double eps = 0.02;
    CpuOptions o; RunInfo info; o.info = &info;
    MCResult r = run_mc(g, European{100.0, 0.05, 1.0}, eps, 64, MCConfig{}, o);
    CHECK(r.std_error < eps);                       // N = 2Var/eps^2 => se ~ eps/sqrt(2)
    CHECK(r.std_error > eps / 3.0);
    CHECK(r.n_samples > 100000);
    CHECK(std::abs(r.price - bs_call(g.S0, 100, g.T, 0.05, g.sigma)) < 5.0 * r.std_error + 0.01);
    CHECK(info.n_nonfinite == 0);
    CHECK_FALSE(info.truncated);
    CHECK_FALSE(info.cancelled);
}

TEST_CASE("run_mc: progreso monótono, estados en orden y snapshot final") {
    struct Rec : ProgressSink {
        std::vector<Snapshot> s;
        void on_snapshot(const Snapshot& x) override { s.push_back(x); }
    } rec;
    GBMParams g; CpuOptions o; o.sink = &rec;
    MCResult r = run_mc(g, European{100.0, 0.05, 1.0}, 0.05, 16, MCConfig{}, o);
    REQUIRE(rec.s.size() >= 3);
    CHECK(rec.s.front().stage == Stage::Pilot);
    CHECK(rec.s.back().stage == Stage::Done);
    CHECK(rec.s.back().is_final);
    CHECK(rec.s.back().mean == r.price);
    for (size_t i = 1; i < rec.s.size(); i++) {
        CHECK(rec.s[i].seq == rec.s[i - 1].seq + 1);
        if (rec.s[i].stage == Stage::Main && rec.s[i - 1].stage == Stage::Main)
            CHECK(rec.s[i].n_done >= rec.s[i - 1].n_done);
    }
}

TEST_CASE("run_mc: cancelación cooperativa desde el sink") {
    struct Canceller : ProgressSink {
        int seen = 0;
        void on_snapshot(const Snapshot& s) override { if (s.stage == Stage::Main) seen++; }
        bool should_cancel() const override { return seen >= 2; }
    } c;
    GBMParams g; CpuOptions o; o.sink = &c; RunInfo info; o.info = &info;
    MCResult r = run_mc(g, European{100.0, 0.05, 1.0}, 0.001, 64, MCConfig{}, o);   // pediría ~4e8 caminos
    CHECK(info.cancelled);
    CHECK(r.n_samples < 5000000);
    CHECK(r.time_s < 10.0);
}

TEST_CASE("run_mc: max_seconds devuelve un resultado parcial marcado como truncado") {
    GBMParams g; CpuOptions o; o.max_seconds = 0.2; RunInfo info; o.info = &info;
    MCResult r = run_mc(g, European{100.0, 0.05, 1.0}, 0.001, 64, MCConfig{}, o);
    CHECK(info.truncated);
    CHECK(r.time_s < 10.0);
    CHECK(r.n_samples > 0);
}

} // TEST_SUITE

TEST_SUITE("stat") {

TEST_CASE("el error estándar reportado coincide con la desviación empírica entre semillas") {
    GBMParams g; PayoffVariant pv = European{100.0, 0.05, 1.0};
    const int R = 30; const long long N = 20000;
    std::vector<double> means; double se_sum = 0;
    for (int r = 0; r < R; r++) {
        auto e = fixed(g, pv, 16, N, 1000u + r);
        means.push_back(e.mean); se_sum += e.se;
    }
    double m = 0; for (double v : means) m += v; m /= R;
    double var = 0; for (double v : means) var += (v - m) * (v - m); var /= (R - 1);
    double ratio = std::sqrt(var) / (se_sum / R);
    CHECK(ratio > 0.6);
    CHECK(ratio < 1.5);
}

TEST_CASE("el error estándar escala como 1/sqrt(N)") {
    GBMParams g; PayoffVariant pv = European{100.0, 0.05, 1.0};
    auto a = fixed(g, pv, 16, 100000, 1u);
    auto b = fixed(g, pv, 16, 1600000, 1u);
    CHECK(a.se / b.se == doctest::Approx(4.0).epsilon(0.05));
}

} // TEST_SUITE

TEST_SUITE("determinism") {

TEST_CASE("run_mc y run_mc_fixed idénticos bit a bit con 1, 3, 8 y 16 hilos") {
    GBMParams g; PayoffVariant pv = Asian{100.0};
    HestonParams h; h.compute_cholesky();
    MultiDupireParams b; b.n = 6; b.uncorrelated = true; b.S0.assign(6, 100.0);
    struct Case { ModelVariant mv; PayoffVariant pv; int n; } cases[] = {
        {g, pv, 32}, {h, European{100.0, 0.05, 1.0}, 16}, {b, Basket{100.0, 0.05, 1.0, 6}, 8}};
    for (const auto& c : cases) {
        MCResult ref = run_mc(c.mv, c.pv, 0.1, c.n, MCConfig{}, [] { CpuOptions o; o.threads = 1; return o; }());
        auto fx = fixed(c.mv, c.pv, c.n, 150000, 3u, 1);
        for (int t : {3, 8, 16}) {
            CpuOptions o; o.threads = t;
            MCResult r = run_mc(c.mv, c.pv, 0.1, c.n, MCConfig{}, o);
            CHECK(bits_equal(r.price, ref.price));
            CHECK(bits_equal(r.std_error, ref.std_error));
            CHECK(r.n_samples == ref.n_samples);
            auto f2 = fixed(c.mv, c.pv, c.n, 150000, 3u, t);
            CHECK(bits_equal(f2.mean, fx.mean));
            CHECK(bits_equal(f2.se, fx.se));
        }
    }
}

} // TEST_SUITE

TEST_SUITE("determinism") {

// Con un ProgressSink las rondas son finas (curva de convergencia); sin él, gruesas (menos esperas
// entre rondas). El resultado no puede depender de eso: los chunks se funden uno a uno en orden.
TEST_CASE("run_mc: mismo resultado bit a bit con y sin ProgressSink (rondas finas o gruesas)") {
    struct Count : ProgressSink {
        long long calls = 0;
        void on_snapshot(const Snapshot&) override { ++calls; }
    };
    GBMParams g; PayoffVariant pv = Asian{100.0};
    for (int threads : {1, 5}) {
        CpuOptions o; o.threads = threads;
        const MCResult a = run_mc(g, pv, 0.01, 32, MCConfig{}, o);
        Count c; o.sink = &c;
        const MCResult b = run_mc(g, pv, 0.01, 32, MCConfig{}, o);
        CHECK(bits_equal(a.price, b.price));
        CHECK(bits_equal(a.std_error, b.std_error));
        CHECK(a.n_samples == b.n_samples);
        CHECK(c.calls > 10);
    }
}

TEST_CASE("make_wave_ends: cubre todos los chunks, crece y respeta el tope (fino y grueso)") {
    for (bool fine : {true, false}) {
        for (long long n : {1LL, 2LL, 63LL, 64LL, 65LL, 1000LL, 5000LL, 123457LL}) {
            const auto ends = make_wave_ends(n, fine);
            REQUIRE(!ends.empty());
            CHECK(ends.back() == n);
            long long prev = 0;
            for (long long e : ends) {
                CHECK(e > prev);
                CHECK(e - prev <= (fine ? 256 : kMaxWaveChunks));
                prev = e;
            }
        }
        // el reparto es una función solo de n (determinista)
        CHECK(make_wave_ends(5000, fine) == make_wave_ends(5000, fine));
    }
    CHECK(make_wave_ends(100000, false).size() < make_wave_ends(100000, true).size() / 4);
}

} // TEST_SUITE
