// QMC en CPU: fuentes de ruido (Sobol, BB, PCA), run_qmc, determinismo y propiedades.
#include "doctest.h"
#include "cpu/mc_cpu.hpp"
#include "cpu/qmc_cpu.hpp"
#include "cpu/qmc_noise.hpp"
#include "utils.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace mc::cpu;

namespace {

bool bits_equal(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// Fuente de prueba: el camino j entrega el vector unitario e_(j mod D).
class UnitNoise final : public NoiseSource {
public:
    explicit UnitNoise(int D) : D_(D) {}
    int dim() const override { return D_; }
    std::unique_ptr<NoiseStream> open(uint64_t first, uint64_t) const override {
        struct S final : NoiseStream {
            int D; uint64_t next;
            void fill(int n, double* Z, int ld) override {
                for (int p = 0; p < n; p++, next++) {
                    for (int d = 0; d < D; d++) Z[(size_t)d * ld + p] = 0.0;
                    Z[(size_t)(next % D) * ld + p] = 1.0;
                }
            }
        };
        auto s = std::make_unique<S>(); s->D = D_; s->next = first;
        return s;
    }
private:
    int D_;
};

// Lee `count` caminos de una fuente con bloques de tamaño `block` (ld = block)
std::vector<std::vector<double>> read_paths(const NoiseSource& src, uint64_t first, int count, int block) {
    std::vector<std::vector<double>> out;
    auto st = src.open(first, (uint64_t)count);
    std::vector<double> Z((size_t)src.dim() * block);
    for (int done = 0; done < count; done += block) {
        const int n = std::min(block, count - done);
        st->fill(n, Z.data(), block);
        for (int p = 0; p < n; p++) {
            std::vector<double> v(src.dim());
            for (int d = 0; d < src.dim(); d++) v[d] = Z[(size_t)d * block + p];
            out.push_back(std::move(v));
        }
    }
    return out;
}

} // namespace

TEST_SUITE("fast") {

TEST_CASE("SobolNoise == Phi^-1 de la versión ingenua, con cualquier alineación y tamaño de bloque") {
    const int D = 24; const uint32_t salt = 99;
    ScrambledSobol sob(salt, D);
    SobolNoise src(sob, 0.5);
    for (uint64_t first : {uint64_t(0), uint64_t(5), uint64_t(1000), uint64_t(1) << 20}) {
        for (int block : {1, 3, 8, 64}) {
            auto paths = read_paths(src, first, 70, block);
            for (int p : {0, 1, 7, 8, 63, 69})
                for (int d : {0, 1, 11, 23}) {
                    double expect = 0.5 * norm_inv_cdf(sobol_word_to_u(scrambled_sobol_naive(salt, d, first + p)));
                    REQUIRE(bits_equal(paths[p][d], expect));
                }
        }
    }
}

TEST_CASE("Brownian Bridge (flujo por bloques) == bb_apply sobre vectores unitarios") {
    for (int N : {2, 8, 64}) {
        const double T = 1.5;
        const BBData& bb = bb_precompute(N, T);
        UnitNoise unit(N);
        BrownianBridgeNoise src(unit, bb);
        auto paths = read_paths(src, 0, N + 5, 7);       // incluye vueltas (j mod N) y bloques parciales
        for (int j = 0; j < N + 5; j++) {
            std::vector<double> z(N, 0.0), dw(N);
            z[j % N] = 1.0;
            bb_apply(bb, z.data(), dw.data(), 1);
            for (int i = 0; i < N; i++) REQUIRE(paths[j][i] == doctest::Approx(dw[i]).epsilon(1e-14).scale(1e-3));
        }
    }
}

TEST_CASE("PCA (GEMM por bloques) == columnas de M_pca sobre vectores unitarios") {
    for (int m : {4, 33, 64, 100}) {
        const PCAData& pca = pca_compute(m, 1.0);
        UnitNoise unit(m);
        PcaNoise src(unit, pca);
        auto paths = read_paths(src, 0, m + 3, 5);
        for (int j = 0; j < m + 3; j++)
            for (int i = 0; i < m; i++)
                REQUIRE(paths[j][i] == doctest::Approx(pca.M_pca[(size_t)i + (size_t)(j % m) * m]).epsilon(1e-14).scale(1e-3));
    }
}

TEST_CASE("Los flujos transformados entregan los mismos caminos con distinta alineación de apertura") {
    const BBData& bb = bb_precompute(16, 1.0);
    ScrambledSobol sob(5, 16);
    SobolNoise base(sob, 1.0);
    BrownianBridgeNoise src(base, bb);
    auto a = read_paths(src, 0, 150, 8);
    auto b = read_paths(src, 70, 80, 13);      // empieza en medio de un "bloque de transformación"
    for (int p = 0; p < 80; p++)
        for (int i = 0; i < 16; i++) REQUIRE(bits_equal(a[70 + p][i], b[p][i]));
}

TEST_CASE("run_qmc: validaciones de entrada") {
    GBMParams g; HestonParams h; h.compute_cholesky();
    PayoffVariant pv = European{100.0, 0.05, 1.0};
    CHECK_THROWS_AS(run_qmc(g, pv, 0.1, 24, QMCConfig{}, NoiseMode::BrownianBridge), std::invalid_argument);
    CHECK_THROWS_AS(run_qmc(h, pv, 0.1, 16, QMCConfig{}, NoiseMode::PCA), std::invalid_argument);
    QMCConfig c1; c1.R = 1;
    CHECK_THROWS_AS(run_qmc(g, pv, 0.1, 16, c1), std::invalid_argument);
    CHECK_THROWS_AS(run_qmc(g, pv, -1.0, 16), std::invalid_argument);
}

TEST_CASE("run_qmc: progreso por duplicaciones") {
    struct Rec : ProgressSink {
        std::vector<Snapshot> s; std::vector<std::vector<double>> reps;
        void on_snapshot(const Snapshot& x) override {
            s.push_back(x);
            reps.emplace_back(x.replica_means, x.replica_means + x.n_replicas);
        }
    } rec;
    GBMParams g; CpuOptions o; o.sink = &rec;
    QMCConfig c; c.R = 8; c.n0 = 256;
    MCResult r = run_qmc(g, European{100.0, 0.05, 1.0}, 0.01, 16, c, NoiseMode::Raw, o);
    REQUIRE(rec.s.size() >= 3);
    CHECK(rec.s.back().stage == Stage::Done);
    CHECK(rec.s.back().is_final);
    long long prev = 0;
    for (size_t i = 0; i + 1 < rec.s.size(); i++) {
        CHECK(rec.s[i].stage == Stage::Doubling);
        CHECK(rec.s[i].n_replicas == 8);
        CHECK(rec.s[i].n_per_replica == 256LL << i);       // potencias de 2 desde n0
        CHECK(rec.s[i].n_done > prev);
        prev = rec.s[i].n_done;
    }
    CHECK(rec.s[rec.s.size() - 2].mean == r.price);
}

TEST_CASE("run_qmc: max_seconds y cancelación") {
    GBMParams g; PayoffVariant pv = European{100.0, 0.05, 1.0};
    {
        CpuOptions o; o.max_seconds = 0.3; RunInfo info; o.info = &info;
        MCResult r = run_qmc(g, pv, 1e-6, 16, QMCConfig{}, NoiseMode::Raw, o);
        CHECK(info.truncated);
        CHECK(r.time_s < 20.0);
        CHECK(std::isfinite(r.price));
    }
    {
        struct C : ProgressSink {
            int n = 0;
            void on_snapshot(const Snapshot& s) override { if (s.stage == Stage::Doubling) n++; }
            bool should_cancel() const override { return n >= 2; }
        } c;
        CpuOptions o; o.sink = &c; RunInfo info; o.info = &info;
        run_qmc(g, pv, 1e-6, 16, QMCConfig{}, NoiseMode::Raw, o);
        CHECK(info.cancelled);
    }
}

TEST_CASE("run_qmc: D > D_MAX_SOBOL usa normales pseudoaleatorias") {
    HestonParams h; h.compute_cholesky();                      // dim 2 => D = 2*n_steps
    QMCConfig c; c.R = 4; c.n0 = 64; c.max_doublings = 1;
    MCResult r = run_qmc(h, European{0.0, 0.0, 1.0}, 1e9, 10001, c, NoiseMode::Raw);
    CHECK(r.n_samples == 4 * 64);
    CHECK(std::abs(r.price - 100.0 * std::pow(1.0 + 0.05 / 10001, 10001)) < 30.0);
}

} // TEST_SUITE

TEST_SUITE("stat") {

TEST_CASE("QMC Raw, BB y PCA concuerdan con Black-Scholes (Euler, 64 pasos)") {
    GBMParams g; PayoffVariant pv = European{100.0, 0.05, 1.0};
    const double bs = bs_call(g.S0, 100.0, g.T, 0.05, g.sigma);
    for (NoiseMode mode : {NoiseMode::Raw, NoiseMode::BrownianBridge, NoiseMode::PCA}) {
        MCResult r = run_qmc(g, pv, 0.005, 64, QMCConfig{}, mode);
        CHECK_MESSAGE(std::abs(r.price - bs) < 5.0 * r.std_error + 0.01,
                      "modo " << (int)mode << " precio " << r.price << " se " << r.std_error);
    }
}

TEST_CASE("QMC con BB reduce la varianza respecto a MC al mismo nº de caminos") {
    GBMParams g; PayoffVariant pv = European{100.0, 0.05, 1.0};
    QMCConfig c; c.R = 16; c.n0 = 1024; c.max_doublings = 4;     // 16 * 16384 = 262144 caminos
    MCResult q = run_qmc(g, pv, 1e-9, 64, c, NoiseMode::BrownianBridge);
    auto [m, v] = run_mc_fixed(g, pv, 64, q.n_samples, 7u);
    const double se_mc = std::sqrt(v);
    CHECK(q.std_error < se_mc / 5.0);
}

TEST_CASE("QMC: el error estándar de réplicas es honesto (comparado con la dispersión entre semillas)") {
    GBMParams g; PayoffVariant pv = European{100.0, 0.05, 1.0};
    std::vector<double> prices; double se_sum = 0;
    for (unsigned s = 0; s < 20; s++) {
        QMCConfig c; c.R = 16; c.n0 = 512; c.max_doublings = 3; c.seed = 100 + 1000 * s;
        MCResult r = run_qmc(g, pv, 1e-9, 16, c, NoiseMode::BrownianBridge);
        prices.push_back(r.price); se_sum += r.std_error;
    }
    double m = 0; for (double v : prices) m += v; m /= prices.size();
    double var = 0; for (double v : prices) var += (v - m) * (v - m); var /= prices.size() - 1;
    const double ratio = std::sqrt(var) / (se_sum / prices.size());
    CHECK(ratio > 0.5);
    CHECK(ratio < 1.8);
}

TEST_CASE("QMC Raw con Heston y con cesta concuerda con MC") {
    {
        HestonParams h; h.compute_cholesky(); PayoffVariant pv = European{100.0, 0.05, 1.0};
        MCResult q = run_qmc(h, pv, 0.01, 32, QMCConfig{}, NoiseMode::Raw);
        auto [m, v] = run_mc_fixed(h, pv, 32, 3000000, 5u);
        CHECK(std::abs(q.price - m) < 5.0 * std::sqrt(q.std_error * q.std_error + v));
    }
    {
        MultiDupireParams b; b.n = 8; b.uncorrelated = true; b.S0.assign(8, 100.0);
        PayoffVariant pv = Basket{100.0, 0.05, 1.0, 8};
        MCResult q = run_qmc(b, pv, 0.02, 4, QMCConfig{}, NoiseMode::Raw);
        auto [m, v] = run_mc_fixed(b, pv, 4, 1000000, 5u);
        CHECK(std::abs(q.price - m) < 5.0 * std::sqrt(q.std_error * q.std_error + v));
    }
}

} // TEST_SUITE

TEST_SUITE("determinism") {

TEST_CASE("run_qmc idéntico bit a bit con 1, 3, 8 y 16 hilos (Raw, BB, PCA)") {
    GBMParams g; PayoffVariant pv = Asian{100.0};
    QMCConfig c; c.R = 8; c.n0 = 512; c.max_doublings = 4;
    for (NoiseMode mode : {NoiseMode::Raw, NoiseMode::BrownianBridge, NoiseMode::PCA}) {
        CpuOptions o1; o1.threads = 1;
        MCResult ref = run_qmc(g, pv, 1e-9, 32, c, mode, o1);
        for (int t : {3, 8, 16}) {
            CpuOptions o; o.threads = t;
            MCResult r = run_qmc(g, pv, 1e-9, 32, c, mode, o);
            CHECK(bits_equal(r.price, ref.price));
            CHECK(bits_equal(r.std_error, ref.std_error));
            CHECK(r.n_samples == ref.n_samples);
        }
    }
}

} // TEST_SUITE

TEST_SUITE("fast") {

// Las transformaciones BB y PCA (y la inversa de la normal que las alimenta) tienen versión AVX2:
// todo el flujo Sobol -> normales -> transformación debe dar los mismos bits con y sin ella.
TEST_CASE("QMC: Sobol, Brownian Bridge y PCA identicos bit a bit con y sin AVX2") {
    if (simd_level_available() != SimdLevel::Avx2) { MESSAGE("sin AVX2: nada que comparar"); return; }
    for (int D : {8, 64, 100}) {
        const double T = 1.0, sqrt_h = std::sqrt(T / D);
        ScrambledSobol sob(1234u, D);
        const BBData& bb = bb_precompute(D, T);
        const PCAData& pca = pca_compute(D, T);
        SobolNoise raw(sob, sqrt_h), unit(sob, 1.0);
        BrownianBridgeNoise bbn(unit, bb);
        PcaNoise pcan(unit, pca);
        const NoiseSource* sources[] = {&raw, &bbn, &pcan};
        for (const NoiseSource* src : sources) {
            set_simd_level(SimdLevel::Scalar);
            const auto ref = read_paths(*src, 5, 150, 8);     // 150 caminos: más de dos bloques de transformación
            set_simd_level(SimdLevel::Avx2);
            const auto got = read_paths(*src, 5, 150, 8);
            REQUIRE(ref.size() == got.size());
            bool same = true;
            for (size_t i = 0; i < ref.size(); i++)
                if (std::memcmp(ref[i].data(), got[i].data(), sizeof(double) * ref[i].size()) != 0) same = false;
            CHECK_MESSAGE(same, "D=" << D << " fuente=" << (int)(src == &raw ? 0 : src == &bbn ? 1 : 2));
        }
    }
    set_simd_level(simd_level_available());
}

} // TEST_SUITE
