// Fórmulas analíticas, Brownian Bridge, PCA, caché memoizada y estadísticos corriendo.

#include "doctest.h"
#include "utils.hpp"

#include <cmath>
#include <numbers>
#include <thread>
#include <vector>

namespace {

// Matriz de incrementos B (N x N): columna j = dW cuando Z = e_j. Entonces dW = B Z y
// Cov(dW) = B B^T, que debe ser h*I (incrementos brownianos independientes de varianza h).
std::vector<double> bb_matrix(int N, double T) {
    const BBData& bb = bb_precompute(N, T);
    std::vector<double> B((size_t)N * N, 0.0); // column-major: B[i + j*N]
    std::vector<double> z(N), dw(N);
    for (int j = 0; j < N; j++) {
        std::fill(z.begin(), z.end(), 0.0);
        z[j] = 1.0;
        bb_apply(bb, z.data(), dw.data(), 1);
        for (int i = 0; i < N; i++) B[(size_t)j * N + i] = dw[i];
    }
    return B;
}

double max_dev_from_scaled_identity(const std::vector<double>& M, int N, double h) {
    double worst = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double acc = 0.0;
            for (int k = 0; k < N; k++) acc += M[(size_t)k * N + i] * M[(size_t)k * N + j];
            worst = std::max(worst, std::abs(acc - (i == j ? h : 0.0)));
        }
    return worst;
}

} // namespace

TEST_SUITE("fast") {

TEST_CASE("bs_call: valor de referencia y limites") {
    // S=100, K=100, T=1, r=5%, sigma=20%  ->  10.4505835722
    CHECK(bs_call(100, 100, 1.0, 0.05, 0.20) == doctest::Approx(10.4505835722).epsilon(1e-9));
    // Paridad call-put limite: sigma -> 0 da el valor intrinseco descontado
    CHECK(bs_call(120, 100, 1.0, 0.05, 1e-9) == doctest::Approx(120 - 100 * std::exp(-0.05)).epsilon(1e-9));
    CHECK(bs_call(100, 100, 0.0, 0.05, 0.2) == doctest::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("geom_asian_analytic: limite N=1 es una call europea con S_T lognormal") {
    // Con n=1, G = S_T: la formula debe coincidir con Black-Scholes sin descuento al
    // plantear mu como tasa libre de riesgo (precio = e^{rT} * BS call).
    double S0 = 100, K = 95, T = 1.0, mu = 0.03, sigma = 0.25;
    double bs_undisc = bs_call(S0, K, T, mu, sigma) * std::exp(mu * T);
    CHECK(geom_asian_analytic(S0, K, T, mu, sigma, 1) == doctest::Approx(bs_undisc).epsilon(1e-9));
}

TEST_CASE("RunningStats: Welford coincide con el calculo directo") {
    RunningStats rs;
    std::vector<double> xs = {1.5, -2.0, 3.25, 0.0, 7.5, 4.125};
    for (double x : xs) rs.update(x);
    double mean = 0; for (double x : xs) mean += x; mean /= xs.size();
    double var = 0; for (double x : xs) var += (x - mean) * (x - mean); var /= (xs.size() - 1);
    CHECK(rs.mean == doctest::Approx(mean).epsilon(1e-14));
    CHECK(rs.variance() == doctest::Approx(var).epsilon(1e-13));
    CHECK(rs.std_error() == doctest::Approx(std::sqrt(var / xs.size())).epsilon(1e-13));
}

TEST_CASE("Brownian Bridge: B*B^T = h*I (incrementos independientes)") {
    for (int N : {2, 4, 16, 64}) {
        double T = 1.5, h = T / N;
        auto B = bb_matrix(N, T);
        CHECK(max_dev_from_scaled_identity(B, N, h) < 1e-13);
    }
}

TEST_CASE("Brownian Bridge: primer normal fija el extremo terminal W_T") {
    const int N = 8; const double T = 2.0;
    const BBData& bb = bb_precompute(N, T);
    std::vector<double> z(N, 0.0), dw(N);
    z[0] = 1.0;
    bb_apply(bb, z.data(), dw.data(), 1);
    double WT = 0; for (double d : dw) WT += d;
    CHECK(WT == doctest::Approx(std::sqrt(T)).epsilon(1e-14));
}

TEST_CASE("PCA: M*M^T = h*I y varianza explicada decreciente") {
    for (int m : {2, 8, 32, 128}) {
        double T = 1.0, h = T / m;
        const PCAData& p = pca_compute(m, T);
        REQUIRE(p.m == m);
        REQUIRE(p.M_pca.size() == (size_t)m * m);
        // M_pca es column-major: elemento (i,k) en i + k*m; dW = Z * M^T => Cov = M M^T.
        CHECK(max_dev_from_scaled_identity(p.M_pca, m, h) < 1e-12);
        // La energia de la columna k (componente k) decrece con k para el MB.
        // Se reconstruye la covarianza acumulada: sum_i (cum M)_{ik}^2 = lambda_k.
        double prev = 1e300;
        for (int k = 0; k < m; k++) {
            double cum = 0.0, lambda = 0.0;
            for (int i = 0; i < m; i++) { cum += p.M_pca[(size_t)k * m + i]; lambda += cum * cum; }
            CHECK(lambda < prev);
            prev = lambda;
        }
    }
}

TEST_CASE("PCA: la tabla de senos coincide con la formula directa") {
    const int m = 64; const double T = 1.0;
    const PCAData& p = pca_compute(m, T);
    const double pi = std::numbers::pi, h = T / m;
    const double nrm = std::sqrt(4.0 / (2.0 * m + 1.0));
    double worst = 0.0;
    for (int k = 0; k < m; k++) {
        double sk = std::sin((2.0 * k + 1.0) * pi / (2.0 * (2.0 * m + 1.0)));
        double sq = std::sqrt(h) / (2.0 * std::abs(sk));
        double prev = 0.0;
        for (int i = 0; i < m; i++) {
            double cum = nrm * std::sin((2.0 * k + 1.0) * (i + 1) * pi / (2.0 * m + 1.0)) * sq;
            worst = std::max(worst, std::abs((cum - prev) - p.M_pca[(size_t)k * m + i]));
            prev = cum;
        }
    }
    CHECK(worst < 1e-12);
    // y la version float32 que sube la GPU es el redondeo de la de double
    for (size_t i = 0; i < p.M_pca.size(); i += 17)
        CHECK(p.M_pca_f32[i] == static_cast<float>(p.M_pca[i]));
}

TEST_CASE("bb_precompute / pca_compute: caché estable y segura entre hilos") {
    const BBData*  bb0  = &bb_precompute(32, 1.0);
    const PCAData* pca0 = &pca_compute(32, 1.0);
    std::vector<std::thread> ts;
    std::vector<int> ok(8, 0);
    for (int t = 0; t < 8; t++)
        ts.emplace_back([&, t] {
            bool good = true;
            for (int rep = 0; rep < 50; rep++) {
                good &= (&bb_precompute(32, 1.0) == bb0);
                good &= (&pca_compute(32, 1.0) == pca0);
                // otras claves en paralelo: inserciones concurrentes en el mapa
                const BBData& b = bb_precompute(8 << (t % 3), 1.0 + 0.001 * t);
                good &= (b.N == (8 << (t % 3)));
            }
            ok[t] = good ? 1 : 0;
        });
    for (auto& th : ts) th.join();
    for (int t = 0; t < 8; t++) CHECK(ok[t] == 1);
    // la referencia del principio sigue siendo valida tras inserciones (nodos estables)
    CHECK(bb0->N == 32);
    CHECK(pca0->m == 32);
}

TEST_CASE("estimar_c1_richardson: sesgo lineal conocido") {
    // sim(n) = P + c*(T/n): sesgo(h) = c*h -> c1 = 1/c devuelto por la rutina
    const double T = 1.0, P = 5.0, c = 0.8;
    auto sim = [&](int n_steps, long long, unsigned) { return P + c * (T / n_steps); };
    double c1 = estimar_c1_richardson(sim, T, 8, 1000);
    CHECK(c1 == doctest::Approx(1.0 / c).epsilon(1e-9));
}

} // TEST_SUITE
