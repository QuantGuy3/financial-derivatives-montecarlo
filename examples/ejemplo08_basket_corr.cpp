#include <cstdlib>
#include "../methods_cuda.cuh"
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>

// Cholesky triangular inferior (Banachiewicz) de C PSD (in-place → L)
static void cholesky(std::vector<double>& L, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = L[i * n + j];
            for (int k = 0; k < j; ++k) s -= L[i * n + k] * L[j * n + k];
            if (i == j)
                L[i * n + i] = std::sqrt(std::max(s, 0.0));
            else
                L[i * n + j] = (L[j * n + j] > 1e-15) ? s / L[j * n + j] : 0.0;
        }
        // Zero upper triangle
        for (int j = i + 1; j < n; ++j) L[i * n + j] = 0.0;
    }
}

int main(int argc, char** argv) {
    // ── Parámetros ───────────────────────────────────────────────────────────
    const int    N_ASSETS = 100;
    double eps = (argc > 1) ? atof(argv[1]) : 0.05;
    const int    M    = 2;
    const int    L_MAX = 10;

    // ── Matriz de correlación aleatoria (seed=0): C = A*A^T/n, regularizada ──
    std::mt19937_64 rng(0u);
    std::normal_distribution<double> nd(0.0, 1.0);

    // A[N_ASSETS × N_ASSETS] ~ N(0,1)
    std::vector<double> A(N_ASSETS * N_ASSETS);
    for (auto& v : A) v = nd(rng);

    // C = A * A^T / N_ASSETS  (PSD por construcción)
    std::vector<double> C(N_ASSETS * N_ASSETS, 0.0);
    for (int i = 0; i < N_ASSETS; ++i)
        for (int j = 0; j < N_ASSETS; ++j)
            for (int k = 0; k < N_ASSETS; ++k)
                C[i * N_ASSETS + j] += A[i * N_ASSETS + k] * A[j * N_ASSETS + k];
    for (auto& v : C) v /= N_ASSETS;

    // Regularizar: sumar 0.01 * I para garantizar PD
    for (int i = 0; i < N_ASSETS; ++i) C[i * N_ASSETS + i] += 0.01;

    // Normalizar a matriz de correlación: rho[i,j] = C[i,j] / sqrt(C[i,i]*C[j,j])
    std::vector<double> diag(N_ASSETS);
    for (int i = 0; i < N_ASSETS; ++i) diag[i] = std::sqrt(C[i * N_ASSETS + i]);
    for (int i = 0; i < N_ASSETS; ++i)
        for (int j = 0; j < N_ASSETS; ++j)
            C[i * N_ASSETS + j] /= (diag[i] * diag[j]);

    // Cholesky L de la matriz de correlación
    std::vector<double> L_chol = C;
    cholesky(L_chol, N_ASSETS);

    // ── Modelo ───────────────────────────────────────────────────────────────
    MultiDupireParams basket;
    basket.n            = N_ASSETS;
    basket.mu           = 0.05;
    basket.sigma0       = 0.20;
    basket.alpha        = 0.5;
    basket.beta_d       = 0.7;
    basket.T            = 1.0;
    basket.uncorrelated = false;
    basket.S0.assign(N_ASSETS, 100.0);
    basket.L = L_chol;               // Cholesky en fila principal

    const double K = 100.0;
    const double r = basket.mu;

    ModelVariant  mv = basket;
    PayoffVariant pv = Basket{K, r, basket.T, N_ASSETS};

    // ── Referencia (MC 5k, n_steps limitado a D_MAX/n = 200) ────────────────
    const int N_REF_STEPS = std::min(200, D_MAX_SOBOL / N_ASSETS);
    double price_ref = run_mc_fixed(mv, pv, N_REF_STEPS, 5000, 99u).first;

    // ── c1 por Richardson ────────────────────────────────────────────────────
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return run_mc_fixed(mv, pv, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, basket.T, 4, 10000);
    int n_steps = std::max(1, (int)std::ceil(std::sqrt(2.0) * basket.T / (eps * c1)));
    n_steps = std::min(n_steps, N_REF_STEPS);

    // ── Configuración ────────────────────────────────────────────────────────
    MCConfig   mc_cfg;
    MLMCConfig ml_cfg;  ml_cfg.M = M;  ml_cfg.max_L = L_MAX;
    QMCConfig  qmc_cfg;

    // ── Métodos ──────────────────────────────────────────────────────────────
    std::vector<TableRow> rows;
    auto add = [&](const char* name, MCResult r) {
        rows.push_back({name, r.price, r.std_error, r.n_samples, r.time_s});
    };

    // MLMC/MLQMC no soportan MultiDupire (cestas): no hay kernel MLMC multi-activo.
    add("MC",
        run_mc_cuda(mv, pv, eps, n_steps, mc_cfg));
    add("QMC Raw",
        run_qmc_cuda(mv, pv, eps, n_steps, qmc_cfg, NoiseMode::Raw));

    // ── Tabla ─────────────────────────────────────────────────────────────────
    print_table(rows, price_ref, eps,
                "Ejemplo 08: Basket Dupire Correlated (n=100)");

    return 0;
}
