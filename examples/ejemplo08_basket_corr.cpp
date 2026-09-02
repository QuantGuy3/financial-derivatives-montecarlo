#include <cstdlib>
#include "../methods_cuda.cuh"
#include <cmath>
#include <algorithm>
#include <vector>
#include <functional>
#include <string>
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
        for (int j = i + 1; j < n; ++j) L[i * n + j] = 0.0;
    }
}

int main(int argc, char** argv) {
    // ── Parámetros ───────────────────────────────────────────────────────────
    const int    N_ASSETS = 100;
    // ── Matriz de correlación aleatoria (seed=0): C = A*A^T/n, regularizada ──
    std::mt19937_64 rng(0u);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::vector<double> A(N_ASSETS * N_ASSETS);
    for (auto& v : A) v = nd(rng);

    std::vector<double> C(N_ASSETS * N_ASSETS, 0.0);
    for (int i = 0; i < N_ASSETS; ++i)
        for (int j = 0; j < N_ASSETS; ++j)
            for (int k = 0; k < N_ASSETS; ++k)
                C[i * N_ASSETS + j] += A[i * N_ASSETS + k] * A[j * N_ASSETS + k];
    for (auto& v : C) v /= N_ASSETS;

    for (int i = 0; i < N_ASSETS; ++i) C[i * N_ASSETS + i] += 0.01;

    std::vector<double> diag(N_ASSETS);
    for (int i = 0; i < N_ASSETS; ++i) diag[i] = std::sqrt(C[i * N_ASSETS + i]);
    for (int i = 0; i < N_ASSETS; ++i)
        for (int j = 0; j < N_ASSETS; ++j)
            C[i * N_ASSETS + j] /= (diag[i] * diag[j]);

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

    auto steps_for_eps = [&](double eps) {
        int n = std::max(1, (int)std::ceil(std::sqrt(2.0) * basket.T / (eps * c1)));
        return std::min(n, N_REF_STEPS);
    };

    // ── Configuración ────────────────────────────────────────────────────────
    MCConfig   mc_cfg;
    QMCConfig  qmc_cfg;

    // ── CORRECCIÓN AL DISEÑO DE TAREA 2: barrido de eps con corte por método ──
    double eps_finest = (argc > 1) ? atof(argv[1]) : 0.0001;
    std::vector<double> eps_list = eps_scale_125(eps_finest);

    // MLMC/MLQMC no soportan MultiDupire (cestas): no hay kernel MLMC multi-activo.
    std::vector<SweepMethod> methods;

    methods.push_back({"MC", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps);
        MCConfig c = mc_cfg; c.seed += so;
        return run_mc_cuda(mv, pv, eps, ns, c);
    }});
    methods.push_back({"QMC Raw", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps);
        QMCConfig c = qmc_cfg; c.seed += so;
        return run_qmc_cuda(mv, pv, eps, ns, c, NoiseMode::Raw);
    }});

    run_precision_sweep("ejemplo08_basket_corr", methods, price_ref, eps_list);

    return 0;
}
