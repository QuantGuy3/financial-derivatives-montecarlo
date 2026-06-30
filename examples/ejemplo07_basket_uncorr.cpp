#include <cstdlib>
#include "../methods_cuda.cuh"
#include <cmath>
#include <algorithm>
#include <vector>

int main(int argc, char** argv) {
    // ── Parámetros ───────────────────────────────────────────────────────────
    const int    N_ASSETS = 1000;
    double eps = (argc > 1) ? atof(argv[1]) : 0.01;
    const int    M    = 2;
    const int    L_MAX = 10;

    MultiDupireParams basket;
    basket.n           = N_ASSETS;
    basket.mu          = 0.05;
    basket.sigma0      = 0.20;
    basket.alpha       = 0.5;
    basket.beta_d      = 0.7;
    basket.T           = 1.0;
    basket.uncorrelated = true;       // rho = I, omite multiplicación Cholesky
    basket.S0.assign(N_ASSETS, 100.0);
    basket.L.clear();                 // vacío → correlación identidad

    const double K  = 100.0;
    const double r  = basket.mu;

    ModelVariant  mv = basket;
    PayoffVariant pv = Basket{K, r, basket.T, N_ASSETS};

    // ── Referencia (MC 5k, n = D_MAX/n_assets = 20) ─────────────────────────
    // Límite Sobol: D_MAX_SOBOL = 20000, dim = n_assets * n_steps ≤ 20000
    const int N_REF_STEPS = D_MAX_SOBOL / N_ASSETS;   // = 20
    double price_ref = run_mc_fixed(mv, pv, N_REF_STEPS, 5000, 99u).first;

    // ── c1 por Richardson ────────────────────────────────────────────────────
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return run_mc_fixed(mv, pv, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, basket.T, 4, 5000);
    int n_steps = std::max(1, (int)std::ceil(std::sqrt(2.0) * basket.T / (eps * c1)));
    n_steps = std::min(n_steps, N_REF_STEPS);          // limitado por dimensión Sobol

    // ── Configuración ────────────────────────────────────────────────────────
    // Cesta grande: se usa modo Raw para todas las variantes QMC
    MCConfig   mc_cfg;
    MLMCConfig ml_cfg;  ml_cfg.M = M;  ml_cfg.max_L = L_MAX;
    QMCConfig  qmc_cfg;

    // ── Métodos ──────────────────────────────────────────────────────────────
    std::vector<TableRow> rows;
    auto add = [&](const char* name, MCResult r) {
        rows.push_back({name, r.price, r.std_error, r.n_samples, r.time_s});
    };

    add("MC",
        run_mc_cuda(mv, pv, eps, n_steps, mc_cfg));
    add("MLMC",
        run_mlmc_cuda(mv, pv, eps, ml_cfg));
    add("QMC Raw",
        run_qmc_cuda(mv, pv, eps, n_steps, qmc_cfg, NoiseMode::Raw));
    add("MLQMC Raw",
        run_mlqmc_cuda(mv, pv, eps, ml_cfg, qmc_cfg, NoiseMode::Raw));

    // ── Tabla ─────────────────────────────────────────────────────────────────
    print_table(rows, price_ref, eps,
                "Ejemplo 07: Basket Dupire Uncorrelated (n=1000)");

    return 0;
}
