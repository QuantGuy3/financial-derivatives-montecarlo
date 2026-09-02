#include <cstdlib>
#include "../methods_cuda.cuh"
#include <cmath>
#include <algorithm>
#include <vector>
#include <functional>
#include <string>

int main(int argc, char** argv) {
    // ── Parámetros ───────────────────────────────────────────────────────────
    const int    N_ASSETS = 1000;
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

    // n_steps limitado por la dimensión de Sobol (n_assets * n_steps <= D_MAX_SOBOL):
    // no se usa next_pow2 aqui, a diferencia de los ejemplos GBM de un activo.
    auto steps_for_eps = [&](double eps) {
        int n = std::max(1, (int)std::ceil(std::sqrt(2.0) * basket.T / (eps * c1)));
        return std::min(n, N_REF_STEPS);
    };

    // ── Configuración ────────────────────────────────────────────────────────
    // Cesta grande: se usa modo Raw para todas las variantes QMC. MLMC/MLQMC
    // no soportan MultiDupire (cestas): no hay kernel MLMC multi-activo.
    MCConfig   mc_cfg;
    QMCConfig  qmc_cfg;

    // ── CORRECCIÓN AL DISEÑO DE TAREA 2: barrido de eps con corte por método ──
    double eps_finest = (argc > 1) ? atof(argv[1]) : 0.0001;
    std::vector<double> eps_list = eps_scale_125(eps_finest);

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

    run_precision_sweep("ejemplo07_basket_uncorr", methods, price_ref, eps_list);

    return 0;
}
