#include <cstdlib>
#include "../methods_cuda.cuh"
#include <cmath>
#include <algorithm>
#include <vector>
#include <functional>
#include <string>

static int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

static int steps_for_eps(double eps, double c1, double T) {
    int n = next_pow2(std::max(1, (int)std::ceil(std::sqrt(2.0) * T / (eps * c1))));
    return std::min(n, 1 << 11);
}

int main(int argc, char** argv) {
    // ── Parámetros ───────────────────────────────────────────────────────────
    HestonParams hes;
    hes.S0    = 100.0;
    hes.mu    = 0.05;
    hes.kappa = 2.0;
    hes.theta = 0.04;
    hes.xi    = 0.5;
    hes.rho   = -0.9;
    hes.v0    = 0.04;
    hes.T     = 1.0;
    hes.compute_cholesky();

    const double K   = 100.0;
    const double r   = hes.mu;
    const int    M   = 2;
    const int    L_MAX = 10;

    ModelVariant  mv = hes;
    PayoffVariant pv = European{K, r, hes.T};

    // ── Referencia (MC 500k trayectorias con n_steps=256) ────────────────────
    double price_ref = run_mc_fixed(mv, pv, 256, 500000, 99u).first;

    // ── c1 por Richardson ────────────────────────────────────────────────────
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return run_mc_fixed(mv, pv, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, hes.T, 4, 20000);

    // ── Configuración ────────────────────────────────────────────────────────
    // Heston d=2: BB/PCA no prácticos para Sobol 2D → solo Raw
    MCConfig   mc_cfg;
    MLMCConfig ml_cfg;  ml_cfg.M = M;  ml_cfg.max_L = L_MAX;
    QMCConfig  qmc_cfg;

    // ── CORRECCIÓN AL DISEÑO DE TAREA 2: barrido de eps con corte por método ──
    // El eps por defecto de este ejemplo era mayor (0.10, d=2 Sobol sigue
    // siendo efectivo a esa escala); se usa como tope superior del barrido
    // en vez del 0.05 general si no se pasa argv[1].
    double eps_finest = (argc > 1) ? atof(argv[1]) : 0.0001;
    std::vector<double> eps_list = eps_scale_125(eps_finest);
    // Ejemplo original arrancaba en 0.10 en vez de 0.05: se antepone ese nivel
    // extra si la escala estandar no llega tan grueso.
    if (eps_list.empty() || eps_list.front() < 0.10) eps_list.insert(eps_list.begin(), 0.10);

    // ── Métodos (4: MC, MLMC, QMC Raw, MLQMC Raw) ────────────────────────────
    std::vector<SweepMethod> methods;

    methods.push_back({"MC", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, hes.T);
        MCConfig c = mc_cfg; c.seed += so;
        return run_mc_cuda(mv, pv, eps, ns, c);
    }});
    methods.push_back({"MLMC", [&](unsigned so, double eps) {
        MLMCConfig c = ml_cfg; c.seed += so;
        return run_mlmc_cuda(mv, pv, eps, c);
    }});
    methods.push_back({"QMC Raw", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, hes.T);
        QMCConfig c = qmc_cfg; c.seed += so;
        return run_qmc_cuda(mv, pv, eps, ns, c, NoiseMode::Raw);
    }});
    methods.push_back({"MLQMC Raw", [&](unsigned so, double eps) {
        QMCConfig c = qmc_cfg; c.seed += so;
        return run_mlqmc_cuda(mv, pv, eps, ml_cfg, c, NoiseMode::Raw);
    }});

    run_precision_sweep("ejemplo05_heston", methods, price_ref, eps_list);

    return 0;
}
