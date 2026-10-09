#include <cstdlib>
#include "common.hpp"
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
    mc_examples::init(argc, argv);
    // ── Parámetros ───────────────────────────────────────────────────────────
    // sigma_loc(S,t) = sigma0 * exp(-alpha*t) * (S/S0)^(beta_d-1)
    //               = 0.20   * exp(-0.5*t)   * (S/100)^(-0.3)
    DupireLocalParams dup;
    dup.S0     = 100.0;
    dup.mu     = 0.05;
    dup.sigma0 = 0.20;
    dup.alpha  = 0.5;
    dup.beta_d = 0.7;   // exponente: beta_d - 1 = -0.3
    dup.T      = 1.0;

    const double K   = 100.0;
    const double r   = dup.mu;
    const int    M   = 2;
    const int    L_MAX = 10;

    ModelVariant  mv = dup;
    PayoffVariant pv = European{K, r, dup.T};

    // ── Referencia (MC 500k, n=256) ──────────────────────────────────────────
    double price_ref = mc::run_mc_fixed(mv, pv, 256, 500000, 99u).first;

    // ── c1 por Richardson ────────────────────────────────────────────────────
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return mc::run_mc_fixed(mv, pv, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, dup.T, 4, 50000);


    // ── Configuración ────────────────────────────────────────────────────────
    MCConfig   mc_cfg;
    MLMCConfig ml_cfg;  ml_cfg.M = M;  ml_cfg.max_L = L_MAX;
    QMCConfig  qmc_cfg;

    // ── CORRECCIÓN AL DISEÑO DE TAREA 2: barrido de eps con corte por método ──
    double eps_finest = (argc > 1) ? atof(argv[1]) : 0.0001;
    std::vector<double> eps_list = eps_scale_125(eps_finest);

    std::vector<SweepMethod> methods;

    methods.push_back({"MC", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, dup.T);
        MCConfig c = mc_cfg; c.seed += so;
        return mc::run_mc(mv, pv, eps, ns, c);
    }});
    methods.push_back({"MLMC", [&](unsigned so, double eps) {
        MLMCConfig c = ml_cfg; c.seed += so;
        return mc::run_mlmc(mv, pv, eps, c);
    }});
    methods.push_back({"QMC Raw", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, dup.T);
        QMCConfig c = qmc_cfg; c.seed += so;
        return mc::run_qmc(mv, pv, eps, ns, c, NoiseMode::Raw);
    }});
    methods.push_back({"QMC BB", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, dup.T);
        QMCConfig c = qmc_cfg; c.seed += so;
        return mc::run_qmc(mv, pv, eps, ns, c, NoiseMode::BrownianBridge);
    }});
    methods.push_back({"QMC PCA", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, dup.T);
        QMCConfig c = qmc_cfg; c.seed += so;
        return mc::run_qmc(mv, pv, eps, ns, c, NoiseMode::PCA);
    }});
    methods.push_back({"MLQMC Raw", [&](unsigned so, double eps) {
        QMCConfig c = qmc_cfg; c.seed += so;
        return mc::run_mlqmc(mv, pv, eps, ml_cfg, c, NoiseMode::Raw);
    }});
    methods.push_back({"MLQMC BB", [&](unsigned so, double eps) {
        QMCConfig c = qmc_cfg; c.seed += so;
        return mc::run_mlqmc(mv, pv, eps, ml_cfg, c, NoiseMode::BrownianBridge);
    }});
    methods.push_back({"MLQMC PCA", [&](unsigned so, double eps) {
        QMCConfig c = qmc_cfg; c.seed += so;
        return mc::run_mlqmc(mv, pv, eps, ml_cfg, c, NoiseMode::PCA);
    }});

    run_precision_sweep("ejemplo06_dupire", methods, price_ref, eps_list);

    return 0;
}
