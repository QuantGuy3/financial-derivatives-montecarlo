#include <cstdlib>
#include "../methods_cuda.cuh"
#include "../referencias.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <functional>
#include <string>

// Ejemplo 14: call europea bajo SABR (β = 0.7). La fórmula de Hagan es solo
// una aproximación asintótica, así que la referencia del barrido es un MC de
// alta resolución (8 × 500k trayectorias, 256 pasos); el valor de Hagan se
// imprime como información (su error suele ser de unos céntimos).

static int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

static int steps_for_eps(double eps, double c1, double T) {
    int n = next_pow2(std::max(1, (int)std::ceil(std::sqrt(2.0) * T / (eps * c1))));
    return std::min(n, 1 << 11);
}

int main(int argc, char** argv) {
    // ── Parámetros ───────────────────────────────────────────────────────────
    SABRParams sab;
    sab.S0    = 100.0;
    sab.mu    = 0.05;
    sab.alpha = 0.20;   // vol en el forward inicial
    sab.beta  = 0.7;
    sab.nu    = 0.4;
    sab.rho   = -0.5;
    sab.T     = 1.0;

    const double K   = 100.0;
    const double r   = sab.mu;
    const int    M   = 2;
    const int    L_MAX = 10;

    ModelVariant  mv = sab;
    PayoffVariant pv = European{K, r, sab.T};

    // ── Referencia (MC de alta resolución) ───────────────────────────────────
    double suma = 0.0;
    const int LOTES = 8;
    for (int b = 0; b < LOTES; ++b) suma += run_mc_fixed(mv, pv, 256, 500000, 99u + 17u * b).first;
    double price_ref = suma / LOTES;
    double hagan = ref::sabr_precio(true, sab.S0, K, sab.T, r, 0.0, sab.alpha, sab.beta, sab.rho, sab.nu);
    std::printf("Referencia SABR: MC(4M, 256 pasos) = %.6f ; aproximación de Hagan = %.6f\n", price_ref, hagan);

    // ── c1 por Richardson ────────────────────────────────────────────────────
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return run_mc_fixed(mv, pv, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, sab.T, 4, 20000);

    // ── Configuración ────────────────────────────────────────────────────────
    // 2 normales por paso (precio, volatilidad): solo Raw
    MCConfig   mc_cfg;
    MLMCConfig ml_cfg;  ml_cfg.M = M;  ml_cfg.max_L = L_MAX;
    QMCConfig  qmc_cfg;

    double eps_finest = (argc > 1) ? atof(argv[1]) : 0.0001;
    std::vector<double> eps_list = eps_scale_125(eps_finest);
    if (eps_list.empty() || eps_list.front() < 0.10) eps_list.insert(eps_list.begin(), 0.10);

    std::vector<SweepMethod> methods;
    methods.push_back({"MC", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, sab.T);
        MCConfig c = mc_cfg; c.seed += so;
        return run_mc_cuda(mv, pv, eps, ns, c);
    }});
    methods.push_back({"MLMC", [&](unsigned so, double eps) {
        MLMCConfig c = ml_cfg; c.seed += so;
        return run_mlmc_cuda(mv, pv, eps, c);
    }});
    methods.push_back({"QMC Raw", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, sab.T);
        QMCConfig c = qmc_cfg; c.seed += so;
        return run_qmc_cuda(mv, pv, eps, ns, c, NoiseMode::Raw);
    }});
    methods.push_back({"MLQMC Raw", [&](unsigned so, double eps) {
        QMCConfig c = qmc_cfg; c.seed += so;
        return run_mlqmc_cuda(mv, pv, eps, ml_cfg, c, NoiseMode::Raw);
    }});

    run_precision_sweep("ejemplo14_sabr", methods, price_ref, eps_list);
    return 0;
}
