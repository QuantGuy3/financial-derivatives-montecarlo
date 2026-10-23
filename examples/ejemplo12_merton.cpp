#include <cstdlib>
#include "../methods_cuda.cuh"
#include "../referencias.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <functional>
#include <string>

// Ejemplo 12: call europea bajo Merton (difusión con saltos lognormales).
// El paso logarítmico es exacto (no hay sesgo de discretización), así que el
// error frente a la referencia es solo estadístico. Referencia: serie de Poisson
// de precios Black-Scholes (Merton 1976), contrastada con Fourier.

static int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

static int steps_for_eps(double eps, double c1, double T) {
    int n = next_pow2(std::max(1, (int)std::ceil(std::sqrt(2.0) * T / (eps * c1))));
    return std::min(n, 1 << 11);
}

int main(int argc, char** argv) {
    // ── Parámetros ───────────────────────────────────────────────────────────
    MertonParams mer;
    mer.S0     = 100.0;
    mer.mu     = 0.05;   // r (sin dividendos)
    mer.sigma  = 0.15;
    mer.lambda = 0.5;    // un salto cada dos años en media
    mer.muJ    = -0.10;  // caída media ~9.5%
    mer.sigmaJ = 0.15;
    mer.T      = 1.0;

    const double K   = 100.0;
    const double r   = mer.mu;
    const int    M   = 2;
    const int    L_MAX = 10;

    ModelVariant  mv = mer;
    PayoffVariant pv = European{K, r, mer.T};

    // ── Referencia exacta ────────────────────────────────────────────────────
    double price_ref = ref::merton_precio(true, mer.S0, K, mer.T, r, 0.0, mer.sigma,
                                          mer.lambda, mer.muJ, mer.sigmaJ);
    double price_fourier = ref::precio_fourier(true,
        ref::fc_merton(mer.sigma, mer.lambda, mer.muJ, mer.sigmaJ, mer.T), mer.S0, K, mer.T, r, 0.0);
    std::printf("Referencia Merton: serie = %.8f, Fourier = %.8f\n", price_ref, price_fourier);

    // ── c1 por Richardson ────────────────────────────────────────────────────
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return run_mc_fixed(mv, pv, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, mer.T, 4, 50000);

    // ── Configuración ────────────────────────────────────────────────────────
    // 3 normales por paso (difusión, nº de saltos, tamaño): BB/PCA no aplican → Raw
    MCConfig   mc_cfg;
    MLMCConfig ml_cfg;  ml_cfg.M = M;  ml_cfg.max_L = L_MAX;
    QMCConfig  qmc_cfg;

    double eps_finest = (argc > 1) ? atof(argv[1]) : 0.0001;
    std::vector<double> eps_list = eps_scale_125(eps_finest);

    std::vector<SweepMethod> methods;
    methods.push_back({"MC", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, mer.T);
        MCConfig c = mc_cfg; c.seed += so;
        return run_mc_cuda(mv, pv, eps, ns, c);
    }});
    methods.push_back({"MLMC", [&](unsigned so, double eps) {
        MLMCConfig c = ml_cfg; c.seed += so;
        return run_mlmc_cuda(mv, pv, eps, c);
    }});
    methods.push_back({"QMC Raw", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, mer.T);
        QMCConfig c = qmc_cfg; c.seed += so;
        return run_qmc_cuda(mv, pv, eps, ns, c, NoiseMode::Raw);
    }});
    methods.push_back({"MLQMC Raw", [&](unsigned so, double eps) {
        QMCConfig c = qmc_cfg; c.seed += so;
        return run_mlqmc_cuda(mv, pv, eps, ml_cfg, c, NoiseMode::Raw);
    }});

    run_precision_sweep("ejemplo12_merton", methods, price_ref, eps_list);
    return 0;
}
