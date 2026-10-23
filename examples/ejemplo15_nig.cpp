#include <cstdlib>
#include "../methods_cuda.cuh"
#include "../referencias.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <functional>
#include <string>

// Ejemplo 15: call europea bajo NIG (proceso de Lévy de saltos puros, colas
// gruesas y asimetría negativa). El incremento NIG se simula de forma exacta
// (subordinador IG por Michael-Schucany-Haas + browniano condicionado), así que
// no hay sesgo de discretización. Referencia: Fourier (Lewis) con la función
// característica NIG.

static int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

static int steps_for_eps(double eps, double c1, double T) {
    int n = next_pow2(std::max(1, (int)std::ceil(std::sqrt(2.0) * T / (eps * c1))));
    return std::min(n, 1 << 11);
}

int main(int argc, char** argv) {
    // ── Parámetros ───────────────────────────────────────────────────────────
    NIGParams nig;
    nig.S0    = 100.0;
    nig.mu    = 0.05;
    nig.sigma = 0.20;
    nig.theta = -0.10;   // asimetría negativa
    nig.kappa = 0.20;    // varianza del subordinador (curtosis)
    nig.T     = 1.0;

    const double K   = 100.0;
    const double r   = nig.mu;
    const int    M   = 2;
    const int    L_MAX = 10;

    ModelVariant  mv = nig;
    PayoffVariant pv = European{K, r, nig.T};

    // ── Referencia (Fourier) ─────────────────────────────────────────────────
    double price_ref = ref::precio_fourier(true, ref::fc_nig(nig.sigma, nig.theta, nig.kappa, nig.T),
                                           nig.S0, K, nig.T, r, 0.0);
    std::printf("Referencia NIG (Fourier) = %.8f\n", price_ref);

    // ── c1 por Richardson ────────────────────────────────────────────────────
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return run_mc_fixed(mv, pv, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, nig.T, 4, 50000);

    // ── Configuración ────────────────────────────────────────────────────────
    // 3 normales por paso (browniano condicionado, normal y uniforme del IG): solo Raw
    MCConfig   mc_cfg;
    MLMCConfig ml_cfg;  ml_cfg.M = M;  ml_cfg.max_L = L_MAX;
    QMCConfig  qmc_cfg;

    double eps_finest = (argc > 1) ? atof(argv[1]) : 0.0001;
    std::vector<double> eps_list = eps_scale_125(eps_finest);

    std::vector<SweepMethod> methods;
    methods.push_back({"MC", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, nig.T);
        MCConfig c = mc_cfg; c.seed += so;
        return run_mc_cuda(mv, pv, eps, ns, c);
    }});
    methods.push_back({"MLMC", [&](unsigned so, double eps) {
        MLMCConfig c = ml_cfg; c.seed += so;
        return run_mlmc_cuda(mv, pv, eps, c);
    }});
    methods.push_back({"QMC Raw", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, nig.T);
        QMCConfig c = qmc_cfg; c.seed += so;
        return run_qmc_cuda(mv, pv, eps, ns, c, NoiseMode::Raw);
    }});
    methods.push_back({"MLQMC Raw", [&](unsigned so, double eps) {
        QMCConfig c = qmc_cfg; c.seed += so;
        return run_mlqmc_cuda(mv, pv, eps, ml_cfg, c, NoiseMode::Raw);
    }});

    run_precision_sweep("ejemplo15_nig", methods, price_ref, eps_list);
    return 0;
}
