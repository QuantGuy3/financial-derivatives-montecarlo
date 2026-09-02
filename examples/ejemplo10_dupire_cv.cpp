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
    DupireLocalParams dup;
    dup.S0     = 100.0;
    dup.mu     = 0.05;
    dup.sigma0 = 0.20;
    dup.alpha  = 0.5;
    dup.beta_d = 0.7;
    dup.T      = 1.0;

    // Variable de control: GBM con sigma efectiva = sigma0 = 0.20
    GBMParams gbm_ctrl;
    gbm_ctrl.S0    = dup.S0;
    gbm_ctrl.mu    = dup.mu;
    gbm_ctrl.sigma = dup.sigma0;
    gbm_ctrl.T     = dup.T;

    const double K   = 100.0;
    const double r   = dup.mu;

    ModelVariant  mv_main = dup;
    ModelVariant  mv_ctrl = gbm_ctrl;
    PayoffVariant pv      = European{K, r, dup.T};

    // ── c1 por Richardson (basado en el modelo Dupire principal) ─────────────
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return run_mc_fixed(mv_main, pv, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, dup.T, 4, 50000);

    // ── CORRECCIÓN AL DISEÑO DE TAREA 2: barrido de eps con corte por método ──
    double eps_finest = (argc > 1) ? atof(argv[1]) : 0.0001;
    std::vector<double> eps_list = eps_scale_125(eps_finest);

    // ── Referencia: MC 500k para European Dupire ──────────────────────────────
    double price_ref = run_mc_fixed(mv_main, pv, 256, 500000, 99u).first;

    // ── E[ctrl payoff] analítico bajo GBM (Black-Scholes, no depende de eps) ──
    double E_ctrl = bs_call(gbm_ctrl.S0, K, gbm_ctrl.T, r, gbm_ctrl.sigma);

    // ── Configuración ────────────────────────────────────────────────────────
    MCConfig  mc_cfg;
    QMCConfig qmc_cfg;
    MLMCConfig ml_cfg;

    // beta se recalcula por nivel de eps (via n_steps, mismo Z para Dupire y GBM).
    auto beta_for_eps = [&](double eps) {
        int ns = steps_for_eps(eps, c1, dup.T);
        CVPilot pilot = cv_pilot(mv_main, mv_ctrl, pv, pv, E_ctrl, ns, 50000);
        return pilot.beta;
    };

    // ── Métodos: MC puro, MC con variable de control, QMC+CV, MLMC ──────────
    std::vector<SweepMethod> methods;

    methods.push_back({"MC (Dupire)", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, dup.T);
        MCConfig c = mc_cfg; c.seed += so;
        return run_mc_cuda(mv_main, pv, eps, ns, c);
    }});
    methods.push_back({"MC + CV (GBM)", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, dup.T);
        double beta = beta_for_eps(eps);
        MCConfig c = mc_cfg; c.seed += so;
        return run_mc_cv_cuda(mv_main, mv_ctrl, pv, pv, E_ctrl, beta, eps, ns, c);
    }});
    methods.push_back({"QMC + CV (GBM)", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, dup.T);
        double beta = beta_for_eps(eps);
        QMCConfig c = qmc_cfg; c.seed += so;
        return run_qmc_cv_cuda(mv_main, mv_ctrl, pv, pv, E_ctrl, beta, eps, ns, c);
    }});
    methods.push_back({"MLMC", [&](unsigned so, double eps) {
        MLMCConfig c = ml_cfg; c.seed += so;
        return run_mlmc_cuda(mv_main, pv, eps, c);
    }});

    run_precision_sweep("ejemplo10_dupire_cv", methods, price_ref, eps_list);

    return 0;
}
