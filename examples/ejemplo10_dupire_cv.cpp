#include <cstdlib>
#include "../methods_cuda.cuh"
#include <cmath>
#include <algorithm>
#include <vector>

static int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

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
    double eps = (argc > 1) ? atof(argv[1]) : 0.05;

    ModelVariant  mv_main = dup;
    ModelVariant  mv_ctrl = gbm_ctrl;
    PayoffVariant pv      = European{K, r, dup.T};

    // ── Referencia: MC 500k para European Dupire ──────────────────────────────
    double price_ref = run_mc_fixed(mv_main, pv, 256, 500000, 99u).first;

    // ── E[ctrl payoff] analítico bajo GBM (Black-Scholes) ────────────────────
    double E_ctrl = bs_call(gbm_ctrl.S0, K, gbm_ctrl.T, r, gbm_ctrl.sigma);

    // ── c1 + n_steps (basados en el modelo Dupire principal) ─────────────────
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return run_mc_fixed(mv_main, pv, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, dup.T, 4, 50000);
    int n_steps = next_pow2(std::max(1, (int)std::ceil(std::sqrt(2.0) * dup.T / (eps * c1))));
    n_steps = std::min(n_steps, 1 << 11);

    // ── Estima beta con piloto (50k trayectorias, mismo Z para Dupire y GBM) ──
    CVPilot pilot = cv_pilot(mv_main, mv_ctrl, pv, pv, E_ctrl, n_steps, 50000);
    double  beta  = pilot.beta;

    // ── Configuración ────────────────────────────────────────────────────────
    MCConfig  mc_cfg;
    QMCConfig qmc_cfg;
    MLMCConfig ml_cfg;

    // ── Métodos: MC puro, MC con variable de control, MLMC ───────────────────
    std::vector<TableRow> rows;
    auto add = [&](const char* name, MCResult r) {
        rows.push_back({name, r.price, r.std_error, r.n_samples, r.time_s});
    };

    add("MC (Dupire)",
        run_mc_cuda(mv_main, pv, eps, n_steps, mc_cfg));
    add("MC + CV (GBM)",
        run_mc_cv_cuda(mv_main, mv_ctrl, pv, pv, E_ctrl, beta, eps, n_steps, mc_cfg));
    add("QMC + CV (GBM)",
        run_qmc_cv_cuda(mv_main, mv_ctrl, pv, pv, E_ctrl, beta, eps, n_steps, qmc_cfg));
    add("MLMC",
        run_mlmc_cuda(mv_main, pv, eps, ml_cfg));

    // ── Tabla ─────────────────────────────────────────────────────────────────
    print_table(rows, price_ref, eps,
                "Ejemplo 10: Dupire Local-Vol + Control Variate (GBM)");

    return 0;
}
