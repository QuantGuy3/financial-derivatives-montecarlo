#include <cstdlib>
#include "../methods_cuda.cuh"
#include <cmath>
#include <algorithm>
#include <vector>

static int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

int main(int argc, char** argv) {
    // ── Parámetros ───────────────────────────────────────────────────────────
    GBMParams gbm;                    // S0=100, mu=0.05, sigma=0.20, T=1.0
    const double K   = 100.0;
    double eps = (argc > 1) ? atof(argv[1]) : 0.05;

    // Payoff principal: Asian aritmética
    // Variable de control: Asian geométrica (valor esperado analítico conocido)
    ModelVariant  mv       = gbm;
    PayoffVariant pv_main  = Asian{K};
    PayoffVariant pv_ctrl  = GeomAsian{K};

    // ── n_steps por Richardson (para MC, MC+CV y QMC+CV) ────────────────────
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return run_mc_fixed(mv, pv_main, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, gbm.T, 4, 50000);
    int n_steps = next_pow2(std::max(1, (int)std::ceil(std::sqrt(2.0) * gbm.T / (eps * c1))));
    n_steps = std::min(n_steps, 1 << 11);

    // ── Referencia: MC 500k con el mismo n_steps ─────────────────────────────
    double price_ref = run_mc_fixed(mv, pv_main, n_steps, 500000, 99u).first;

    // ── Valor esperado analítico de la variable de control ────────────────────
    // geom_asian_analytic usa la fórmula cerrada para la Asian geométrica
    // discreta con n_steps fechas de monitorización, consistente con el kernel
    double E_ctrl = geom_asian_analytic(gbm.S0, K, gbm.T, gbm.mu, gbm.sigma, n_steps);

    // ── Estima beta con piloto (50k trayectorias) ─────────────────────────────
    CVPilot pilot = cv_pilot(mv, mv, pv_main, pv_ctrl, E_ctrl, n_steps, 50000);
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

    add("MC",
        run_mc_cuda(mv, pv_main, eps, n_steps, mc_cfg));
    add("MC+CV(beta=0)",
        run_mc_cv_cuda(mv, mv, pv_main, pv_ctrl, E_ctrl, 0.0, eps, n_steps, mc_cfg));
    add("MC + CV (geom)",
        run_mc_cv_cuda(mv, mv, pv_main, pv_ctrl, E_ctrl, beta, eps, n_steps, mc_cfg));
    add("QMC + CV (geom)",
        run_qmc_cv_cuda(mv, mv, pv_main, pv_ctrl, E_ctrl, beta, eps, n_steps, qmc_cfg));
    add("MLMC",
        run_mlmc_cuda(mv, pv_main, eps, ml_cfg));

    // ── Tabla ─────────────────────────────────────────────────────────────────
    print_table(rows, price_ref, eps,
                "Ejemplo 09: Asian GBM + Control Variate (Geometric Asian)");

    return 0;
}
