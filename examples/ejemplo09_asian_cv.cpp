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
    GBMParams gbm;                    // S0=100, mu=0.05, sigma=0.20, T=1.0
    const double K   = 100.0;
    const int    M_LEVELS = 2;
    const int    L_MAX    = 10;

    // Payoff principal: Asian aritmética
    // Variable de control: Asian geométrica (valor esperado analítico conocido)
    ModelVariant  mv       = gbm;
    PayoffVariant pv_main  = Asian{K};
    PayoffVariant pv_ctrl  = GeomAsian{K};

    // ── c1 por Richardson (independiente de eps) ─────────────────────────────
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return mc::run_mc_fixed(mv, pv_main, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, gbm.T, 4, 50000);

    // ── CORRECCIÓN AL DISEÑO DE TAREA 2: barrido de eps con corte por método ──
    double eps_finest = (argc > 1) ? atof(argv[1]) : 0.0001;
    std::vector<double> eps_list = eps_scale_125(eps_finest);

    // ── Referencia: MC 500k, con n_steps del nivel de eps MAS FINO a intentar
    // (mayor resolucion posible dentro del barrido), fijo para toda la tabla.
    int n_steps_ref = steps_for_eps(eps_list.back(), c1, gbm.T);
    double price_ref = mc::run_mc_fixed(mv, pv_main, n_steps_ref, 500000, 99u).first;


    // ── Configuración ────────────────────────────────────────────────────────
    MCConfig  mc_cfg;
    QMCConfig qmc_cfg;
    MLMCConfig ml_cfg;  ml_cfg.M = M_LEVELS;  ml_cfg.max_L = L_MAX;

    // n_steps, E_ctrl (analitico) y beta (piloto 50k) dependen del nivel de
    // eps via n_steps: se recalculan en cada nivel del barrido (una vez por
    // llamada de metodo; el piloto es barato, 50k trayectorias).
    struct CVPrep { int n_steps; double E_ctrl; double beta; };
    auto prep_cv = [&](double eps) -> CVPrep {
        int ns = steps_for_eps(eps, c1, gbm.T);
        double Ec = geom_asian_analytic(gbm.S0, K, gbm.T, gbm.mu, gbm.sigma, ns);
        CVPilot pilot = mc::cv_pilot(mv, mv, pv_main, pv_ctrl, Ec, ns, 50000);
        return {ns, Ec, pilot.beta};
    };

    // ── Métodos: MC puro, MC/QMC/MLMC/MLQMC con variable de control ─────────
    std::vector<SweepMethod> methods;

    methods.push_back({"MC", [&](unsigned so, double eps) {
        int ns = steps_for_eps(eps, c1, gbm.T);
        MCConfig c = mc_cfg; c.seed += so;
        return mc::run_mc(mv, pv_main, eps, ns, c);
    }});
    methods.push_back({"MC+CV(beta=0)", [&](unsigned so, double eps) {
        auto p = prep_cv(eps);
        MCConfig c = mc_cfg; c.seed += so;
        return mc::run_mc_cv(mv, mv, pv_main, pv_ctrl, p.E_ctrl, 0.0, eps, p.n_steps, c);
    }});
    methods.push_back({"MC + CV (geom)", [&](unsigned so, double eps) {
        auto p = prep_cv(eps);
        MCConfig c = mc_cfg; c.seed += so;
        return mc::run_mc_cv(mv, mv, pv_main, pv_ctrl, p.E_ctrl, p.beta, eps, p.n_steps, c);
    }});
    // ── QMC + CV (geom): 3 construcciones de trayectoria ─────────────────────
    methods.push_back({"QMC+CV Raw", [&](unsigned so, double eps) {
        auto p = prep_cv(eps);
        QMCConfig c = qmc_cfg; c.seed += so;
        return mc::run_qmc_cv(mv, mv, pv_main, pv_ctrl, p.E_ctrl, p.beta, eps, p.n_steps, c,
                                NoiseMode::Raw);
    }});
    methods.push_back({"QMC+CV BB", [&](unsigned so, double eps) {
        auto p = prep_cv(eps);
        QMCConfig c = qmc_cfg; c.seed += so;
        return mc::run_qmc_cv(mv, mv, pv_main, pv_ctrl, p.E_ctrl, p.beta, eps, p.n_steps, c,
                                      NoiseMode::BrownianBridge);
    }});
    methods.push_back({"QMC+CV PCA", [&](unsigned so, double eps) {
        auto p = prep_cv(eps);
        QMCConfig c = qmc_cfg; c.seed += so;
        return mc::run_qmc_cv(mv, mv, pv_main, pv_ctrl, p.E_ctrl, p.beta, eps, p.n_steps, c,
                                      NoiseMode::PCA);
    }});
    methods.push_back({"MLMC", [&](unsigned so, double eps) {
        MLMCConfig c = ml_cfg; c.seed += so;
        return mc::run_mlmc(mv, pv_main, eps, c);
    }});
    // NUEVO (TAREA 1a): control variate aplicado nivel a nivel dentro de MLMC/MLQMC.
    // Ver kernel_mlmc_cv_gbm_asian y run_mlmc_cv_cuda/run_mlqmc_cv_cuda en
    // methods_cuda.cu. Cambio 1: ambas usan el patron de streams concurrentes
    // por nivel de run_mlmc_cuda/run_mlqmc_cuda. E_ctrl/beta para MLMC+CV se
    // basan en el n_steps de referencia mas fino del barrido (MLMC no toma
    // n_steps explicito, pero el analisis de beta/E_ctrl si necesita uno).
    methods.push_back({"MLMC + CV (geom)", [&](unsigned so, double eps) {
        auto p = prep_cv(eps);
        MLMCConfig c = ml_cfg; c.seed += so;
        return mc::run_mlmc_cv(mv, mv, pv_main, pv_ctrl, p.E_ctrl, p.beta, eps, c);
    }});
    // ── MLQMC + CV (geom): 3 construcciones de trayectoria ───────────────────
    methods.push_back({"MLQMC+CV Raw", [&](unsigned so, double eps) {
        auto p = prep_cv(eps);
        QMCConfig c = qmc_cfg; c.seed += so;
        return mc::run_mlqmc_cv(mv, mv, pv_main, pv_ctrl, p.E_ctrl, p.beta, eps, ml_cfg, c,
                                  NoiseMode::Raw);
    }});
    methods.push_back({"MLQMC+CV BB", [&](unsigned so, double eps) {
        auto p = prep_cv(eps);
        QMCConfig c = qmc_cfg; c.seed += so;
        return mc::run_mlqmc_cv(mv, mv, pv_main, pv_ctrl, p.E_ctrl, p.beta, eps, ml_cfg, c,
                                  NoiseMode::BrownianBridge);
    }});
    methods.push_back({"MLQMC+CV PCA", [&](unsigned so, double eps) {
        auto p = prep_cv(eps);
        QMCConfig c = qmc_cfg; c.seed += so;
        return mc::run_mlqmc_cv(mv, mv, pv_main, pv_ctrl, p.E_ctrl, p.beta, eps, ml_cfg, c,
                                  NoiseMode::PCA);
    }});

    run_precision_sweep("ejemplo09_asian_cv", methods, price_ref, eps_list);

    // ── Limpieza ─────────────────────────────────────────────────────────────
    return 0;
}
