#include <cstdlib>
#include "../methods_cuda.cuh"
#include <cmath>
#include <algorithm>
#include <vector>
#include <functional>
#include <string>

// n_steps de un solo nivel para MC "plano" (formula de Richardson estandar).
static int steps_for_eps_mc(double eps, double c1, double T) {
    int n = std::max(1, (int)std::ceil(std::sqrt(2.0) * T / (eps * c1)));
    return std::min(n, 1 << 11);
}

// n_steps que usa INTERNAMENTE run_qmc_is_cuda/run_mlmc_is_cuda a nivel fino
// (ver methods_cuda.cu: n_steps = max(4, ceil(T/eps))), distinto de la formula
// de Richardson de arriba. El BB/PCA que se le pasa a run_qmc_is_cuda debe
// precalcularse con ESTE n_steps (no el de Richardson) para que las tablas de
// indices/pesos del Brownian Bridge tengan el tamaño que el kernel espera.
static int steps_for_eps_is(double eps, double T) {
    int n = std::max(4, (int)std::ceil(T / eps));
    int p = 1; while (p < n) p <<= 1;
    return std::min(1 << 11, p); // potencia de 2, y tope 2^11 como el resto de ejemplos
}

int main(int argc, char** argv) {
    // ── Parámetros ───────────────────────────────────────────────────────────
    GBMParams gbm;
    gbm.S0    = 100.0;
    gbm.mu    = 0.05;
    gbm.sigma = 0.20;
    gbm.T     = 1.0;

    const double K   = 180.0;         // muy fuera del dinero
    const double r   = gbm.mu;

    European payoff{K, r, gbm.T};

    // ── Desplazamiento IS: z* = (log(K/S0) - (mu - sigma²/2)*T) / (sigma*sqrt(T)) ─
    double log_moneyness = std::log(K / gbm.S0);
    double drift         = (gbm.mu - 0.5 * gbm.sigma * gbm.sigma) * gbm.T;
    double z_star        = (log_moneyness - drift) / (gbm.sigma * std::sqrt(gbm.T));

    // ── c1 por Richardson (IS no cambia el sesgo de discretización) ──────────
    ModelVariant  mv = gbm;
    PayoffVariant pv = European{100.0, r, gbm.T};  // ATM payoff para estimar c1
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return run_mc_fixed(mv, pv, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, gbm.T, 8, 50000);

    const int M_LEVELS = 2;
    const int L_MAX    = 10;

    // ── BB / PCA por nivel de MLQMC (2^l pasos, eps-independiente) ───────────
    std::vector<DeviceBBData*>  dev_bb_list(L_MAX + 1);
    std::vector<DevicePCAData*> dev_pca_list(L_MAX + 1);
    for (int l = 0; l <= L_MAX; ++l) {
        int nl = 1 << l;
        dev_bb_list[l]  = bb_upload(bb_precompute(nl, gbm.T));
        dev_pca_list[l] = pca_upload(pca_compute(nl, gbm.T));
    }

    // ── Configuración ────────────────────────────────────────────────────────
    MCConfig  mc_cfg;
    MCConfig  is_cfg = mc_cfg;   // mismo batching; el kernel IS usa z_star internamente
    QMCConfig qmc_cfg;
    MLMCConfig ml_cfg;  ml_cfg.M = M_LEVELS;  ml_cfg.max_L = L_MAX;

    // ── CORRECCIÓN AL DISEÑO DE TAREA 2: barrido de eps con corte por método ──
    // (eps por defecto original era 0.002, mas fino que el 0.05 general: se usa
    // como tope si no se pasa argv[1]).
    double eps_finest = (argc > 1) ? atof(argv[1]) : 0.002;
    std::vector<double> eps_list = eps_scale_125(eps_finest);

    // ── Referencia (Black-Scholes exacto, no depende de eps) ─────────────────
    double price_ref = bs_call(gbm.S0, K, gbm.T, r, gbm.sigma);

    // ── Métodos (MC estándar, IS, QMC+IS×3, MLMC+IS, MLQMC+IS×3) ─────────────
    std::vector<SweepMethod> methods;

    methods.push_back({"MC (plain)", [&](unsigned so, double eps) {
        int ns = steps_for_eps_mc(eps, c1, gbm.T);
        MCConfig c = mc_cfg; c.seed += so;
        return run_mc_cuda(mv, PayoffVariant{payoff}, eps, ns, c);
    }});
    methods.push_back({"IS", [&](unsigned so, double eps) {
        MCConfig c = is_cfg; c.seed += so;
        return run_is_cuda(gbm, payoff, z_star, eps, c);
    }});
    // NUEVO (TAREA 1b): mismo desplazamiento de IS, pero con Sobol (QMC) o
    // acoplado nivel a nivel dentro de MLMC. Ver kernel_mlmc_is_gbm_dw y
    // run_qmc_is_cuda/run_mlmc_is_cuda en methods_cuda.cu.
    // ── QMC + IS: 3 construcciones de trayectoria ────────────────────────────
    methods.push_back({"QMC+IS Raw", [&](unsigned so, double eps) {
        QMCConfig c = qmc_cfg; c.seed += so;
        return run_qmc_is_cuda(gbm, payoff, z_star, eps, c, NoiseMode::Raw);
    }});
    methods.push_back({"QMC+IS BB", [&](unsigned so, double eps) {
        int ns = steps_for_eps_is(eps, gbm.T);
        DeviceBBData* dbb = bb_upload(bb_precompute(ns, gbm.T));
        QMCConfig c = qmc_cfg; c.seed += so;
        MCResult r = run_qmc_is_cuda(gbm, payoff, z_star, eps, c,
                                      NoiseMode::BrownianBridge, dbb);
        bb_free(dbb);
        return r;
    }});
    methods.push_back({"QMC+IS PCA", [&](unsigned so, double eps) {
        int ns = steps_for_eps_is(eps, gbm.T);
        DevicePCAData* dpca = pca_upload(pca_compute(ns, gbm.T));
        QMCConfig c = qmc_cfg; c.seed += so;
        MCResult r = run_qmc_is_cuda(gbm, payoff, z_star, eps, c,
                                      NoiseMode::PCA, nullptr, dpca);
        pca_free(dpca);
        return r;
    }});
    // Cambio 1: run_mlmc_is_cuda ahora usa el mismo patrón de streams
    // concurrentes por nivel que run_mlmc_cuda.
    methods.push_back({"MLMC + IS", [&](unsigned so, double eps) {
        MLMCConfig c = ml_cfg; c.seed += so;
        return run_mlmc_is_cuda(gbm, payoff, z_star, eps, c);
    }});
    // ── MLQMC + IS (NUEVO): 3 construcciones de trayectoria ──────────────────
    methods.push_back({"MLQMC+IS Raw", [&](unsigned so, double eps) {
        QMCConfig c = qmc_cfg; c.seed += so;
        return run_mlqmc_is_cuda(gbm, payoff, z_star, eps, ml_cfg, c, NoiseMode::Raw);
    }});
    methods.push_back({"MLQMC+IS BB", [&](unsigned so, double eps) {
        QMCConfig c = qmc_cfg; c.seed += so;
        return run_mlqmc_is_cuda(gbm, payoff, z_star, eps, ml_cfg, c,
                                  NoiseMode::BrownianBridge, dev_bb_list);
    }});
    methods.push_back({"MLQMC+IS PCA", [&](unsigned so, double eps) {
        QMCConfig c = qmc_cfg; c.seed += so;
        return run_mlqmc_is_cuda(gbm, payoff, z_star, eps, ml_cfg, c,
                                  NoiseMode::PCA, {}, dev_pca_list);
    }});

    run_precision_sweep("ejemplo11_otm_is", methods, price_ref, eps_list);

    // ── Limpieza ─────────────────────────────────────────────────────────────
    for (int l = 0; l <= L_MAX; ++l) {
        bb_free(dev_bb_list[l]);
        pca_free(dev_pca_list[l]);
    }
    return 0;
}
