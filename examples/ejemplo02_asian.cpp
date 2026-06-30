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
    const int    M   = 2;
    const int    L_MAX = 10;

    ModelVariant  mv = gbm;
    PayoffVariant pv = Asian{K};

    // ── Referencia (MC 500k, n=64) ───────────────────────────────────────────
    double price_ref = run_mc_fixed(mv, pv, 64, 500000, 99u).first;

    // ── c1 por Richardson ────────────────────────────────────────────────────
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return run_mc_fixed(mv, pv, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, gbm.T, 4, 50000);
    int n_steps = next_pow2(std::max(1, (int)std::ceil(std::sqrt(2.0) * gbm.T / (eps * c1))));
    n_steps = std::min(n_steps, 1 << 11);

    // ── BB / PCA para QMC ────────────────────────────────────────────────────
    auto bbd  = bb_precompute(n_steps, gbm.T);
    auto pcad = pca_compute(n_steps, gbm.T);
    DeviceBBData*  dev_bb  = bb_upload(bbd);
    DevicePCAData* dev_pca = pca_upload(pcad);

    std::vector<DeviceBBData*>  dev_bb_list(L_MAX);
    std::vector<DevicePCAData*> dev_pca_list(L_MAX);
    for (int l = 0; l < L_MAX; ++l) {
        int nl = 1 << l;
        dev_bb_list[l]  = bb_upload(bb_precompute(nl, gbm.T));
        dev_pca_list[l] = pca_upload(pca_compute(nl, gbm.T));
    }

    // ── Configuración ────────────────────────────────────────────────────────
    MCConfig   mc_cfg;
    MLMCConfig ml_cfg;  ml_cfg.M = M;  ml_cfg.max_L = L_MAX;
    QMCConfig  qmc_cfg;

    // ── Métodos ──────────────────────────────────────────────────────────────
    std::vector<TableRow> rows;
    auto add = [&](const char* name, MCResult r) {
        rows.push_back({name, r.price, r.std_error, r.n_samples, r.time_s});
    };

    std::string skip = (argc > 2) ? argv[2] : "";
    bool no_mc   = skip.find("nomc") != std::string::npos;
    bool no_mlmc = skip.find("noml") != std::string::npos;
    bool no_qraw = skip.find("noqr") != std::string::npos;
    bool no_qmc  = skip.find("noqmc") != std::string::npos;
    if (!no_mc) add("MC",
        run_mc_cuda(mv, pv, eps, n_steps, mc_cfg));
    if (!no_mlmc) add("MLMC",
        run_mlmc_cuda(mv, pv, eps, ml_cfg));
    if (!no_qraw && !no_qmc) add("QMC Raw",
        run_qmc_cuda(mv, pv, eps, n_steps, qmc_cfg, NoiseMode::Raw));
    if (!no_qmc) add("QMC BB",
        run_qmc_cuda(mv, pv, eps, n_steps, qmc_cfg, NoiseMode::BrownianBridge, dev_bb));
    if (!no_qmc) add("QMC PCA",
        run_qmc_cuda(mv, pv, eps, n_steps, qmc_cfg, NoiseMode::PCA, nullptr, dev_pca));
    add("MLQMC Raw",
        run_mlqmc_cuda(mv, pv, eps, ml_cfg, qmc_cfg, NoiseMode::Raw));
    add("MLQMC BB",
        run_mlqmc_cuda(mv, pv, eps, ml_cfg, qmc_cfg, NoiseMode::BrownianBridge, dev_bb_list));
    add("MLQMC PCA",
        run_mlqmc_cuda(mv, pv, eps, ml_cfg, qmc_cfg, NoiseMode::PCA, {}, dev_pca_list));

    // ── Tabla ─────────────────────────────────────────────────────────────────
    print_table(rows, price_ref, eps, "Ejemplo 02: Asian GBM");

    // ── Limpieza ─────────────────────────────────────────────────────────────
    bb_free(dev_bb);
    pca_free(dev_pca);
    for (int l = 0; l < L_MAX; ++l) {
        bb_free(dev_bb_list[l]);
        pca_free(dev_pca_list[l]);
    }
    return 0;
}
