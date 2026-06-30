#include <cstdlib>
#include "../methods_cuda.cuh"
#include <cmath>
#include <algorithm>
#include <vector>

static int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

int main(int argc, char** argv) {
    // ── Parámetros ───────────────────────────────────────────────────────────
    HestonParams hes;
    hes.S0    = 100.0;
    hes.mu    = 0.05;
    hes.kappa = 2.0;
    hes.theta = 0.04;
    hes.xi    = 0.5;
    hes.rho   = -0.9;
    hes.v0    = 0.04;
    hes.T     = 1.0;
    hes.compute_cholesky();

    const double K   = 100.0;
    const double r   = hes.mu;
    double eps = (argc > 1) ? atof(argv[1]) : 0.10;          // eps mayor: d=2 Sobol sigue siendo efectivo
    const int    M   = 2;
    const int    L_MAX = 10;

    ModelVariant  mv = hes;
    PayoffVariant pv = European{K, r, hes.T};

    // ── Referencia (MC 500k trayectorias con n_steps=256) ────────────────────
    double price_ref = run_mc_fixed(mv, pv, 256, 500000, 99u).first;

    // ── c1 por Richardson ────────────────────────────────────────────────────
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return run_mc_fixed(mv, pv, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, hes.T, 4, 20000);
    int n_steps = next_pow2(std::max(1, (int)std::ceil(std::sqrt(2.0) * hes.T / (eps * c1))));
    n_steps = std::min(n_steps, 1 << 11);

    // ── Configuración ────────────────────────────────────────────────────────
    // Heston d=2: BB/PCA no prácticos para Sobol 2D → solo Raw
    MCConfig   mc_cfg;
    MLMCConfig ml_cfg;  ml_cfg.M = M;  ml_cfg.max_L = L_MAX;
    QMCConfig  qmc_cfg;

    // ── Métodos (4: MC, MLMC, QMC Raw, MLQMC Raw) ────────────────────────────
    std::vector<TableRow> rows;
    auto add = [&](const char* name, MCResult r) {
        rows.push_back({name, r.price, r.std_error, r.n_samples, r.time_s});
    };

    add("MC",
        run_mc_cuda(mv, pv, eps, n_steps, mc_cfg));
    add("MLMC",
        run_mlmc_cuda(mv, pv, eps, ml_cfg));
    add("QMC Raw",
        run_qmc_cuda(mv, pv, eps, n_steps, qmc_cfg, NoiseMode::Raw));
    add("MLQMC Raw",
        run_mlqmc_cuda(mv, pv, eps, ml_cfg, qmc_cfg, NoiseMode::Raw));

    // ── Tabla ─────────────────────────────────────────────────────────────────
    print_table(rows, price_ref, eps, "Ejemplo 05: European Heston (QE discretization)");

    return 0;
}
