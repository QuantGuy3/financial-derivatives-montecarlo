#include <cstdlib>
#include "../methods_cuda.cuh"
#include <cmath>
#include <algorithm>
#include <vector>

int main(int argc, char** argv) {
    // ── Parámetros ───────────────────────────────────────────────────────────
    GBMParams gbm;
    gbm.S0    = 100.0;
    gbm.mu    = 0.05;
    gbm.sigma = 0.20;
    gbm.T     = 1.0;

    const double K   = 180.0;         // muy fuera del dinero
    const double r   = gbm.mu;
    double eps = (argc > 1) ? atof(argv[1]) : 0.002;

    European payoff{K, r, gbm.T};

    // ── Referencia (Black-Scholes exacto) ────────────────────────────────────
    double price_ref = bs_call(gbm.S0, K, gbm.T, r, gbm.sigma);

    // ── Desplazamiento IS: z* = (log(K/S0) - (mu - sigma²/2)*T) / (sigma*sqrt(T)) ─
    double log_moneyness = std::log(K / gbm.S0);
    double drift         = (gbm.mu - 0.5 * gbm.sigma * gbm.sigma) * gbm.T;
    double z_star        = (log_moneyness - drift) / (gbm.sigma * std::sqrt(gbm.T));

    // ── n_steps por Richardson (IS no cambia el sesgo de discretización) ─────
    ModelVariant  mv = gbm;
    PayoffVariant pv = European{100.0, r, gbm.T};  // ATM payoff para estimar c1
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return run_mc_fixed(mv, pv, ns, np, s).first;
    };
    double c1 = estimar_c1_richardson(sim_fn, gbm.T, 8, 50000);
    int n_steps = std::max(1, (int)std::ceil(std::sqrt(2.0) * gbm.T / (eps * c1)));
    n_steps = std::min(n_steps, 1 << 11);

    // ── Configuración ────────────────────────────────────────────────────────
    MCConfig mc_cfg;
    MCConfig is_cfg = mc_cfg;   // mismo batching; el kernel IS usa z_star internamente

    // ── Métodos (MC estándar + IS) ────────────────────────────────────────────
    std::vector<TableRow> rows;
    auto add = [&](const std::string& name, MCResult r) {
        rows.push_back({name, r.price, r.std_error, r.n_samples, r.time_s});
    };

    add("MC (plain)",
        run_mc_cuda(mv, PayoffVariant{payoff}, eps, n_steps, mc_cfg));
    add("IS (z*=" + std::to_string(z_star).substr(0, 4) + ")",
        run_is_cuda(gbm, payoff, z_star, eps, is_cfg));

    // ── Tabla ─────────────────────────────────────────────────────────────────
    print_table(rows, price_ref, eps,
                "Ejemplo 11: Deep OTM European + Importance Sampling (K=180)");

    return 0;
}
