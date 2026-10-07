#include "bench_workloads.hpp"

#include "../cpu/mc_cpu.hpp"

namespace mcbench {

namespace {

// Carga de MC con N caminos fijos: run_mc_fixed (sin piloto ni planificación) para medir solo
// el motor de simulación.
Workload mc_fixed(const std::string& id, const std::string& desc, ModelVariant mv, PayoffVariant pv,
                  int n_steps, long long N, int noise_dim) {
    Workload w;
    w.id = id; w.desc = desc; w.paths = (double)N; w.steps = (double)n_steps * noise_dim;
    w.run = [=] { mc::cpu::run_mc_fixed(mv, pv, n_steps, N, 1u); };
    return w;
}

} // namespace

std::vector<Workload> make_workloads(bool quick) {
    const long long S = quick ? 16 : 1;   // divisor de tamaño
    std::vector<Workload> w;

    GBMParams gbm;
    HestonParams hes; hes.compute_cholesky();
    DupireLocalParams dup;

    w.push_back(mc_fixed("W1_gbm_euro_64", "MC GBM europea, 64 pasos, 2^22 caminos", gbm,
                         European{100.0, 0.05, 1.0}, 64, (1LL << 22) / S, 1));
    w.push_back(mc_fixed("W2_gbm_asian_256", "MC GBM asiática, 256 pasos, 2^20 caminos", gbm,
                         Asian{100.0}, 256, (1LL << 20) / S, 1));
    w.push_back(mc_fixed("W3_dupire_euro_256", "MC Dupire europea, 256 pasos, 2^19 caminos", dup,
                         European{100.0, 0.05, 1.0}, 256, (1LL << 19) / S, 1));
    w.push_back(mc_fixed("W4_heston_euro_128", "MC Heston europea, 128 pasos, 2^20 caminos", hes,
                         European{100.0, 0.05, 1.0}, 128, (1LL << 20) / S, 2));
    return w;
}

} // namespace mcbench
