#include "bench_workloads.hpp"

#include <cmath>

#include "../cpu/mc_cpu.hpp"
#include "../cpu/mlmc_cpu.hpp"
#include "../cpu/qmc_cpu.hpp"

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

// MC con N caminos fijos y generador de normales elegido (para comparar A/B en el mismo binario)
Workload mc_fixed_normal(const std::string& id, const std::string& desc, ModelVariant mv, PayoffVariant pv,
                         int n_steps, long long N, int noise_dim, mc::cpu::NormalMethod nm) {
    Workload w;
    w.id = id; w.desc = desc; w.paths = (double)N; w.steps = (double)n_steps * noise_dim;
    w.run = [=] {
        mc::cpu::CpuOptions o; o.normal = nm;
        mc::cpu::run_mc_fixed(mv, pv, n_steps, N, 1u, o);
    };
    return w;
}

// QMC con puntos fijos: eps inalcanzable y max_doublings acotado => R * n0 * 2^(d-1) puntos
Workload qmc_fixed(const std::string& id, const std::string& desc, ModelVariant mv, PayoffVariant pv, int n_steps,
                   int R, int n0, int doublings, NoiseMode mode) {
    Workload w;
    w.id = id; w.desc = desc;
    w.paths = (double)R * n0 * std::pow(2.0, doublings - 1);
    w.steps = n_steps;
    w.run = [=] {
        QMCConfig c; c.R = R; c.n0 = n0; c.max_doublings = doublings;
        mc::cpu::run_qmc(mv, pv, 1e-12, n_steps, c, mode);
    };
    return w;
}

// MLMC / MLQMC a eps fijo: el nº de muestras es determinista (misma semilla), se mide una vez al crear la carga
Workload ml_workload(const std::string& id, const std::string& desc, ModelVariant mv, PayoffVariant pv, double eps,
                     bool qmc, mc::cpu::NormalMethod nm) {
    Workload w;
    w.id = id; w.desc = desc; w.steps = 1.0;
    auto fn = [=] {
        mc::cpu::CpuOptions o; o.normal = nm;
        return qmc ? mc::cpu::run_mlqmc(mv, pv, eps, MLMCConfig{}, QMCConfig{}, NoiseMode::Raw, o)
                   : mc::cpu::run_mlmc(mv, pv, eps, MLMCConfig{}, o);
    };
    w.paths = (double)fn().n_samples;
    w.run = [=] { fn(); };
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
    w.push_back(mc_fixed_normal("W1b_gbm_euro_64_boxmuller", "W1 con Box-Muller (referencia A/B)", gbm,
                                European{100.0, 0.05, 1.0}, 64, (1LL << 22) / S, 1, mc::cpu::NormalMethod::BoxMuller));
    w.push_back(mc_fixed("W4_heston_euro_128", "MC Heston europea, 128 pasos, 2^20 caminos", hes,
                         European{100.0, 0.05, 1.0}, 128, (1LL << 20) / S, 2));
    w.push_back(qmc_fixed("W5_qmc_bb_64", "QMC BB, 64 pasos, R=32, 2^19 puntos", gbm, European{100.0, 0.05, 1.0}, 64,
                          32, quick ? 128 : 2048, quick ? 3 : 4, NoiseMode::BrownianBridge));
    w.push_back(qmc_fixed("W6_qmc_pca_256", "QMC PCA, 256 pasos, R=8, 2^14 puntos", gbm, European{100.0, 0.05, 1.0}, 256,
                          8, quick ? 64 : 512, quick ? 3 : 3, NoiseMode::PCA));
    w.push_back(ml_workload("W7_mlmc_euro_5e-3", "MLMC GBM europea eps=5e-3", gbm, European{100.0, 0.05, 1.0},
                            quick ? 2e-2 : 5e-3, false, mc::cpu::NormalMethod::Ziggurat));
    w.push_back(ml_workload("W7b_mlmc_euro_5e-3_boxmuller", "W7 con Box-Muller (referencia A/B)", gbm, European{100.0, 0.05, 1.0},
                            quick ? 2e-2 : 5e-3, false, mc::cpu::NormalMethod::BoxMuller));
    w.push_back(ml_workload("W8_mlqmc_euro_5e-3", "MLQMC GBM europea eps=5e-3 (Sobol)", gbm, European{100.0, 0.05, 1.0},
                            quick ? 2e-2 : 5e-3, true, mc::cpu::NormalMethod::Ziggurat));
    return w;
}

} // namespace mcbench
