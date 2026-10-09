#pragma once
// Quasi-Monte Carlo en CPU multihilo (equivalente a run_qmc_cuda): R réplicas de Sobol con
// scrambling de Hong-Hickernell independiente, duplicando los puntos por réplica hasta que
// var_of_means < eps²/2 (o max_doublings).

#include "mc_cpu.hpp"

namespace mc::cpu {

// mode = Raw: normales de Sobol directas al esquema de Euler.
// mode = BrownianBridge: puente browniano (GBM/Dupire con n_steps potencia de 2).
// mode = PCA: componentes principales del movimiento browniano (GBM/Dupire).
// Lanza SobolLimitReached si harían falta más de 2^32 puntos por réplica.
// Si D = n_steps*dim(ruido) supera D_MAX_SOBOL se usan normales pseudoaleatorias (como la GPU).
MCResult run_qmc(const ModelVariant& model, const PayoffVariant& payoff,
                 double eps, int n_steps, const QMCConfig& cfg = {},
                 NoiseMode mode = NoiseMode::Raw, const CpuOptions& opt = {});

// Igual que run_qmc pero evaluando `eval` (CV / IS). Con IS el n_steps lo fija el llamante.
MCResult run_qmc_eval(const ModelVariant& model, const PayoffVariant& payoff, const EvalSpec& eval,
                      double eps, int n_steps, const QMCConfig& cfg = {},
                      NoiseMode mode = NoiseMode::Raw, const CpuOptions& opt = {});

} // namespace mc::cpu
