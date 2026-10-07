#pragma once
// Multilevel Monte Carlo (Giles 2008) y Multilevel QMC en CPU multihilo
// (equivalentes a run_mlmc_cuda / run_mlqmc_cuda).
//
// Un único bucle de Giles (piloto -> asignación óptima N_l -> refinamiento -> test de sesgo
// -> añadir nivel) sirve para los dos: MLMC es el caso de una réplica con ruido
// pseudoaleatorio; MLQMC usa R réplicas de Sobol scrambleado por (nivel, réplica).
// Todos los niveles pendientes de una iteración se ejecutan en una sola región paralela
// (equivalente a los streams por nivel de la GPU), balanceando el nivel 0 (muchos caminos
// baratos) con el nivel L (pocos caminos caros).

#include "mc_cpu.hpp"

namespace mc::cpu {

// Las cestas multi-activo no están soportadas (tampoco en la GPU).
MCResult run_mlmc(const ModelVariant& model, const PayoffVariant& payoff,
                  double eps, const MLMCConfig& cfg = {}, const CpuOptions& opt = {});

// mode = Raw: normales de Sobol directas. BrownianBridge/PCA: solo ruido 1D (GBM/Dupire) y M = 2.
// Los puntos de cada (nivel, réplica) se prolongan de una ronda a otra (offset acumulado).
MCResult run_mlqmc(const ModelVariant& model, const PayoffVariant& payoff,
                   double eps, const MLMCConfig& ml_cfg = {}, const QMCConfig& qmc_cfg = {},
                   NoiseMode mode = NoiseMode::Raw, const CpuOptions& opt = {});

} // namespace mc::cpu
