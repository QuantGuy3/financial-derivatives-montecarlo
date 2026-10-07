#pragma once
// Reducción de varianza en CPU: variables de control (CV) e importance sampling (IS), para MC,
// QMC, MLMC y MLQMC (equivalentes a run_*_cv_cuda / run_*_is_cuda / cv_pilot).
//
//  * CV: solo las dos parejas que implementa la GPU.
//      GBM + Asian aritmética      con control Asian geométrica (E_ctrl = geom_asian_analytic)
//      Dupire local + europea      con control GBM(sigma0) europea (E_ctrl dado por el llamante)
//    El control se evalúa sobre los MISMOS caminos (ctrl_model / ctrl_payoff se ignoran, como en
//    la GPU). Y_cv = Y_main - beta·(Y_ctrl - E_ctrl).
//  * IS: GBM + call europea, desplazamiento total z_star (z_star/sqrt(n_pasos) por paso).

#include "mlmc_cpu.hpp"
#include "qmc_cpu.hpp"

namespace mc::cpu {

// beta* = Cov(Y_main, Y_ctrl)/Var(Y_ctrl) (recortado a [0,5]) estimado con N_pilot caminos.
CVPilot cv_pilot(const ModelVariant& main_model, const ModelVariant& ctrl_model,
                 const PayoffVariant& main_payoff, const PayoffVariant& ctrl_payoff,
                 double E_ctrl, int n_steps, int N_pilot, unsigned seed = 0,
                 const CpuOptions& opt = {});

MCResult run_mc_cv(const ModelVariant& main_model, const ModelVariant& ctrl_model,
                   const PayoffVariant& main_payoff, const PayoffVariant& ctrl_payoff,
                   double E_ctrl, double beta, double eps, int n_steps,
                   const MCConfig& cfg = {}, const CpuOptions& opt = {});

MCResult run_qmc_cv(const ModelVariant& main_model, const ModelVariant& ctrl_model,
                    const PayoffVariant& main_payoff, const PayoffVariant& ctrl_payoff,
                    double E_ctrl, double beta, double eps, int n_steps,
                    const QMCConfig& cfg = {}, NoiseMode mode = NoiseMode::Raw,
                    const CpuOptions& opt = {});

// Solo GBM + Asian (como la GPU).
MCResult run_mlmc_cv(const ModelVariant& main_model, const ModelVariant& ctrl_model,
                     const PayoffVariant& main_payoff, const PayoffVariant& ctrl_payoff,
                     double E_ctrl, double beta, double eps,
                     const MLMCConfig& cfg = {}, const CpuOptions& opt = {});

MCResult run_mlqmc_cv(const ModelVariant& main_model, const ModelVariant& ctrl_model,
                      const PayoffVariant& main_payoff, const PayoffVariant& ctrl_payoff,
                      double E_ctrl, double beta, double eps,
                      const MLMCConfig& ml_cfg = {}, const QMCConfig& qmc_cfg = {},
                      NoiseMode mode = NoiseMode::Raw, const CpuOptions& opt = {});

// n_steps que usan run_is / run_qmc_is: min(2048, pow2_ceil(max(4, ceil(T/eps)))).
int is_n_steps(double T, double eps);

MCResult run_is(const GBMParams& model, const European& payoff, double z_star, double eps,
                const MCConfig& cfg = {}, const CpuOptions& opt = {});

MCResult run_qmc_is(const GBMParams& model, const European& payoff, double z_star, double eps,
                    const QMCConfig& cfg = {}, NoiseMode mode = NoiseMode::Raw,
                    const CpuOptions& opt = {});

MCResult run_mlmc_is(const GBMParams& model, const European& payoff, double z_star, double eps,
                     const MLMCConfig& cfg = {}, const CpuOptions& opt = {});

MCResult run_mlqmc_is(const GBMParams& model, const European& payoff, double z_star, double eps,
                      const MLMCConfig& ml_cfg = {}, const QMCConfig& qmc_cfg = {},
                      NoiseMode mode = NoiseMode::Raw, const CpuOptions& opt = {});

} // namespace mc::cpu
