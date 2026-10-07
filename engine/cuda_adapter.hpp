#pragma once
// Adaptador del backend CUDA: envuelve las funciones run_*_cuda de methods_cuda.cu ocultando los
// manejadores de dispositivo de BB/PCA (se suben una vez por (N, T) y se reutilizan).
// Es C++ plano (no incluye cabeceras de CUDA), así que se compila siempre; los símbolos reales
// los aporta mc_cuda (con nvcc) o engine/cuda_stub.cpp (sin CUDA, lanzan una excepción).

#include "../methods_cuda.cuh"

namespace mc::detail {

bool cuda_compiled_in();      // ¿el binario incluye el backend CUDA real?
bool cuda_present();          // compilado y con al menos un dispositivo
void cuda_release_cache();    // libera los datos BB/PCA subidos al dispositivo

MCResult cuda_mc(const ModelVariant&, const PayoffVariant&, double eps, int n_steps, const MCConfig&);
MCResult cuda_qmc(const ModelVariant&, const PayoffVariant&, double eps, int n_steps, const QMCConfig&, NoiseMode);
MCResult cuda_mlmc(const ModelVariant&, const PayoffVariant&, double eps, const MLMCConfig&);
MCResult cuda_mlqmc(const ModelVariant&, const PayoffVariant&, double eps, const MLMCConfig&, const QMCConfig&, NoiseMode);
std::pair<double, double> cuda_mc_fixed(const ModelVariant&, const PayoffVariant&, int n_steps, long long n_paths, unsigned seed);

CVPilot cuda_cv_pilot(const ModelVariant& main_model, const ModelVariant& ctrl_model,
                      const PayoffVariant& main_payoff, const PayoffVariant& ctrl_payoff,
                      double E_ctrl, int n_steps, int N_pilot, unsigned seed);
MCResult cuda_mc_cv(const ModelVariant&, const ModelVariant&, const PayoffVariant&, const PayoffVariant&,
                    double E_ctrl, double beta, double eps, int n_steps, const MCConfig&);
MCResult cuda_qmc_cv(const ModelVariant&, const ModelVariant&, const PayoffVariant&, const PayoffVariant&,
                     double E_ctrl, double beta, double eps, int n_steps, const QMCConfig&, NoiseMode);
MCResult cuda_mlmc_cv(const ModelVariant&, const ModelVariant&, const PayoffVariant&, const PayoffVariant&,
                      double E_ctrl, double beta, double eps, const MLMCConfig&);
MCResult cuda_mlqmc_cv(const ModelVariant&, const ModelVariant&, const PayoffVariant&, const PayoffVariant&,
                       double E_ctrl, double beta, double eps, const MLMCConfig&, const QMCConfig&, NoiseMode);

MCResult cuda_is(const GBMParams&, const European&, double z_star, double eps, const MCConfig&);
MCResult cuda_qmc_is(const GBMParams&, const European&, double z_star, double eps, const QMCConfig&, NoiseMode);
MCResult cuda_mlmc_is(const GBMParams&, const European&, double z_star, double eps, const MLMCConfig&);
MCResult cuda_mlqmc_is(const GBMParams&, const European&, double z_star, double eps,
                       const MLMCConfig&, const QMCConfig&, NoiseMode);

} // namespace mc::detail
