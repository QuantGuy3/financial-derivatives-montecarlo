#pragma once
#include "models.hpp"
#include "payoffs.hpp"
#include "utils.hpp"
#include "mc_types.hpp"
#include "sweep.hpp"

// ¿Hay al menos un dispositivo CUDA utilizable? (lo usa engine/ para elegir el backend)
bool cuda_device_available();

// --------------------- //
// Datos BB y PCA en GPU //
// --------------------- //

struct DeviceBBData;
struct DevicePCAData;

// Sube los pesos del Brownian Bridge al dispositivo
DeviceBBData*  bb_upload(const BBData& bb);
void           bb_free(DeviceBBData*);

// Sube la matriz PCA al dispositivo (en float16 para Tensor Cores)
DevicePCAData* pca_upload(const PCAData& pca);
void           pca_free(DevicePCAData*);


// ----------------------- //
// Métodos GPU principales //
// ----------------------- //

// Monte Carlo estándar
MCResult run_mc_cuda(const ModelVariant& model, const PayoffVariant& payoff,
                     double eps, int n_steps, const MCConfig& cfg = {});

// Multilevel Monte Carlo (Giles 2008)
MCResult run_mlmc_cuda(const ModelVariant& model, const PayoffVariant& payoff,
                       double eps, const MLMCConfig& cfg = {});

// Quasi-Monte Carlo con secuencias Sobol scrambled (R réplicas)
// mode = Raw:             normales Sobol directas al esquema de Euler
// mode = BrownianBridge:  transformada BB en GPU antes del Euler
// mode = PCA:             multiplicación con Tensor Cores (cuBLAS F16→F32)
MCResult run_qmc_cuda(const ModelVariant& model, const PayoffVariant& payoff,
                      double eps, int n_steps,
                      const QMCConfig& cfg  = {},
                      NoiseMode mode        = NoiseMode::Raw,
                      DeviceBBData*  dev_bb  = nullptr,
                      DevicePCAData* dev_pca = nullptr);

// Multilevel QMC (un DeviceBBData / DevicePCAData por nivel)
MCResult run_mlqmc_cuda(const ModelVariant& model, const PayoffVariant& payoff,
                        double eps,
                        const MLMCConfig& ml_cfg  = {},
                        const QMCConfig&  qmc_cfg = {},
                        NoiseMode mode             = NoiseMode::Raw,
                        std::vector<DeviceBBData*>  bb_list  = {},
                        std::vector<DevicePCAData*> pca_list = {});

// Variables de control
// E_ctrl = valor esperado analítico del payoff de control
// beta   = Cov(Y_main, Y_ctrl) / Var(Y_ctrl), estimado en el piloto

CVPilot cv_pilot(const ModelVariant& main_model,
                 const ModelVariant& ctrl_model,
                 const PayoffVariant& main_payoff,
                 const PayoffVariant& ctrl_payoff,
                 double E_ctrl, int n_steps,
                 int N_pilot, unsigned seed = 0);

MCResult run_mc_cv_cuda(const ModelVariant& main_model,
                        const ModelVariant& ctrl_model,
                        const PayoffVariant& main_payoff,
                        const PayoffVariant& ctrl_payoff,
                        double E_ctrl, double beta,
                        double eps, int n_steps,
                        const MCConfig& cfg = {});

MCResult run_qmc_cv_cuda(const ModelVariant& main_model,
                         const ModelVariant& ctrl_model,
                         const PayoffVariant& main_payoff,
                         const PayoffVariant& ctrl_payoff,
                         double E_ctrl, double beta,
                         double eps, int n_steps,
                         const QMCConfig& cfg = {},
                         NoiseMode mode = NoiseMode::Raw,
                         DeviceBBData* dev_bb = nullptr,
                         DevicePCAData* dev_pca = nullptr);

// MLMC + variable de control (solo GBM: Asian aritmética + control Asian
// geométrica, análogo a kernel_gbm_asian_cv pero aplicado nivel a nivel dentro
// del esquema telescópico de MLMC en vez de una sola vez al final; ver
// kernel_mlmc_cv_gbm_asian en methods_cuda.cu).
MCResult run_mlmc_cv_cuda(const ModelVariant& main_model,
                          const ModelVariant& ctrl_model,
                          const PayoffVariant& main_payoff,
                          const PayoffVariant& ctrl_payoff,
                          double E_ctrl, double beta,
                          double eps, const MLMCConfig& cfg = {});

// MLQMC + variable de control: igual que run_mlmc_cv_cuda pero generando el
// ruido de cada nivel con Sobol scrambled (R réplicas) en vez de pseudoaleatorio.
// Solo modo Raw (igual que run_qmc_cv_cuda, que tampoco admite BB/PCA).
MCResult run_mlqmc_cv_cuda(const ModelVariant& main_model,
                           const ModelVariant& ctrl_model,
                           const PayoffVariant& main_payoff,
                           const PayoffVariant& ctrl_payoff,
                           double E_ctrl, double beta,
                           double eps, const MLMCConfig& ml_cfg,
                           const QMCConfig& qmc_cfg = {},
                           NoiseMode mode = NoiseMode::Raw,
                           std::vector<DeviceBBData*>  bb_list  = {},
                           std::vector<DevicePCAData*> pca_list = {});

// Importance Sampling (solo GBM + call europea)
MCResult run_is_cuda(const GBMParams& model, const European& payoff,
                     double z_star, double eps, const MCConfig& cfg = {});

// Importance Sampling + QMC: mismo desplazamiento z_star de kernel_is_gbm,
// pero alimentado con normales Sobol scrambled (mismo generador que usa
// run_qmc_cuda) en vez de pseudoaleatorias.
MCResult run_qmc_is_cuda(const GBMParams& model, const European& payoff,
                         double z_star, double eps, const QMCConfig& cfg = {},
                         NoiseMode mode = NoiseMode::Raw,
                         DeviceBBData* dev_bb = nullptr,
                         DevicePCAData* dev_pca = nullptr);

// Importance Sampling + MLMC: aplica el mismo cambio de medida (shift z_star)
// sobre la trayectoria fina de cada nivel, y la gruesa se construye agregando
// esos mismos incrementos ya desplazados (mismo acoplamiento que kernel_mlmc);
// el ratio de verosimilitud se calcula una vez por trayectoria a partir del
// desplazamiento total y pondera por igual fino y grueso (ver
// kernel_mlmc_is_gbm en methods_cuda.cu).
MCResult run_mlmc_is_cuda(const GBMParams& model, const European& payoff,
                          double z_star, double eps, const MLMCConfig& cfg = {});

// Importance Sampling + MLQMC: igual que run_mlmc_is_cuda pero generando el
// ruido de cada nivel con Sobol scrambled (R réplicas), con las 3
// construcciones de trayectoria (Raw/BB/PCA) igual que run_mlqmc_cv_cuda.
// Restringido a GBM + call europea (igual que el resto de variantes IS).
MCResult run_mlqmc_is_cuda(const GBMParams& model, const European& payoff,
                           double z_star, double eps,
                           const MLMCConfig& ml_cfg, const QMCConfig& qmc_cfg = {},
                           NoiseMode mode = NoiseMode::Raw,
                           std::vector<DeviceBBData*>  bb_list  = {},
                           std::vector<DevicePCAData*> pca_list = {});

// Ejecuta N trayectorias con n_steps pasos (sin varianza adaptativa)
std::pair<double, double> run_mc_fixed(const ModelVariant& model,
                                       const PayoffVariant& payoff,
                                       int n_steps, long long n_paths,
                                       unsigned seed);
