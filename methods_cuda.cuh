#pragma once
#include "models.hpp"
#include "payoffs.hpp"
#include "utils.hpp"

// ------------------------ //
// Structs de configuración //
// ------------------------ //

struct MCConfig {
    long long N_batch = 1LL << 16; // Trayectorias por lote GPU
    int       pilot_n = 10000;
    unsigned  seed    = 123u;
};

struct QMCConfig {
    // Réplicas Sobol: bloques disjuntos consecutivos de UNA misma secuencia scrambled
    // (mismo scrambling en todas, distinto carril de offset), no randomizaciones
    // independientes entre sí. var_of_means entre réplicas es una aproximación
    // práctica del error, no el estimador insesgado de un RQMC con shifts propios
    // por réplica.
    int R            = 32;
    int max_doublings = 20; // Máximo de duplicaciones del número de puntos
};

struct MLMCConfig {
    int M      = 2;   // Factor de refinamiento entre niveles
    int max_L  = 10;  // Número máximo de niveles
    int pilot_n = 400; // Trayectorias piloto por nivel
};

// Modo de transformación del ruido antes de aplicar el esquema de Euler
enum class NoiseMode { Raw, BrownianBridge, PCA };


// -------------------- //
// Struct de resultados //
// -------------------- //

struct MCResult {
    double    price     = 0.0;
    double    std_error = 0.0;
    long long n_samples = 0;
    double    time_s    = 0.0;
};


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
struct CVPilot { double beta, var_plain, var_cv; };

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
                         const QMCConfig& cfg = {});

// Importance Sampling (solo GBM + call europea)
MCResult run_is_cuda(const GBMParams& model, const European& payoff,
                     double z_star, double eps, const MCConfig& cfg = {});

// Ejecuta N trayectorias con n_steps pasos (sin varianza adaptativa)
std::pair<double, double> run_mc_fixed(const ModelVariant& model,
                                       const PayoffVariant& payoff,
                                       int n_steps, long long n_paths,
                                       unsigned seed);
