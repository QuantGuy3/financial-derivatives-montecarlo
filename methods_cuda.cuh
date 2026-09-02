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
    // Réplicas Sobol: cada una recibe su propia matriz triangular de
    // Hong-Hickernell (scrambling sobre F_2) y su propio desplazamiento
    // digital, calculados a mano vía la API de dispositivo de cuRAND (ver
    // gen_scrambled_sobol_normal_replica/hh_scramble en methods_cuda.cu), no
    // solo un carril de offset distinto sobre un scramble fijo compartido.
    // var_of_means entre réplicas es así el estimador insesgado descrito en
    // la observación 3.19 de la memoria (Owen, 1997).
    int R            = 32;
    int max_doublings = 20; // Máximo de duplicaciones del número de puntos
    // Semilla base para el generador pseudoaleatorio / los scrambles Sobol.
    // Se usa como cfg.seed (Raw) o como sal de scramble por réplica (Sobol),
    // reemplazando la constante 42u que antes iba fija en el código: así los
    // ejemplos pueden repetir un método con semillas distintas (ver TAREA 2).
    unsigned seed = 42u;
};

struct MLMCConfig {
    int M      = 2;   // Factor de refinamiento entre niveles
    int max_L  = 10;  // Número máximo de niveles
    int pilot_n = 400; // Trayectorias piloto por nivel
    // Semilla base de la cadena de semillas por nivel/iteración (antes un 42u
    // fijo dentro de run_mlmc_cuda/run_mlqmc_cuda). Permite repetir MLMC con
    // distinta semilla sin tocar el motor (ver TAREA 2).
    unsigned seed = 42u;
};

// Modo de transformación del ruido antes de aplicar el esquema de Euler
enum class NoiseMode { Raw, BrownianBridge, PCA };


// -------------------- //
// Struct de resultados //
// -------------------- //

// Se lanza cuando un metodo Sobol/QMC necesitaria mas de 2^32 puntos por
// replica y choca con el limite de 32 bits del offset de la API de
// dispositivo de cuRAND. El barrido lo captura y marca el metodo como
// no aplicable a ese eps (--) en vez de abortar el ejecutable.
struct SobolLimitReached {};

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

struct SweepMethod {
    std::string name;
    // run_once(seed_offset, eps): una sola corrida de este metodo al nivel
    // de precision `eps` dado, con semilla desplazada por seed_offset.
    std::function<MCResult(unsigned seed_offset, double eps)> run_once;
};

void run_precision_sweep(const std::string& example_name,
                         std::vector<SweepMethod>& methods,
                         double price_ref,
                         const std::vector<double>& eps_list,
                         double T_BUDGET_S = 20.0,
                         int R_MAX = 30,
                         int R_MIN = 10,
                         double SE_REL = 0.01);

// Escala redonda "1-2-5" descendente (la misma que ya asumia gen_informe.py
// en sus comentarios), recortada por abajo al eps mas fino solicitado
// (eps_finest, tipicamente argv[1] si se paso; si no se pasa nada se usa el
// valor mas fino ya predefinido en la escala). Si eps_finest es mas fino que
// el ultimo valor de la escala, se anade el propio eps_finest al final para
// no dejar de intentar la precision que pidio el usuario.
std::vector<double> eps_scale_125(double eps_finest = 0.0001);
