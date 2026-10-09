#pragma once
// Fachada del motor: un único punto de entrada para CPU (multihilo) y GPU (CUDA).
//
// Dos capas:
//  1. Funciones espejo de la API de la GPU (mc::run_mc, mc::run_qmc, ...): mismos argumentos que
//     run_*_cuda sin los manejadores de dispositivo de BB/PCA (se gestionan dentro). El backend y
//     los hilos salen de RunOptions (por defecto, de default_options(), que fijan los ejemplos con
//     --backend / --threads).
//  2. mc::run(model, payoff, RunSpec, RunOptions): planifica (n_steps con Richardson, beta y E_ctrl
//     de las variables de control, z_star del importance sampling) y despacha; devuelve un
//     RunReport. Es lo que usa la GUI.

#include "../mc_progress.hpp"
#include "../mc_types.hpp"
#include "../models.hpp"
#include "../payoffs.hpp"
#include "../sweep.hpp"
#include "../utils.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace mc {

enum class Backend { Cpu, Cuda };

const char* backend_name(Backend b);
Backend backend_from_string(const std::string& s);    // "cpu" | "cuda" | "gpu"; lanza si es desconocido

// ¿Este binario incluye CUDA y hay un dispositivo utilizable?
bool cuda_available();
int  hardware_threads();

struct RunOptions {
    Backend       backend = Backend::Cpu;
    int           threads = 0;          // CPU: 0 = todos los hilos del sistema
    ProgressSink* sink = nullptr;       // progreso/cancelación (solo CPU; la GPU emite inicio y fin)
    double        max_seconds = 0.0;    // CPU: tope de tiempo, resultado parcial marcado 'truncated'
    // Salidas opcionales (CPU)
    bool*         truncated = nullptr;
    bool*         cancelled = nullptr;
    long long*    n_nonfinite = nullptr;
    std::vector<LevelStat>* levels = nullptr;
};

// Opciones usadas cuando no se pasan explícitas (hilo único: se fijan al arrancar un ejecutable).
RunOptions& default_options();

// ---- capa 1: espejo de la API de la GPU -----------------------------------------------------------

MCResult run_mc(const ModelVariant& model, const PayoffVariant& payoff, double eps, int n_steps,
                const MCConfig& cfg = {}, const RunOptions& opt = default_options());

MCResult run_qmc(const ModelVariant& model, const PayoffVariant& payoff, double eps, int n_steps,
                 const QMCConfig& cfg = {}, NoiseMode mode = NoiseMode::Raw,
                 const RunOptions& opt = default_options());

MCResult run_mlmc(const ModelVariant& model, const PayoffVariant& payoff, double eps,
                  const MLMCConfig& cfg = {}, const RunOptions& opt = default_options());

MCResult run_mlqmc(const ModelVariant& model, const PayoffVariant& payoff, double eps,
                   const MLMCConfig& ml_cfg = {}, const QMCConfig& qmc_cfg = {},
                   NoiseMode mode = NoiseMode::Raw, const RunOptions& opt = default_options());

std::pair<double, double> run_mc_fixed(const ModelVariant& model, const PayoffVariant& payoff,
                                       int n_steps, long long n_paths, unsigned seed,
                                       const RunOptions& opt = default_options());

CVPilot cv_pilot(const ModelVariant& main_model, const ModelVariant& ctrl_model,
                 const PayoffVariant& main_payoff, const PayoffVariant& ctrl_payoff,
                 double E_ctrl, int n_steps, int N_pilot, unsigned seed = 0,
                 const RunOptions& opt = default_options());

MCResult run_mc_cv(const ModelVariant& main_model, const ModelVariant& ctrl_model,
                   const PayoffVariant& main_payoff, const PayoffVariant& ctrl_payoff,
                   double E_ctrl, double beta, double eps, int n_steps,
                   const MCConfig& cfg = {}, const RunOptions& opt = default_options());

MCResult run_qmc_cv(const ModelVariant& main_model, const ModelVariant& ctrl_model,
                    const PayoffVariant& main_payoff, const PayoffVariant& ctrl_payoff,
                    double E_ctrl, double beta, double eps, int n_steps,
                    const QMCConfig& cfg = {}, NoiseMode mode = NoiseMode::Raw,
                    const RunOptions& opt = default_options());

MCResult run_mlmc_cv(const ModelVariant& main_model, const ModelVariant& ctrl_model,
                     const PayoffVariant& main_payoff, const PayoffVariant& ctrl_payoff,
                     double E_ctrl, double beta, double eps,
                     const MLMCConfig& cfg = {}, const RunOptions& opt = default_options());

MCResult run_mlqmc_cv(const ModelVariant& main_model, const ModelVariant& ctrl_model,
                      const PayoffVariant& main_payoff, const PayoffVariant& ctrl_payoff,
                      double E_ctrl, double beta, double eps,
                      const MLMCConfig& ml_cfg = {}, const QMCConfig& qmc_cfg = {},
                      NoiseMode mode = NoiseMode::Raw, const RunOptions& opt = default_options());

MCResult run_is(const GBMParams& model, const European& payoff, double z_star, double eps,
                const MCConfig& cfg = {}, const RunOptions& opt = default_options());

MCResult run_qmc_is(const GBMParams& model, const European& payoff, double z_star, double eps,
                    const QMCConfig& cfg = {}, NoiseMode mode = NoiseMode::Raw,
                    const RunOptions& opt = default_options());

MCResult run_mlmc_is(const GBMParams& model, const European& payoff, double z_star, double eps,
                     const MLMCConfig& cfg = {}, const RunOptions& opt = default_options());

MCResult run_mlqmc_is(const GBMParams& model, const European& payoff, double z_star, double eps,
                      const MLMCConfig& ml_cfg = {}, const QMCConfig& qmc_cfg = {},
                      NoiseMode mode = NoiseMode::Raw, const RunOptions& opt = default_options());

// ---- capa 2: ejecución planificada ------------------------------------------------------------------

enum class Family { MC, QMC, MLMC, MLQMC };
enum class Variance { None, ControlVariate, ImportanceSampling };

const char* family_name(Family f);
const char* variance_name(Variance v);
const char* noise_name(NoiseMode m);

inline constexpr double kAuto = std::numeric_limits<double>::quiet_NaN();

struct RunSpec {
    Family    family = Family::MC;
    NoiseMode noise = NoiseMode::Raw;            // QMC / MLQMC
    Variance  variance = Variance::None;
    double    eps = 0.01;
    int       n_steps = 0;                       // 0 = planificar (MC y QMC); MLMC lo ignora
    // CV: kAuto = E_ctrl analítico del control y beta de un piloto
    double    E_ctrl = kAuto;
    double    beta = kAuto;
    int       cv_pilot_n = 50000;
    // IS: kAuto = heurística según la moneyness
    double    z_star = kAuto;
    MCConfig   mc;
    QMCConfig  qmc;
    MLMCConfig mlmc;
};

struct RunReport {
    MCResult  result;
    Backend   backend = Backend::Cpu;
    int       threads = 1;
    int       n_steps = 0;                       // pasos usados (MC/QMC)
    double    c1 = 0.0;                          // constante de Richardson usada para planificar (0 si no)
    double    beta = 0.0, E_ctrl = 0.0, z_star = 0.0;
    bool      truncated = false, cancelled = false;
    long long n_nonfinite = 0;
    std::vector<LevelStat> levels;               // MLMC / MLQMC
};

// n_steps recomendado para MC/QMC a la precisión eps (c1 por Richardson, potencia de 2 <= 2^11;
// las cestas se limitan por D_MAX_SOBOL). Lo usa run(); expuesto para la GUI.
int plan_n_steps(const ModelVariant& model, const PayoffVariant& payoff, double eps,
                 const RunOptions& opt = default_options(), double* c1_out = nullptr);

RunReport run(const ModelVariant& model, const PayoffVariant& payoff, const RunSpec& spec,
              const RunOptions& opt = {});

} // namespace mc
