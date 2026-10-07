#pragma once
// Monte Carlo estándar en CPU multihilo (equivalente a run_mc_cuda / run_mc_fixed).

#include "../mc_progress.hpp"
#include "../mc_types.hpp"
#include "../models.hpp"
#include "../payoffs.hpp"
#include "normal.hpp"

#include <vector>

namespace mc::cpu {

// Información adicional de una ejecución (rellenada si CpuOptions::info != nullptr).
struct RunInfo {
    bool      truncated = false;    // parada por max_seconds: el resultado es parcial
    bool      cancelled = false;    // parada por ProgressSink::should_cancel()
    long long n_nonfinite = 0;      // caminos con payoff no finito, excluidos de la muestra
    int       threads = 1;
    std::vector<LevelStat> levels;  // MLMC / MLQMC: estadísticos finales por nivel
};

struct CpuOptions {
    int           threads = 0;        // >0: fija el tamaño del pool global antes de empezar; 0: el actual
    ProgressSink* sink = nullptr;     // progreso en vivo y cancelación (hilo coordinador)
    double        max_seconds = 0.0;  // >0: tope de tiempo; devuelve el resultado parcial (truncated)
    NormalMethod  normal = NormalMethod::BoxMuller;
    RunInfo*      info = nullptr;
};

// MC con N = 2·Var/eps² caminos (Var estimada con un piloto de cfg.pilot_n caminos).
// cfg.N_batch no se usa en CPU: el reparto en chunks es automático.
MCResult run_mc(const ModelVariant& model, const PayoffVariant& payoff,
                double eps, int n_steps, const MCConfig& cfg = {},
                const CpuOptions& opt = {});

// N caminos con n_steps pasos, sin varianza adaptativa. Devuelve {media, var_de_la_media}
// con var = max(0, E[Y²] - media²)/N (igual que run_mc_fixed de la GPU).
std::pair<double, double> run_mc_fixed(const ModelVariant& model, const PayoffVariant& payoff,
                                       int n_steps, long long n_paths, unsigned seed,
                                       const CpuOptions& opt = {});

} // namespace mc::cpu
