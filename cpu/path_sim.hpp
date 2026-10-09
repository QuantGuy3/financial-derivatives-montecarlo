#pragma once
// Simulador de caminos por chunks y reducción determinista en rondas ("waves").
//
// Un chunk = rango de caminos consecutivos [first, first+count) que una tarea del pool simula
// por bloques de kLanes. Cada chunk entrega su ChunkAcc; los chunks se funden SIEMPRE uno a
// uno en orden de índice (Chan), de modo que el resultado es idéntico bit a bit para
// cualquier nº de hilos y cualquier partición en rondas.

#include "../mc_progress.hpp"
#include "kernels.hpp"
#include "noise.hpp"
#include "reduce.hpp"
#include "thread_pool.hpp"

#include <chrono>
#include <functional>
#include <vector>

namespace mc::cpu {

// Resultado de un chunk
struct ChunkAcc {
    ShiftedAcc acc;            // estadístico principal (payoff, o payoff con CV/IS aplicado)
    ShiftedAcc2 pair;          // CV: (Y_principal, Y_control) para estimar beta en el piloto
    long long nonfinite = 0;   // caminos con payoff no finito (se excluyen de la muestra)
};

// Memoria de trabajo por hilo
struct Scratch {
    std::vector<double> Z;       // D * kLanes
    std::vector<double> S;       // n_assets * kLanes (cesta)
};

// Cuántos caminos por chunk: ~0.5-2 ms de trabajo, dependiente solo del coste por camino.
inline int chunk_paths_for(double work_per_path) {
    constexpr double kChunkWork = 262144.0;   // "pasos-dimensión" por chunk
    double c = kChunkWork / std::max(1.0, work_per_path);
    int n = (int)std::clamp(c, 64.0, 8192.0);
    return (n / kLanes) * kLanes;
}

// Finales acumulados (en nº de chunks) de cada ronda. Deterministas (no dependen del nº de hilos).
//   fine = true:  1, x1.25, tope 256. Es el reparto de la curva de convergencia: ~10 puntos por
//                 década de N. Lo usa quien tiene un ProgressSink escuchando.
//   fine = false: 64, x2, tope 2048. Cuando nadie mira el progreso solo hace falta poder cancelar,
//                 y cada frontera de ronda cuesta: los hilos esperan al último chunk de la ronda.
// El reparto no cambia el resultado: los chunks se funden uno a uno en orden de índice.
inline constexpr long long kMaxWaveChunks = 2048;
std::vector<long long> make_wave_ends(long long n_chunks, bool fine = true);

class PathSim {
public:
    // noise.dim() debe ser n_steps * model.noise_dim. Las referencias deben sobrevivir al objeto.
    PathSim(const CpuModel& m, const CpuPayoff& p, int n_steps, const NoiseSource& noise,
            const EvalSpec& eval = {});

    int n_steps() const { return kc_.n_steps; }
    int chunk_paths() const { return chunk_paths_; }

    // Simula los caminos first..first+count-1 y acumula sus payoffs.
    void run_chunk(uint64_t first, int count, Scratch& s, ChunkAcc& out) const;

private:
    const CpuModel& m_;
    const CpuPayoff& p_;
    const NoiseSource& noise_;
    KCtx kc_;
    SingleFn fn_ = nullptr;
    EvalFn eval_fn_ = nullptr;     // CV / IS (nullptr = payoff simple)
    EvalSpec eval_;
    bool multi_ = false;
    int chunk_paths_ = 64;
};

// Resultado de simular un rango de caminos
struct RangeResult {
    Moments moments;
    Cov2 pair;                 // CV: covarianza (Y_principal, Y_control) de los caminos finitos
    long long nonfinite = 0;
    bool stopped = false;      // parada anticipada (cancelación o tiempo máximo)
};

// Simula los caminos [first, first+N) en rondas, fundiendo en orden. Tras cada ronda llama a
// on_wave(moments_acumulados, n_hechos) -> false para parar. fine_waves elige el reparto en rondas
// (ver make_wave_ends); sin on_wave siempre se usa el grueso.
RangeResult simulate_range(ThreadPool& pool, const PathSim& sim, uint64_t first, long long N,
                           const std::function<bool(const Moments&, long long)>& on_wave, bool fine_waves = true);

// ---- niveles MLMC (fino/grueso acoplados) ---------------------------------------------------------

// Resultado de un chunk de un nivel: corrección dY = Yf - Yc y valor fino Yf.
struct ChunkAcc2 {
    ShiftedAcc dY, Yf;
    long long nonfinite = 0;
};

class CoupledSim {
public:
    // noise.dim() debe ser n_fine * model.noise_dim y entregar el incremento FINO ya escalado.
    CoupledSim(const CpuModel& m, const CpuPayoff& p, int level, int M, const NoiseSource& noise,
               const EvalSpec& eval = {});

    int n_fine() const { return kc_.n_fine; }
    int chunk_paths() const { return chunk_paths_; }
    void run_chunk(uint64_t first, int count, Scratch& s, ChunkAcc2& out) const;

private:
    const NoiseSource& noise_;
    CKCtx kc_;
    CoupledFn fn_ = nullptr;
    int chunk_paths_ = 64;
};

using Clock = std::chrono::steady_clock;
inline double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

} // namespace mc::cpu
