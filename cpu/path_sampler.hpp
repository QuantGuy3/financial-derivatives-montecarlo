#pragma once
// Muestreo de trayectorias completas S_t (y v_t en Heston) para visualizarlas en la GUI.
//
// Usa EXACTAMENTE los mismos pasos de Euler que el motor (cpu/kernels.hpp) y las mismas fuentes
// de ruido (pseudoaleatorio o Sobol scrambleado, con construcción Raw / Brownian Bridge / PCA),
// así que lo que se dibuja es la dinámica que se simula, no una ilustración aparte.

#include "../mc_types.hpp"
#include "../models.hpp"
#include "../payoffs.hpp"

#include <cstdint>
#include <vector>

namespace mc::cpu {

struct SampleOptions {
    int      n_paths = 5;
    int      n_steps = 128;
    uint64_t seed = 1;
    bool     sobol = false;                          // true: puntos Sobol scrambleados en vez de pseudoaleatorios
    NoiseMode construction = NoiseMode::Raw;         // Raw | BrownianBridge | PCA (modelos de ruido 1D)
    int      max_assets_shown = 5;                   // cestas: activos dibujados
};

struct SampledPath {
    std::vector<double> S;          // S_0..S_n (en cestas: la media de la cesta)
    std::vector<double> V;          // Heston: v_0..v_n
    std::vector<double> run;        // estadístico corriente del payoff: media (Asian), media geométrica, mín o máx
    std::vector<std::vector<double>> assets;   // cestas: primeros activos
    double payoff = 0.0;            // valor del payoff de este camino (con descuento y corrección BGK, como el motor)
    bool   knocked_out = false;     // Barrier: el máximo alcanzó la barrera
};

struct SampledPaths {
    std::vector<double> t;          // t_0..t_n
    std::vector<SampledPath> paths;
    int n_steps = 0;
};

// Trayectorias independientes (el camino i usa el índice i del flujo de ruido).
SampledPaths sample_paths(const ModelVariant& model, const PayoffVariant& payoff, const SampleOptions& opt);

// Par acoplado de MLMC en el nivel `level` (M^level pasos finos, M^(level-1) gruesos) con el MISMO
// ruido browniano: lo que hace que Var[P_l - P_{l-1}] decaiga.
struct CoupledPair {
    std::vector<double> t_fine, S_fine, t_coarse, S_coarse;
    int n_fine = 0, n_coarse = 0;
    double payoff_fine = 0.0, payoff_coarse = 0.0;
};
CoupledPair sample_coupled(const ModelVariant& model, const PayoffVariant& payoff, int level, int M,
                           uint64_t seed, int index = 0);

// Abanico de percentiles de S_t y valores terminales de `n_fan` trayectorias pseudoaleatorias.
struct FanBands {
    std::vector<double> t, p05, p25, p50, p75, p95, mean;
    std::vector<double> terminal;   // S_T de cada trayectoria
    std::vector<double> payoffs;    // payoff de cada trayectoria
    double payoff_mean = 0.0;
};
FanBands sample_fan(const ModelVariant& model, const PayoffVariant& payoff, int n_steps, int n_fan, uint64_t seed);

} // namespace mc::cpu
