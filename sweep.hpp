#pragma once
// Barrido de precisión por método (independiente del backend).
#include "mc_types.hpp"
#include <functional>
#include <string>
#include <vector>

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
