#pragma once
// Registro de cargas de trabajo de bench_cpu. Cada carga ejecuta una simulación de tamaño fijo
// (mismos caminos en cada repetición) para que el tiempo sea comparable entre commits.

#include <functional>
#include <string>
#include <vector>

namespace mcbench {

struct Workload {
    std::string id;
    std::string desc;
    double paths = 0;               // caminos simulados por ejecución
    double steps = 1;               // pasos por camino
    bool single_thread_only = false;
    std::function<void()> run;
};

// quick=true reduce los tamaños para una comprobación rápida (no comparable con la normal).
std::vector<Workload> make_workloads(bool quick);

} // namespace mcbench
