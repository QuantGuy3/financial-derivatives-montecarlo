#pragma once
// Utilidades comunes de los ejemplos: selección del backend y del nº de hilos.
//
//   ejemploNN [eps_fino] [nomc|noml|noqr|noqmc] [--backend=cpu|cuda] [--threads=N]
//
// Los flags se retiran de argv, así que los argumentos posicionales de cada ejemplo
// (argv[1] = eps más fino, argv[2] = métodos a omitir) siguen funcionando igual.
// Por defecto se usa CUDA si el binario la incluye y hay GPU; si no, la CPU con todos los hilos.

#include "../engine/api.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace mc_examples {

inline void init(int& argc, char** argv) {
    mc::RunOptions& opt = mc::default_options();
    opt.backend = mc::cuda_available() ? mc::Backend::Cuda : mc::Backend::Cpu;

    int w = 1;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto value = [&](const char* flag) -> std::string {
            const std::string f = flag;
            if (a.rfind(f + "=", 0) == 0) return a.substr(f.size() + 1);
            if (a == f && i + 1 < argc) return argv[++i];
            return "";
        };
        if (a.rfind("--backend", 0) == 0) {
            try { opt.backend = mc::backend_from_string(value("--backend")); }
            catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); std::exit(2); }
        } else if (a.rfind("--threads", 0) == 0) {
            opt.threads = std::atoi(value("--threads").c_str());
        } else if (a == "--help" || a == "-h") {
            std::printf("uso: %s [eps_fino] [nomc|noml|noqr|noqmc] [--backend=cpu|cuda] [--threads=N]\n", argv[0]);
            std::exit(0);
        } else {
            argv[w++] = argv[i];
        }
    }
    argc = w;

    if (opt.backend == mc::Backend::Cuda && !mc::cuda_available()) {
        std::fprintf(stderr, "backend CUDA no disponible (binario sin CUDA o sin GPU); use --backend=cpu\n");
        std::exit(2);
    }
    std::fprintf(stderr, "[backend] %s", mc::backend_name(opt.backend));
    if (opt.backend == mc::Backend::Cpu)
        std::fprintf(stderr, ", %d hilos\n", opt.threads > 0 ? opt.threads : mc::hardware_threads());
    else
        std::fprintf(stderr, "\n");
}

} // namespace mc_examples
