// Paridad CPU <-> GPU. Etiqueta CTest "cuda": solo tiene sentido en un equipo con GPU NVIDIA (se
// valida con tools/colab_validate.sh); en builds sin CUDA los casos se omiten con un aviso.
//
//  * Sobol: si existe el volcado de la GPU (variable de entorno MC_SOBOL_DUMP, generado por
//    tools/sobol_dump.cu), los primeros 4096 puntos sin scrambling deben coincidir bit a bit con el
//    generador de la CPU. Esto confirma el orden Gray-code y las tablas Joe-Kuo de cuRAND.
//  * Precios: cada familia de métodos ejecutada en GPU (float) debe coincidir con la CPU (double)
//    dentro de sus errores estándar combinados.
#include "doctest.h"
#include "cpu/sobol.hpp"
#include "engine/api.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>

namespace {
bool gpu_ready() { return mc::cuda_available(); }

struct Pair { MCResult cpu, gpu; };
Pair both(const std::function<MCResult(const mc::RunOptions&)>& run) {
    mc::RunOptions c; c.backend = mc::Backend::Cpu;
    mc::RunOptions g; g.backend = mc::Backend::Cuda;
    return {run(c), run(g)};
}
bool agree(const Pair& p, double slack_abs) {
    const double se = std::sqrt(p.cpu.std_error * p.cpu.std_error + p.gpu.std_error * p.gpu.std_error);
    return std::abs(p.cpu.price - p.gpu.price) < 5.0 * se + slack_abs;
}
} // namespace

TEST_SUITE("cuda") {

TEST_CASE("Sobol de cuRAND == Sobol de la CPU (primeros 4096 puntos, sin scrambling)") {
    const char* path = std::getenv("MC_SOBOL_DUMP");
    if (!path) { MESSAGE("MC_SOBOL_DUMP no definida: se omite (genere el volcado con tools/sobol_dump.cu)"); return; }
    std::ifstream f(path);
    REQUIRE_MESSAGE(f.good(), "no se pudo abrir " << path);
    std::string line;
    int lines = 0;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        int dim; ss >> dim;
        unsigned w; int n = 0;
        while (ss >> w) {
            INFO("dim " << dim << " punto " << n);
            REQUIRE(w == mc::cpu::sobol_raw(dim, (uint64_t)n));
            ++n;
        }
        CHECK(n == 4096);
        ++lines;
    }
    CHECK(lines == 6);
}

TEST_CASE("Precios CPU vs GPU: MC, QMC, MLMC y MLQMC (europea GBM)") {
    if (!gpu_ready()) { MESSAGE("sin GPU: se omite"); return; }
    GBMParams g; PayoffVariant pv = European{100.0, 0.05, 1.0};
    CHECK(agree(both([&](const mc::RunOptions& o) { return mc::run_mc(g, pv, 0.01, 64, MCConfig{}, o); }), 0.02));
    for (NoiseMode m : {NoiseMode::Raw, NoiseMode::BrownianBridge, NoiseMode::PCA})
        CHECK(agree(both([&](const mc::RunOptions& o) { return mc::run_qmc(g, pv, 0.005, 64, QMCConfig{}, m, o); }), 0.02));
    CHECK(agree(both([&](const mc::RunOptions& o) { return mc::run_mlmc(g, pv, 0.01, MLMCConfig{}, o); }), 0.02));
    CHECK(agree(both([&](const mc::RunOptions& o) { return mc::run_mlqmc(g, pv, 0.01, MLMCConfig{}, QMCConfig{}, NoiseMode::Raw, o); }), 0.02));
}

TEST_CASE("Precios CPU vs GPU: Heston, Dupire, asiática con control e IS") {
    if (!gpu_ready()) { MESSAGE("sin GPU: se omite"); return; }
    HestonParams h; h.compute_cholesky(); DupireLocalParams d; GBMParams g;
    CHECK(agree(both([&](const mc::RunOptions& o) { return mc::run_mc(h, European{100.0, 0.05, 1.0}, 0.02, 64, MCConfig{}, o); }), 0.03));
    CHECK(agree(both([&](const mc::RunOptions& o) { return mc::run_mc(d, European{100.0, 0.05, 1.0}, 0.02, 64, MCConfig{}, o); }), 0.03));
    const double Ec = geom_asian_analytic(g.S0, 100.0, g.T, g.mu, g.sigma, 64);
    CHECK(agree(both([&](const mc::RunOptions& o) {
        return mc::run_mc_cv(g, g, Asian{100.0}, GeomAsian{100.0}, Ec, 1.0, 0.01, 64, MCConfig{}, o); }), 0.03));
    CHECK(agree(both([&](const mc::RunOptions& o) {
        return mc::run_is(g, European{150.0, 0.05, 1.0}, 1.9, 0.01, MCConfig{}, o); }), 0.01));
}

} // TEST_SUITE
