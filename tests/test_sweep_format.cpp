// El formato de salida de run_precision_sweep lo parsea gen_informe.py (regex de filas).
// Los ficheros golden se generaron con la implementación ORIGINAL (std::format) sobre los
// mismos métodos sintéticos, así que esto garantiza que el port a printf es idéntico byte
// a byte (salvo el tiempo de pared, que se enmascara).

#include "doctest.h"
#include "mc_format.hpp"
#include "sweep.hpp"
#include "utils.hpp"

#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    REQUIRE_MESSAGE(f.good(), "no se pudo abrir " << path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string run_driver(double eps_finest) {
    std::string out;
    mc_output_sink() = [&](const std::string& s) { out += s; out += '\n'; };

    std::vector<SweepMethod> m;
    m.push_back({"MC", [](unsigned so, double eps) {
        MCResult r; r.price = 10.45 + 0.0003 * std::sin(so * 0.01 + eps * 100);
        r.std_error = eps / 3; r.n_samples = (long long)(2.0 / (eps * eps));
        r.time_s = 0.00001 / (eps * eps); return r; }});
    m.push_back({"MLMC (nombre largo para desborde de columna)", [](unsigned so, double eps) {
        MCResult r; r.price = 10.45 + 0.0002 * std::cos(so * 0.02 + eps * 50);
        r.std_error = eps / 4; r.n_samples = (long long)(1.0 / eps);
        r.time_s = 0.0001 / eps; return r; }});
    m.push_back({"QMC Raw", [](unsigned, double eps) -> MCResult {
        if (eps < 0.001) throw SobolLimitReached{};
        MCResult r; r.price = 10.451; r.std_error = eps / 10; r.n_samples = 1 << 20; r.time_s = 0.5; return r; }});
    m.push_back({"Lento", [](unsigned, double eps) {
        MCResult r; r.price = 10.5; r.std_error = eps; r.n_samples = 5;
        r.time_s = 2.0 / (eps * 100); return r; }});

    const double ref = 10.450584;
    run_precision_sweep("ejemplo_fake", m, ref, eps_scale_125(eps_finest));
    run_precision_sweep("ejemplo_fake2", m, ref, eps_scale_125(0.00033), 1e-9, 4, 2, 0.5);
    mc_output_sink() = nullptr;
    return std::regex_replace(out, std::regex("wall=[0-9.]+s"), "wall=X");
}

} // namespace

TEST_SUITE("fast") {

TEST_CASE("run_precision_sweep: salida identica al golden (eps fino 1e-4)") {
    CHECK(run_driver(0.0001) == read_file(std::string(MC_TEST_DIR) + "/golden/sweep_fake_run1.txt"));
}

TEST_CASE("run_precision_sweep: salida identica al golden (eps fino 2e-3)") {
    CHECK(run_driver(0.002) == read_file(std::string(MC_TEST_DIR) + "/golden/sweep_fake_run2.txt"));
}

TEST_CASE("eps_scale_125: escala 1-2-5 y recorte por abajo") {
    auto v = eps_scale_125(0.001);
    REQUIRE(v.size() == 6);
    CHECK(v.front() == doctest::Approx(0.05));
    CHECK(v.back()  == doctest::Approx(0.001));
    auto w = eps_scale_125(0.000003);        // mas fino que la escala predefinida
    CHECK(w.back() == doctest::Approx(0.000003));
    CHECK(w.size() == 13);
}

TEST_CASE("mc_shortest equivale a '{}' de std::format") {
    CHECK(mc_shortest(0.05) == "0.05");
    CHECK(mc_shortest(0.001) == "0.001");
    CHECK(mc_shortest(0.0001) == "1e-04");   // la forma mas corta (como "{}" de std::format)
    CHECK(mc_shortest(0.00001) == "1e-05");
    CHECK(mc_shortest(5e-5) == "5e-05");
    CHECK(mc_shortest(0.1) == "0.1");
}

} // TEST_SUITE
