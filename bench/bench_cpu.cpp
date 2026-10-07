// Harness de benchmarks del motor CPU.
//
// Metodología: 2 repeticiones de calentamiento + N repeticiones medidas (por defecto 7) de
// cada carga; se reportan mediana, mínimo, IQR y CV% del tiempo de pared, y el rendimiento
// derivado (Mcaminos/s y ns por paso-camino). Salida en tabla y, opcionalmente, JSON con
// información de la máquina/compilación/commit (tools/bench_report.py lo convierte en
// docs/perf/*.md).
//
// Uso:  bench_cpu [--filter texto] [--reps N] [--threads a,b,c] [--json fichero] [--quick] [--list]

#include "../cpu/mc_cpu.hpp"
#include "../cpu/thread_pool.hpp"
#include "bench_workloads.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef MC_GIT_HASH
#define MC_GIT_HASH "desconocido"
#endif
#ifndef MC_BUILD_FLAGS
#define MC_BUILD_FLAGS ""
#endif

namespace {

struct Stats {
    double median = 0, min = 0, q1 = 0, q3 = 0, mean = 0, cv_pct = 0;
    int n = 0;
};

double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const double pos = q * (v.size() - 1);
    const size_t lo = (size_t)pos;
    const size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

Stats summarize(const std::vector<double>& t) {
    Stats s;
    s.n = (int)t.size();
    s.median = quantile(t, 0.5);
    s.min = *std::min_element(t.begin(), t.end());
    s.q1 = quantile(t, 0.25);
    s.q3 = quantile(t, 0.75);
    double m = 0; for (double x : t) m += x; m /= t.size();
    double v = 0; for (double x : t) v += (x - m) * (x - m); v /= std::max<size_t>(1, t.size() - 1);
    s.mean = m;
    s.cv_pct = m > 0 ? 100.0 * std::sqrt(v) / m : 0.0;
    return s;
}

std::string cpu_name() {
    if (const char* p = std::getenv("MC_CPU_NAME")) return p;   // nombre legible, opcional
#if defined(_WIN32)
    if (const char* p = std::getenv("PROCESSOR_IDENTIFIER")) return p;
#else
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
#endif
    return "desconocida";
}

std::string compiler_name() {
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_VER);
#else
    return "desconocido";
#endif
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o;
}

struct Row {
    std::string id, desc;
    int threads;
    double paths;          // caminos por repetición
    double steps;          // pasos por camino (para ns/paso)
    Stats st;
};

} // namespace

int main(int argc, char** argv) {
    std::string filter, json_path;
    int reps = 7;
    bool quick = false, list_only = false;
    std::vector<int> thread_list;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--filter") filter = next();
        else if (a == "--reps") reps = std::max(3, std::atoi(next().c_str()));
        else if (a == "--json") json_path = next();
        else if (a == "--quick") quick = true;
        else if (a == "--list") list_only = true;
        else if (a == "--threads") {
            std::stringstream ss(next()); std::string tok;
            while (std::getline(ss, tok, ',')) if (!tok.empty()) thread_list.push_back(std::atoi(tok.c_str()));
        } else {
            std::fprintf(stderr, "argumento desconocido: %s\n", a.c_str());
            return 2;
        }
    }
    const int hw = (int)std::max(1u, std::thread::hardware_concurrency());
    if (thread_list.empty()) thread_list = {hw};
    if (quick) reps = 3;

    auto workloads = mcbench::make_workloads(quick);
    if (list_only) {
        for (auto& w : workloads) std::printf("%-22s %s\n", w.id.c_str(), w.desc.c_str());
        return 0;
    }

    std::printf("bench_cpu | cpu: %s | hilos HW: %d | %s | commit %s\n", cpu_name().c_str(), hw,
                compiler_name().c_str(), MC_GIT_HASH);
    std::printf("%-22s %7s %10s %10s %9s %8s %12s %10s\n", "carga", "hilos", "mediana(s)", "min(s)",
                "IQR(s)", "CV(%)", "Mcaminos/s", "ns/paso");

    std::vector<Row> rows;
    for (auto& w : workloads) {
        if (!filter.empty() && w.id.find(filter) == std::string::npos) continue;
        for (int th : (w.single_thread_only ? std::vector<int>{1} : thread_list)) {
            mc::cpu::set_num_threads(th);
            for (int i = 0; i < 2; i++) w.run();   // calentamiento
            std::vector<double> t;
            for (int i = 0; i < reps; i++) {
                auto t0 = std::chrono::steady_clock::now();
                w.run();
                t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            Row r{w.id, w.desc, th, w.paths, w.steps, summarize(t)};
            rows.push_back(r);
            const double mpps = r.paths / r.st.median / 1e6;
            const double ns_step = r.st.median / (r.paths * r.steps) * 1e9;
            std::printf("%-22s %7d %10.4f %10.4f %9.4f %8.2f %12.3f %10.3f\n", r.id.c_str(), th,
                        r.st.median, r.st.min, r.st.q3 - r.st.q1, r.st.cv_pct, mpps, ns_step);
            std::fflush(stdout);
        }
    }

    if (!json_path.empty()) {
        std::ofstream f(json_path);
        f << "{\n  \"machine\": {\"cpu\": \"" << json_escape(cpu_name()) << "\", \"hw_threads\": " << hw
          << ", \"compiler\": \"" << json_escape(compiler_name()) << "\", \"flags\": \""
          << json_escape(MC_BUILD_FLAGS) << "\", \"commit\": \"" << MC_GIT_HASH << "\"},\n"
          << "  \"reps\": " << reps << ",\n  \"results\": [\n";
        for (size_t i = 0; i < rows.size(); i++) {
            const Row& r = rows[i];
            f << "    {\"id\": \"" << r.id << "\", \"threads\": " << r.threads << ", \"paths\": " << r.paths
              << ", \"steps\": " << r.steps << ", \"median_s\": " << r.st.median << ", \"min_s\": " << r.st.min
              << ", \"q1_s\": " << r.st.q1 << ", \"q3_s\": " << r.st.q3 << ", \"cv_pct\": " << r.st.cv_pct
              << ", \"mpaths_per_s\": " << r.paths / r.st.median / 1e6 << "}" << (i + 1 < rows.size() ? "," : "") << "\n";
        }
        f << "  ]\n}\n";
        std::printf("JSON escrito en %s\n", json_path.c_str());
    }
    return 0;
}
