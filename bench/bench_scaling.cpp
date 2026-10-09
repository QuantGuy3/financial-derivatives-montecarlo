// Sonda de escalado por hilos: separa el techo del hardware de la sobrecarga del software.
//
// La misma carga (MC GBM europea, 64 pasos, 1024 chunks de 4096 caminos) se ejecuta de tres formas:
//   raw     T hilos propios, reparto estático (1024/T chunks cada uno), arrancan a la vez. Sin pool,
//           sin rondas, sin reducción: es el techo de lo que da la máquina con T hilos.
//   rawpin  igual, con cada hilo fijado a un núcleo físico distinto (a un procesador lógico distinto
//           si T supera el nº de núcleos). Mide si el planificador coloca mal los hilos.
//   flat    el pool del motor en UNA sola ronda (parallel_for de los 1024 chunks).
//   engine  simulate_range tal cual (rondas crecientes + reducción en orden).
//
// engine - flat = coste de las rondas; flat - raw = coste del pool (despertar, reparto dinámico).
//
// Uso: bench_scaling [--threads 1,2,4,8,16] [--reps 7] [--steps 64] [--log2paths 22] [--boxmuller]

#include "../cpu/mc_cpu.hpp"
#include "../cpu/noise.hpp"
#include "../cpu/params.hpp"
#include "../cpu/path_sim.hpp"
#include "../cpu/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace mc::cpu;
using SteadyClock = std::chrono::steady_clock;

namespace {

// Un procesador lógico por núcleo físico primero; después los hermanos SMT.
std::vector<int> logical_order() {
    std::vector<int> first, rest;
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> info(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (GetLogicalProcessorInformation(info.data(), &len)) {
        for (const auto& e : info) {
            if (e.Relationship != RelationProcessorCore) continue;
            bool is_first = true;
            for (int b = 0; b < 64; b++) {
                if (!(e.ProcessorMask & (1ULL << b))) continue;
                (is_first ? first : rest).push_back(b);
                is_first = false;
            }
        }
    }
#endif
    if (first.empty()) {
        const int hc = (int)std::max(1u, std::thread::hardware_concurrency());
        for (int i = 0; i < hc; i++) first.push_back(i);
    }
    first.insert(first.end(), rest.begin(), rest.end());
    return first;
}

void pin_current_thread(int logical) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << logical);
#else
    (void)logical;
#endif
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Setup {
    CpuModel m;
    CpuPayoff p;
    int n_steps;
    long long N;
    NormalMethod nm;
};

// T hilos propios con reparto estático. Devuelve el tiempo de pared desde la señal de salida.
double run_raw(const Setup& s, int T, bool pin, double* checksum) {
    const double sqrt_h = std::sqrt(s.m.T / s.n_steps);
    RngNoise noise(1u, Stream::Main, 0, s.n_steps * s.m.noise_dim, sqrt_h, s.nm);
    PathSim sim(s.m, s.p, s.n_steps, noise);
    const long long cp = sim.chunk_paths();
    const long long n_chunks = (s.N + cp - 1) / cp;
    const auto order = logical_order();

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<Padded<ChunkAcc>> acc((size_t)T);
    std::vector<std::thread> th;
    for (int t = 0; t < T; t++) {
        th.emplace_back([&, t] {
            if (pin) pin_current_thread(order[(size_t)t % order.size()]);
            Scratch sc;
            ready.fetch_add(1);
            while (!go.load(std::memory_order_acquire)) {}
            const long long c0 = n_chunks * t / T, c1 = n_chunks * (t + 1) / T;
            for (long long c = c0; c < c1; c++) {
                const long long p0 = c * cp;
                sim.run_chunk((uint64_t)p0, (int)std::min<long long>(cp, s.N - p0), sc, acc[(size_t)t].v);
            }
        });
    }
    while (ready.load() < T) std::this_thread::yield();
    const auto t0 = SteadyClock::now();
    go.store(true, std::memory_order_release);
    for (auto& x : th) x.join();
    const double dt = std::chrono::duration<double>(SteadyClock::now() - t0).count();
    double sum = 0;
    for (auto& a : acc) sum += a.v.acc.to_moments().mean;
    *checksum = sum;
    return dt;
}

// Aritmética entera pura (xoshiro256++, sin memoria ni coma flotante): T hilos, mismo trabajo total.
// Si esto escala igual que "raw", el techo es la frecuencia del procesador y no el código.
double run_spin(int T, bool pin, double* checksum) {
    const long long total = 1LL << 29;
    const auto order = logical_order();
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<Padded<uint64_t>> acc((size_t)T);
    std::vector<std::thread> th;
    for (int t = 0; t < T; t++) {
        th.emplace_back([&, t] {
            if (pin) pin_current_thread(order[(size_t)t % order.size()]);
            Xoshiro256pp g = Xoshiro256pp::for_path(7, Stream::Test, 0, (uint64_t)t);
            ready.fetch_add(1);
            while (!go.load(std::memory_order_acquire)) {}
            uint64_t x = 0;
            for (long long i = 0, n = total / T; i < n; i++) x += g.next();
            acc[(size_t)t].v = x;
        });
    }
    while (ready.load() < T) std::this_thread::yield();
    const auto t0 = SteadyClock::now();
    go.store(true, std::memory_order_release);
    for (auto& x : th) x.join();
    const double dt = std::chrono::duration<double>(SteadyClock::now() - t0).count();
    uint64_t sum = 0;
    for (auto& a : acc) sum += a.v;
    *checksum = (double)sum;
    return dt;
}

// Latencia de reparto del pool: `rounds` rondas de T tareas de ~`task_us` microsegundos.
// Devuelve microsegundos por ronda (lo ideal es task_us).
double run_dispatch(int T, int rounds, double task_us) {
    set_num_threads(T);
    ThreadPool& pool = global_pool();
    // calibra el nº de iteraciones de la tarea
    auto burn = [](long long it) {
        Xoshiro256pp g = Xoshiro256pp::for_path(3, Stream::Test, 0, 0);
        uint64_t x = 0;
        for (long long i = 0; i < it; i++) x += g.next();
        return x;
    };
    const auto c0 = SteadyClock::now();
    volatile uint64_t sink = burn(1 << 22);
    (void)sink;
    const double ns_it = std::chrono::duration<double, std::nano>(SteadyClock::now() - c0).count() / (1 << 22);
    const long long it = std::max<long long>(1, (long long)(task_us * 1000.0 / ns_it));
    std::vector<Padded<uint64_t>> out((size_t)T);
    const auto t0 = SteadyClock::now();
    for (int r = 0; r < rounds; r++)
        pool.parallel_for((size_t)T, [&](size_t t, int) { out[t].v += burn(it); });
    return std::chrono::duration<double, std::micro>(SteadyClock::now() - t0).count() / rounds;
}

// El pool del motor en una sola ronda (sin reducción por rondas).
double run_flat(const Setup& s, int T, double* checksum) {
    set_num_threads(T);
    ThreadPool& pool = global_pool();
    const double sqrt_h = std::sqrt(s.m.T / s.n_steps);
    RngNoise noise(1u, Stream::Main, 0, s.n_steps * s.m.noise_dim, sqrt_h, s.nm);
    PathSim sim(s.m, s.p, s.n_steps, noise);
    const long long cp = sim.chunk_paths();
    const long long n_chunks = (s.N + cp - 1) / cp;
    std::vector<Scratch> scratch((size_t)pool.threads());
    std::vector<Padded<ChunkAcc>> slots((size_t)n_chunks);
    const auto t0 = SteadyClock::now();
    pool.parallel_for((size_t)n_chunks, [&](size_t c, int worker) {
        const long long p0 = (long long)c * cp;
        sim.run_chunk((uint64_t)p0, (int)std::min<long long>(cp, s.N - p0), scratch[(size_t)worker], slots[c].v);
    });
    Moments mo;
    for (auto& sl : slots) mo.merge(sl.v.acc.to_moments());
    const double dt = std::chrono::duration<double>(SteadyClock::now() - t0).count();
    *checksum = mo.mean;
    return dt;
}

double run_engine(const Setup& s, int T, double* checksum) {
    set_num_threads(T);
    ThreadPool& pool = global_pool();
    const double sqrt_h = std::sqrt(s.m.T / s.n_steps);
    RngNoise noise(1u, Stream::Main, 0, s.n_steps * s.m.noise_dim, sqrt_h, s.nm);
    PathSim sim(s.m, s.p, s.n_steps, noise);
    const auto t0 = SteadyClock::now();
    RangeResult r = simulate_range(pool, sim, 0, s.N, nullptr);
    const double dt = std::chrono::duration<double>(SteadyClock::now() - t0).count();
    *checksum = r.moments.mean;
    return dt;
}

std::vector<int> parse_list(const std::string& s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        v.push_back(std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return v;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> threads = {1, 2, 4, 8, 16};
    int reps = 7, n_steps = 64, log2paths = 22;
    NormalMethod nm = NormalMethod::Ziggurat;
    bool dispatch = false;
    std::vector<std::string> modes = {"raw", "rawpin", "flat", "engine"};
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&] { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--threads") threads = parse_list(next());
        else if (a == "--reps") reps = std::max(3, std::atoi(next().c_str()));
        else if (a == "--steps") n_steps = std::atoi(next().c_str());
        else if (a == "--log2paths") log2paths = std::atoi(next().c_str());
        else if (a == "--boxmuller") nm = NormalMethod::BoxMuller;
        else if (a == "--dispatch") dispatch = true;
        else if (a == "--modes") {
            modes.clear();
            const std::string s = next();
            size_t p = 0;
            while (p < s.size()) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                modes.push_back(s.substr(p, q - p));
                p = q + 1;
            }
        }
    }

    if (dispatch) {
        std::printf("bench_scaling --dispatch | microsegundos por ronda de T tareas (ideal = duracion de la tarea)\n");
        std::printf("%5s %12s %12s %12s\n", "hilos", "tarea 5us", "tarea 50us", "tarea 500us");
        for (int T : threads) {
            double r[3];
            const double us[3] = {5, 50, 500};
            for (int k = 0; k < 3; k++) {
                run_dispatch(T, 200, us[k]);   // calentamiento
                std::vector<double> v;
                for (int i = 0; i < reps; i++) v.push_back(run_dispatch(T, k == 2 ? 300 : 2000, us[k]));
                r[k] = median(v);
            }
            std::printf("%5d %12.1f %12.1f %12.1f\n", T, r[0], r[1], r[2]);
            std::fflush(stdout);
        }
        return 0;
    }

    GBMParams gbm;
    Setup s{make_cpu_model(gbm), make_cpu_payoff(European{100.0, 0.05, 1.0}), n_steps, 1LL << log2paths, nm};

    std::printf("bench_scaling | %d pasos | 2^%d caminos | %s | mediana de %d (intercaladas)\n", n_steps, log2paths,
                nm == NormalMethod::Ziggurat ? "Ziggurat" : "Box-Muller", reps);
    std::printf("%-8s %5s %10s %10s %9s %7s\n", "modo", "hilos", "mediana(s)", "min(s)", "acelera", "efic.%");

    // Las repeticiones se intercalan entre modos para que un cambio de frecuencia afecte a todos.
    std::vector<double> t1((size_t)modes.size(), 0.0);
    for (int T : threads) {
        std::vector<std::vector<double>> t(modes.size());
        double chk = 0;
        for (int r = -1; r < reps; r++) {   // r = -1: calentamiento
            for (size_t mi = 0; mi < modes.size(); mi++) {
                const std::string& mode = modes[mi];
                double dt = 0;
                if (mode == "raw") dt = run_raw(s, T, false, &chk);
                else if (mode == "rawpin") dt = run_raw(s, T, true, &chk);
                else if (mode == "spin") dt = run_spin(T, false, &chk);
                else if (mode == "spinpin") dt = run_spin(T, true, &chk);
                else if (mode == "flat") dt = run_flat(s, T, &chk);
                else dt = run_engine(s, T, &chk);
                if (r >= 0) t[mi].push_back(dt);
            }
        }
        for (size_t mi = 0; mi < modes.size(); mi++) {
            const double med = median(t[mi]);
            const double mn = *std::min_element(t[mi].begin(), t[mi].end());
            if (T == threads.front()) t1[mi] = med * T;
            const double sp = t1[mi] / med;
            std::printf("%-8s %5d %10.4f %10.4f %8.2fx %7.0f\n", modes[mi].c_str(), T, med, mn, sp, 100.0 * sp / T);
        }
        std::fflush(stdout);
    }
    return 0;
}
