#include "qmc_cpu.hpp"

#include "params.hpp"
#include "path_sim.hpp"
#include "qmc_noise.hpp"
#include "sobol.hpp"
#include "thread_pool.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace mc::cpu {

namespace {

uint64_t pow2_floor(uint64_t x) {
    uint64_t p = 1;
    while (p * 2 <= x) p *= 2;
    return p;
}

// Puntos por tarea (potencia de 2 para que los rangos de cada duplicación queden alineados).
// Mínimo 1024: amortiza el coste O(32·D) de reconstruir el estado de Sobol al inicio.
uint64_t qmc_chunk_points(int D) {
    const double c = 262144.0 / std::max(1, D);
    return pow2_floor((uint64_t)std::clamp(c, 1024.0, 65536.0));
}

bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

} // namespace

MCResult run_qmc(const ModelVariant& model, const PayoffVariant& payoff,
                 double eps, int n_steps, const QMCConfig& cfg, NoiseMode mode, const CpuOptions& opt) {
    const auto t0 = Clock::now();
    if (opt.threads > 0) set_num_threads(opt.threads);
    ThreadPool& pool = global_pool();
    const CpuModel m = make_cpu_model(model);
    const CpuPayoff p = make_cpu_payoff(payoff);
    if (n_steps < 1) throw std::invalid_argument("run_qmc: n_steps debe ser >= 1");
    if (!(eps > 0.0)) throw std::invalid_argument("run_qmc: eps debe ser > 0");
    if (cfg.R < 2) throw std::invalid_argument("run_qmc: se necesitan al menos 2 réplicas (R >= 2)");
    if (mode != NoiseMode::Raw) {
        if (m.noise_dim != 1)
            throw std::invalid_argument("run_qmc: BB/PCA solo están soportados para ruido de dimensión 1 (GBM/Dupire)");
        if (mode == NoiseMode::BrownianBridge && !is_pow2(n_steps))
            throw std::invalid_argument("run_qmc: el Brownian Bridge exige n_steps potencia de 2");
    }

    const int R = cfg.R;
    const int D = n_steps * m.noise_dim;
    const double sqrt_h = std::sqrt(m.T / n_steps);
    const bool use_sobol = (D <= D_MAX_SOBOL);
    const double raw_scale = (mode == NoiseMode::Raw) ? sqrt_h : 1.0;   // BB/PCA ya incluyen sqrt(T)

    // ---- ruido por réplica ------------------------------------------------------------------
    std::vector<std::unique_ptr<ScrambledSobol>> sob(R);
    if (use_sobol)
        pool.parallel_for((size_t)R, [&](size_t r, int) {
            sob[r] = std::make_unique<ScrambledSobol>(cfg.seed + (unsigned)r, D);
        });
    std::vector<std::unique_ptr<NoiseSource>> base(R), wrapped(R);
    std::vector<std::unique_ptr<PathSim>> sims(R);
    const BBData* bb = (mode == NoiseMode::BrownianBridge) ? &bb_precompute(n_steps, m.T) : nullptr;
    const PCAData* pca = (mode == NoiseMode::PCA) ? &pca_compute(n_steps, m.T) : nullptr;
    for (int r = 0; r < R; r++) {
        if (use_sobol) base[r] = std::make_unique<SobolNoise>(*sob[r], raw_scale);
        else base[r] = std::make_unique<RngNoise>(cfg.seed + (unsigned)r, Stream::Main, 0, D, raw_scale, opt.normal);
        const NoiseSource* src = base[r].get();
        if (bb) { wrapped[r] = std::make_unique<BrownianBridgeNoise>(*base[r], *bb); src = wrapped[r].get(); }
        else if (pca) { wrapped[r] = std::make_unique<PcaNoise>(*base[r], *pca); src = wrapped[r].get(); }
        sims[r] = std::make_unique<PathSim>(m, p, n_steps, *src);
    }

    // ---- bucle de duplicaciones -----------------------------------------------------------
    RunInfo info;
    info.threads = pool.threads();
    uint64_t seq = 0;
    std::vector<Moments> mom(R);
    std::vector<uint64_t> n_done(R, 0);
    std::vector<double> replica_means(R, 0.0);
    std::vector<Scratch> scratch(pool.threads());
    std::vector<Padded<ChunkAcc>> slots(256);

    uint64_t n_per_rep = (uint64_t)std::max(64, cfg.n0);
    if (use_sobol) n_per_rep = pow2_floor(n_per_rep);
    const uint64_t qcp = qmc_chunk_points(D);

    double var_of_means = 1e30, grand_mean = 0.0;
    long long total_N = 0;
    bool stopped = false;

    auto emit = [&](Stage st, int doubling, bool final_) {
        if (!opt.sink) return;
        Snapshot s;
        s.stage = st; s.seq = seq++; s.elapsed_s = seconds_since(t0);
        s.n_done = total_N; s.mean = grand_mean; s.std_error = std::sqrt(var_of_means);
        s.eps_target = eps; s.doubling = doubling; s.n_per_replica = (long long)n_per_rep;
        s.replica_means = replica_means.data(); s.n_replicas = R; s.is_final = final_;
        opt.sink->on_snapshot(s);
    };

    struct Task { int r; uint64_t first; int count; };

    for (int doublings = 0; doublings < cfg.max_doublings && !stopped; doublings++) {
        if (use_sobol && n_per_rep > (uint64_t)UINT_MAX) throw SobolLimitReached{};

        std::vector<Task> tasks;
        for (int r = 0; r < R; r++)
            for (uint64_t a = n_done[r]; a < n_per_rep; a += qcp)
                tasks.push_back({r, a, (int)std::min<uint64_t>(qcp, n_per_rep - a)});

        bool complete = true;   // ¿se ejecutaron todas las tareas de esta duplicación?
        for (size_t g0 = 0; g0 < tasks.size(); g0 += slots.size()) {
            const size_t g1 = std::min(tasks.size(), g0 + slots.size());
            for (size_t i = 0; i < g1 - g0; i++) slots[i].v = ChunkAcc{};
            pool.parallel_for(g1 - g0, [&](size_t i, int worker) {
                const Task& t = tasks[g0 + i];
                sims[t.r]->run_chunk(t.first, t.count, scratch[worker], slots[i].v);
            });
            // fusión en orden de tarea (réplica-mayor, chunk-menor): determinista
            for (size_t i = 0; i < g1 - g0; i++) {
                const Task& t = tasks[g0 + i];
                mom[t.r].merge(slots[i].v.acc.to_moments());
                info.n_nonfinite += slots[i].v.nonfinite;
            }
            if (opt.sink && opt.sink->should_cancel()) info.cancelled = true;
            else if (opt.max_seconds > 0.0 && seconds_since(t0) > opt.max_seconds) info.truncated = true;
            if (info.cancelled || info.truncated) {
                stopped = true;
                if (g1 < tasks.size()) complete = false;
                break;
            }
        }

        if (!complete) {
            // Duplicación a medias: sus réplicas tienen distinto nº de puntos, así que se
            // conserva el resultado de la última duplicación completa. Si no hubo ninguna, se
            // estima con lo acumulado (solo réplicas con datos).
            if (total_N == 0) {
                double mean = 0.0; int used = 0;
                for (int r = 0; r < R; r++) {
                    replica_means[r] = mom[r].mean;
                    if (mom[r].n > 0) { mean += mom[r].mean; ++used; }
                    total_N += mom[r].n;
                }
                grand_mean = used ? mean / used : 0.0;
                var_of_means = 0.0;
                if (used > 1) {
                    double v = 0.0;
                    for (int r = 0; r < R; r++) if (mom[r].n > 0) v += (mom[r].mean - grand_mean) * (mom[r].mean - grand_mean);
                    var_of_means = v / (used - 1) / used;
                }
            }
            break;
        }
        for (int r = 0; r < R; r++) { n_done[r] = n_per_rep; replica_means[r] = mom[r].mean; }

        double mean = 0.0;
        for (double v : replica_means) mean += v;
        mean /= R;
        double v = 0.0;
        for (double rv : replica_means) v += (rv - mean) * (rv - mean);
        v = v / (R - 1);
        var_of_means = v / R;
        grand_mean = mean;
        total_N = (long long)R * (long long)n_per_rep;
        emit(Stage::Doubling, doublings, false);

        if (var_of_means < eps * eps / 2.0) break;
        if (stopped) break;
        n_per_rep *= 2;
    }

    const double t_s = seconds_since(t0);
    emit(Stage::Done, -1, true);
    if (opt.info) *opt.info = std::move(info);
    return {grand_mean, std::sqrt(var_of_means), total_N, t_s};
}

} // namespace mc::cpu
