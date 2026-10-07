#include "mlmc_cpu.hpp"

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
#include <string>

namespace mc::cpu {

namespace {

// ---- motores de nivel -----------------------------------------------------------------------------

struct LTask {
    int l;
    int r;            // réplica (0 en MLMC)
    uint64_t first;   // índice del primer camino/punto dentro de (nivel, réplica)
    int count;
};

// Abstrae de dónde salen las muestras de un nivel (pseudoaleatorias o Sobol por réplica).
class LevelEngine {
public:
    virtual ~LevelEngine() = default;
    virtual int replicas() const = 0;
    // Añade las tareas para `extra` muestras nuevas por réplica del nivel l y avanza el offset.
    // Solo se llama desde el hilo coordinador.
    virtual void plan(int l, long long extra, std::vector<LTask>& out) = 0;
    // Thread-safe una vez planificado.
    virtual void run(const LTask& t, Scratch& s, ChunkAcc2& out) const = 0;
};

int pow_int(int M, int l) {
    long long v = 1;
    for (int i = 0; i < l; i++) {
        v *= M;
        if (v > (1 << 24)) throw std::invalid_argument("MLMC: n_fine = M^l demasiado grande");
    }
    return (int)v;
}

class MlmcEngine final : public LevelEngine {
public:
    MlmcEngine(const CpuModel& m, const CpuPayoff& p, const MLMCConfig& cfg, NormalMethod normal)
        : m_(m), p_(p), cfg_(cfg), normal_(normal), lev_((size_t)cfg.max_L + 1) {}

    int replicas() const override { return 1; }

    void plan(int l, long long extra, std::vector<LTask>& out) override {
        Level& L = get(l);
        const uint64_t cp = (uint64_t)L.sim->chunk_paths();
        for (uint64_t a = 0; a < (uint64_t)extra; a += cp)
            out.push_back({l, 0, L.next + a, (int)std::min<uint64_t>(cp, (uint64_t)extra - a)});
        L.next += (uint64_t)extra;
    }

    void run(const LTask& t, Scratch& s, ChunkAcc2& out) const override {
        lev_[(size_t)t.l].sim->run_chunk(t.first, t.count, s, out);
    }

private:
    struct Level {
        std::unique_ptr<RngNoise> noise;
        std::unique_ptr<CoupledSim> sim;
        uint64_t next = 0;
    };

    Level& get(int l) {
        Level& L = lev_[(size_t)l];
        if (!L.sim) {
            const int n_fine = pow_int(cfg_.M, l);
            const double sqrt_hf = std::sqrt(m_.T / n_fine);
            L.noise = std::make_unique<RngNoise>(cfg_.seed, Stream::Main, (uint64_t)l,
                                                 n_fine * m_.noise_dim, sqrt_hf, normal_);
            L.sim = std::make_unique<CoupledSim>(m_, p_, l, cfg_.M, *L.noise);
        }
        return L;
    }

    const CpuModel& m_;
    const CpuPayoff& p_;
    MLMCConfig cfg_;
    NormalMethod normal_;
    std::vector<Level> lev_;
};

class MlqmcEngine final : public LevelEngine {
public:
    MlqmcEngine(const CpuModel& m, const CpuPayoff& p, const MLMCConfig& ml, const QMCConfig& q,
                NoiseMode mode, NormalMethod normal, ThreadPool& pool)
        : m_(m), p_(p), ml_(ml), q_(q), mode_(mode), normal_(normal), pool_(pool),
          lev_((size_t)ml.max_L + 1) {}

    int replicas() const override { return q_.R; }

    void plan(int l, long long extra, std::vector<LTask>& out) override {
        Level& L = get(l);
        const int D = L.D;
        const uint64_t cp = L.chunk;
        for (int r = 0; r < q_.R; r++) {
            if (L.use_sobol && L.next[(size_t)r] + (uint64_t)extra > (uint64_t)UINT_MAX) throw SobolLimitReached{};
            for (uint64_t a = 0; a < (uint64_t)extra; a += cp)
                out.push_back({l, r, L.next[(size_t)r] + a, (int)std::min<uint64_t>(cp, (uint64_t)extra - a)});
            L.next[(size_t)r] += (uint64_t)extra;
        }
        (void)D;
    }

    void run(const LTask& t, Scratch& s, ChunkAcc2& out) const override {
        lev_[(size_t)t.l].sims[(size_t)t.r]->run_chunk(t.first, t.count, s, out);
    }

private:
    struct Level {
        int D = 0;
        bool use_sobol = true;
        uint64_t chunk = 1024;
        std::vector<std::unique_ptr<ScrambledSobol>> sob;
        std::vector<std::unique_ptr<NoiseSource>> base, wrapped;
        std::vector<std::unique_ptr<CoupledSim>> sims;
        std::vector<uint64_t> next;
        bool built = false;
    };

    Level& get(int l) {
        Level& L = lev_[(size_t)l];
        if (L.built) return L;
        const int R = q_.R;
        const int n_fine = pow_int(ml_.M, l);
        L.D = n_fine * m_.noise_dim;
        L.use_sobol = (L.D <= D_MAX_SOBOL);
        const double raw_scale = (mode_ == NoiseMode::Raw) ? std::sqrt(m_.T / n_fine) : 1.0;
        {   // puntos por tarea: potencia de 2 en [1024, 65536]
            double c = std::clamp(262144.0 / std::max(1, L.D), 1024.0, 65536.0);
            uint64_t p2 = 1; while (p2 * 2 <= (uint64_t)c) p2 *= 2;
            L.chunk = p2;
        }
        L.sob.resize((size_t)R); L.base.resize((size_t)R); L.wrapped.resize((size_t)R);
        L.sims.resize((size_t)R); L.next.assign((size_t)R, 0);
        const unsigned long long salt_base = (unsigned long long)l * (unsigned long long)R;

        if (L.use_sobol)
            pool_.parallel_for((size_t)R, [&](size_t r, int) {
                L.sob[r] = std::make_unique<ScrambledSobol>(q_.seed + (unsigned)(salt_base + r), L.D);
            });

        const BBData* bb = (mode_ == NoiseMode::BrownianBridge) ? &bb_precompute(n_fine, m_.T) : nullptr;
        const PCAData* pca = (mode_ == NoiseMode::PCA) ? &pca_compute(n_fine, m_.T) : nullptr;
        for (int r = 0; r < R; r++) {
            if (L.use_sobol) L.base[(size_t)r] = std::make_unique<SobolNoise>(*L.sob[(size_t)r], raw_scale);
            else L.base[(size_t)r] = std::make_unique<RngNoise>(q_.seed + (unsigned)(salt_base + r), Stream::Main,
                                                                (uint64_t)l, L.D, raw_scale, normal_);
            const NoiseSource* src = L.base[(size_t)r].get();
            if (bb) { L.wrapped[(size_t)r] = std::make_unique<BrownianBridgeNoise>(*L.base[(size_t)r], *bb); src = L.wrapped[(size_t)r].get(); }
            else if (pca) { L.wrapped[(size_t)r] = std::make_unique<PcaNoise>(*L.base[(size_t)r], *pca); src = L.wrapped[(size_t)r].get(); }
            L.sims[(size_t)r] = std::make_unique<CoupledSim>(m_, p_, l, ml_.M, *src);
        }
        L.built = true;
        return L;
    }

    const CpuModel& m_;
    const CpuPayoff& p_;
    MLMCConfig ml_;
    QMCConfig q_;
    NoiseMode mode_;
    NormalMethod normal_;
    ThreadPool& pool_;
    std::vector<Level> lev_;
};

// ---- bucle de Giles --------------------------------------------------------------------------------

MCResult giles(LevelEngine& eng, const MLMCConfig& cfg, double eps, const CpuOptions& opt,
               ThreadPool& pool, Clock::time_point t0, RunInfo& info) {
    const int M = cfg.M;
    const int max_L = cfg.max_L;
    const int R = eng.replicas();
    const bool qmc_style = (R > 1);

    std::vector<double> E((size_t)max_L + 1, 0.0), var((size_t)max_L + 1, 0.0);
    std::vector<long long> N((size_t)max_L + 1, 0);       // muestras por réplica
    std::vector<Moments> tot((size_t)max_L + 1);           // MLMC: acumulado por nivel
    std::vector<Scratch> scratch(pool.threads());
    std::vector<Padded<ChunkAcc2>> slots(256);
    std::vector<LevelStat> stats;
    uint64_t seq = 0;
    int L = std::min(2, max_L);

    auto estimate = [&](double& price, double& se, long long& n_total) {
        price = 0.0; double vs = 0.0; n_total = 0;
        for (int l = 0; l <= L; l++) {
            price += E[(size_t)l];
            if (N[(size_t)l] > 0) vs += var[(size_t)l] / ((double)N[(size_t)l] * (double)R);
            n_total += N[(size_t)l] * R;
        }
        se = std::sqrt(vs);
    };
    auto emit = [&](Stage st, bool final_) {
        if (!opt.sink) return;
        stats.clear();
        for (int l = 0; l <= L; l++)
            stats.push_back({l, N[(size_t)l] * R, E[(size_t)l], var[(size_t)l], std::pow((double)M, l)});
        Snapshot s;
        double price, se; long long nt;
        estimate(price, se, nt);
        s.stage = st; s.seq = seq++; s.elapsed_s = seconds_since(t0);
        s.n_done = nt; s.mean = price; s.std_error = se; s.eps_target = eps;
        s.L = L; s.levels = stats.data(); s.n_levels = (int)stats.size(); s.is_final = final_;
        opt.sink->on_snapshot(s);
    };

    // Ejecuta una ronda: `req` = (nivel, muestras nuevas por réplica). Devuelve false si se
    // interrumpe (cancelación o tiempo): la ronda a medias se descarta.
    auto run_round = [&](const std::vector<std::pair<int, long long>>& req) -> bool {
        std::vector<LTask> tasks;
        for (const auto& [l, n] : req) eng.plan(l, n, tasks);
        std::vector<std::vector<Moments>> rm((size_t)max_L + 1);
        for (const auto& [l, n] : req) rm[(size_t)l].assign((size_t)R, Moments{});

        for (size_t g0 = 0; g0 < tasks.size(); g0 += slots.size()) {
            const size_t g1 = std::min(tasks.size(), g0 + slots.size());
            for (size_t i = 0; i < g1 - g0; i++) slots[i].v = ChunkAcc2{};
            pool.parallel_for(g1 - g0, [&](size_t i, int worker) {
                eng.run(tasks[g0 + i], scratch[worker], slots[i].v);
            });
            for (size_t i = 0; i < g1 - g0; i++) {   // fusión en orden de tarea: determinista
                const LTask& t = tasks[g0 + i];
                rm[(size_t)t.l][(size_t)t.r].merge(slots[i].v.dY.to_moments());
                info.n_nonfinite += slots[i].v.nonfinite;
            }
            if (opt.sink && opt.sink->should_cancel()) { info.cancelled = true; return false; }
            if (opt.max_seconds > 0.0 && seconds_since(t0) > opt.max_seconds) { info.truncated = true; return false; }
        }

        for (const auto& [l, n] : req) {
            const size_t li = (size_t)l;
            if (!qmc_style) {
                tot[li].merge(rm[li][0]);
                E[li] = tot[li].mean;
                var[li] = tot[li].variance_biased();
                N[li] = tot[li].n;
            } else {
                // Medias de réplica de esta ronda -> (media, varianza entre réplicas) como la GPU
                double em = 0.0;
                for (const auto& mo : rm[li]) em += mo.mean;
                em /= R;
                double v = 0.0;
                for (const auto& mo : rm[li]) v += (mo.mean - em) * (mo.mean - em);
                v = (R > 1) ? v / (R - 1) : 0.0;
                const double wnew = (double)n, wold = (double)N[li];
                const double sig2_new = v * wnew;
                if (N[li] == 0) { E[li] = em; var[li] = sig2_new; N[li] = n; }
                else {
                    E[li] = (E[li] * wold + em * wnew) / (wold + wnew);
                    var[li] = (var[li] * wold + sig2_new * wnew) / (wold + wnew);
                    N[li] += n;
                }
            }
        }
        return true;
    };

    // ---- piloto de los niveles 0..L ------------------------------------------------------------
    bool ok = true;
    {
        std::vector<std::pair<int, long long>> req;
        for (int l = 0; l <= L; l++) req.push_back({l, (long long)cfg.pilot_n});
        ok = run_round(req);
        if (ok) emit(Stage::Pilot, false);
    }

    // ---- bucle adaptativo (Giles 2008, Teorema 1) ---------------------------------------------------
    bool converged = false;
    int iter = 0;
    while (ok && !converged && L <= max_L && iter++ < 50) {
        double sum_term = 0.0;
        for (int l = 0; l <= L; l++)
            sum_term += std::sqrt(var[(size_t)l] * std::pow((double)M, l));

        std::vector<std::pair<int, long long>> req;
        for (int l = 0; l <= L; l++) {
            const double C_l = std::pow((double)M, l);
            long long N_opt = (long long)std::ceil(2.0 / ((double)R * eps * eps)
                                                   * std::sqrt(var[(size_t)l] / C_l) * sum_term);
            N_opt = std::max(N_opt, 100LL);
            if (N_opt > N[(size_t)l]) req.push_back({l, N_opt - N[(size_t)l]});
        }
        if (!req.empty()) {
            ok = run_round(req);
            if (!ok) break;
        }

        // Criterio de convergencia: sesgo estimado < eps/sqrt(2)
        double bias_est = std::abs(E[(size_t)L]) / std::max(M - 1, 1);
        if (L >= 1) bias_est = std::max(bias_est, std::abs(E[(size_t)L - 1]) * M / std::max(M * (M - 1), 1));
        converged = (bias_est < eps / std::sqrt(2.0));

        if (!converged && L < max_L) {
            L++;
            ok = run_round({{L, (long long)cfg.pilot_n}});
            if (!ok) { --L; break; }
        } else if (!converged && req.empty()) {
            break;   // L == max_L y nada que refinar: punto fijo
        }
        emit(Stage::Level, false);
    }

    double price, se; long long n_total;
    estimate(price, se, n_total);
    const double t_s = seconds_since(t0);
    info.levels.clear();
    for (int l = 0; l <= L; l++)
        info.levels.push_back({l, N[(size_t)l] * R, E[(size_t)l], var[(size_t)l], std::pow((double)M, l)});
    emit(Stage::Done, true);
    return {price, se, n_total, t_s};
}

void check_ml(const CpuModel& m, double eps, const MLMCConfig& cfg) {
    if (m.kind == ModelKind::MultiDupire)
        throw std::invalid_argument("MLMC/MLQMC: las cestas multi-activo no están soportadas "
                                    "(no hay núcleo MLMC multi-activo)");
    if (!(eps > 0.0)) throw std::invalid_argument("MLMC: eps debe ser > 0");
    if (cfg.M < 2) throw std::invalid_argument("MLMC: M (factor de refinamiento) debe ser >= 2");
    if (cfg.max_L < 1) throw std::invalid_argument("MLMC: max_L debe ser >= 1");
    if (cfg.pilot_n < 2) throw std::invalid_argument("MLMC: pilot_n debe ser >= 2");
}

} // namespace

MCResult run_mlmc(const ModelVariant& model, const PayoffVariant& payoff,
                  double eps, const MLMCConfig& cfg, const CpuOptions& opt) {
    const auto t0 = Clock::now();
    if (opt.threads > 0) set_num_threads(opt.threads);
    ThreadPool& pool = global_pool();
    const CpuModel m = make_cpu_model(model);
    const CpuPayoff p = make_cpu_payoff(payoff);
    check_ml(m, eps, cfg);

    RunInfo info; info.threads = pool.threads();
    MlmcEngine eng(m, p, cfg, opt.normal);
    MCResult r = giles(eng, cfg, eps, opt, pool, t0, info);
    if (opt.info) *opt.info = std::move(info);
    return r;
}

MCResult run_mlqmc(const ModelVariant& model, const PayoffVariant& payoff,
                   double eps, const MLMCConfig& ml_cfg, const QMCConfig& qmc_cfg,
                   NoiseMode mode, const CpuOptions& opt) {
    const auto t0 = Clock::now();
    if (opt.threads > 0) set_num_threads(opt.threads);
    ThreadPool& pool = global_pool();
    const CpuModel m = make_cpu_model(model);
    const CpuPayoff p = make_cpu_payoff(payoff);
    check_ml(m, eps, ml_cfg);
    if (qmc_cfg.R < 2) throw std::invalid_argument("run_mlqmc: se necesitan al menos 2 réplicas (R >= 2)");
    if (mode != NoiseMode::Raw && m.noise_dim != 1)
        throw std::invalid_argument("run_mlqmc: Brownian Bridge/PCA solo soportados para ruido de dimensión 1 (no Heston)");

    RunInfo info; info.threads = pool.threads();
    MlqmcEngine eng(m, p, ml_cfg, qmc_cfg, mode, opt.normal, pool);
    MCResult r = giles(eng, ml_cfg, eps, opt, pool, t0, info);
    if (opt.info) *opt.info = std::move(info);
    return r;
}

} // namespace mc::cpu
