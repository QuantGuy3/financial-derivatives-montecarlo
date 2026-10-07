#include "mc_cpu.hpp"

#include "noise.hpp"
#include "params.hpp"
#include "path_sim.hpp"
#include "thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mc::cpu {

namespace {

ThreadPool& pool_for(const CpuOptions& opt) {
    if (opt.threads > 0) set_num_threads(opt.threads);
    return global_pool();
}

void fill_info(const CpuOptions& opt, RunInfo&& tmp) {
    if (opt.info) *opt.info = std::move(tmp);
}

} // namespace

std::pair<double, double> run_mc_fixed(const ModelVariant& model, const PayoffVariant& payoff,
                                       int n_steps, long long n_paths, unsigned seed,
                                       const CpuOptions& opt) {
    ThreadPool& pool = pool_for(opt);
    const CpuModel m = make_cpu_model(model);
    const CpuPayoff p = make_cpu_payoff(payoff);
    const double sqrt_h = std::sqrt(m.T / n_steps);
    RngNoise noise(seed, Stream::Main, 0, n_steps * m.noise_dim, sqrt_h, opt.normal);
    PathSim sim(m, p, n_steps, noise);
    RangeResult r = simulate_range(pool, sim, 0, n_paths, nullptr);
    const double N = (double)std::max<long long>(r.moments.n, 1);
    const double var = r.moments.variance_biased() / N;   // max(0, E[Y²]-mean²)/N
    RunInfo info; info.n_nonfinite = r.nonfinite; info.threads = pool.threads();
    fill_info(opt, std::move(info));
    return {r.moments.mean, var};
}

MCResult run_mc(const ModelVariant& model, const PayoffVariant& payoff,
                double eps, int n_steps, const MCConfig& cfg, const CpuOptions& opt) {
    return run_mc_eval(model, payoff, EvalSpec{}, eps, n_steps, cfg, opt);
}

MCResult run_mc_eval(const ModelVariant& model, const PayoffVariant& payoff, const EvalSpec& eval,
                     double eps, int n_steps, const MCConfig& cfg, const CpuOptions& opt) {
    const auto t0 = Clock::now();
    ThreadPool& pool = pool_for(opt);
    const CpuModel m = make_cpu_model(model);
    const CpuPayoff p = make_cpu_payoff(payoff);
    if (n_steps < 1) throw std::invalid_argument("run_mc: n_steps debe ser >= 1");
    if (!(eps > 0.0)) throw std::invalid_argument("run_mc: eps debe ser > 0");
    const double sqrt_h = std::sqrt(m.T / n_steps);
    const int D = n_steps * m.noise_dim;

    RunInfo info;
    info.threads = pool.threads();
    uint64_t seq = 0;
    auto emit = [&](Stage st, long long n_done, const Moments& mo, bool final_) {
        if (!opt.sink) return;
        Snapshot s;
        s.stage = st; s.seq = seq++; s.elapsed_s = seconds_since(t0);
        s.n_done = n_done; s.mean = mo.mean; s.std_error = std::sqrt(mo.variance_biased() / std::max<long long>(mo.n, 1));
        s.eps_target = eps; s.is_final = final_;
        opt.sink->on_snapshot(s);
    };
    auto should_stop = [&]() {
        if (opt.sink && opt.sink->should_cancel()) { info.cancelled = true; return true; }
        if (opt.max_seconds > 0.0 && seconds_since(t0) > opt.max_seconds) { info.truncated = true; return true; }
        return false;
    };

    // ---- piloto: estima la varianza de la muestra ------------------------------------------
    RngNoise pilot_noise(cfg.seed, Stream::Pilot, 0, D, sqrt_h, opt.normal);
    PathSim pilot_sim(m, p, n_steps, pilot_noise, eval);
    RangeResult pr = simulate_range(pool, pilot_sim, 0, cfg.pilot_n, nullptr);
    info.n_nonfinite += pr.nonfinite;
    const double sample_var = pr.moments.variance_biased();
    emit(Stage::Pilot, cfg.pilot_n, pr.moments, false);

    long long N_needed = (long long)std::ceil(2.0 * sample_var / (eps * eps));
    N_needed = std::max<long long>(N_needed, cfg.pilot_n);

    // ---- corrida principal --------------------------------------------------------------------
    RngNoise noise(cfg.seed, Stream::Main, 0, D, sqrt_h, opt.normal);
    PathSim sim(m, p, n_steps, noise, eval);
    RangeResult mr = simulate_range(pool, sim, 0, N_needed,
        [&](const Moments& mo, long long n_done) {
            emit(Stage::Main, n_done, mo, false);
            return !should_stop();
        });
    info.n_nonfinite += mr.nonfinite;

    const long long N_done = std::max<long long>(mr.moments.n, 1);
    const double mean = mr.moments.mean;
    const double var = mr.moments.variance_biased() / (double)N_done;
    const double t_s = seconds_since(t0);
    emit(Stage::Done, N_done, mr.moments, true);
    fill_info(opt, std::move(info));
    return {mean, std::sqrt(var), mr.moments.n, t_s};
}

} // namespace mc::cpu
