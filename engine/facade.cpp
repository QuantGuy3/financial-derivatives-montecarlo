// Capa 1 de la fachada: despacha cada llamada al motor CPU o al adaptador CUDA.
#include "api.hpp"

#include "../cpu/mc_cpu.hpp"
#include "../cpu/mlmc_cpu.hpp"
#include "../cpu/qmc_cpu.hpp"
#include "../cpu/thread_pool.hpp"
#include "../cpu/vr_cpu.hpp"
#include "cuda_adapter.hpp"

#include <chrono>
#include <stdexcept>
#include <thread>

namespace mc {

const char* backend_name(Backend b) { return b == Backend::Cuda ? "cuda" : "cpu"; }

Backend backend_from_string(const std::string& s) {
    if (s == "cpu" || s == "CPU") return Backend::Cpu;
    if (s == "cuda" || s == "gpu" || s == "CUDA" || s == "GPU") return Backend::Cuda;
    throw std::invalid_argument("backend desconocido: '" + s + "' (use cpu o cuda)");
}

bool cuda_available() { return detail::cuda_present(); }
int hardware_threads() { return (int)std::max(1u, std::thread::hardware_concurrency()); }

RunOptions& default_options() {
    static RunOptions opt;
    return opt;
}

namespace {

// Opciones de la CPU + publicación de las salidas opcionales al terminar (también si hay excepción).
struct CpuCall {
    cpu::CpuOptions co;
    cpu::RunInfo info;
    const RunOptions& ro;

    explicit CpuCall(const RunOptions& r) : ro(r) {
        co.threads = r.threads;
        co.sink = r.sink;
        co.max_seconds = r.max_seconds;
        co.info = &info;
    }
    ~CpuCall() {
        if (ro.truncated) *ro.truncated = info.truncated;
        if (ro.cancelled) *ro.cancelled = info.cancelled;
        if (ro.n_nonfinite) *ro.n_nonfinite = info.n_nonfinite;
        if (ro.levels) *ro.levels = info.levels;
    }
};

// La GPU no emite progreso intermedio: solo un snapshot final con el resultado.
MCResult gpu_done(const RunOptions& opt, const MCResult& r) {
    if (opt.sink) {
        Snapshot s;
        s.stage = Stage::Done; s.elapsed_s = r.time_s; s.n_done = r.n_samples;
        s.mean = r.price; s.std_error = r.std_error; s.is_final = true;
        opt.sink->on_snapshot(s);
    }
    return r;
}

#define MC_DISPATCH(opt, cuda_expr, cpu_expr)                     \
    do {                                                          \
        if ((opt).backend == Backend::Cuda) return gpu_done((opt), (cuda_expr)); \
        CpuCall call(opt);                                        \
        return (cpu_expr);                                        \
    } while (0)

} // namespace

MCResult run_mc(const ModelVariant& m, const PayoffVariant& p, double eps, int n_steps,
                const MCConfig& cfg, const RunOptions& opt) {
    MC_DISPATCH(opt, detail::cuda_mc(m, p, eps, n_steps, cfg), cpu::run_mc(m, p, eps, n_steps, cfg, call.co));
}

MCResult run_qmc(const ModelVariant& m, const PayoffVariant& p, double eps, int n_steps,
                 const QMCConfig& cfg, NoiseMode mode, const RunOptions& opt) {
    MC_DISPATCH(opt, detail::cuda_qmc(m, p, eps, n_steps, cfg, mode),
                cpu::run_qmc(m, p, eps, n_steps, cfg, mode, call.co));
}

MCResult run_mlmc(const ModelVariant& m, const PayoffVariant& p, double eps, const MLMCConfig& cfg,
                  const RunOptions& opt) {
    MC_DISPATCH(opt, detail::cuda_mlmc(m, p, eps, cfg), cpu::run_mlmc(m, p, eps, cfg, call.co));
}

MCResult run_mlqmc(const ModelVariant& m, const PayoffVariant& p, double eps, const MLMCConfig& ml,
                   const QMCConfig& q, NoiseMode mode, const RunOptions& opt) {
    MC_DISPATCH(opt, detail::cuda_mlqmc(m, p, eps, ml, q, mode), cpu::run_mlqmc(m, p, eps, ml, q, mode, call.co));
}

std::pair<double, double> run_mc_fixed(const ModelVariant& m, const PayoffVariant& p, int n_steps,
                                       long long n_paths, unsigned seed, const RunOptions& opt) {
    if (opt.backend == Backend::Cuda) return detail::cuda_mc_fixed(m, p, n_steps, n_paths, seed);
    CpuCall call(opt);
    call.co.sink = nullptr;
    call.co.max_seconds = 0.0;
    return cpu::run_mc_fixed(m, p, n_steps, n_paths, seed, call.co);
}

CVPilot cv_pilot(const ModelVariant& mm, const ModelVariant& cm, const PayoffVariant& mp,
                 const PayoffVariant& cp, double E_ctrl, int n_steps, int N_pilot, unsigned seed,
                 const RunOptions& opt) {
    if (opt.backend == Backend::Cuda) return detail::cuda_cv_pilot(mm, cm, mp, cp, E_ctrl, n_steps, N_pilot, seed);
    CpuCall call(opt);
    call.co.sink = nullptr;
    call.co.max_seconds = 0.0;
    return cpu::cv_pilot(mm, cm, mp, cp, E_ctrl, n_steps, N_pilot, seed, call.co);
}

MCResult run_mc_cv(const ModelVariant& mm, const ModelVariant& cm, const PayoffVariant& mp,
                   const PayoffVariant& cp, double E_ctrl, double beta, double eps, int n_steps,
                   const MCConfig& cfg, const RunOptions& opt) {
    MC_DISPATCH(opt, detail::cuda_mc_cv(mm, cm, mp, cp, E_ctrl, beta, eps, n_steps, cfg),
                cpu::run_mc_cv(mm, cm, mp, cp, E_ctrl, beta, eps, n_steps, cfg, call.co));
}

MCResult run_qmc_cv(const ModelVariant& mm, const ModelVariant& cm, const PayoffVariant& mp,
                    const PayoffVariant& cp, double E_ctrl, double beta, double eps, int n_steps,
                    const QMCConfig& cfg, NoiseMode mode, const RunOptions& opt) {
    MC_DISPATCH(opt, detail::cuda_qmc_cv(mm, cm, mp, cp, E_ctrl, beta, eps, n_steps, cfg, mode),
                cpu::run_qmc_cv(mm, cm, mp, cp, E_ctrl, beta, eps, n_steps, cfg, mode, call.co));
}

MCResult run_mlmc_cv(const ModelVariant& mm, const ModelVariant& cm, const PayoffVariant& mp,
                     const PayoffVariant& cp, double E_ctrl, double beta, double eps,
                     const MLMCConfig& cfg, const RunOptions& opt) {
    MC_DISPATCH(opt, detail::cuda_mlmc_cv(mm, cm, mp, cp, E_ctrl, beta, eps, cfg),
                cpu::run_mlmc_cv(mm, cm, mp, cp, E_ctrl, beta, eps, cfg, call.co));
}

MCResult run_mlqmc_cv(const ModelVariant& mm, const ModelVariant& cm, const PayoffVariant& mp,
                      const PayoffVariant& cp, double E_ctrl, double beta, double eps,
                      const MLMCConfig& ml, const QMCConfig& q, NoiseMode mode, const RunOptions& opt) {
    MC_DISPATCH(opt, detail::cuda_mlqmc_cv(mm, cm, mp, cp, E_ctrl, beta, eps, ml, q, mode),
                cpu::run_mlqmc_cv(mm, cm, mp, cp, E_ctrl, beta, eps, ml, q, mode, call.co));
}

MCResult run_is(const GBMParams& m, const European& p, double z_star, double eps, const MCConfig& cfg,
                const RunOptions& opt) {
    MC_DISPATCH(opt, detail::cuda_is(m, p, z_star, eps, cfg), cpu::run_is(m, p, z_star, eps, cfg, call.co));
}

MCResult run_qmc_is(const GBMParams& m, const European& p, double z_star, double eps, const QMCConfig& cfg,
                    NoiseMode mode, const RunOptions& opt) {
    MC_DISPATCH(opt, detail::cuda_qmc_is(m, p, z_star, eps, cfg, mode),
                cpu::run_qmc_is(m, p, z_star, eps, cfg, mode, call.co));
}

MCResult run_mlmc_is(const GBMParams& m, const European& p, double z_star, double eps,
                     const MLMCConfig& cfg, const RunOptions& opt) {
    MC_DISPATCH(opt, detail::cuda_mlmc_is(m, p, z_star, eps, cfg), cpu::run_mlmc_is(m, p, z_star, eps, cfg, call.co));
}

MCResult run_mlqmc_is(const GBMParams& m, const European& p, double z_star, double eps,
                      const MLMCConfig& ml, const QMCConfig& q, NoiseMode mode, const RunOptions& opt) {
    MC_DISPATCH(opt, detail::cuda_mlqmc_is(m, p, z_star, eps, ml, q, mode),
                cpu::run_mlqmc_is(m, p, z_star, eps, ml, q, mode, call.co));
}

} // namespace mc
