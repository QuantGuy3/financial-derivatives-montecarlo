#include "vr_cpu.hpp"

#include "noise.hpp"
#include "path_sim.hpp"
#include "thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mc::cpu {

namespace {

// Elige el tipo de CV según el modelo principal (como cv_pilot / run_mc_cv_cuda).
EvalSpec cv_spec(const ModelVariant& main_model, const PayoffVariant& main_payoff,
                 double E_ctrl, double beta) {
    EvalSpec e;
    e.beta = beta;
    e.E_ctrl = E_ctrl;
    if (std::holds_alternative<GBMParams>(main_model) && std::holds_alternative<Asian>(main_payoff))
        e.kind = EvalSpec::Kind::CvAsianGeom;
    else if (std::holds_alternative<DupireLocalParams>(main_model) && std::holds_alternative<European>(main_payoff))
        e.kind = EvalSpec::Kind::CvDupireGbm;
    else
        throw std::invalid_argument("CV: solo GBM+Asian (control geométrica) y Dupire+europea (control GBM)");
    return e;
}

EvalSpec is_spec(double z_star) {
    EvalSpec e;
    e.kind = EvalSpec::Kind::IsGbmCall;
    e.z_star = z_star;
    return e;
}

int pow2_ceil_i(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

} // namespace

CVPilot cv_pilot(const ModelVariant& main_model, const ModelVariant& /*ctrl_model*/,
                 const PayoffVariant& main_payoff, const PayoffVariant& /*ctrl_payoff*/,
                 double E_ctrl, int n_steps, int N_pilot, unsigned seed, const CpuOptions& opt) {
    if (opt.threads > 0) set_num_threads(opt.threads);
    ThreadPool& pool = global_pool();
    const EvalSpec ev = cv_spec(main_model, main_payoff, E_ctrl, /*beta=*/0.0);
    const CpuModel m = make_cpu_model(main_model);
    const CpuPayoff p = make_cpu_payoff(main_payoff);
    RngNoise noise(seed, Stream::Pilot, 0, n_steps * m.noise_dim, std::sqrt(m.T / n_steps), opt.normal);
    PathSim sim(m, p, n_steps, noise, ev);
    RangeResult r = simulate_range(pool, sim, 0, N_pilot, nullptr);

    const double var_main = std::max(0.0, r.pair.var_x());
    const double var_ctrl = std::max(1e-30, r.pair.var_y());
    const double cov = r.pair.cov_xy();
    CVPilot out;
    out.beta = std::clamp(cov / var_ctrl, 0.0, 5.0);
    out.var_plain = var_main;
    out.var_cv = std::max(0.0, var_main - cov * cov / var_ctrl);
    return out;
}

MCResult run_mc_cv(const ModelVariant& main_model, const ModelVariant&, const PayoffVariant& main_payoff,
                   const PayoffVariant&, double E_ctrl, double beta, double eps, int n_steps,
                   const MCConfig& cfg, const CpuOptions& opt) {
    return run_mc_eval(main_model, main_payoff, cv_spec(main_model, main_payoff, E_ctrl, beta),
                       eps, n_steps, cfg, opt);
}

MCResult run_qmc_cv(const ModelVariant& main_model, const ModelVariant&, const PayoffVariant& main_payoff,
                    const PayoffVariant&, double E_ctrl, double beta, double eps, int n_steps,
                    const QMCConfig& cfg, NoiseMode mode, const CpuOptions& opt) {
    return run_qmc_eval(main_model, main_payoff, cv_spec(main_model, main_payoff, E_ctrl, beta),
                        eps, n_steps, cfg, mode, opt);
}

MCResult run_mlmc_cv(const ModelVariant& main_model, const ModelVariant&, const PayoffVariant& main_payoff,
                     const PayoffVariant&, double E_ctrl, double beta, double eps,
                     const MLMCConfig& cfg, const CpuOptions& opt) {
    return run_mlmc_eval(main_model, main_payoff, cv_spec(main_model, main_payoff, E_ctrl, beta),
                         eps, cfg, opt);
}

MCResult run_mlqmc_cv(const ModelVariant& main_model, const ModelVariant&, const PayoffVariant& main_payoff,
                      const PayoffVariant&, double E_ctrl, double beta, double eps,
                      const MLMCConfig& ml_cfg, const QMCConfig& qmc_cfg, NoiseMode mode,
                      const CpuOptions& opt) {
    return run_mlqmc_eval(main_model, main_payoff, cv_spec(main_model, main_payoff, E_ctrl, beta),
                          eps, ml_cfg, qmc_cfg, mode, opt);
}

int is_n_steps(double T, double eps) {
    return std::min(1 << 11, pow2_ceil_i(std::max(4, (int)std::ceil(T / eps))));
}

MCResult run_is(const GBMParams& model, const European& payoff, double z_star, double eps,
                const MCConfig& cfg, const CpuOptions& opt) {
    return run_mc_eval(model, payoff, is_spec(z_star), eps, is_n_steps(model.T, eps), cfg, opt);
}

MCResult run_qmc_is(const GBMParams& model, const European& payoff, double z_star, double eps,
                    const QMCConfig& cfg, NoiseMode mode, const CpuOptions& opt) {
    return run_qmc_eval(model, payoff, is_spec(z_star), eps, is_n_steps(model.T, eps), cfg, mode, opt);
}

MCResult run_mlmc_is(const GBMParams& model, const European& payoff, double z_star, double eps,
                     const MLMCConfig& cfg, const CpuOptions& opt) {
    return run_mlmc_eval(model, payoff, is_spec(z_star), eps, cfg, opt);
}

MCResult run_mlqmc_is(const GBMParams& model, const European& payoff, double z_star, double eps,
                      const MLMCConfig& ml_cfg, const QMCConfig& qmc_cfg, NoiseMode mode,
                      const CpuOptions& opt) {
    return run_mlqmc_eval(model, payoff, is_spec(z_star), eps, ml_cfg, qmc_cfg, mode, opt);
}

} // namespace mc::cpu
