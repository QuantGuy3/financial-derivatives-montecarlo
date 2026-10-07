// Capa 2 de la fachada: ejecución planificada (mc::run) y nombres legibles.
#include "api.hpp"

#include "../cpu/params.hpp"
#include "../cpu/thread_pool.hpp"
#include "../cpu/vr_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mc {

const char* family_name(Family f) {
    switch (f) {
    case Family::MC:    return "MC";
    case Family::QMC:   return "QMC";
    case Family::MLMC:  return "MLMC";
    case Family::MLQMC: return "MLQMC";
    }
    return "?";
}

const char* variance_name(Variance v) {
    switch (v) {
    case Variance::None: return "none";
    case Variance::ControlVariate: return "cv";
    case Variance::ImportanceSampling: return "is";
    }
    return "?";
}

const char* noise_name(NoiseMode m) {
    switch (m) {
    case NoiseMode::Raw: return "raw";
    case NoiseMode::BrownianBridge: return "bb";
    case NoiseMode::PCA: return "pca";
    }
    return "?";
}

namespace {

int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

RunOptions quiet(const RunOptions& o) {
    RunOptions q = o;
    q.sink = nullptr;
    q.max_seconds = 0.0;
    q.truncated = q.cancelled = nullptr;
    q.n_nonfinite = nullptr;
    q.levels = nullptr;
    return q;
}

} // namespace

int plan_n_steps(const ModelVariant& model, const PayoffVariant& payoff, double eps,
                 const RunOptions& opt, double* c1_out) {
    if (!(eps > 0.0)) throw std::invalid_argument("plan_n_steps: eps debe ser > 0");
    const double T = model_T(model);
    const RunOptions q = quiet(opt);

    // c1 tal que sesgo(h) ≈ h/c1 (extrapolación de Richardson, como los ejemplos del repositorio)
    auto sim_fn = [&](int ns, long long np, unsigned s) -> double {
        return mc::run_mc_fixed(model, payoff, ns, np, s, q).first;
    };
    const double c1 = estimar_c1_richardson(sim_fn, T, 8, 50000);
    if (c1_out) *c1_out = c1;

    int n = next_pow2(std::max(1, (int)std::ceil(std::sqrt(2.0) * T / (eps * c1))));
    n = std::min(n, 1 << 11);
    if (const auto* b = std::get_if<MultiDupireParams>(&model)) {
        // La dimensión (activos × pasos) está acotada por Sobol; sin potencia de 2 (como el ejemplo 07/08)
        n = std::max(1, std::min((int)std::ceil(std::sqrt(2.0) * T / (eps * c1)), D_MAX_SOBOL / std::max(1, b->n)));
    }
    return std::max(n, 1);
}

namespace {

// E_ctrl analítico del control de cada pareja soportada por la GPU/CPU.
double analytic_control(const ModelVariant& model, const PayoffVariant& payoff, int n_steps_ref) {
    if (const auto* g = std::get_if<GBMParams>(&model)) {
        if (const auto* a = std::get_if<Asian>(&payoff))
            return geom_asian_analytic(g->S0, a->K, g->T, g->mu, g->sigma, n_steps_ref);
    } else if (const auto* d = std::get_if<DupireLocalParams>(&model)) {
        if (const auto* e = std::get_if<European>(&payoff))
            return bs_call(d->S0, e->K, d->T, e->r, d->sigma0);
    }
    throw std::invalid_argument("variable de control: solo GBM+Asian (control geométrica) y Dupire+europea (control GBM)");
}

double default_z_star(const GBMParams& g, const European& e) {
    const double s = g.sigma * std::sqrt(g.T);
    const double z = (std::log(e.K / g.S0) - (g.mu - 0.5 * g.sigma * g.sigma) * g.T) / s;
    return std::clamp(z, 0.0, 4.0);
}

} // namespace

RunReport run(const ModelVariant& model, const PayoffVariant& payoff, const RunSpec& spec,
              const RunOptions& opt_in) {
    if (!(spec.eps > 0.0)) throw std::invalid_argument("run: eps debe ser > 0");
    RunReport rep;
    rep.backend = opt_in.backend;
    rep.threads = opt_in.backend == Backend::Cpu ? (opt_in.threads > 0 ? opt_in.threads : hardware_threads()) : 1;
    if (opt_in.backend == Backend::Cuda && !cuda_available())
        throw std::runtime_error("backend CUDA no disponible en este equipo");

    // Recoge las salidas opcionales del motor
    RunOptions opt = opt_in;
    std::vector<LevelStat> levels;
    bool trunc = false, canc = false;
    long long nnf = 0;
    opt.levels = &levels; opt.truncated = &trunc; opt.cancelled = &canc; opt.n_nonfinite = &nnf;

    const bool ml = (spec.family == Family::MLMC || spec.family == Family::MLQMC);
    const bool single_level = !ml;

    // ---- pasos --------------------------------------------------------------------------------
    int ns = spec.n_steps;
    double c1 = 0.0;
    if (spec.variance == Variance::ImportanceSampling) {
        const auto* g = std::get_if<GBMParams>(&model);
        if (!g || !std::holds_alternative<European>(payoff))
            throw std::invalid_argument("importance sampling: solo GBM + call europea");
        if (single_level) ns = cpu::is_n_steps(g->T, spec.eps);
    } else if (ns <= 0 && (single_level || spec.variance == Variance::ControlVariate)) {
        ns = plan_n_steps(model, payoff, spec.eps, opt_in, &c1);
    }
    rep.n_steps = single_level ? ns : 0;
    rep.c1 = c1;

    // ---- despacho -----------------------------------------------------------------------------
    switch (spec.variance) {
    case Variance::None:
        switch (spec.family) {
        case Family::MC:    rep.result = run_mc(model, payoff, spec.eps, ns, spec.mc, opt); break;
        case Family::QMC:   rep.result = run_qmc(model, payoff, spec.eps, ns, spec.qmc, spec.noise, opt); break;
        case Family::MLMC:  rep.result = run_mlmc(model, payoff, spec.eps, spec.mlmc, opt); break;
        case Family::MLQMC: rep.result = run_mlqmc(model, payoff, spec.eps, spec.mlmc, spec.qmc, spec.noise, opt); break;
        }
        break;

    case Variance::ControlVariate: {
        rep.E_ctrl = std::isnan(spec.E_ctrl) ? analytic_control(model, payoff, ns) : spec.E_ctrl;
        if (std::isnan(spec.beta)) {
            CVPilot p = cv_pilot(model, model, payoff, payoff, rep.E_ctrl, ns, spec.cv_pilot_n, spec.mc.seed, opt);
            rep.beta = p.beta;
        } else {
            rep.beta = spec.beta;
        }
        switch (spec.family) {
        case Family::MC:    rep.result = run_mc_cv(model, model, payoff, payoff, rep.E_ctrl, rep.beta, spec.eps, ns, spec.mc, opt); break;
        case Family::QMC:   rep.result = run_qmc_cv(model, model, payoff, payoff, rep.E_ctrl, rep.beta, spec.eps, ns, spec.qmc, spec.noise, opt); break;
        case Family::MLMC:  rep.result = run_mlmc_cv(model, model, payoff, payoff, rep.E_ctrl, rep.beta, spec.eps, spec.mlmc, opt); break;
        case Family::MLQMC: rep.result = run_mlqmc_cv(model, model, payoff, payoff, rep.E_ctrl, rep.beta, spec.eps, spec.mlmc, spec.qmc, spec.noise, opt); break;
        }
        break;
    }

    case Variance::ImportanceSampling: {
        const GBMParams& g = std::get<GBMParams>(model);
        const European& e = std::get<European>(payoff);
        rep.z_star = std::isnan(spec.z_star) ? default_z_star(g, e) : spec.z_star;
        switch (spec.family) {
        case Family::MC:    rep.result = run_is(g, e, rep.z_star, spec.eps, spec.mc, opt); break;
        case Family::QMC:   rep.result = run_qmc_is(g, e, rep.z_star, spec.eps, spec.qmc, spec.noise, opt); break;
        case Family::MLMC:  rep.result = run_mlmc_is(g, e, rep.z_star, spec.eps, spec.mlmc, opt); break;
        case Family::MLQMC: rep.result = run_mlqmc_is(g, e, rep.z_star, spec.eps, spec.mlmc, spec.qmc, spec.noise, opt); break;
        }
        break;
    }
    }

    rep.truncated = trunc; rep.cancelled = canc; rep.n_nonfinite = nnf;
    rep.levels = std::move(levels);
    return rep;
}

} // namespace mc
