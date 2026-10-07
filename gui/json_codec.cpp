#include "json_codec.hpp"

#include "../cpu/path_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mc::gui {

namespace {

[[noreturn]] void bad(const std::string& msg) { throw std::invalid_argument(msg); }

double num(const json& j, const char* key, double def, double lo, double hi, const char* label = nullptr) {
    double v = def;
    if (j.contains(key) && !j[key].is_null()) {
        if (!j[key].is_number()) bad(std::string("el parámetro '") + key + "' debe ser numérico");
        v = j[key].get<double>();
    }
    if (!std::isfinite(v) || v < lo || v > hi)
        bad(std::string("parámetro '") + (label ? label : key) + "' fuera de rango [" + std::to_string(lo) + ", " +
            std::to_string(hi) + "]: " + std::to_string(v));
    return v;
}

int inum(const json& j, const char* key, int def, int lo, int hi) {
    return (int)std::llround(num(j, key, def, lo, hi));
}

std::string str(const json& j, const char* key, const std::string& def) {
    if (!j.contains(key) || j[key].is_null()) return def;
    if (!j[key].is_string()) bad(std::string("el campo '") + key + "' debe ser texto");
    return j[key].get<std::string>();
}

double r6(double x) { return std::isfinite(x) ? std::round(x * 1e6) / 1e6 : 0.0; }
double r4(double x) { return std::isfinite(x) ? std::round(x * 1e4) / 1e4 : 0.0; }

std::vector<double> cholesky_equicorr(int n, double rho) {
    std::vector<double> L((size_t)n * n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = (i == j) ? 1.0 : rho;
            for (int k = 0; k < j; k++) s -= L[(size_t)i * n + k] * L[(size_t)j * n + k];
            L[(size_t)i * n + j] = (i == j) ? std::sqrt(std::max(s, 1e-12)) : s / L[(size_t)j * n + j];
        }
    }
    return L;
}

ModelVariant parse_model(const json& j, std::string& type) {
    if (!j.is_object()) bad("falta el objeto 'model'");
    type = str(j, "type", "");
    if (type == "gbm") {
        GBMParams g;
        g.S0 = num(j, "S0", 100.0, 1e-3, 1e6);
        g.mu = num(j, "mu", 0.05, -1.0, 1.0);
        g.sigma = num(j, "sigma", 0.20, 1e-3, 5.0);
        g.T = num(j, "T", 1.0, 1e-3, 30.0);
        return g;
    }
    if (type == "heston") {
        HestonParams h;
        h.S0 = num(j, "S0", 100.0, 1e-3, 1e6);
        h.mu = num(j, "mu", 0.05, -1.0, 1.0);
        h.kappa = num(j, "kappa", 2.0, 1e-3, 50.0);
        h.theta = num(j, "theta", 0.04, 1e-4, 4.0);
        h.xi = num(j, "xi", 0.5, 1e-3, 5.0);
        h.rho = num(j, "rho", -0.9, -0.999, 0.999);
        h.v0 = num(j, "v0", 0.04, 1e-4, 4.0);
        h.T = num(j, "T", 1.0, 1e-2, 30.0);
        h.compute_cholesky();
        return h;
    }
    if (type == "dupire") {
        DupireLocalParams d;
        d.S0 = num(j, "S0", 100.0, 1e-3, 1e6);
        d.mu = num(j, "mu", 0.05, -1.0, 1.0);
        d.sigma0 = num(j, "sigma0", 0.20, 1e-3, 5.0);
        d.alpha = num(j, "alpha", 0.5, 0.0, 10.0);
        d.beta_d = num(j, "beta", 0.7, 0.05, 2.0);
        d.T = num(j, "T", 1.0, 1e-3, 30.0);
        return d;
    }
    if (type == "basket") {
        MultiDupireParams b;
        b.n = inum(j, "n", 20, 2, 500);
        b.mu = num(j, "mu", 0.05, -1.0, 1.0);
        b.sigma0 = num(j, "sigma0", 0.20, 1e-3, 5.0);
        b.alpha = num(j, "alpha", 0.5, 0.0, 10.0);
        b.beta_d = num(j, "beta", 0.7, 0.05, 2.0);
        b.T = num(j, "T", 1.0, 1e-3, 30.0);
        const double S0 = num(j, "S0", 100.0, 1e-3, 1e6);
        const double rho = num(j, "rho", 0.0, 0.0, 0.99);
        b.S0.assign((size_t)b.n, S0);
        b.uncorrelated = (rho == 0.0);
        if (!b.uncorrelated) b.L = cholesky_equicorr(b.n, rho);
        return b;
    }
    bad("modelo desconocido: '" + type + "' (use gbm, heston, dupire o basket)");
}

PayoffVariant parse_payoff(const json& j, const ModelVariant& model, std::string& type) {
    if (!j.is_object()) bad("falta el objeto 'payoff'");
    type = str(j, "type", "");
    const double T = model_T(model);
    const double sigma_default = std::visit([](const auto& m) -> double {
        using Tm = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<Tm, GBMParams>) return m.sigma;
        else if constexpr (std::is_same_v<Tm, DupireLocalParams> || std::is_same_v<Tm, MultiDupireParams>) return m.sigma0;
        else return std::sqrt(m.theta);
    }, model);
    const double S0 = model_S0(model);
    const bool is_basket_model = std::holds_alternative<MultiDupireParams>(model);
    if (is_basket_model != (type == "basket"))
        bad(is_basket_model ? "la cesta multi-activo solo admite el payoff 'basket'"
                            : "el payoff 'basket' solo se puede usar con el modelo 'basket'");

    if (type == "european") return European{num(j, "K", S0, 1e-3, 1e7), num(j, "r", 0.05, -1.0, 1.0), T};
    if (type == "asian")    return Asian{num(j, "K", S0, 1e-3, 1e7)};
    if (type == "geomasian")return GeomAsian{num(j, "K", S0, 1e-3, 1e7)};
    if (type == "lookback") return Lookback{num(j, "sigma", sigma_default, 1e-3, 5.0)};
    if (type == "barrier") {
        const double K = num(j, "K", S0, 1e-3, 1e7);
        const double B = num(j, "B", 1.2 * S0, 1e-3, 1e7);
        return Barrier{K, B, num(j, "sigma", sigma_default, 1e-3, 5.0), num(j, "r", 0.05, -1.0, 1.0), T};
    }
    if (type == "basket") {
        const int n = std::get<MultiDupireParams>(model).n;
        return Basket{num(j, "K", S0, 1e-3, 1e7), num(j, "r", 0.05, -1.0, 1.0), T, n};
    }
    bad("payoff desconocido: '" + type + "' (european, asian, geomasian, lookback, barrier, basket)");
}

mc::Family parse_family(const std::string& s) {
    if (s == "mc") return mc::Family::MC;
    if (s == "qmc") return mc::Family::QMC;
    if (s == "mlmc") return mc::Family::MLMC;
    if (s == "mlqmc") return mc::Family::MLQMC;
    bad("familia de método desconocida: '" + s + "' (mc, qmc, mlmc, mlqmc)");
}

NoiseMode parse_noise(const std::string& s) {
    if (s == "raw") return NoiseMode::Raw;
    if (s == "bb") return NoiseMode::BrownianBridge;
    if (s == "pca") return NoiseMode::PCA;
    bad("construcción de ruido desconocida: '" + s + "' (raw, bb, pca)");
}

mc::Variance parse_variance(const std::string& s) {
    if (s == "none") return mc::Variance::None;
    if (s == "cv") return mc::Variance::ControlVariate;
    if (s == "is") return mc::Variance::ImportanceSampling;
    bad("reducción de varianza desconocida: '" + s + "' (none, cv, is)");
}

} // namespace

ParsedRun parse_run_request(const json& body, int hw_threads, bool cuda_ok) {
    if (!body.is_object()) bad("el cuerpo de la petición debe ser un objeto JSON");
    ParsedRun r;
    r.model = parse_model(body.value("model", json::object()), r.model_type);
    r.payoff = parse_payoff(body.value("payoff", json::object()), r.model, r.payoff_type);

    const json m = body.value("method", json::object());
    mc::RunSpec& s = r.spec;
    s.family = parse_family(str(m, "family", "mc"));
    s.noise = parse_noise(str(m, "noise", "raw"));
    s.variance = parse_variance(str(m, "variance", "none"));
    s.eps = num(m, "eps", 0.01, 1e-5, 10.0);
    s.n_steps = inum(m, "n_steps", 0, 0, 4096);
    s.mc.seed = (unsigned)num(body, "seed", 123.0, 0.0, 4294967295.0);
    s.mc.pilot_n = inum(m, "pilot_n", 10000, 100, 5000000);
    s.qmc.seed = s.mc.seed;
    s.qmc.R = inum(m, "R", 32, 2, 128);
    s.qmc.n0 = inum(m, "n0", 256, 64, 1 << 20);
    s.qmc.max_doublings = inum(m, "max_doublings", 20, 1, 30);
    s.mlmc.seed = s.mc.seed;
    s.mlmc.M = inum(m, "M", 2, 2, 8);
    s.mlmc.max_L = inum(m, "max_L", 10, 2, 14);
    s.mlmc.pilot_n = inum(m, "ml_pilot_n", 400, 50, 200000);
    s.cv_pilot_n = inum(m, "cv_pilot_n", 50000, 1000, 5000000);
    if (m.contains("z_star") && !m["z_star"].is_null()) s.z_star = num(m, "z_star", 0.0, -10.0, 10.0);
    if (m.contains("beta") && !m["beta"].is_null()) s.beta = num(m, "beta", 1.0, -10.0, 10.0);

    // combinaciones soportadas por el motor
    const bool path_ml = (s.family == mc::Family::MLMC || s.family == mc::Family::MLQMC);
    if (r.model_type == "basket" && path_ml) bad("MLMC/MLQMC no están soportados para cestas multi-activo");
    if (s.noise != NoiseMode::Raw && (s.family == mc::Family::MC || s.family == mc::Family::MLMC))
        s.noise = NoiseMode::Raw;                                  // la construcción solo afecta a QMC/MLQMC
    if (s.noise != NoiseMode::Raw && (r.model_type == "heston" || r.model_type == "basket"))
        bad("Brownian Bridge y PCA solo están disponibles para modelos de ruido 1D (GBM, Dupire)");
    if (s.variance == mc::Variance::ImportanceSampling && !(r.model_type == "gbm" && r.payoff_type == "european"))
        bad("importance sampling: solo GBM + call europea");
    if (s.variance == mc::Variance::ControlVariate &&
        !((r.model_type == "gbm" && r.payoff_type == "asian") || (r.model_type == "dupire" && r.payoff_type == "european")))
        bad("variable de control: solo GBM + asiática aritmética o Dupire + europea");
    if (s.variance == mc::Variance::ControlVariate && path_ml && r.model_type != "gbm")
        bad("MLMC con variable de control: solo GBM + asiática");

    // dimensión máxima de la simulación (memoria/tiempo razonables en una máquina de escritorio)
    if (s.n_steps > 0) {
        const long long dim = (long long)s.n_steps * model_noise_dim(r.model);
        if (dim > 40000) bad("n_steps * dimensión del ruido supera el límite de 40000");
    }

    r.opt.backend = mc::backend_from_string(str(body, "backend", "cpu"));
    if (r.opt.backend == mc::Backend::Cuda && !cuda_ok) bad("el backend CUDA no está disponible en este equipo");
    r.opt.threads = inum(body, "threads", 0, 0, std::max(1, hw_threads));
    r.opt.max_seconds = num(body, "max_seconds", 0.0, 0.0, 86400.0);

    r.echo = body;
    return r;
}

// ---------------------------------------------------------------------------------------------------
// Capacidades (formulario) y presets
// ---------------------------------------------------------------------------------------------------

namespace {

json P(const char* key, double def, double lo, double hi, double step, const char* scale = "lin") {
    return json{{"key", key}, {"default", def}, {"min", lo}, {"max", hi}, {"step", step}, {"scale", scale}};
}

} // namespace

json capabilities_json(int hw_threads, bool cuda_ok) {
    json caps;
    caps["version"] = "1.0";
    caps["backends"] = json::array({
        json{{"id", "cpu"}, {"available", true}, {"threads", hw_threads}},
        json{{"id", "cuda"}, {"available", cuda_ok}, {"threads", 1}}});

    caps["models"] = json::array({
        json{{"id", "gbm"}, {"params", json::array({P("S0", 100, 1, 1000, 1), P("mu", 0.05, -0.2, 0.5, 0.005),
                                                    P("sigma", 0.2, 0.01, 1.5, 0.005), P("T", 1.0, 0.05, 10, 0.05)})}},
        json{{"id", "heston"}, {"params", json::array({P("S0", 100, 1, 1000, 1), P("mu", 0.05, -0.2, 0.5, 0.005),
                                                       P("kappa", 2.0, 0.1, 10, 0.1), P("theta", 0.04, 0.005, 0.5, 0.005),
                                                       P("xi", 0.5, 0.05, 2.0, 0.05), P("rho", -0.9, -0.99, 0.99, 0.01),
                                                       P("v0", 0.04, 0.005, 0.5, 0.005), P("T", 1.0, 0.05, 10, 0.05)})}},
        json{{"id", "dupire"}, {"params", json::array({P("S0", 100, 1, 1000, 1), P("mu", 0.05, -0.2, 0.5, 0.005),
                                                       P("sigma0", 0.2, 0.01, 1.5, 0.005), P("alpha", 0.5, 0.0, 3.0, 0.05),
                                                       P("beta", 0.7, 0.1, 1.5, 0.05), P("T", 1.0, 0.05, 10, 0.05)})}},
        json{{"id", "basket"}, {"params", json::array({P("n", 20, 2, 200, 1), P("S0", 100, 1, 1000, 1),
                                                       P("mu", 0.05, -0.2, 0.5, 0.005), P("sigma0", 0.2, 0.01, 1.5, 0.005),
                                                       P("alpha", 0.5, 0.0, 3.0, 0.05), P("beta", 0.7, 0.1, 1.5, 0.05),
                                                       P("rho", 0.0, 0.0, 0.95, 0.05), P("T", 1.0, 0.05, 10, 0.05)})}}});

    caps["payoffs"] = json::array({
        json{{"id", "european"}, {"params", json::array({P("K", 100, 1, 500, 1), P("r", 0.05, -0.1, 0.5, 0.005)})}},
        json{{"id", "asian"}, {"params", json::array({P("K", 100, 1, 500, 1)})}},
        json{{"id", "geomasian"}, {"params", json::array({P("K", 100, 1, 500, 1)})}},
        json{{"id", "lookback"}, {"params", json::array({P("sigma", 0.2, 0.01, 1.5, 0.005)})}},
        json{{"id", "barrier"}, {"params", json::array({P("K", 100, 1, 500, 1), P("B", 120, 1, 1000, 1),
                                                        P("sigma", 0.2, 0.01, 1.5, 0.005), P("r", 0.05, -0.1, 0.5, 0.005)})}},
        json{{"id", "basket"}, {"params", json::array({P("K", 100, 1, 500, 1), P("r", 0.05, -0.1, 0.5, 0.005)})}}});

    caps["limits"] = json{{"eps_min", 1e-5}, {"eps_max", 1.0}, {"max_paths_shown", 200}, {"max_steps_shown", 2048},
                          {"max_fan", 5000}, {"max_assets_shown", 5}};

    auto preset = [](const char* id, const char* es, const char* en, json model, json payoff, json method) {
        return json{{"id", id}, {"label", json{{"es", es}, {"en", en}}}, {"model", std::move(model)},
                    {"payoff", std::move(payoff)}, {"method", std::move(method)}};
    };
    json gbm = {{"type", "gbm"}, {"S0", 100}, {"mu", 0.05}, {"sigma", 0.2}, {"T", 1.0}};
    json dup = {{"type", "dupire"}, {"S0", 100}, {"mu", 0.05}, {"sigma0", 0.2}, {"alpha", 0.5}, {"beta", 0.7}, {"T", 1.0}};
    json hes = {{"type", "heston"}, {"S0", 100}, {"mu", 0.05}, {"kappa", 2.0}, {"theta", 0.04}, {"xi", 0.5},
                {"rho", -0.9}, {"v0", 0.04}, {"T", 1.0}};
    json bsk = {{"type", "basket"}, {"n", 20}, {"S0", 100}, {"mu", 0.05}, {"sigma0", 0.2}, {"alpha", 0.5},
                {"beta", 0.7}, {"rho", 0.0}, {"T", 1.0}};
    caps["presets"] = json::array({
        preset("e01", "1 · Call europea (GBM)", "1 · European call (GBM)", gbm,
               {{"type", "european"}, {"K", 100}, {"r", 0.05}}, {{"family", "qmc"}, {"noise", "bb"}, {"eps", 0.01}}),
        preset("e02", "2 · Asiática aritmética", "2 · Arithmetic Asian", gbm,
               {{"type", "asian"}, {"K", 100}}, {{"family", "mlmc"}, {"eps", 0.01}}),
        preset("e03", "3 · Lookback", "3 · Lookback", gbm,
               {{"type", "lookback"}, {"sigma", 0.2}}, {{"family", "mlmc"}, {"eps", 0.02}}),
        preset("e04", "4 · Barrera up-and-out", "4 · Up-and-out barrier", gbm,
               {{"type", "barrier"}, {"K", 100}, {"B", 120}, {"sigma", 0.2}, {"r", 0.05}}, {{"family", "mc"}, {"eps", 0.02}}),
        preset("e05", "5 · Heston", "5 · Heston", hes,
               {{"type", "european"}, {"K", 100}, {"r", 0.05}}, {{"family", "mlqmc"}, {"eps", 0.02}}),
        preset("e06", "6 · Volatilidad local (Dupire)", "6 · Local vol (Dupire)", dup,
               {{"type", "european"}, {"K", 100}, {"r", 0.05}}, {{"family", "qmc"}, {"noise", "bb"}, {"eps", 0.01}}),
        preset("e07", "7 · Cesta sin correlación", "7 · Uncorrelated basket", bsk,
               {{"type", "basket"}, {"K", 100}, {"r", 0.05}}, {{"family", "qmc"}, {"eps", 0.05}}),
        preset("e08", "8 · Cesta correlacionada", "8 · Correlated basket",
               json{{"type", "basket"}, {"n", 20}, {"S0", 100}, {"mu", 0.05}, {"sigma0", 0.2}, {"alpha", 0.5},
                    {"beta", 0.7}, {"rho", 0.5}, {"T", 1.0}},
               {{"type", "basket"}, {"K", 100}, {"r", 0.05}}, {{"family", "mc"}, {"eps", 0.05}}),
        preset("e09", "9 · Asiática + variable de control", "9 · Asian + control variate", gbm,
               {{"type", "asian"}, {"K", 100}}, {{"family", "mc"}, {"variance", "cv"}, {"eps", 0.005}}),
        preset("e10", "10 · Dupire + control GBM", "10 · Dupire + GBM control", dup,
               {{"type", "european"}, {"K", 100}, {"r", 0.05}}, {{"family", "mc"}, {"variance", "cv"}, {"eps", 0.01}}),
        preset("e11", "11 · Call muy OTM + importance sampling", "11 · Deep OTM call + importance sampling", gbm,
               {{"type", "european"}, {"K", 180}, {"r", 0.05}}, {{"family", "mc"}, {"variance", "is"}, {"eps", 0.002}})});
    return caps;
}

// ---------------------------------------------------------------------------------------------------
// Eventos
// ---------------------------------------------------------------------------------------------------

namespace {

const char* stage_name(Stage s) {
    switch (s) {
    case Stage::Plan: return "plan";
    case Stage::Pilot: return "pilot";
    case Stage::Main: return "main";
    case Stage::Level: return "level";
    case Stage::Doubling: return "doubling";
    case Stage::Done: return "done";
    }
    return "?";
}

} // namespace

json snapshot_json(const Snapshot& s) {
    json j{{"type", "progress"}, {"seq", s.seq}, {"stage", stage_name(s.stage)}, {"elapsed", r4(s.elapsed_s)},
           {"n_done", s.n_done}, {"mean", s.mean}, {"se", s.std_error}, {"eps", s.eps_target}};
    if (s.n_steps > 0) j["n_steps"] = s.n_steps;
    if (s.L >= 0) {
        j["L"] = s.L;
        json lv = json::array();
        for (int i = 0; i < s.n_levels; i++)
            lv.push_back({{"l", s.levels[i].l}, {"N", s.levels[i].N}, {"E", s.levels[i].E}, {"V", s.levels[i].V},
                          {"cost", s.levels[i].cost}});
        j["levels"] = std::move(lv);
    }
    if (s.doubling >= 0) {
        j["doubling"] = s.doubling;
        j["n_per_replica"] = s.n_per_replica;
        if (s.replica_means && s.n_replicas > 0)
            j["replica_means"] = std::vector<double>(s.replica_means, s.replica_means + s.n_replicas);
    }
    return j;
}

json report_json(const ParsedRun& run, const mc::RunReport& rep, const std::optional<ReferencePrice>& ref) {
    const MCResult& r = rep.result;
    json j{{"type", "result"}, {"price", r.price}, {"std_error", r.std_error}, {"n_samples", r.n_samples},
           {"time_s", r4(r.time_s)}, {"backend", mc::backend_name(rep.backend)}, {"threads", rep.threads},
           {"n_steps", rep.n_steps}, {"truncated", rep.truncated}, {"cancelled", rep.cancelled},
           {"n_nonfinite", rep.n_nonfinite}, {"eps", run.spec.eps},
           {"family", mc::family_name(run.spec.family)}, {"variance", mc::variance_name(run.spec.variance)},
           {"noise", mc::noise_name(run.spec.noise)}};
    j["ci95"] = json::array({r.price - 1.96 * r.std_error, r.price + 1.96 * r.std_error});
    j["mpaths_per_s"] = r.time_s > 0 ? r4(r.n_samples / r.time_s / 1e6) : 0.0;
    if (rep.c1 > 0) j["c1"] = rep.c1;
    if (run.spec.variance == mc::Variance::ControlVariate) { j["beta"] = rep.beta; j["E_ctrl"] = rep.E_ctrl; }
    if (run.spec.variance == mc::Variance::ImportanceSampling) j["z_star"] = rep.z_star;
    if (!rep.levels.empty()) {
        json lv = json::array();
        for (const auto& l : rep.levels) lv.push_back({{"l", l.l}, {"N", l.N}, {"E", l.E}, {"V", l.V}, {"cost", l.cost}});
        j["levels"] = std::move(lv);
    }
    if (ref) {
        j["reference"] = {{"value", ref->value}, {"kind", ref->kind}};
        j["abs_error"] = std::abs(r.price - ref->value);
        j["z_score"] = r.std_error > 0 ? (r.price - ref->value) / r.std_error : 0.0;
        j["within"] = std::abs(r.price - ref->value) < 2.0 * run.spec.eps;
    } else {
        j["reference"] = nullptr;
    }
    return j;
}

// ---------------------------------------------------------------------------------------------------
// POST /api/paths
// ---------------------------------------------------------------------------------------------------

json paths_response(const json& body) {
    if (!body.is_object()) bad("el cuerpo debe ser un objeto JSON");
    std::string mt, pt;
    const ModelVariant model = parse_model(body.value("model", json::object()), mt);
    const PayoffVariant payoff = parse_payoff(body.value("payoff", json::object()), model, pt);

    mc::cpu::SampleOptions so;
    so.n_steps = inum(body, "n_steps", 128, 2, 2048);
    so.n_paths = inum(body, "n_paths", 5, 1, 200);
    so.seed = (uint64_t)num(body, "seed", 1.0, 0.0, 9e15);
    so.sobol = body.value("sobol", false);
    so.construction = parse_noise(str(body, "construction", "raw"));
    if ((long long)so.n_steps * model_noise_dim(model) > 40000) bad("n_steps * dimensión supera el límite de 40000");

    json out;
    mc::cpu::SampledPaths sp = mc::cpu::sample_paths(model, payoff, so);
    out["t"] = json::array();
    for (double t : sp.t) out["t"].push_back(r6(t));
    out["paths"] = json::array();
    for (const auto& p : sp.paths) {
        json jp;
        jp["S"] = json::array();
        for (double v : p.S) jp["S"].push_back(r4(v));
        if (!p.V.empty()) { jp["V"] = json::array(); for (double v : p.V) jp["V"].push_back(r6(v)); }
        if (!p.run.empty()) { jp["run"] = json::array(); for (double v : p.run) jp["run"].push_back(r4(v)); }
        if (!p.assets.empty()) {
            jp["assets"] = json::array();
            for (const auto& a : p.assets) {
                json ja = json::array();
                for (double v : a) ja.push_back(r4(v));
                jp["assets"].push_back(std::move(ja));
            }
        }
        jp["payoff"] = r4(p.payoff);
        jp["knocked"] = p.knocked_out;
        out["paths"].push_back(std::move(jp));
    }

    // Líneas de referencia del payoff
    json marks = json::object();
    std::visit([&](const auto& pv) {
        using Tp = std::decay_t<decltype(pv)>;
        if constexpr (std::is_same_v<Tp, European> || std::is_same_v<Tp, Asian> || std::is_same_v<Tp, GeomAsian> ||
                      std::is_same_v<Tp, Barrier> || std::is_same_v<Tp, Basket>)
            marks["strike"] = pv.K;
        if constexpr (std::is_same_v<Tp, Barrier>) marks["barrier"] = pv.B;
    }, payoff);
    marks["S0"] = model_S0(model);
    out["marks"] = marks;
    out["model_type"] = mt;
    out["payoff_type"] = pt;

    const int n_fan = inum(body, "fan", 0, 0, 5000);
    if (n_fan >= 20) {
        const auto fb = mc::cpu::sample_fan(model, payoff, std::min(so.n_steps, 256), n_fan, so.seed + 101);
        json f;
        f["t"] = json::array();
        for (double t : fb.t) f["t"].push_back(r6(t));
        auto arr = [&](const char* key, const std::vector<double>& v) {
            f[key] = json::array();
            for (double x : v) f[key].push_back(r4(x));
        };
        arr("p05", fb.p05); arr("p25", fb.p25); arr("p50", fb.p50); arr("p75", fb.p75); arr("p95", fb.p95);
        arr("mean", fb.mean);
        // histograma del valor terminal (40 cajas entre los percentiles 0.5% y 99.5%)
        std::vector<double> term = fb.terminal;
        std::sort(term.begin(), term.end());
        const double lo = term[(size_t)(0.005 * (term.size() - 1))], hi = term[(size_t)(0.995 * (term.size() - 1))];
        const int bins = 40;
        std::vector<int> cnt(bins, 0);
        for (double v : term) {
            if (v < lo || v > hi) continue;
            int b = (int)std::min<double>(bins - 1, std::floor((v - lo) / (hi - lo + 1e-12) * bins));
            cnt[(size_t)b]++;
        }
        f["hist"] = {{"lo", r4(lo)}, {"hi", r4(hi)}, {"counts", cnt}};
        f["payoff_mean"] = r4(fb.payoff_mean);
        out["fan"] = std::move(f);
    }

    if (body.contains("coupled") && body["coupled"].is_object()) {
        const json& c = body["coupled"];
        const int level = inum(c, "level", 4, 1, 10), M = inum(c, "M", 2, 2, 8);
        const auto cp = mc::cpu::sample_coupled(model, payoff, level, M, so.seed, inum(c, "index", 0, 0, 1000000));
        json jc;
        auto vec = [&](const std::vector<double>& v, bool six) {
            json a = json::array();
            for (double x : v) a.push_back(six ? r6(x) : r4(x));
            return a;
        };
        jc["t_fine"] = vec(cp.t_fine, true); jc["S_fine"] = vec(cp.S_fine, false);
        jc["t_coarse"] = vec(cp.t_coarse, true); jc["S_coarse"] = vec(cp.S_coarse, false);
        jc["payoff_fine"] = r4(cp.payoff_fine); jc["payoff_coarse"] = r4(cp.payoff_coarse);
        jc["level"] = level; jc["M"] = M;
        out["coupled"] = std::move(jc);
    }
    return out;
}

} // namespace mc::gui
