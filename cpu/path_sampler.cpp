#include "path_sampler.hpp"

#include "kernels.hpp"
#include "noise.hpp"
#include "params.hpp"
#include "qmc_noise.hpp"
#include "sobol.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace mc::cpu {

namespace {

bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

// Payoff evaluado en tiempo de ejecución con las MISMAS funciones plantilla que el motor.
struct PayoffEval {
    const CpuPayoff& p;
    int n_steps;
    double h;

    double init(double S0) const { return p.kind == PayoffKind::Lookback ? S0 : 0.0; }

    void update(double& run, double S) const {
        switch (p.kind) {
        case PayoffKind::Asian:     running_update<PayoffKind::Asian>(run, S); break;
        case PayoffKind::GeomAsian: running_update<PayoffKind::GeomAsian>(run, S); break;
        case PayoffKind::Lookback:  running_update<PayoffKind::Lookback>(run, S); break;
        case PayoffKind::Barrier:   running_update<PayoffKind::Barrier>(run, S); break;
        default: break;
        }
    }

    // Valor que se dibuja del estadístico corriente tras k pasos
    double display(double run, int k) const {
        switch (p.kind) {
        case PayoffKind::Asian:     return k > 0 ? run / k : 0.0;
        case PayoffKind::GeomAsian: return k > 0 ? std::exp(run / k) : 0.0;
        default:                    return run;
        }
    }

    double finish(double S_T, double run) const {
        switch (p.kind) {
        case PayoffKind::European: return terminal_payoff<PayoffKind::European>(p, S_T, run, n_steps);
        case PayoffKind::Asian:    return terminal_payoff<PayoffKind::Asian>(p, S_T, run, n_steps);
        case PayoffKind::GeomAsian:return terminal_payoff<PayoffKind::GeomAsian>(p, S_T, run, n_steps);
        case PayoffKind::Lookback:
            apply_bgk<PayoffKind::Lookback>(p, run, h);
            return terminal_payoff<PayoffKind::Lookback>(p, S_T, run, n_steps);
        case PayoffKind::Barrier:
            apply_bgk<PayoffKind::Barrier>(p, run, h);
            return terminal_payoff<PayoffKind::Barrier>(p, S_T, run, n_steps);
        case PayoffKind::Basket:   return terminal_payoff<PayoffKind::Basket>(p, S_T, run, n_steps);
        }
        return 0.0;
    }
};

// Cadena de ruido (pseudoaleatorio / Sobol, con construcción) que vive lo que dura el muestreo.
struct NoiseChain {
    std::unique_ptr<ScrambledSobol> sob;
    std::unique_ptr<NoiseSource> base, wrapped;
    const NoiseSource* src = nullptr;

    NoiseChain(const CpuModel& m, int n_steps, uint64_t seed, bool sobol, NoiseMode construction) {
        const int D = n_steps * m.noise_dim;
        if (construction != NoiseMode::Raw) {
            if (m.noise_dim != 1)
                throw std::invalid_argument("BB/PCA solo para modelos con ruido de dimensión 1 (GBM, Dupire)");
            if (construction == NoiseMode::BrownianBridge && !is_pow2(n_steps))
                throw std::invalid_argument("Brownian Bridge: n_steps debe ser potencia de 2");
        }
        const double scale = (construction == NoiseMode::Raw) ? std::sqrt(m.T / n_steps) : 1.0;
        if (sobol && D <= D_MAX_SOBOL) {
            sob = std::make_unique<ScrambledSobol>((uint32_t)seed, D);
            base = std::make_unique<SobolNoise>(*sob, scale);
        } else {
            base = std::make_unique<RngNoise>(seed, Stream::Paths, 0, D, scale);
        }
        src = base.get();
        if (construction == NoiseMode::BrownianBridge)
            wrapped = std::make_unique<BrownianBridgeNoise>(*base, bb_precompute(n_steps, m.T));
        else if (construction == NoiseMode::PCA)
            wrapped = std::make_unique<PcaNoise>(*base, pca_compute(n_steps, m.T));
        if (wrapped) src = wrapped.get();
    }

    // Incrementos brownianos (D valores, orden k*dim + c) del camino de índice `idx`.
    void path_noise(uint64_t idx, std::vector<double>& Z) const {
        Z.assign((size_t)src->dim(), 0.0);
        auto st = src->open(idx, 1);
        st->fill(1, Z.data(), 1);
    }
};

// Correlación de la cesta (Cholesky) in situ sobre los n_assets incrementos de un paso.
void correlate_step(const CpuModel& m, double* dw) {
    if (m.uncorrelated) return;
    const int n = m.n_assets;
    for (int a = n - 1; a >= 0; a--) {
        double acc = 0.0;
        for (int b = 0; b <= a; b++) acc += m.L[(size_t)a * n + b] * dw[b];
        dw[a] = acc;
    }
}

} // namespace

SampledPaths sample_paths(const ModelVariant& mv, const PayoffVariant& pv, const SampleOptions& o) {
    if (o.n_steps < 1) throw std::invalid_argument("n_steps debe ser >= 1");
    if (o.n_paths < 1) throw std::invalid_argument("n_paths debe ser >= 1");
    const CpuModel m = make_cpu_model(mv);
    const CpuPayoff p = make_cpu_payoff(pv);
    const int n = o.n_steps;
    const double h = m.T / n;
    const double em = std::exp(-m.kappa * h);
    const PayoffEval pe{p, n, h};
    NoiseChain chain(m, n, o.seed, o.sobol, o.construction);

    SampledPaths out;
    out.n_steps = n;
    out.t.resize((size_t)n + 1);
    for (int k = 0; k <= n; k++) out.t[(size_t)k] = k * h;

    std::vector<double> Z;
    const bool multi = (m.kind == ModelKind::MultiDupire);
    const int shown = multi ? std::min(m.n_assets, std::max(0, o.max_assets_shown)) : 0;

    for (int i = 0; i < o.n_paths; i++) {
        chain.path_noise((uint64_t)i, Z);
        SampledPath sp;
        sp.S.assign((size_t)n + 1, m.S0);
        sp.run.assign((size_t)n + 1, 0.0);
        const bool has_run = (p.kind == PayoffKind::Asian || p.kind == PayoffKind::GeomAsian ||
                              p.kind == PayoffKind::Lookback || p.kind == PayoffKind::Barrier);
        double run = pe.init(m.S0);
        sp.run[0] = m.S0;

        if (multi) {
            const int na = m.n_assets;
            std::vector<double> Sa(m.S0v);
            sp.assets.assign((size_t)shown, std::vector<double>((size_t)n + 1));
            for (int a = 0; a < shown; a++) sp.assets[(size_t)a][0] = Sa[(size_t)a];
            sp.S[0] = std::accumulate(Sa.begin(), Sa.end(), 0.0) / na;
            for (int k = 0; k < n; k++) {
                double* dw = Z.data() + (size_t)k * na;
                correlate_step(m, dw);
                const double e_t = std::exp(-m.alpha * (k * h));
                for (int a = 0; a < na; a++) Sa[(size_t)a] = euler_dupire(m, Sa[(size_t)a], dw[a], h, e_t, m.S0v[(size_t)a]);
                for (int a = 0; a < shown; a++) sp.assets[(size_t)a][(size_t)k + 1] = Sa[(size_t)a];
                sp.S[(size_t)k + 1] = std::accumulate(Sa.begin(), Sa.end(), 0.0) / na;
            }
            sp.payoff = pe.finish(sp.S[(size_t)n], 0.0);
            sp.run.clear();
        } else {
            double S = m.S0, V = m.v0;
            if (m.kind == ModelKind::Heston) { sp.V.assign((size_t)n + 1, m.v0); }
            for (int k = 0; k < n; k++) {
                if (m.kind == ModelKind::GBM) {
                    S = euler_gbm(m, S, Z[(size_t)k], h);
                } else if (m.kind == ModelKind::Dupire) {
                    S = euler_dupire(m, S, Z[(size_t)k], h, std::exp(-m.alpha * (k * h)), m.S0);
                } else {
                    const double a1 = Z[(size_t)k * 2], a2 = Z[(size_t)k * 2 + 1];
                    euler_heston(m, S, V, a1, m.l21 * a1 + m.l22 * a2, h, em);
                    sp.V[(size_t)k + 1] = V;
                }
                sp.S[(size_t)k + 1] = S;
                if (has_run) {
                    pe.update(run, S);
                    sp.run[(size_t)k + 1] = pe.display(run, k + 1);
                    if (p.kind == PayoffKind::Barrier && run >= p.B) sp.knocked_out = true;
                }
            }
            sp.payoff = pe.finish(S, run);
            if (!has_run) sp.run.clear();
        }
        out.paths.push_back(std::move(sp));
    }
    return out;
}

CoupledPair sample_coupled(const ModelVariant& mv, const PayoffVariant& pv, int level, int M,
                           uint64_t seed, int index) {
    const CpuModel m = make_cpu_model(mv);
    const CpuPayoff p = make_cpu_payoff(pv);
    if (m.kind == ModelKind::MultiDupire) throw std::invalid_argument("MLMC: las cestas no están soportadas");
    if (level < 1 || M < 2) throw std::invalid_argument("par acoplado: level >= 1 y M >= 2");
    const CKCtx c = make_ckctx(m, p, level, M);
    const int dim = m.noise_dim;
    RngNoise noise(seed, Stream::Paths, (uint64_t)level, c.n_fine * dim, std::sqrt(c.h_f));
    std::vector<double> Z((size_t)c.n_fine * dim);
    auto st = noise.open((uint64_t)index, 1);
    st->fill(1, Z.data(), 1);

    CoupledPair cp;
    cp.n_fine = c.n_fine; cp.n_coarse = c.n_coarse;
    cp.t_fine.resize((size_t)c.n_fine + 1); cp.S_fine.assign((size_t)c.n_fine + 1, m.S0);
    cp.t_coarse.resize((size_t)c.n_coarse + 1); cp.S_coarse.assign((size_t)c.n_coarse + 1, m.S0);
    for (int k = 0; k <= c.n_fine; k++) cp.t_fine[(size_t)k] = k * c.h_f;
    for (int k = 0; k <= c.n_coarse; k++) cp.t_coarse[(size_t)k] = k * c.h_c;

    double Sf = m.S0, Vf = m.v0, Sc = m.S0, Vc = m.v0, acc1 = 0.0, acc2 = 0.0;
    int ck = 0;
    for (int k = 0; k < c.n_fine; k++) {
        if (m.kind == ModelKind::GBM) {
            acc1 += Z[(size_t)k];
            Sf = euler_gbm(m, Sf, Z[(size_t)k], c.h_f);
        } else if (m.kind == ModelKind::Dupire) {
            acc1 += Z[(size_t)k];
            Sf = euler_dupire(m, Sf, Z[(size_t)k], c.h_f, std::exp(-m.alpha * (k * c.h_f)), m.S0);
        } else {
            const double a1 = Z[(size_t)k * 2], a2 = Z[(size_t)k * 2 + 1];
            const double dw2 = m.l21 * a1 + m.l22 * a2;
            acc1 += a1; acc2 += dw2;
            euler_heston(m, Sf, Vf, a1, dw2, c.h_f, c.em_f);
        }
        cp.S_fine[(size_t)k + 1] = Sf;
        if ((k + 1) % M == 0) {
            if (m.kind == ModelKind::GBM) Sc = Sc + m.mu * Sc * c.h_c + m.sigma * Sc * acc1;
            else if (m.kind == ModelKind::Dupire) Sc = euler_dupire(m, Sc, acc1, c.h_c, std::exp(-m.alpha * (ck * c.h_c)), m.S0);
            else euler_heston(m, Sc, Vc, acc1, acc2, c.h_c, c.em_c);
            acc1 = acc2 = 0.0;
            ++ck;
            cp.S_coarse[(size_t)ck] = Sc;
        }
    }
    cp.payoff_fine = terminal_payoff<PayoffKind::European>(p, Sf, 0.0, c.n_fine);
    cp.payoff_coarse = terminal_payoff<PayoffKind::European>(p, Sc, 0.0, c.n_coarse);
    return cp;
}

FanBands sample_fan(const ModelVariant& mv, const PayoffVariant& pv, int n_steps, int n_fan, uint64_t seed) {
    SampleOptions o;
    o.n_paths = std::max(2, n_fan); o.n_steps = n_steps; o.seed = seed; o.sobol = false;
    o.max_assets_shown = 0;
    SampledPaths sp = sample_paths(mv, pv, o);

    FanBands fb;
    fb.t = sp.t;
    const size_t cols = sp.t.size(), rows = sp.paths.size();
    fb.p05.resize(cols); fb.p25.resize(cols); fb.p50.resize(cols); fb.p75.resize(cols); fb.p95.resize(cols);
    fb.mean.resize(cols);
    std::vector<double> col(rows);
    auto quant = [&](double q) {
        const size_t idx = (size_t)std::llround(q * (double)(rows - 1));
        std::nth_element(col.begin(), col.begin() + (long)idx, col.end());
        return col[idx];
    };
    for (size_t k = 0; k < cols; k++) {
        double mean = 0.0;
        for (size_t i = 0; i < rows; i++) { col[i] = sp.paths[i].S[k]; mean += col[i]; }
        fb.mean[k] = mean / (double)rows;
        fb.p05[k] = quant(0.05); fb.p25[k] = quant(0.25); fb.p50[k] = quant(0.50);
        fb.p75[k] = quant(0.75); fb.p95[k] = quant(0.95);
    }
    fb.terminal.resize(rows); fb.payoffs.resize(rows);
    double pm = 0.0;
    for (size_t i = 0; i < rows; i++) {
        fb.terminal[i] = sp.paths[i].S.back();
        fb.payoffs[i] = sp.paths[i].payoff;
        pm += fb.payoffs[i];
    }
    fb.payoff_mean = pm / (double)rows;
    return fb;
}

} // namespace mc::cpu
