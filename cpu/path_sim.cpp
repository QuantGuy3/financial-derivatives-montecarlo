#include "path_sim.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace mc::cpu {

std::vector<long long> make_wave_ends(long long n_chunks, bool fine) {
    std::vector<long long> ends;
    long long c = 0, wave = fine ? 1 : 64;
    while (c < n_chunks) {
        c = std::min(n_chunks, c + wave);
        ends.push_back(c);
        wave = fine ? std::min<long long>(256, std::max(wave + 1, wave * 5 / 4))
                    : std::min<long long>(kMaxWaveChunks, wave * 2);
    }
    return ends;
}

PathSim::PathSim(const CpuModel& m, const CpuPayoff& p, int n_steps, const NoiseSource& noise,
                 const EvalSpec& eval)
    : m_(m), p_(p), noise_(noise), kc_(make_kctx(m, p, n_steps, eval)), eval_(eval) {
    if (n_steps < 1) throw std::invalid_argument("PathSim: n_steps debe ser >= 1");
    if (noise.dim() != n_steps * m.noise_dim)
        throw std::invalid_argument("PathSim: la dimensión de la fuente de ruido no coincide con n_steps*noise_dim");
    multi_ = (m.kind == ModelKind::MultiDupire);
    if (eval.kind != EvalSpec::Kind::Plain) {
        // Solo las parejas que implementa la GPU (kernel_gbm_asian_cv, kernel_dupire_gbm_cv, kernel_is_gbm*)
        const bool ok =
            (eval.kind == EvalSpec::Kind::CvAsianGeom && m.kind == ModelKind::GBM && p.kind == PayoffKind::Asian) ||
            (eval.kind == EvalSpec::Kind::CvDupireGbm && m.kind == ModelKind::Dupire && p.kind == PayoffKind::European) ||
            (eval.kind == EvalSpec::Kind::IsGbmCall && m.kind == ModelKind::GBM && p.kind == PayoffKind::European);
        if (!ok) throw std::invalid_argument("PathSim: reducción de varianza no soportada para este modelo/payoff");
        eval_fn_ = select_eval_kernel(eval.kind);
    } else if (!multi_) {
        fn_ = select_single_kernel(m.kind, p.kind);
    } else if (p.kind != PayoffKind::Basket) {
        throw std::invalid_argument("PathSim: la cesta multi-activo solo admite el payoff Basket");
    }
    double work = (double)n_steps * m.noise_dim;
    if (multi_ && !m.uncorrelated) work *= std::max(1.0, m.n_assets / 2.0);   // Cholesky por paso
    chunk_paths_ = chunk_paths_for(work);
}

namespace {

// dW <- L * dW por paso, en sitio. Recorrer a descendente permite sobrescribir: la fila a solo
// lee las entradas b <= a, que aún no se han modificado.
void correlate_block(const CpuModel& m, double* Z, int ld, int n_steps) {
    constexpr int W = kLanes;
    const int n = m.n_assets;
    for (int k = 0; k < n_steps; k++) {
        double* base = Z + (size_t)k * n * ld;
        for (int a = n - 1; a >= 0; a--) {
            double accv[W] = {};
            const double* La = m.L.data() + (size_t)a * n;
            for (int b = 0; b <= a; b++) {
                const double lb = La[b];
                const double* in = base + (size_t)b * ld;
                for (int l = 0; l < W; l++) accv[l] += lb * in[l];
            }
            double* out = base + (size_t)a * ld;
            for (int l = 0; l < W; l++) out[l] = accv[l];
        }
    }
}

} // namespace

void PathSim::run_chunk(uint64_t first, int count, Scratch& s, ChunkAcc& out) const {
    constexpr int W = kLanes;
    const int D = noise_.dim();
    if ((int)s.Z.size() < D * W) s.Z.resize((size_t)D * W);
    if (multi_ && (int)s.S.size() < m_.n_assets * W) s.S.resize((size_t)m_.n_assets * W);
    double* Z = s.Z.data();
    double Y[W], Y1[W];
    std::unique_ptr<NoiseStream> stream = noise_.open(first, (uint64_t)count);
    const bool cv = (eval_.kind == EvalSpec::Kind::CvAsianGeom || eval_.kind == EvalSpec::Kind::CvDupireGbm);

    for (int b = 0; b < count; b += W) {
        const int n = std::min(W, count - b);
        if (n < W) std::memset(Z, 0, sizeof(double) * (size_t)D * W);   // carriles de relleno
        stream->fill(n, Z, W);
        if (eval_fn_) {
            eval_fn_(kc_, Z, W, Y, Y1);
            for (int l = 0; l < n; l++) {
                if (cv) {
                    const double ycv = Y[l] - eval_.beta * (Y1[l] - eval_.E_ctrl);
                    if (std::isfinite(ycv) && std::isfinite(Y[l]) && std::isfinite(Y1[l])) {
                        out.acc.add(ycv);
                        out.pair.add(Y[l], Y1[l]);
                    } else ++out.nonfinite;
                } else {
                    if (std::isfinite(Y[l])) out.acc.add(Y[l]);
                    else ++out.nonfinite;
                }
            }
            continue;
        }
        if (multi_) {
            if (!m_.uncorrelated) correlate_block(m_, Z, W, kc_.n_steps);
            kernel_basket(kc_, Z, W, Y, s.S.data());
        } else {
            fn_(kc_, Z, W, Y);
        }
        for (int l = 0; l < n; l++) {
            if (std::isfinite(Y[l])) out.acc.add(Y[l]);
            else ++out.nonfinite;
        }
    }
}

CoupledSim::CoupledSim(const CpuModel& m, const CpuPayoff& p, int level, int M, const NoiseSource& noise,
                       const EvalSpec& eval)
    : noise_(noise), kc_(make_ckctx(m, p, level, M, eval)) {
    if (m.kind == ModelKind::MultiDupire)
        throw std::invalid_argument("MLMC: las cestas multi-activo no están soportadas");
    if (noise.dim() != kc_.n_fine * m.noise_dim)
        throw std::invalid_argument("CoupledSim: la dimensión de la fuente de ruido no coincide con n_fine*noise_dim");
    if (eval.kind == EvalSpec::Kind::Plain) {
        fn_ = select_coupled_kernel(m.kind, p.kind);
    } else {
        const bool ok =
            (eval.kind == EvalSpec::Kind::CvAsianGeom && m.kind == ModelKind::GBM && p.kind == PayoffKind::Asian) ||
            (eval.kind == EvalSpec::Kind::IsGbmCall && m.kind == ModelKind::GBM && p.kind == PayoffKind::European);
        if (!ok) throw std::invalid_argument("CoupledSim: reducción de varianza no soportada para este modelo/payoff");
        fn_ = select_coupled_eval_kernel(eval.kind);
    }
    chunk_paths_ = chunk_paths_for((double)kc_.n_fine * m.noise_dim * 1.5);
}

void CoupledSim::run_chunk(uint64_t first, int count, Scratch& s, ChunkAcc2& out) const {
    constexpr int W = kLanes;
    const int D = noise_.dim();
    if ((int)s.Z.size() < D * W) s.Z.resize((size_t)D * W);
    double* Z = s.Z.data();
    double Yf[W], Yc[W];
    std::unique_ptr<NoiseStream> stream = noise_.open(first, (uint64_t)count);

    for (int b = 0; b < count; b += W) {
        const int n = std::min(W, count - b);
        if (n < W) std::memset(Z, 0, sizeof(double) * (size_t)D * W);
        stream->fill(n, Z, W);
        fn_(kc_, Z, W, Yf, Yc);
        for (int l = 0; l < n; l++) {
            const double dy = Yf[l] - Yc[l];
            if (std::isfinite(dy) && std::isfinite(Yf[l])) { out.dY.add(dy); out.Yf.add(Yf[l]); }
            else ++out.nonfinite;
        }
    }
}

RangeResult simulate_range(ThreadPool& pool, const PathSim& sim, uint64_t first, long long N,
                           const std::function<bool(const Moments&, long long)>& on_wave, bool fine_waves) {
    RangeResult res;
    if (N <= 0) return res;
    const long long cp = sim.chunk_paths();
    const long long n_chunks = (N + cp - 1) / cp;
    const auto ends = make_wave_ends(n_chunks, fine_waves && on_wave);

    std::vector<Scratch> scratch(pool.threads());
    std::vector<Padded<ChunkAcc>> slots((size_t)std::min<long long>(n_chunks, kMaxWaveChunks));

    long long done_chunks = 0;
    for (long long wave_end : ends) {
        const long long wave_chunks = wave_end - done_chunks;
        const long long c0 = done_chunks;
        for (auto& sl : slots) sl.v = ChunkAcc{};
        pool.parallel_for((size_t)wave_chunks, [&](size_t t, int worker) {
            const long long c = c0 + (long long)t;
            const long long p0 = c * cp;
            const int count = (int)std::min<long long>(cp, N - p0);
            sim.run_chunk(first + (uint64_t)p0, count, scratch[worker], slots[t].v);
        });
        // Fusión en orden de índice de chunk (una a una, para que el resultado no dependa de
        // cómo se agrupan los chunks en rondas).
        for (long long t = 0; t < wave_chunks; t++) {
            res.moments.merge(slots[(size_t)t].v.acc.to_moments());
            res.pair.merge(slots[(size_t)t].v.pair.to_cov2());
            res.nonfinite += slots[(size_t)t].v.nonfinite;
        }
        done_chunks = wave_end;
        const long long n_done = std::min<long long>(done_chunks * cp, N);
        if (on_wave && !on_wave(res.moments, n_done)) {
            res.stopped = (done_chunks < n_chunks);
            return res;
        }
    }
    return res;
}

} // namespace mc::cpu
