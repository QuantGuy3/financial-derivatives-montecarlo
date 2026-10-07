#include "path_sim.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace mc::cpu {

std::vector<long long> make_wave_ends(long long n_chunks) {
    constexpr long long kMaxWave = 256;
    std::vector<long long> ends;
    long long c = 0, wave = 1;
    while (c < n_chunks) {
        c = std::min(n_chunks, c + wave);
        ends.push_back(c);
        wave = std::min(kMaxWave, std::max(wave + 1, wave * 5 / 4));
    }
    return ends;
}

PathSim::PathSim(const CpuModel& m, const CpuPayoff& p, int n_steps, const NoiseSource& noise)
    : m_(m), p_(p), noise_(noise), kc_(make_kctx(m, p, n_steps)) {
    if (n_steps < 1) throw std::invalid_argument("PathSim: n_steps debe ser >= 1");
    if (noise.dim() != n_steps * m.noise_dim)
        throw std::invalid_argument("PathSim: la dimensión de la fuente de ruido no coincide con n_steps*noise_dim");
    multi_ = (m.kind == ModelKind::MultiDupire);
    if (!multi_) fn_ = select_single_kernel(m.kind, p.kind);
    else if (p.kind != PayoffKind::Basket)
        throw std::invalid_argument("PathSim: la cesta multi-activo solo admite el payoff Basket");
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
    double Y[W];
    std::unique_ptr<NoiseStream> stream = noise_.open(first, (uint64_t)count);

    for (int b = 0; b < count; b += W) {
        const int n = std::min(W, count - b);
        if (n < W) std::memset(Z, 0, sizeof(double) * (size_t)D * W);   // carriles de relleno
        stream->fill(n, Z, W);
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

CoupledSim::CoupledSim(const CpuModel& m, const CpuPayoff& p, int level, int M, const NoiseSource& noise)
    : noise_(noise), kc_(make_ckctx(m, p, level, M)) {
    if (m.kind == ModelKind::MultiDupire)
        throw std::invalid_argument("MLMC: las cestas multi-activo no están soportadas");
    if (noise.dim() != kc_.n_fine * m.noise_dim)
        throw std::invalid_argument("CoupledSim: la dimensión de la fuente de ruido no coincide con n_fine*noise_dim");
    fn_ = select_coupled_kernel(m.kind, p.kind);
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
                           const std::function<bool(const Moments&, long long)>& on_wave) {
    RangeResult res;
    if (N <= 0) return res;
    const long long cp = sim.chunk_paths();
    const long long n_chunks = (N + cp - 1) / cp;
    const auto ends = make_wave_ends(n_chunks);

    std::vector<Scratch> scratch(pool.threads());
    std::vector<Padded<ChunkAcc>> slots((size_t)std::min<long long>(n_chunks, 256));

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
