// Microbenchmarks de las piezas del motor CPU (un hilo): cada experimento compara la variante
// original/ingenua con la optimizada y da ns por elemento. Se usan para decidir qué optimizaciones
// se conservan (criterio: >= 5 % en la carga objetivo y sin regresiones; ver docs/perf/).
//
// Uso: bench_micro [--filter texto] [--reps N] [--json fichero]

#include "../cpu/mc_cpu.hpp"
#include "../cpu/noise.hpp"
#include "../cpu/normal.hpp"
#include "../cpu/path_sim.hpp"
#include "../cpu/qmc_noise.hpp"
#include "../cpu/rng.hpp"
#include "../cpu/sobol.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <numbers>
#include <string>
#include <vector>

using namespace mc::cpu;

namespace {

volatile double g_sink = 0.0;
volatile uint64_t g_isink = 0;

// Tiempo mínimo (s) de `reps` ejecuciones de fn (tras un calentamiento)
double best_of(int reps, const std::function<void()>& fn) {
    fn();
    double best = 1e300;
    for (int i = 0; i < reps; i++) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

struct Row { std::string group, name; double ns; double speedup_vs_first; std::string unit; };
std::vector<Row> rows;
std::string filter;
int reps = 7;

void report(const std::string& group, const std::vector<std::pair<std::string, double>>& variants) {
    // La unidad va entre paréntesis en el nombre del grupo: "(ns/normal)", "(ms)", ...
    const auto lp = group.rfind('('), rp = group.rfind(')');
    const std::string unit = (lp != std::string::npos && rp != std::string::npos && rp > lp) ? group.substr(lp + 1, rp - lp - 1) : "";
    for (size_t i = 0; i < variants.size(); i++) {
        Row r{group, variants[i].first, variants[i].second, variants[0].second / variants[i].second, unit};
        rows.push_back(r);
        std::printf("%-40s %-36s %12.3f   x%.2f\n", group.c_str(), r.name.c_str(), r.ns, r.speedup_vs_first);
    }
    std::fflush(stdout);
}

bool selected(const std::string& g) { return filter.empty() || g.find(filter) != std::string::npos; }

// --- 1. Generadores de normales (ns por normal) ------------------------------------------------------
void bench_normals() {
    if (!selected("normales")) return;
    const int paths = 20000, per_path = 64;
    std::vector<double> z(per_path);
    auto run = [&](NormalMethod m) {
        return best_of(reps, [&] {
            double acc = 0;
            for (int p = 0; p < paths; p++) {
                auto g = Xoshiro256pp::for_path(1, Stream::Test, 0, (uint64_t)p);
                fill_normals(m, g, z.data(), per_path);
                acc += z[per_path - 1];
            }
            g_sink = acc;
        }) / ((double)paths * per_path) * 1e9;
    };
    report("normales (ns/normal)", {{"Box-Muller (log+sqrt+sincos)", run(NormalMethod::BoxMuller)},
                                    {"CDF inversa AS 241", run(NormalMethod::InverseCdf)},
                                    {"Ziggurat 256 capas", run(NormalMethod::Ziggurat)}});
}

// --- 1c. Potencia de la volatilidad local de Dupire: pow(x, b) frente a exp(b*log(x)) (ns por llamada) ---------
void bench_pow() {
    if (!selected("potencia")) return;
    const int N = 1 << 16;
    std::vector<double> x((size_t)N);
    auto g = Xoshiro256pp::for_path(2, Stream::Test, 0, 0);
    for (auto& v : x) v = 0.7 + 0.6 * g.uniform();          // S/S0 en un rango típico
    const double b = -0.3;
    double a_pow = best_of(reps, [&] { double a = 0; for (int i = 0; i < N; i++) a += std::pow(x[(size_t)i], b); g_sink = a; }) / N * 1e9;
    double a_exp = best_of(reps, [&] { double a = 0; for (int i = 0; i < N; i++) a += std::exp(b * std::log(x[(size_t)i])); g_sink = a; }) / N * 1e9;
    report("potencia (S/S0)^(beta-1) (ns/llamada)", {{"std::pow", a_pow}, {"std::exp(b*std::log(x))", a_exp}});
}

// --- 1d. De extremo a extremo (1 hilo), A/B INTERCALADO: robusto frente a cambios de frecuencia de la CPU ----------
void bench_e2e() {
    if (!selected("e2e")) return;
    GBMParams g; PayoffVariant pv = European{100.0, 0.05, 1.0};
    const long long N = 1 << 19;
    const int steps = 64;
    auto timed = [&](NormalMethod m) {
        mc::cpu::CpuOptions o; o.normal = m; o.threads = 1;
        auto t0 = std::chrono::steady_clock::now();
        auto r = mc::cpu::run_mc_fixed(g, pv, steps, N, 1u, o);
        g_sink = r.first;
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    double best_bm = 1e300, best_zig = 1e300;
    timed(NormalMethod::Ziggurat);     // calentamiento
    for (int i = 0; i < reps; i++) {
        best_bm = std::min(best_bm, timed(NormalMethod::BoxMuller));
        best_zig = std::min(best_zig, timed(NormalMethod::Ziggurat));
    }
    const double per_step = 1e9 / ((double)N * steps);
    report("MC GBM europea 64 pasos 1 hilo (ns/paso)", {{"Box-Muller", best_bm * per_step}, {"Ziggurat", best_zig * per_step}});
}

// --- 1e. Desglose de un bloque de MC (ns por paso-camino): dónde se va el tiempo de W1/W2 ---------------------
//   ruido (antes): un generador por camino, normales a un búfer y transposición al layout de los núcleos
//   ruido (ahora): kLanes generadores intercalados escribiendo directamente (RngNoise)
//   núcleo: kernel_single sobre un bloque de ruido fijo
//   total: PathSim::run_chunk (ruido + núcleo + acumulación)
void bench_breakdown() {
    if (!selected("desglose")) return;
    struct Case { const char* name; PayoffVariant pv; int steps; };
    GBMParams gbm;
    for (const Case& c : {Case{"GBM europea 64 pasos", European{100.0, 0.05, 1.0}, 64},
                          Case{"GBM asiática 256 pasos", Asian{100.0}, 256}}) {
        const CpuModel m = make_cpu_model(gbm);
        const CpuPayoff pp = make_cpu_payoff(c.pv);
        const int D = c.steps, W = kLanes;
        const int blocks = (1 << 21) / (D * W);
        const double scale = std::sqrt(m.T / D);
        const double per = 1e9 / ((double)blocks * W * D);
        std::vector<double> Z((size_t)D * W), tmp((size_t)D);

        double t_old = best_of(reps, [&] {
            uint64_t next = 0; double acc = 0;
            for (int b = 0; b < blocks; b++) {
                for (int l = 0; l < W; l++) {
                    Xoshiro256pp g = Xoshiro256pp::for_path(1, Stream::Main, 0, next++);
                    fill_normals(NormalMethod::Ziggurat, g, tmp.data(), D);
                    for (int d = 0; d < D; d++) Z[(size_t)d * W + l] = tmp[(size_t)d] * scale;
                }
                acc += Z[(size_t)D * W - 1];
            }
            g_sink = acc;
        }) * per;

        RngNoise noise(1, Stream::Main, 0, D, scale, NormalMethod::Ziggurat);
        auto time_noise = [&](SimdLevel lv) {
            set_simd_level(lv);
            const double t = best_of(reps, [&] {
                auto st = noise.open(0, (uint64_t)blocks * W);
                double acc = 0;
                for (int b = 0; b < blocks; b++) { st->fill(W, Z.data(), W); acc += Z[(size_t)D * W - 1]; }
                g_sink = acc;
            }) * per;
            set_simd_level(simd_level_available());
            return t;
        };
        const double t_scalar = time_noise(SimdLevel::Scalar);
        const double t_avx2 = time_noise(SimdLevel::Avx2);

        const KCtx kc = make_kctx(m, pp, D);
        SingleFn fn = select_single_kernel(m.kind, pp.kind);
        double Y[kLanes];
        double t_ker = best_of(reps, [&] {
            double acc = 0;
            for (int b = 0; b < blocks; b++) { fn(kc, Z.data(), W, Y); acc += Y[0]; }
            g_sink = acc;
        }) * per;

        PathSim sim(m, pp, D, noise);
        Scratch sc;
        double t_tot = best_of(reps, [&] {
            ChunkAcc out;
            const int cp = sim.chunk_paths();
            for (long long p0 = 0; p0 < (long long)blocks * W; p0 += cp)
                sim.run_chunk((uint64_t)p0, (int)std::min<long long>(cp, (long long)blocks * W - p0), sc, out);
            g_sink = out.acc.to_moments().mean;
        }) * per;

        report(std::string("desglose ") + c.name + " (ns/paso)",
               {{"ruido: 1 generador por camino + transposición", t_old},
                {"ruido por bloque, escalar", t_scalar},
                {simd_level_available() == SimdLevel::Avx2 ? "ruido por bloque, AVX2" : "ruido por bloque, AVX2 (no disponible)", t_avx2},
                {"núcleo (Euler + payoff)", t_ker},
                {"total run_chunk", t_tot}});
    }
}

// --- 2. Sobol scrambleado: versión ingenua de la GPU frente a Gray-code con V' = L·V ------------------------
void bench_sobol() {
    if (!selected("sobol")) return;
    const int D = 64, N = 1 << 14;
    const uint32_t salt = 7;
    // (a) solo palabras de 32 bits (sin inversa normal): aísla el coste del scrambling
    double naive_words = best_of(reps, [&] {
        uint32_t acc = 0;
        for (int n = 0; n < N; n++)
            for (int d = 0; d < D; d++) acc ^= scrambled_sobol_naive(salt, d, (uint64_t)n);
        g_isink = acc;
    }) / ((double)N * D) * 1e9;
    ScrambledSobol sob(salt, D);
    double fast_words = best_of(reps, [&] {
        std::vector<uint32_t> x(D);
        sob.state_at(0, x.data());
        uint32_t acc = 0;
        for (int n = 0; n < N; n++) {
            if (n) sob.advance((uint64_t)n, x.data());
            for (int d = 0; d < D; d++) acc ^= x[d];
        }
        g_isink = acc;
    }) / ((double)N * D) * 1e9;
    report("sobol palabras (ns/punto·dim)", {{"ingenuo: 32 hashes + 32 popcounts", naive_words}, {"Gray-code, V' = L·V (1 XOR)", fast_words}});

    // (b) con la inversa de la normal incluida (lo que consume el motor)
    double naive_norm = best_of(reps, [&] {
        double acc = 0;
        for (int n = 0; n < N; n++)
            for (int d = 0; d < D; d++) acc += norm_inv_cdf(sobol_word_to_u(scrambled_sobol_naive(salt, d, (uint64_t)n)));
        g_sink = acc;
    }) / ((double)N * D) * 1e9;
    double fast_norm = best_of(reps, [&] {
        std::vector<double> z((size_t)D * 64);
        double acc = 0;
        for (int n0 = 0; n0 < N; n0 += 64) { sob.fill_normals((uint64_t)n0, 64, z.data()); acc += z[(size_t)D * 64 - 1]; }
        g_sink = acc;
    }) / ((double)N * D) * 1e9;
    report("sobol normales (ns/punto·dim)", {{"ingenuo + AS 241", naive_norm}, {"Gray-code + AS 241", fast_norm}});
}

// --- 3. Transformaciones BB y PCA por bloques (ns por camino) --------------------------------------------------
void bench_transforms() {
    if (!selected("transformaciones")) return;
    for (int m : {64, 256, 1024}) {
        const int paths = (m >= 1024) ? 256 : 1024;
        // PCA: GEMM ingenuo (un camino cada vez) frente al GEMM por bloques de qmc_noise.cpp
        const PCAData& pca = pca_compute(m, 1.0);
        std::vector<double> z((size_t)m), out((size_t)m);
        RngNoise rng(5, Stream::Test, 0, m, 1.0);
        double naive = best_of(reps, [&] {
            auto st = rng.open(0, (uint64_t)paths);
            double acc = 0;
            for (int p = 0; p < paths; p++) {
                st->fill(1, z.data(), 1);
                for (int i = 0; i < m; i++) {
                    double s = 0;
                    for (int k = 0; k < m; k++) s += pca.M_pca[(size_t)i + (size_t)k * m] * z[(size_t)k];   // acceso con zancada
                    out[(size_t)i] = s;
                }
                acc += out[(size_t)m - 1];
            }
            g_sink = acc;
        }) / paths * 1e9;
        PcaNoise pn(rng, pca);
        std::vector<double> blk((size_t)m * 8);
        double blocked = best_of(reps, [&] {
            auto st = pn.open(0, (uint64_t)paths);
            double acc = 0;
            for (int p = 0; p < paths; p += 8) { st->fill(8, blk.data(), 8); acc += blk[blk.size() - 1]; }
            g_sink = acc;
        }) / paths * 1e9;
        report("PCA m=" + std::to_string(m) + " (ns/camino)", {{"producto ingenuo por camino", naive}, {"GEMM por bloques (64 caminos)", blocked}});
    }
    // Brownian Bridge: bb_apply original (un camino, vector por llamada) frente al flujo por bloques
    for (int N : {64, 1024}) {
        const BBData& bb = bb_precompute(N, 1.0);
        const int paths = 4096;
        RngNoise rng(5, Stream::Test, 0, N, 1.0);
        std::vector<double> z((size_t)N * paths), dw((size_t)N * paths);
        {   // normales de entrada, fuera de la medición
            auto st = rng.open(0, (uint64_t)paths);
            std::vector<double> one((size_t)N);
            for (int p = 0; p < paths; p++) { st->fill(1, one.data(), 1); std::copy(one.begin(), one.end(), z.begin() + (long)p * N); }
        }
        double old_apply = best_of(reps, [&] {
            for (int p = 0; p < paths; p++) bb_apply(bb, z.data() + (size_t)p * N, dw.data() + (size_t)p * N, 1);   // asigna W por llamada
            g_sink = dw[(size_t)N * paths - 1];
        }) / paths * 1e9;
        double batched = best_of(reps, [&] {
            bb_apply(bb, z.data(), dw.data(), paths);                                                             // W reutilizado
            g_sink = dw[(size_t)N * paths - 1];
        }) / paths * 1e9;
        BrownianBridgeNoise bn(rng, bb);
        std::vector<double> blk((size_t)N * 8);
        double staged = best_of(reps, [&] {
            auto st = bn.open(0, (uint64_t)paths);
            double acc = 0;
            for (int p = 0; p < paths; p += 8) { st->fill(8, blk.data(), 8); acc += blk[blk.size() - 1]; }
            g_sink = acc;
        }) / paths * 1e9;
        report("BB N=" + std::to_string(N) + " (ns/camino)", {{"bb_apply, 1 llamada por camino", old_apply}, {"bb_apply con W reutilizado", batched},
                                                              {"flujo por bloques (incl. normales)", staged}});
    }
}

// --- 4. Utilidades de host: caché de BB/PCA -----------------------------------------------------------------
void bench_host() {
    if (!selected("host")) return;
    for (int m : {512, 2048}) {
        pca_compute(m, 1.0);   // calienta la caché
        // antes: pca_compute devolvía PCAData POR VALOR (copia de M_pca double + float)
        double by_value = best_of(3, [&] { PCAData copy = pca_compute(m, 1.0); g_sink = copy.M_pca[(size_t)m * m / 2]; }) * 1e3;
        double by_ref = best_of(reps, [&] { const PCAData& r = pca_compute(m, 1.0); g_sink = r.M_pca[(size_t)m * m / 2]; }) * 1e3;
        report("pca_compute caché m=" + std::to_string(m) + " (ms)", {{"por valor (copia, antes)", by_value}, {"por referencia (ahora)", by_ref}});

        // cálculo de la matriz: m^2 llamadas a sin() (antes) frente a tabla de 4m+2 senos (ahora)
        const double pi = std::numbers::pi;
        double direct = best_of(2, [&] {
            const double h = 1.0 / m, nrm = std::sqrt(4.0 / (2.0 * m + 1.0));
            double acc = 0;
            for (int k = 0; k < m; k++) {
                double sk = std::sin((2.0 * k + 1.0) * pi / (2.0 * (2.0 * m + 1.0)));
                double sq = std::sqrt(h) / (2.0 * std::abs(sk));
                double phase = (2.0 * k + 1.0) * pi / (2.0 * m + 1.0);
                for (int i = 0; i < m; i++) acc += nrm * std::sin(phase * (i + 1)) * sq;
            }
            g_sink = acc;
        }) * 1e3;
        const long long period = 4LL * m + 2;
        double table = best_of(3, [&] {
            std::vector<double> tab((size_t)period);
            for (long long j = 0; j < period; j++) tab[(size_t)j] = std::sin(pi * (double)j / (2.0 * m + 1.0));
            const double h = 1.0 / m, nrm = std::sqrt(4.0 / (2.0 * m + 1.0));
            double acc = 0;
            for (int k = 0; k < m; k++) {
                double sk = std::sin((2.0 * k + 1.0) * pi / (2.0 * (2.0 * m + 1.0)));
                double scale = nrm * std::sqrt(h) / (2.0 * std::abs(sk));
                long long step = (2LL * k + 1) % period, idx = step;
                for (int i = 0; i < m; i++) { acc += scale * tab[(size_t)idx]; idx += step; if (idx >= period) idx -= period; }
            }
            g_sink = acc;
        }) * 1e3;
        report("PCA matriz m=" + std::to_string(m) + " (ms)", {{"m^2 llamadas a sin() (antes)", direct}, {"tabla de senos (ahora)", table}});
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string json_path;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (a == "--filter") filter = next();
        else if (a == "--reps") reps = std::max(3, std::atoi(next().c_str()));
        else if (a == "--json") json_path = next();
        else { std::fprintf(stderr, "argumento desconocido: %s\n", a.c_str()); return 2; }
    }
    std::printf("bench_micro (un hilo, mejor de %d repeticiones)\n", reps);
    bench_normals();
    bench_e2e();
    bench_breakdown();
    bench_pow();
    bench_sobol();
    bench_transforms();
    bench_host();
    if (!json_path.empty()) {
        std::ofstream f(json_path);
        f << "[\n";
        for (size_t i = 0; i < rows.size(); i++)
            f << "  {\"group\": \"" << rows[i].group << "\", \"name\": \"" << rows[i].name << "\", \"value\": " << rows[i].ns
              << ", \"unit\": \"" << rows[i].unit << "\", \"speedup\": " << rows[i].speedup_vs_first << "}" << (i + 1 < rows.size() ? "," : "") << "\n";
        f << "]\n";
    }
    return 0;
}
