#include "utils.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <map>
#include <mutex>
#include <numbers>
#include "mc_format.hpp"

// Función de distribución acumulada de la normal estándar
static double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// ----------------------------------- //
// Precio analítico de Black-Scholes   //
// ----------------------------------- //

double bs_call(double S0, double K, double T, double r, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) return std::max(S0 - K * std::exp(-r * T), 0.0);
    double sqrtT = std::sqrt(T);
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    return S0 * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

// ---------------------------------------------------------------- //
// Fórmula analítica de la Asian geométrica bajo GBM sin descuento  //
// ---------------------------------------------------------------- //

double geom_asian_analytic(double S0, double K, double T, double mu,
                           double sigma, int n) {
    // m_G = E[log G_n] = log(S0) + (μ - σ²/2)·T·(n+1)/(2n)
    double m_G = std::log(S0) + (mu - 0.5 * sigma * sigma) * T * (n + 1) / (2.0 * n);

    // Var[log G_n] = σ²·T·(n+1)·(2n+1) / (6n²)
    double sig2_G = sigma * sigma * T * (n + 1) * (2 * n + 1) / (6.0 * n * n);
    double sig_G  = std::sqrt(sig2_G);

    if (sig_G < 1e-12) return std::max(std::exp(m_G) - K, 0.0);
    double d1 = (m_G + sig2_G - std::log(K)) / sig_G;
    double d2 = d1 - sig_G;
    return std::exp(m_G + 0.5 * sig2_G) * norm_cdf(d1) - K * norm_cdf(d2);
}


// ---------------------------------- //
// Precomputación del Brownian Bridge //
// ---------------------------------- //

// bb_precompute / pca_compute son deterministas en (N, T). Los ejemplos las
// llaman hasta 30x por nivel de eps; pca_compute a m grande (n_steps ~ 2048 en eps
// finos) cuesta tiempo y memoria (m^2 doubles). Memoizacion por (N, T): la 1a llamada
// calcula, el resto devuelve una referencia a la misma entrada. Antes se devolvia
// una COPIA por valor (~50 MB a m=2048) y el mapa no era seguro entre hilos; ahora el
// acceso va protegido por un mutex y las referencias son estables (std::map no
// invalida referencias a nodos al insertar).
static BBData bb_precompute_impl(int N, double T);
static PCAData pca_compute_impl(int m, double T);
static long long np_key(int n, double T) {
    return (long long)n * 1000003LL + (long long)std::llround(T * 1e6);
}

const BBData& bb_precompute(int N, double T) {
    static std::mutex mtx;
    static std::map<long long, BBData> memo;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = memo.find(np_key(N, T));
    if (it == memo.end()) {
        auto t0 = std::chrono::steady_clock::now();
        it = memo.emplace(np_key(N, T), bb_precompute_impl(N, T)).first;
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[precompute] bb_precompute N=%d T=%.5f -> %.3fs\n", N, T, dt);
    }
    return it->second;
}

const PCAData& pca_compute(int m, double T) {
    static std::mutex mtx;
    static std::map<long long, PCAData> memo;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = memo.find(np_key(m, T));
    if (it == memo.end()) {
        auto t0 = std::chrono::steady_clock::now();
        it = memo.emplace(np_key(m, T), pca_compute_impl(m, T)).first;
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[precompute] pca_compute m=%d T=%.5f -> %.3fs\n", m, T, dt);
    }
    return it->second;
}

static BBData bb_precompute_impl(int N, double T) {
    assert(N > 0 && (N & (N - 1)) == 0); // N debe ser potencia de 2

    BBData bb;
    bb.N = N;
    bb.T = T;
    bb.map_idx.resize(N, 0);
    bb.left_idx.resize(N, 0);
    bb.right_idx.resize(N, 0);
    bb.weight_left.resize(N, 0.0);
    bb.weight_right.resize(N, 0.0);
    bb.std_dev.resize(N, 0.0);

    std::vector<double> times(N + 1);
    for (int i = 0; i <= N; i++) times[i] = i * T / N;

    // Primer punto: extremo derecho W[N]
    bb.map_idx[0] = N;
    bb.std_dev[0] = std::sqrt(T);

    int actual_step = 1;
    int n_levels = 0;
    for (int tmp = N; tmp > 1; tmp >>= 1) ++n_levels;

    for (int level = 0; level < n_levels; level++) {
        int num_points  = 1 << level;
        int stride      = N >> level;
        int half_stride = stride >> 1;

        for (int j = 0; j < num_points; j++) {
            int L = j * stride;
            int R = (j + 1) * stride;
            int M = L + half_stride;

            bb.map_idx[actual_step]    = M;
            bb.left_idx[actual_step]   = L;
            bb.right_idx[actual_step]  = R;

            double tL = times[L], tR = times[R], tM = times[M];
            bb.weight_left[actual_step]  = (tR - tM) / (tR - tL);
            bb.weight_right[actual_step] = (tM - tL) / (tR - tL);
            bb.std_dev[actual_step]      = std::sqrt((tM - tL) * (tR - tM) / (tR - tL));

            actual_step++;
        }
    }
    return bb;
}

// Transforma Z_(n_sim×N) en incrementos brownianos dW_(n_sim×N)
void bb_apply(const BBData& bb, double* Z, double* dW, int n_sim) {
    const int N = bb.N;
    std::vector<double> W(N + 1, 0.0); // reutilizado entre simulaciones (antes: 1 alloc por sim)
    for (long long sim = 0; sim < n_sim; sim++) {
        double* z   = Z   + sim * (long long)N;   // long long: sim*N desbordaba int
        double* out = dW  + sim * (long long)N;

        W[0] = 0.0;
        W[bb.map_idx[0]] = bb.std_dev[0] * z[0];

        for (int step = 1; step < N; step++) {
            int    m  = bb.map_idx[step];
            int    l  = bb.left_idx[step];
            int    r  = bb.right_idx[step];
            double wl = bb.weight_left[step];
            double wr = bb.weight_right[step];
            double sd = bb.std_dev[step];
            W[m] = wl * W[l] + wr * W[r] + sd * z[step];
        }
        for (int k = 0; k < N; k++) out[k] = W[k + 1] - W[k];
    }
}


// ------------------------------------- //
// PCA (forma cerrada, sin dependencias) //
// ------------------------------------- //

// Descomposicion espectral CERRADA de la covarianza del MB C[i,j]=min(i+1,j+1)*h.
// C = h*A con A_{pq}=min(p,q) (p,q=1..m). A tiene inversa tridiagonal (2 en la
// diagonal salvo la ultima entrada, que es 1) => autovectores seno y autovalores
// analiticos (KL discreta del puente browniano):
//   lambda_k = h / (4 sin^2( (2k+1)pi / (2(2m+1)) ) ),  k=0..m-1  (k=0 el mayor)
//   v_k(i)   = sqrt(4/(2m+1)) * sin( (2k+1)(i+1)pi / (2m+1) )
// Cuesta O(m^2) en vez del O(m^3) de un eigensolver generico (300 s a m=2048).
// Devuelve Mc en column-major: Mc[i + k*m] = v_k(i) * sqrt(lambda_k).
//
// Los senos se leen de una tabla sin(pi*j/(2m+1)), j = 0..4m+1, indexada por
// (2k+1)(i+1) mod (4m+2): m^2 llamadas a sin() -> 4m+2. Ademas es mas exacta que
// evaluar sin() con argumentos grandes.
static std::vector<double> pca_M_pca_cum_closed_form(int m, double T) {
    const double pi  = std::numbers::pi;
    const double h   = T / m;
    const double den = 2.0 * (2.0 * m + 1.0);
    const double nrm = std::sqrt(4.0 / (2.0 * m + 1.0));
    const long long period = 4LL * m + 2;
    std::vector<double> sin_tab(period);
    for (long long j = 0; j < period; j++) sin_tab[j] = std::sin(pi * (double)j / (2.0 * m + 1.0));

    std::vector<double> Mc((size_t)m * m);
    for (int k = 0; k < m; k++) {
        double sk = std::sin((2.0 * k + 1.0) * pi / den);
        double sqrt_lambda = std::sqrt(h) / (2.0 * std::abs(sk));
        double scale = nrm * sqrt_lambda;
        long long step = (2LL * k + 1) % period;
        long long idx = step; // (2k+1)*(i+1) mod period, incremental en i
        double* col = Mc.data() + (size_t)k * m;
        for (int i = 0; i < m; i++) {
            col[i] = scale * sin_tab[idx];
            idx += step; if (idx >= period) idx -= period;
        }
    }
    return Mc;
}

static PCAData pca_compute_impl(int m, double T) {
    // Auto-chequeo unico: la forma cerrada debe reconstruir C a maquina.
    static std::once_flag checked;
    std::call_once(checked, [] {
        const int mc = 137; const double Tc = 1.0, hc = Tc / mc;
        std::vector<double> Mc = pca_M_pca_cum_closed_form(mc, Tc);
        double num = 0.0, den = 0.0;
        for (int i = 0; i < mc; i++)
            for (int j = 0; j < mc; j++) {
                double acc = 0.0;
                for (int k = 0; k < mc; k++) acc += Mc[(size_t)k * mc + i] * Mc[(size_t)k * mc + j];
                double c = std::min(i + 1, j + 1) * hc;
                num += (acc - c) * (acc - c);
                den += c * c;
            }
        double rel = std::sqrt(num) / std::sqrt(den);
        if (!(rel < 1e-9)) {
            std::fprintf(stderr, "[pca] forma cerrada FALLA el auto-chequeo (rel=%.3e)\n", rel);
            std::abort();
        }
    });

    PCAData pca;
    pca.m = m;
    pca.T = T;

    std::vector<double> Mc = pca_M_pca_cum_closed_form(m, T);

    // Operador diferencia por filas: M_pca[0,:] = Mc[0,:]; M_pca[i,:] = Mc[i,:] - Mc[i-1,:]
    // (column-major: el elemento (i,k) esta en i + k*m).
    pca.M_pca.resize((size_t)m * m);
    for (int k = 0; k < m; k++) {
        const double* src = Mc.data() + (size_t)k * m;
        double* dst = pca.M_pca.data() + (size_t)k * m;
        dst[0] = src[0];
        for (int i = 1; i < m; i++) dst[i] = src[i] - src[i - 1];
    }

    pca.M_pca_f32.resize((size_t)m * m);
    for (size_t i = 0; i < (size_t)m * m; i++)
        pca.M_pca_f32[i] = static_cast<float>(pca.M_pca[i]);

    return pca;
}


// ------------------------------- //
// Estimación de c1 por Richardson //
// ------------------------------- //

double estimar_c1_richardson(SimFn sim_fn, double T, int M_rich, int N_pilot, unsigned seed) {
    int n_fine   = std::max(4 * M_rich, 4);
    int n_coarse = n_fine / M_rich;

    double P_fine   = sim_fn(n_fine,   N_pilot, seed);
    double P_coarse = sim_fn(n_coarse, N_pilot, seed + 1);

    double h_fine  = T / n_fine;
    double c_bias  = std::abs(P_coarse - P_fine) / (h_fine * (M_rich - 1));
    if (c_bias < 1e-12) c_bias = 1e-12;
    return 1.0 / c_bias;
}


// ------------------- //
// Tabla de resultados //
// ------------------- //

void print_table(const std::vector<TableRow>& rows, double price_ref,
                 double epsilon, const std::string& example_name) {

    std::puts(mc_sprintf("\n[%s]   epsilon = %s", example_name.c_str(), mc_shortest(epsilon).c_str()).c_str());
    std::puts(mc_sprintf("  Referencia: %.6f\n", price_ref).c_str());

    // Columna "R": repeticiones efectivas con semilla distinta (TAREA 2). Con
    // R>1, StdErr es la desviación típica ENTRE esas R repeticiones dividida
    // por sqrt(R) (ver RunningStats::std_error en utils.hpp), no el error
    // estándar intra-corrida; T(s) es la SUMA de las R corridas (coste real).
    std::puts(mc_sprintf("  %-22s%10s%10s%12s%9s%10s%6s%5s",
                 "Metodo", "Precio", "StdErr", "N", "T(s)", "|Error|", "OK?", "R").c_str());
    std::puts(mc_sprintf("  %s", std::string(84, '-').c_str()).c_str());

    for (const auto& r : rows) {
        double err = std::abs(r.price - price_ref);
        bool   ok  = (err < 2.0 * epsilon);
        std::puts(mc_sprintf("  %-22s%10.4f%10.4f%12lld%9.3f%10.4f%6s%5d",
                     r.method.c_str(), r.price, r.std_error, r.n_samples,
                     r.time_s, err, ok ? "SI" : "NO", r.n_reps).c_str());
    }
    std::puts("");
}
