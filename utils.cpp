#include "utils.hpp"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cassert>
#include <numbers>
#include <print>
#include <Eigen/Eigenvalues>

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

// ----------- //
// Call Heston //
// ----------- //

// Precio de una call en Heston por transformada de Fourier (cuadratura con N=512 puntos).
// Albrecher, Mayer, Schachermayer & Teichmann (2007)

double heston_call_cf(double S0, double K, double T, double r,
                      double kappa, double theta, double xi,
                      double rho, double v0) {
    using cd = std::complex<double>;
    const double pi = std::numbers::pi;

    // Función característica de log(S_T) bajo la medida Q_j
    auto char_fn = [&](cd u, int j) -> cd {
        cd i(0, 1);
        double b  = (j == 1) ? kappa - rho * xi : kappa;
        cd sigma2 = xi * xi;
        cd d = std::sqrt((rho * xi * i * u - b) * (rho * xi * i * u - b)
                         + sigma2 * (i * u + u * u));
        cd g = (b - rho * xi * i * u + d) / (b - rho * xi * i * u - d);
        cd C = r * i * u * T
               + (kappa * theta / sigma2)
               * ((b - rho * xi * i * u + d) * T
                  - 2.0 * std::log((1.0 - g * std::exp(d * T)) / (1.0 - g)));
        cd D = (b - rho * xi * i * u + d) / sigma2
               * (1.0 - std::exp(d * T)) / (1.0 - g * std::exp(d * T));
        return std::exp(C + D * v0 + i * u * std::log(S0));
    };

    // P_j = 0.5 + (1/π) · Re[ ∫₀^∞ e^{-iu·log(K)} · φ_j(u) / (iu) du ]
    const int    N  = 512;
    const double du = 0.05;
    auto integral = [&](int j) {
        double sum = 0.0;
        for (int k = 1; k <= N; k++) {
            double u = (k - 0.5) * du;
            cd iu(0, u);
            cd phi = char_fn(u - (j == 1 ? cd(0, 1) : 0.0), j);
            sum += (std::exp(-iu * std::log(K)) * phi / iu).real() * du;
        }
        return 0.5 + sum / pi;
    };

    double P1 = integral(1);
    double P2 = integral(2);
    return S0 * P1 - K * std::exp(-r * T) * P2;
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

BBData bb_precompute(int N, double T) {
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
    for (int sim = 0; sim < n_sim; sim++) {
        double* z   = Z   + sim * N;
        double* out = dW  + sim * N;

        std::vector<double> W(N + 1, 0.0);
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


// -------------------- //
// PCA mediante Eigen   //
// -------------------- //

PCAData pca_compute(int m, double T) {
    PCAData pca;
    pca.m = m;
    pca.T = T;

    double h = T / m;

    // Matriz de covarianza del BM: C[i,j] = min(i+1, j+1) * h
    Eigen::MatrixXd C(m, m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            C(i, j) = std::min(i + 1, j + 1) * h;

    // Eigendescomposición simétrica
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(C);

    // Ordenar de mayor a menor eigenvalor
    Eigen::VectorXd eigenvalues  = solver.eigenvalues().reverse();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors().rowwise().reverse();

    // M_pca_cum[i,k] = sqrt(λ_k) · v_k[i]
    Eigen::MatrixXd M_pca_cum = eigenvectors *
                                 eigenvalues.cwiseMax(0.0).cwiseSqrt().asDiagonal();

    // Operador diferencia: M_pca[0,:] = M_pca_cum[0,:]; M_pca[i,:] = M_pca_cum[i,:] - M_pca_cum[i-1,:]
    Eigen::MatrixXd M_pca = M_pca_cum;
    M_pca.bottomRows(m - 1) -= M_pca_cum.topRows(m - 1);

    pca.M_pca.resize(m * m);
    Eigen::Map<Eigen::MatrixXd>(pca.M_pca.data(), m, m) = M_pca;

    pca.M_pca_f32.resize(m * m);
    for (int i = 0; i < m * m; i++)
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

    std::println("\n[{}]   epsilon = {}", example_name, epsilon);
    std::println("  Referencia: {:.6f}\n", price_ref);

    std::println("  {:<22}{:>10}{:>10}{:>12}{:>9}{:>10}{:>6}",
                 "Metodo", "Precio", "StdErr", "N", "T(s)", "|Error|", "OK?");
    std::println("  {}", std::string(79, '-'));

    for (const auto& r : rows) {
        double err = std::abs(r.price - price_ref);
        bool   ok  = (err < 2.0 * epsilon);
        std::println("  {:<22}{:>10.4f}{:>10.4f}{:>12}{:>9.3f}{:>10.4f}{:>6}",
                     r.method, r.price, r.std_error, r.n_samples,
                     r.time_s, err, ok ? "SI" : "NO");
    }
    std::println("");
}
