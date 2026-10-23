// Movimiento browniano fraccionario (Davies-Harte y Cholesky) y modelos de
// Black-Scholes fraccionario y mixto.

#include "fraccional.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace frac {

double autocov_fgn(double H, long k) {
    const double a = std::fabs((double)k), h2 = 2.0 * H;
    return 0.5 * (std::pow(a + 1.0, h2) - 2.0 * std::pow(a, h2) + std::pow(std::fabs(a - 1.0), h2));
}

GeneradorFBM::GeneradorFBM(double H, int n, double T, Metodo m) : H_(H), T_(T), n_(n), metodo_(m) {
    if (!(H > 0.0 && H < 1.0)) throw std::invalid_argument("fBm: hace falta 0 < H < 1");
    if (n < 1) throw std::invalid_argument("fBm: hace falta al menos un paso");
    if (metodo_ == Metodo::DaviesHarte) {
        size_t N = 1;
        while (N < (size_t)n) N <<= 1;
        const size_t M = 2 * N;
        std::vector<cd> c(M);
        for (size_t k = 0; k <= N; ++k) c[k] = autocov_fgn(H, (long)k);
        for (size_t k = 1; k < N; ++k) c[M - k] = c[k];
        fft(c, false);
        double lmax = 0.0, lmin = 0.0;
        for (const auto& v : c) {
            lmax = std::max(lmax, v.real());
            lmin = std::min(lmin, v.real());
        }
        if (lmin < -1e-10 * lmax) {
            metodo_ = Metodo::Cholesky; // la inmersión no es definida positiva (no ocurre con el ruido fraccionario)
        } else {
            raiz_.resize(M);
            for (size_t k = 0; k < M; ++k) {
                const double l = std::max(c[k].real(), 0.0);
                raiz_[k] = (k == 0 || k == N) ? std::sqrt(l / M) : std::sqrt(l / (2.0 * M));
            }
        }
    }
    if (metodo_ == Metodo::Cholesky) {
        const size_t N = (size_t)n;
        L_.assign(N * N, 0.0);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double s = autocov_fgn(H, (long)i - (long)j);
                for (size_t k = 0; k < j; ++k) s -= L_[i * N + k] * L_[j * N + k];
                if (i == j) {
                    if (s <= 0.0) throw std::runtime_error("fBm: la covarianza no es definida positiva");
                    L_[i * N + i] = std::sqrt(s);
                } else {
                    L_[i * N + j] = s / L_[j * N + j];
                }
            }
        }
    }
}

void GeneradorFBM::generar(std::mt19937_64& gen, std::vector<double>& B) const {
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> X((size_t)n_);
    if (metodo_ == Metodo::DaviesHarte) {
        const size_t M = raiz_.size(), N = M / 2;
        std::vector<cd> V(M);
        V[0] = raiz_[0] * nd(gen);
        V[N] = raiz_[N] * nd(gen);
        for (size_t k = 1; k < N; ++k) {
            const double a = nd(gen), b = nd(gen);
            V[k] = raiz_[k] * cd(a, b);
            V[M - k] = std::conj(V[k]);
        }
        fft(V, false);
        for (int j = 0; j < n_; ++j) X[(size_t)j] = V[(size_t)j].real();
    } else {
        const size_t N = (size_t)n_;
        std::vector<double> z(N);
        for (auto& v : z) v = nd(gen);
        for (size_t i = 0; i < N; ++i) {
            double s = 0.0;
            for (size_t k = 0; k <= i; ++k) s += L_[i * N + k] * z[k];
            X[i] = s;
        }
    }
    // B^H en la rejilla: suma de incrementos (cada uno con varianza (T/n)^{2H})
    const double esc = std::pow(T_ / n_, H_);
    B.assign((size_t)n_ + 1, 0.0);
    for (int k = 0; k < n_; ++k) B[(size_t)k + 1] = B[(size_t)k] + esc * X[(size_t)k];
}

// ===================================================================== //
// Black-Scholes fraccionario y mixto                                     //
// ===================================================================== //

double FBS::varianza(double t) const {
    return sigma * sigma * t + sigma_h * sigma_h * std::pow(t, 2.0 * H);
}

double fbs_precio(const FBS& p, bool call, double S0, double K, double T, double r, double q) {
    const double F = S0 * std::exp((r - q) * T), DF = std::exp(-r * T);
    return black(call, F, K, DF, p.varianza(T));
}

namespace {
// Covarianza de ln S en t y s: σ² min(t,s) + σ_H²·½(t^{2H} + s^{2H} - |t-s|^{2H})
double cov_log(const FBS& p, double t, double s) {
    const double h2 = 2.0 * p.H;
    return p.sigma * p.sigma * std::min(t, s) +
           p.sigma_h * p.sigma_h * 0.5 * (std::pow(t, h2) + std::pow(s, h2) - std::pow(std::fabs(t - s), h2));
}
} // namespace

double fbs_asiatica_geometrica(const FBS& p, bool call, double S0, double K, double T, double r, double q,
                               int n_obs) {
    if (n_obs < 1) throw std::invalid_argument("asiática: hace falta al menos una fecha");
    double m = std::log(S0), s2 = 0.0;
    for (int i = 1; i <= n_obs; ++i) {
        const double ti = T * i / n_obs;
        m += ((r - q) * ti - 0.5 * p.varianza(ti)) / n_obs;
        for (int j = 1; j <= n_obs; ++j) s2 += cov_log(p, ti, T * j / n_obs);
    }
    s2 /= (double)n_obs * n_obs;
    // ln G ~ N(m, s²): E[(G - K)⁺] = Black con forward e^{m + s²/2}
    return black(call, std::exp(m + 0.5 * s2), K, std::exp(-r * T), s2);
}

ResultadoAsiaticaMC fbs_asiatica_mc(const FBS& p, bool call, double S0, double K, double T, double r, double q,
                                    int n_obs, long long n, uint64_t semilla, GeneradorFBM::Metodo met) {
    GeneradorFBM gen_h(p.H, n_obs, T, met);
    std::mt19937_64 gen(semilla);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> BH;
    const double dt = T / n_obs, DF = std::exp(-r * T);
    std::vector<double> deriva((size_t)n_obs + 1);
    for (int i = 1; i <= n_obs; ++i) {
        const double ti = T * i / n_obs;
        deriva[(size_t)i] = std::log(S0) + (r - q) * ti - 0.5 * p.varianza(ti);
    }
    const double geo_exacta = fbs_asiatica_geometrica(p, call, S0, K, T, r, q, n_obs);
    double sg = 0, sg2 = 0, sa = 0, sa2 = 0, sga = 0;
    for (long long it = 0; it < n; ++it) {
        gen_h.generar(gen, BH);
        double W = 0.0, suma_log = 0.0, suma = 0.0;
        for (int i = 1; i <= n_obs; ++i) {
            W += std::sqrt(dt) * nd(gen);
            const double lx = deriva[(size_t)i] + p.sigma * W + p.sigma_h * BH[(size_t)i];
            suma_log += lx;
            suma += std::exp(lx);
        }
        const double G = std::exp(suma_log / n_obs), A = suma / n_obs;
        const double pg = DF * std::max(call ? G - K : K - G, 0.0);
        const double pa = DF * std::max(call ? A - K : K - A, 0.0);
        sg += pg; sg2 += pg * pg; sa += pa; sa2 += pa * pa; sga += pg * pa;
    }
    const double N = (double)n;
    ResultadoAsiaticaMC R;
    const double mg = sg / N, ma = sa / N;
    const double vg = std::max(sg2 / N - mg * mg, 0.0), va = std::max(sa2 / N - ma * ma, 0.0);
    const double cga = sga / N - mg * ma;
    R.geometrica = {mg, std::sqrt(vg / N)};
    // Variable de control: la geométrica (con su precio exacto)
    const double b = vg > 0 ? cga / vg : 0.0;
    const double vcv = std::max(va - 2 * b * cga + b * b * vg, 0.0);
    R.aritmetica = {ma - b * (mg - geo_exacta), std::sqrt(vcv / N)};
    return R;
}

} // namespace frac
