#include "referencias.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ref {

namespace {
constexpr double PI = 3.14159265358979323846;
using cd = std::complex<double>;
const cd I(0.0, 1.0);

// Nodos y pesos de Gauss-Legendre de n puntos en [-1, 1] (Newton sobre P_n)
struct GaussLegendre {
    std::vector<double> x, w;
    explicit GaussLegendre(int n) : x((size_t)n), w((size_t)n) {
        for (int i = 0; i < n; ++i) {
            double z = std::cos(PI * (i + 0.75) / (n + 0.5)), dp = 0.0;
            for (int it = 0; it < 100; ++it) {
                double p0 = 1.0, p1 = 0.0;
                for (int j = 1; j <= n; ++j) {
                    double p2 = p1;
                    p1 = p0;
                    p0 = ((2.0 * j - 1.0) * z * p1 - (j - 1.0) * p2) / j;
                }
                dp = n * (z * p0 - p1) / (z * z - 1.0);
                double z1 = z;
                z = z1 - p0 / dp;
                if (std::fabs(z - z1) < 1e-15) break;
            }
            x[(size_t)i] = z;
            w[(size_t)i] = 2.0 / ((1.0 - z * z) * dp * dp);
        }
    }
};

// Integral de f en [a, b] con Gauss-Legendre de 48 puntos en m subintervalos
template<class F>
double gauss_legendre(const F& f, double a, double b, int m = 4) {
    static const GaussLegendre gl(48);
    double h = (b - a) / m, total = 0.0;
    for (int k = 0; k < m; ++k) {
        double c = a + (k + 0.5) * h, r = 0.5 * h, s = 0.0;
        for (size_t i = 0; i < gl.x.size(); ++i) s += gl.w[i] * f(c + r * gl.x[i]);
        total += r * s;
    }
    return total;
}
} // namespace

double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
double norm_pdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * PI); }

// Algoritmo BVNU de A. Genz (2004): P(X > h, Y > k)
static double bvnu(double h, double k, double r) {
    static const double X[3][10] = {
        {-0.9324695142031522, -0.6612093864662647, -0.2386191860831970},
        {-0.9815606342467191, -0.9041172563704750, -0.7699026741943050, -0.5873179542866171,
         -0.3678314989981802, -0.1252334085114692},
        {-0.9931285991850949, -0.9639719272779138, -0.9122344282513259, -0.8391169718222188,
         -0.7463319064601508, -0.6360536807265150, -0.5108670019508271, -0.3737060887154196,
         -0.2277858511416451, -0.07652652113349733}};
    static const double W[3][10] = {
        {0.1713244923791705, 0.3607615730481384, 0.4679139345726904},
        {0.04717533638651177, 0.1069393259953183, 0.1600783285433464, 0.2031674267230659,
         0.2334925365383547, 0.2491470458134029},
        {0.01761400713915212, 0.04060142980038694, 0.06267204833410906, 0.08327674157670475,
         0.1019301198172404, 0.1181945319615184, 0.1316886384491766, 0.1420961093183821,
         0.1491729864726037, 0.1527533871307259}};
    int ng, lg;
    if (std::fabs(r) < 0.3) { ng = 0; lg = 3; }
    else if (std::fabs(r) < 0.75) { ng = 1; lg = 6; }
    else { ng = 2; lg = 10; }
    double hk = h * k, bvn = 0.0;
    if (std::fabs(r) < 0.925) {
        double hs = (h * h + k * k) / 2.0, asr = std::asin(r);
        for (int i = 0; i < lg; ++i) {
            double sn = std::sin(asr * (X[ng][i] + 1.0) / 2.0);
            bvn += W[ng][i] * std::exp((sn * hk - hs) / (1.0 - sn * sn));
            sn = std::sin(asr * (-X[ng][i] + 1.0) / 2.0);
            bvn += W[ng][i] * std::exp((sn * hk - hs) / (1.0 - sn * sn));
        }
        bvn = bvn * asr / (4.0 * PI) + norm_cdf(-h) * norm_cdf(-k);
    } else {
        if (r < 0) { k = -k; hk = -hk; }
        if (std::fabs(r) < 1.0) {
            double as = (1.0 - r) * (1.0 + r), a = std::sqrt(as);
            double bs = (h - k) * (h - k);
            double c = (4.0 - hk) / 8.0, d = (12.0 - hk) / 16.0;
            bvn = a * std::exp(-(bs / as + hk) / 2.0) *
                  (1.0 - c * (bs - as) * (1.0 - d * bs / 5.0) / 3.0 + c * d * as * as / 5.0);
            if (hk > -160.0) {
                double b = std::sqrt(bs);
                bvn -= std::exp(-hk / 2.0) * std::sqrt(2.0 * PI) * norm_cdf(-b / a) * b *
                       (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0);
            }
            a /= 2.0;
            for (int i = 0; i < lg; ++i) {
                double xs = (a * (X[ng][i] + 1.0)) * (a * (X[ng][i] + 1.0));
                double rs = std::sqrt(1.0 - xs);
                bvn += a * W[ng][i] *
                       (std::exp(-bs / (2.0 * xs) - hk / (1.0 + rs)) / rs -
                        std::exp(-(bs / xs + hk) / 2.0) * (1.0 + c * xs * (1.0 + d * xs)));
                xs = as * (-X[ng][i] + 1.0) * (-X[ng][i] + 1.0) / 4.0;
                rs = std::sqrt(1.0 - xs);
                bvn += a * W[ng][i] * std::exp(-(bs / xs + hk) / 2.0) *
                       (std::exp(-hk * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs - (1.0 + c * xs * (1.0 + d * xs)));
            }
            bvn = -bvn / (2.0 * PI);
        }
        if (r > 0) bvn += norm_cdf(-std::max(h, k));
        if (r < 0) bvn = -bvn + std::max(0.0, norm_cdf(-h) - norm_cdf(-k));
    }
    return bvn;
}

double normal_bivariante(double a, double b, double rho) { return bvnu(-a, -b, rho); }

double bs_precio(bool call, double S, double K, double T, double r, double q, double sigma) {
    double dfr = std::exp(-r * T), dfq = std::exp(-q * T);
    if (T <= 0.0 || sigma <= 0.0) {
        double fwd = S * dfq - K * dfr;
        return call ? std::max(fwd, 0.0) : std::max(-fwd, 0.0);
    }
    double sv = sigma * std::sqrt(T);
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sv, d2 = d1 - sv;
    if (call) return S * dfq * norm_cdf(d1) - K * dfr * norm_cdf(d2);
    return K * dfr * norm_cdf(-d2) - S * dfq * norm_cdf(-d1);
}

double bs_digital(bool call, double S, double K, double T, double r, double q, double sigma) {
    double sv = sigma * std::sqrt(T);
    double d2 = (std::log(S / K) + (r - q - 0.5 * sigma * sigma) * T) / sv;
    return std::exp(-r * T) * norm_cdf(call ? d2 : -d2);
}

namespace {
// Términos A, B, C, D de Reiner-Rubinstein (Haug, "The Complete Guide...", 4.17.1)
struct TerminosRR { double A, B, C, D; };
TerminosRR terminos_rr(double S, double X, double H, double T, double r, double q, double v, double phi, double eta) {
    double b = r - q, sv = v * std::sqrt(T);
    double mu = (b - 0.5 * v * v) / (v * v);
    double x1 = std::log(S / X) / sv + (1.0 + mu) * sv;
    double x2 = std::log(S / H) / sv + (1.0 + mu) * sv;
    double y1 = std::log(H * H / (S * X)) / sv + (1.0 + mu) * sv;
    double y2 = std::log(H / S) / sv + (1.0 + mu) * sv;
    double ebr = std::exp((b - r) * T), er = std::exp(-r * T);
    double hs1 = std::pow(H / S, 2.0 * (mu + 1.0)), hs = std::pow(H / S, 2.0 * mu);
    TerminosRR t;
    t.A = phi * S * ebr * norm_cdf(phi * x1) - phi * X * er * norm_cdf(phi * x1 - phi * sv);
    t.B = phi * S * ebr * norm_cdf(phi * x2) - phi * X * er * norm_cdf(phi * x2 - phi * sv);
    t.C = phi * S * ebr * hs1 * norm_cdf(eta * y1) - phi * X * er * hs * norm_cdf(eta * y1 - eta * sv);
    t.D = phi * S * ebr * hs1 * norm_cdf(eta * y2) - phi * X * er * hs * norm_cdf(eta * y2 - eta * sv);
    return t;
}
} // namespace

double bs_put_down_in(double S, double K, double H, double T, double r, double q, double sigma) {
    if (S <= H) return bs_precio(false, S, K, T, r, q, sigma);
    TerminosRR t = terminos_rr(S, K, H, T, r, q, sigma, -1.0, 1.0);
    return (K > H) ? (t.B - t.C + t.D) : t.A;
}

double bs_call_up_out(double S, double K, double H, double T, double r, double q, double sigma) {
    if (S >= H || K >= H) return 0.0;
    TerminosRR t = terminos_rr(S, K, H, T, r, q, sigma, 1.0, -1.0);
    return t.A - t.B + t.C - t.D;
}

double merton_precio(bool call, double S, double K, double T, double r, double q, double sigma,
                     double lambda, double mu_j, double sigma_j) {
    // Condicionando a N = n saltos (N ~ Poisson(λT)): ln S_T es normal con
    // forward F_n = S·e^{(r-q-λk̄)T}·(1+k̄)^n y varianza σ²T + nδ²; el precio
    // condicionado es Black-Scholes con un dividendo equivalente q_n.
    double kbar = std::exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0;
    double lt = lambda * T;
    double suma = 0.0, peso = std::exp(-lt);
    for (int n = 0; n < 400; ++n) {
        if (n > 0) peso *= lt / n;
        double sn = std::sqrt(sigma * sigma + n * sigma_j * sigma_j / T);
        double qn = q + lambda * kbar - n * std::log1p(kbar) / T;
        suma += peso * bs_precio(call, S, K, T, r, qn, sn);
        if (n > lt && peso < 1e-18) break;
    }
    return suma;
}

FuncCar fc_black_scholes(double sigma, double T) {
    return [=](cd u) { return std::exp(-0.5 * sigma * sigma * T * (I * u + u * u)); };
}

FuncCar fc_merton(double sigma, double lambda, double mu_j, double sigma_j, double T) {
    double kbar = std::exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0;
    return [=](cd u) {
        cd ex = -0.5 * sigma * sigma * (I * u + u * u) +
                lambda * (std::exp(I * u * mu_j - 0.5 * sigma_j * sigma_j * u * u) - 1.0) - I * u * lambda * kbar;
        return std::exp(T * ex);
    };
}

FuncCar fc_heston(double v0, double kappa, double theta, double xi, double rho, double T) {
    return [=](cd u) {
        cd iu = I * u;
        cd b = kappa - rho * xi * iu;
        cd d = std::sqrt(b * b + xi * xi * (iu + u * u));
        cd g = (b - d) / (b + d);
        cd e = std::exp(-d * T);
        cd C = kappa * theta / (xi * xi) * ((b - d) * T - 2.0 * std::log((1.0 - g * e) / (1.0 - g)));
        cd D = (b - d) / (xi * xi) * (1.0 - e) / (1.0 - g * e);
        return std::exp(C + D * v0);
    };
}

FuncCar fc_bates(double v0, double kappa, double theta, double xi, double rho,
                 double lambda, double mu_j, double sigma_j, double T) {
    FuncCar h = fc_heston(v0, kappa, theta, xi, rho, T);
    double kbar = std::exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0;
    return [=](cd u) {
        cd salto = lambda * (std::exp(I * u * mu_j - 0.5 * sigma_j * sigma_j * u * u) - 1.0) - I * u * lambda * kbar;
        return h(u) * std::exp(T * salto);
    };
}

FuncCar fc_nig(double sigma, double theta, double kappa, double T) {
    auto psi = [=](cd u) {
        return (1.0 - std::sqrt(1.0 + u * u * sigma * sigma * kappa - 2.0 * I * theta * u * kappa)) / kappa;
    };
    cd psi_mi = psi(cd(0.0, -1.0));
    return [=](cd u) { return std::exp(T * (psi(u) - I * u * psi_mi)); };
}

double precio_fourier(bool call, const FuncCar& phi, double S, double K, double T, double r, double q) {
    const double k = std::log(S / K) + (r - q) * T;
    auto f = [&](double u) {
        cd val = std::exp(I * u * k) * phi(cd(u, -0.5));
        return val.real() / (u * u + 0.25);
    };
    // Tramos de anchura creciente hasta que el integrando sea despreciable
    double total = 0.0, a = 0.0, ancho = 4.0;
    for (int tramo = 0; tramo < 60; ++tramo) {
        double b = a + ancho;
        double parte = gauss_legendre(f, a, b);
        total += parte;
        a = b;
        if (std::fabs(parte) < 1e-16 * std::max(1.0, std::fabs(total)) && std::fabs(f(b)) < 1e-16) break;
        ancho *= 1.4;
    }
    double c = S * std::exp(-q * T) - std::sqrt(S * K) * std::exp(-(r + q) * T / 2.0) / PI * total;
    if (call) return c;
    return c - S * std::exp(-q * T) + K * std::exp(-r * T);
}

double sabr_vol_hagan(double F, double K, double T, double a, double b, double r, double n) {
    double omb = 1.0 - b;
    double corr_t = 1.0;
    if (std::fabs(F - K) < 1e-12 * F) {
        double f1 = std::pow(F, omb);
        corr_t = 1.0 + (omb * omb / 24.0 * a * a / (f1 * f1) + 0.25 * r * b * n * a / f1 +
                        (2.0 - 3.0 * r * r) / 24.0 * n * n) * T;
        return a / f1 * corr_t;
    }
    double lfk = std::log(F / K);
    double fkb = std::pow(F * K, omb / 2.0);
    double z = (n / a) * fkb * lfk;
    double zx = 1.0;
    if (std::fabs(z) > 1e-12) {
        double x = std::log((std::sqrt(1.0 - 2.0 * r * z + z * z) + z - r) / (1.0 - r));
        zx = z / x;
    }
    double denom = fkb * (1.0 + omb * omb / 24.0 * lfk * lfk + std::pow(omb, 4) / 1920.0 * std::pow(lfk, 4));
    corr_t = 1.0 + (omb * omb / 24.0 * a * a / (fkb * fkb) + 0.25 * r * b * n * a / fkb +
                    (2.0 - 3.0 * r * r) / 24.0 * n * n) * T;
    return a / denom * zx * corr_t;
}

double sabr_precio(bool call, double S, double K, double T, double r, double q,
                   double alpha, double beta, double rho, double nu) {
    double F = S * std::exp((r - q) * T);
    double a_abs = alpha * std::pow(F, 1.0 - beta);
    double vol = sabr_vol_hagan(F, K, T, a_abs, beta, rho, nu);
    return bs_precio(call, S, K, T, r, q, vol);
}

} // namespace ref
