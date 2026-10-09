#include "reference_prices.hpp"

#include "utils.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <numbers>

namespace {

double ncdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// Nodos y pesos de Gauss-Legendre de 32 puntos en [-1,1] (calculados por Newton al arrancar).
struct GL32 {
    std::array<double, 32> x, w;
    GL32() {
        const int n = 32;
        for (int i = 0; i < n / 2; i++) {
            double z = std::cos(std::numbers::pi * (i + 0.75) / (n + 0.5));
            double pp = 1.0;
            for (int it = 0; it < 100; it++) {
                double p1 = 1.0, p2 = 0.0;
                for (int j = 0; j < n; j++) {
                    double p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
                }
                pp = n * (z * p1 - p2) / (z * z - 1.0);
                double dz = p1 / pp;
                z -= dz;
                if (std::abs(dz) < 1e-15) break;
            }
            x[i] = -z; x[n - 1 - i] = z;
            w[i] = w[n - 1 - i] = 2.0 / ((1.0 - z * z) * pp * pp);
        }
    }
};

} // namespace

double gbm_call_drift(double S0, double K, double T, double mu, double r, double sigma) {
    const double s = sigma * std::sqrt(T);
    const double d1 = (std::log(S0 / K) + (mu + 0.5 * sigma * sigma) * T) / s;
    const double d2 = d1 - s;
    return std::exp(-r * T) * (S0 * std::exp(mu * T) * ncdf(d1) - K * ncdf(d2));
}

double heston_call(const HestonParams& h, double K, double r) {
    using cd = std::complex<double>;
    static const GL32 gl;
    const double T = h.T, lnS = std::log(h.S0), lnK = std::log(K);
    const double kappa = h.kappa, theta = h.theta, xi = h.xi, rho = h.rho, v0 = h.v0, mu = h.mu;
    const cd I(0.0, 1.0);

    // P_j = 1/2 + (1/pi) ∫_0^∞ Re[ e^{-iu lnK} φ_j(u) / (iu) ] du, con la deriva mu como "tasa"
    auto integrand = [&](double u, int j) -> double {
        const double uj = (j == 1) ? 0.5 : -0.5;
        const double bj = (j == 1) ? kappa - rho * xi : kappa;
        const cd iu = I * u;
        const cd rsi = rho * xi * iu;
        const cd d = std::sqrt((rsi - bj) * (rsi - bj) - xi * xi * (2.0 * uj * iu - u * u));
        const cd g = (bj - rsi - d) / (bj - rsi + d);                 // formulación "little trap"
        const cd edt = std::exp(-d * T);
        const cd C = mu * iu * T + (kappa * theta / (xi * xi)) *
                     ((bj - rsi - d) * T - 2.0 * std::log((1.0 - g * edt) / (1.0 - g)));
        const cd D = ((bj - rsi - d) / (xi * xi)) * (1.0 - edt) / (1.0 - g * edt);
        const cd phi = std::exp(C + D * v0 + iu * lnS);
        return std::real(std::exp(-iu * lnK) * phi / iu);
    };

    auto P = [&](int j) {
        const double U = 250.0;     // el integrando decae exponencialmente; 250 sobra con T>=0.05
        const int panels = 50;
        const double hw = U / panels;
        double sum = 0.0;
        for (int p = 0; p < panels; p++) {
            const double a = p * hw, mid = a + 0.5 * hw;
            for (int i = 0; i < 32; i++) sum += gl.w[i] * 0.5 * hw * integrand(mid + 0.5 * hw * gl.x[i], j);
        }
        return 0.5 + sum / std::numbers::pi;
    };

    const double p1 = P(1), p2 = P(2);
    return std::exp(-r * T) * (h.S0 * std::exp(mu * T) * p1 - K * p2);
}

std::optional<ReferencePrice> reference_price(const ModelVariant& model, const PayoffVariant& payoff,
                                              int n_steps) {
    if (const auto* g = std::get_if<GBMParams>(&model)) {
        if (const auto* e = std::get_if<European>(&payoff))
            return ReferencePrice{gbm_call_drift(g->S0, e->K, g->T, g->mu, e->r, g->sigma),
                                  "Black-Scholes (analítica, tiempo continuo)"};
        if (const auto* a = std::get_if<GeomAsian>(&payoff); a && n_steps > 0)
            return ReferencePrice{geom_asian_analytic(g->S0, a->K, g->T, g->mu, g->sigma, n_steps),
                                  "Asian geométrica discreta (analítica)"};
    } else if (const auto* h = std::get_if<HestonParams>(&model)) {
        if (const auto* e = std::get_if<European>(&payoff))
            return ReferencePrice{heston_call(*h, e->K, e->r), "Heston (inversión de Fourier)"};
    }
    return std::nullopt;
}
