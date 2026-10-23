// Volatilidad rugosa: función característica de rough Heston (Riccati
// fraccionaria) y esquema híbrido de referencia para rough Bergomi.

#include "fraccional.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace frac {

// ===================================================================== //
// Rough Heston                                                           //
// ===================================================================== //

cd fc_rough_heston(const RoughHeston& p, cd u, double T, int N) {
    if (!(T > 0.0)) return cd(1.0, 0.0);
    if (!(p.H > 0.0 && p.H <= 0.5)) throw std::invalid_argument("rough Heston: hace falta 0 < H ≤ 1/2");
    N = std::max(N, 8);
    const double a = p.H + 0.5;
    const cd I(0.0, 1.0);
    const cd A = -0.5 * (u * u + I * u);
    const cd B = I * (p.rho * p.xi) * u - p.kappa;
    const double C = 0.5 * p.xi * p.xi;
    const double d = T / N;

    // Pesos de la regla trapezoidal fraccionaria (Diethelm): con b_m = m^{α+1},
    //   a_{0,n+1} ∝ n^{α+1} - (n-α)(n+1)^α,  a_{j,n+1} ∝ b_{n-j+2} + b_{n-j} - 2b_{n-j+1},  a_{n+1,n+1} ∝ 1
    std::vector<double> b((size_t)N + 2), dd((size_t)N);
    for (int m = 0; m <= N + 1; ++m) b[(size_t)m] = std::pow((double)m, a + 1.0);
    for (int m = 0; m < N; ++m) dd[(size_t)m] = b[(size_t)m + 2] + b[(size_t)m] - 2.0 * b[(size_t)m + 1];
    const double w = std::pow(d, a) / (a * (a + 1.0) * std::tgamma(a));
    const cd Q = 1.0 - w * B;

    std::vector<double> Fr((size_t)N + 1), Fi((size_t)N + 1);
    std::vector<cd> h((size_t)N + 1);
    h[0] = 0.0;
    Fr[0] = A.real();
    Fi[0] = A.imag();
    double Ih_r = 0, Ih_i = 0, IF_r = 0.5 * A.real(), IF_i = 0.5 * A.imag();
    for (int n = 0; n < N; ++n) {
        // Parte explícita (historia)
        const double a0 = b[(size_t)n] - (n - a) * std::pow((double)n + 1.0, a);
        double Rr = a0 * Fr[0], Ri = a0 * Fi[0];
        for (int j = 1; j <= n; ++j) {
            const double c = dd[(size_t)(n - j)];
            Rr += c * Fr[(size_t)j];
            Ri += c * Fi[(size_t)j];
        }
        // h = R + w·F(h): w·C·h² - (1 - w·B)·h + (R + w·A) = 0, raíz que tiende a P/Q
        const cd P = w * cd(Rr, Ri) + w * A;
        cd disc = std::sqrt(Q * Q - 4.0 * w * C * P);
        if ((std::conj(Q) * disc).real() < 0.0) disc = -disc;
        const cd hn = 2.0 * P / (Q + disc);
        const cd Fn = A + B * hn + C * hn * hn;
        h[(size_t)n + 1] = hn;
        Fr[(size_t)n + 1] = Fn.real();
        Fi[(size_t)n + 1] = Fn.imag();
        const double peso = (n + 1 == N) ? 0.5 : 1.0;
        Ih_r += peso * hn.real();
        Ih_i += peso * hn.imag();
        IF_r += peso * Fn.real();
        IF_i += peso * Fn.imag();
    }
    const cd Ih = d * cd(Ih_r, Ih_i), IF = d * cd(IF_r, IF_i);
    return std::exp(p.kappa * p.theta * Ih + p.V0 * IF);
}

double varianza_forward_rh(const RoughHeston& p, double t) {
    if (t <= 0.0) return p.V0;
    const double a = p.H + 0.5;
    return p.theta + (p.V0 - p.theta) * mittag_leffler(a, p.kappa * std::pow(t, a));
}

// ===================================================================== //
// Rough Bergomi: esquema híbrido (Bennedsen, Lunde y Pakkanen 2017)       //
// ===================================================================== //

std::vector<PrecioMC> rbergomi_hibrido(const RoughBergomi& p, double T, int pasos, long long trayectorias,
                                       const std::vector<double>& strikes, uint64_t semilla) {
    if (!(p.H > 0.0 && p.H <= 0.5)) throw std::invalid_argument("rough Bergomi: hace falta 0 < H ≤ 1/2");
    const int n = std::max(pasos, 1);
    const double dt = T / n, a = p.H + 0.5, s2H = std::sqrt(2.0 * p.H);
    const double rc = std::sqrt(std::max(0.0, 1.0 - p.rho * p.rho));
    // Paso cercano exacto: N = β·ΔW + γ·e
    const double beta = s2H * std::pow(dt, p.H - 0.5) / a;
    const double gam = std::pow(dt, p.H) * std::sqrt(std::max(0.0, 1.0 - 2.0 * p.H / (a * a)));
    // Pasos lejanos: g_k = √(2H)·(b_k·dt)^{H-1/2}, b_k = ((k^α - (k-1)^α)/α)^{1/(α-1)}
    size_t NF = 1;
    while (NF < 2 * (size_t)n) NF <<= 1;
    std::vector<cd> G(NF, 0.0);
    if (a < 1.0 - 1e-12) {
        for (int k = 2; k <= n; ++k) {
            const double bk = std::pow((std::pow((double)k, a) - std::pow(k - 1.0, a)) / a, 1.0 / (a - 1.0));
            G[(size_t)k] = s2H * std::pow(bk * dt, p.H - 0.5);
        }
    } else {
        for (int k = 2; k <= n; ++k) G[(size_t)k] = 1.0; // H = 1/2: Y = W
    }
    fft(G, false);
    std::vector<double> ct((size_t)n + 1);
    for (int i = 0; i <= n; ++i) ct[(size_t)i] = -0.5 * p.eta * p.eta * std::pow(dt * i, 2.0 * p.H);

    std::mt19937_64 gen(semilla);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> z1((size_t)n), dW((size_t)n), e((size_t)n);
    std::vector<cd> conv(NF);
    const size_t nk = strikes.size();
    std::vector<double> s1(nk, 0.0), s2(nk, 0.0);
    const long long npares = std::max(1LL, trayectorias / 2);
    const double sq = std::sqrt(dt);
    for (long long it = 0; it < npares; ++it) {
        for (int k = 0; k < n; ++k) {
            z1[(size_t)k] = nd(gen);
            dW[(size_t)k] = sq * (p.rho * z1[(size_t)k] + rc * nd(gen));
            e[(size_t)k] = nd(gen);
        }
        std::fill(conv.begin(), conv.end(), cd(0.0, 0.0));
        for (int k = 0; k < n; ++k) conv[(size_t)k] = dW[(size_t)k];
        fft(conv, false);
        for (size_t k = 0; k < NF; ++k) conv[k] *= G[k];
        fft(conv, true); // conv[i] = Σ_{k≥2} g_k·ΔW_{i-k}
        std::vector<double> pv(2 * nk);
        for (int sgn = 0; sgn < 2; ++sgn) {
            const double sg = sgn ? -1.0 : 1.0;
            double x = 0.0, V = p.xi0;
            for (int i = 0; i < n; ++i) {
                x += -0.5 * V * dt + std::sqrt(V) * sq * sg * z1[(size_t)i];
                const double Y = sg * (beta * dW[(size_t)i] + gam * e[(size_t)i] + conv[(size_t)i + 1].real());
                V = p.xi0 * std::exp(p.eta * Y + ct[(size_t)i + 1]);
            }
            const double S = std::exp(x);
            for (size_t j = 0; j < nk; ++j) pv[(size_t)sgn * nk + j] = std::max(S - strikes[j], 0.0);
        }
        for (size_t j = 0; j < nk; ++j) {
            const double m = 0.5 * (pv[j] + pv[nk + j]);
            s1[j] += m;
            s2[j] += m * m;
        }
    }
    std::vector<PrecioMC> R(nk);
    for (size_t j = 0; j < nk; ++j) {
        const double m = s1[j] / npares, v = std::max(s2[j] / npares - m * m, 0.0);
        R[j] = {m, std::sqrt(v / npares)};
    }
    return R;
}

} // namespace frac
