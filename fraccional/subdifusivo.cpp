// Black-Scholes subdifusivo (Magdziarz 2009): el log-precio forward es un
// browniano con deriva evaluado en el tiempo operacional E_t, la inversa de un
// subordinador α-estable (periodos en los que el precio no se mueve).
//
//   S_t = F_t·exp(σB(E_t) - ½σ²E_t),  E_T =d T^α·U^{-α},  E[e^{-sU}] = e^{-s^α}
//
// Con α = 1 es Black-Scholes. Tres métodos para europeas:
//   cuadratura   U por la representación de Kanter (1975):
//                U = (A(πV)/W)^{(1-α)/α}, V ~ U(0,1), W ~ Exp(1)
//   EDP          ∂_t^α c = ½σ²(c_yy - c_y) (derivada de Caputo en t), esquema L1
//                en malla graduada (Stynes, O'Riordan y Gracia 2017)
//   Monte Carlo  muestreo exacto de E_T con la misma representación

#include "fraccional.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace frac {

namespace {
constexpr double PI = 3.14159265358979323846;

void comprobar(const Subdifusivo& p) {
    if (!(p.alpha > 0.0 && p.alpha <= 1.0)) throw std::invalid_argument("subdifusivo: hace falta 0 < α ≤ 1");
    if (!(p.sigma > 0.0)) throw std::invalid_argument("subdifusivo: la volatilidad debe ser positiva");
}

// ln E_T - α·ln T = (1-α)·(ln W - ln A(φ)), con
// (1-α)·ln A(φ) = ln(sin αφ / sin φ) + (1-α)·ln(sin((1-α)φ) / sin αφ)  (sin desbordamientos)
double log_tiempo(double a, double phi, double lnW, double T) {
    const double sa = std::sin(a * phi), s1 = std::sin(phi), sb = std::sin((1.0 - a) * phi);
    const double lnA1a = std::log(sa / s1) + (1.0 - a) * std::log(sb / sa);
    return a * std::log(T) + (1.0 - a) * lnW - lnA1a;
}

// Nodos de Gauss-Legendre en [-1, 1]
void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
    x.resize((size_t)n);
    w.resize((size_t)n);
    for (int i = 0; i < n; ++i) {
        double z = std::cos(PI * (i + 0.75) / (n + 0.5)), dp = 0.0;
        for (int it = 0; it < 100; ++it) {
            double p0 = 1.0, p1 = 0.0;
            for (int j = 1; j <= n; ++j) {
                const double p2 = p1;
                p1 = p0;
                p0 = ((2.0 * j - 1.0) * z * p1 - (j - 1.0) * p2) / j;
            }
            dp = n * (z * p0 - p1) / (z * z - 1.0);
            const double z1 = z;
            z = z1 - p0 / dp;
            if (std::fabs(z - z1) < 1e-15) break;
        }
        x[(size_t)i] = z;
        w[(size_t)i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
}
} // namespace

double tiempo_operacional_medio(double alpha, double T) { return std::pow(T, alpha) / std::tgamma(1.0 + alpha); }

double subdif_precio(const Subdifusivo& p, bool call, double S0, double K, double T, double r, double q) {
    comprobar(p);
    const double F = S0 * std::exp((r - q) * T), DF = std::exp(-r * T);
    if (p.alpha >= 1.0 - 1e-12) return black(call, F, K, DF, p.sigma * p.sigma * T);
    const double a = p.alpha, s2 = p.sigma * p.sigma;
    // V = 1 - s² (s ∈ (0,1)) suaviza el extremo φ → π, donde E_T → 0 como (1 - V)
    std::vector<double> gx, gw;
    gauss_legendre(64, gx, gw);
    const double hy = 0.08, y0 = -38.0, y1 = 4.2; // W = e^y
    double total = 0.0;
    for (size_t i = 0; i < gx.size(); ++i) {
        const double s = 0.5 * (gx[i] + 1.0), ws = 0.5 * gw[i];
        const double V = 1.0 - s * s, jac = 2.0 * s;
        const double phi = PI * V;
        double inner = 0.0;
        for (double y = y0; y <= y1 + 1e-12; y += hy) {
            const double W = std::exp(y);
            const double peso = std::exp(-W) * W; // e^{-W}·dW/dy
            if (peso < 1e-300) continue;
            const double lt = log_tiempo(a, phi, y, T);
            inner += peso * black(call, F, K, DF, s2 * std::exp(lt));
        }
        total += ws * jac * inner * hy;
    }
    return total;
}

PrecioMC subdif_mc(const Subdifusivo& p, bool call, double S0, double K, double T, double r, double q, long long n,
                   uint64_t semilla) {
    comprobar(p);
    std::mt19937_64 gen(semilla);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::exponential_distribution<double> ed(1.0);
    std::normal_distribution<double> nd(0.0, 1.0);
    const double F = S0 * std::exp((r - q) * T), DF = std::exp(-r * T);
    double s1 = 0, s2 = 0;
    const long long npares = std::max(1LL, n / 2);
    for (long long i = 0; i < npares; ++i) {
        double ET = T;
        if (p.alpha < 1.0) {
            double u;
            do { u = ud(gen); } while (u <= 0.0 || u >= 1.0);
            ET = std::exp(log_tiempo(p.alpha, PI * u, std::log(ed(gen)), T));
        }
        const double z = nd(gen), sv = p.sigma * std::sqrt(ET), d = -0.5 * p.sigma * p.sigma * ET;
        double m = 0.0;
        for (double sg : {1.0, -1.0}) {
            const double S = F * std::exp(d + sg * sv * z);
            m += 0.5 * DF * std::max(call ? S - K : K - S, 0.0);
        }
        s1 += m;
        s2 += m * m;
    }
    const double m = s1 / npares;
    return {m, std::sqrt(std::max(s2 / npares - m * m, 0.0) / npares)};
}

namespace {
// Una resolución de la EDP con el esquema L1 (ny nodos en y, nt pasos en t)
double subdif_edp_base(const Subdifusivo& p, bool call, double S0, double K, double T, double r, double q, int ny,
                       int nt) {
    const double a = p.alpha, s2 = p.sigma * p.sigma;
    const double F = S0 * std::exp((r - q) * T), DF = std::exp(-r * T);
    const double y0 = std::log(F / K);
    // Malla en y = ln(F/K) del forward "congelado": u(y, t) = E[(e^{y+X_t} - 1)⁺] (o la put)
    const double z = p.sigma * std::sqrt(3.0 * tiempo_operacional_medio(a, T)) + 0.05;
    const double yL = std::min(0.0, y0) - 7.0 * z, yR = std::max(0.0, y0) + 7.0 * z;
    ny = std::max(ny, 20);
    nt = std::max(nt, 4);
    const double h = (yR - yL) / ny;
    const int n = ny - 1; // nodos interiores
    auto borde_izq = [&]() { return call ? 0.0 : 1.0 - std::exp(yL); };
    auto borde_dcho = [&]() { return call ? std::exp(yR) - 1.0 : 0.0; };
    // Pago promediado en cada celda
    std::vector<double> u0((size_t)n);
    for (int m = 0; m < n; ++m) {
        const double yi = yL + (m + 1) * h, lo = yi - 0.5 * h, hi = yi + 0.5 * h;
        double v = 0.0;
        if (call && hi > 0.0) {
            const double c0 = std::max(lo, 0.0);
            v = (std::exp(hi) - std::exp(c0) - (hi - c0)) / h;
        } else if (!call && lo < 0.0) {
            const double c0 = std::min(hi, 0.0);
            v = ((c0 - lo) - (std::exp(c0) - std::exp(lo))) / h;
        }
        u0[(size_t)m] = v;
    }
    // Malla temporal graduada: cerca del dinero la solución se comporta como √E[E_t] ~ t^{α/2},
    // y la malla t_n = T(n/N)^γ con γ = (2-α)/(α/2) recupera el orden del esquema L1
    const double gam = (a >= 1.0 - 1e-12) ? 1.0 : std::min(2.0 * (2.0 - a) / a, 6.0);
    std::vector<double> t((size_t)nt + 1);
    for (int k = 0; k <= nt; ++k) t[(size_t)k] = T * std::pow((double)k / nt, gam);
    const double g2a = std::tgamma(2.0 - a);
    // L_h = ½σ²(D2 - D1) tridiagonal
    const double lo_c = 0.5 * s2 * (1.0 / (h * h) + 1.0 / (2.0 * h));
    const double di_c = -s2 / (h * h);
    const double up_c = 0.5 * s2 * (1.0 / (h * h) - 1.0 / (2.0 * h));
    std::vector<std::vector<double>> du; // u^k - u^{k-1}
    du.reserve((size_t)nt);
    std::vector<double> u = u0, rhs((size_t)n), cp((size_t)n), dp((size_t)n);
    const double bL = borde_izq(), bR = borde_dcho();
    for (int nn = 1; nn <= nt; ++nn) {
        const double tn = t[(size_t)nn];
        auto coef = [&](int k) { // a_{n,k}
            const double tk = t[(size_t)k], tk1 = t[(size_t)k - 1];
            if (a >= 1.0 - 1e-12) return k == nn ? 1.0 / (tk - tk1) : 0.0; // α = 1: Euler implícito
            const double dk = (k == nn) ? 0.0 : std::pow(tn - tk, 1.0 - a);
            return (std::pow(tn - tk1, 1.0 - a) - dk) / (g2a * (tk - tk1));
        };
        const double ann = coef(nn);
        for (int m = 0; m < n; ++m) rhs[(size_t)m] = ann * u[(size_t)m];
        for (int k = 1; k < nn; ++k) {
            const double ak = coef(k);
            const std::vector<double>& d = du[(size_t)k - 1];
            for (int m = 0; m < n; ++m) rhs[(size_t)m] -= ak * d[(size_t)m];
        }
        // (a_nn·I - L_h)·u^n = rhs, con los bordes (constantes en t) al lado derecho
        rhs[0] += lo_c * bL;
        rhs[(size_t)n - 1] += up_c * bR;
        const double A = -lo_c, Bd = ann - di_c, C = -up_c;
        cp[0] = C / Bd;
        dp[0] = rhs[0] / Bd;
        for (int m = 1; m < n; ++m) {
            const double den = Bd - A * cp[(size_t)m - 1];
            cp[(size_t)m] = C / den;
            dp[(size_t)m] = (rhs[(size_t)m] - A * dp[(size_t)m - 1]) / den;
        }
        std::vector<double> un((size_t)n);
        un[(size_t)n - 1] = dp[(size_t)n - 1];
        for (int m = n - 2; m >= 0; --m) un[(size_t)m] = dp[(size_t)m] - cp[(size_t)m] * un[(size_t)m + 1];
        std::vector<double> d((size_t)n);
        for (int m = 0; m < n; ++m) d[(size_t)m] = un[(size_t)m] - u[(size_t)m];
        du.push_back(std::move(d));
        u.swap(un);
    }
    // Interpolación cúbica en y0
    const double pos = (y0 - yL) / h;
    int i0 = std::max(0, std::min((int)std::floor(pos) - 1, ny - 3));
    auto valor = [&](int i) {
        if (i <= 0) return bL;
        if (i >= ny) return bR;
        return u[(size_t)i - 1];
    };
    double res = 0.0;
    for (int a1 = 0; a1 < 4; ++a1) {
        double l = 1.0;
        for (int b1 = 0; b1 < 4; ++b1)
            if (b1 != a1) l *= (pos - (i0 + b1)) / (double)(a1 - b1);
        res += l * valor(i0 + a1);
    }
    return DF * K * std::max(res, 0.0);
}
} // namespace

double subdif_edp(const Subdifusivo& p, bool call, double S0, double K, double T, double r, double q, int ny,
                  int nt) {
    comprobar(p);
    ny = std::max(ny, 40);
    nt = std::max(nt, 8);
    // Extrapolación de Richardson: en t con el orden del esquema L1 en malla graduada
    // (2 - α; 1 con α = 1) y en y con orden 2 (diferencias centradas)
    const double pt = (p.alpha < 1.0 - 1e-12) ? 2.0 - p.alpha : 1.0;
    const double ft = std::pow(2.0, pt);
    auto en_t = [&](int n_y) {
        const double a = subdif_edp_base(p, call, S0, K, T, r, q, n_y, nt / 2);
        const double b = subdif_edp_base(p, call, S0, K, T, r, q, n_y, nt);
        return (ft * b - a) / (ft - 1.0);
    };
    const double fino = en_t(ny), grueso = en_t(ny / 2);
    return std::max(fino + (fino - grueso) / 3.0, 0.0);
}

} // namespace frac
