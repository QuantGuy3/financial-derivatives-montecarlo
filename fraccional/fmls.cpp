// FMLS (Finite Moment Log Stable, Carr y Wu 2003): función característica,
// muestreo exacto de Chambers-Mallows-Stuck y EDP fraccionaria en espacio.

#include "fraccional.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace frac {

namespace {
constexpr double PI = 3.14159265358979323846;

void comprobar(const FMLS& p) {
    if (!(p.alpha > 1.0 && p.alpha <= 2.0)) throw std::invalid_argument("FMLS: hace falta 1 < α ≤ 2");
    if (!(p.sigma > 0.0)) throw std::invalid_argument("FMLS: la volatilidad debe ser positiva");
}
} // namespace

double FMLS::escala() const { return sigma / std::sqrt(2.0); }

double FMLS::mu() const { return std::pow(escala(), alpha) / std::cos(PI * alpha / 2.0); }

cd fc_fmls(const FMLS& p, cd u, double T) {
    comprobar(p);
    if (std::abs(u) == 0.0) return cd(1.0, 0.0);
    const cd iu = cd(0.0, 1.0) * u;
    const double sec = 1.0 / std::cos(PI * p.alpha / 2.0);
    // ln φ = iu·μ·T - T·(iu·s)^α·sec(πα/2); con Re(iu) ≥ 0 la rama principal es la correcta
    return std::exp(iu * p.mu() * T - T * std::pow(iu * p.escala(), p.alpha) * sec);
}

double cms_fmls(const FMLS& p, double dt, double u, double W) {
    const double a = p.alpha;
    const double V = PI * (u - 0.5);
    const double ta = std::tan(PI * a / 2.0);
    const double B = std::atan(-ta) / a;                       // β = -1
    const double S = std::pow(1.0 + ta * ta, 1.0 / (2.0 * a));
    const double X = S * std::sin(a * (V + B)) / std::pow(std::cos(V), 1.0 / a) *
                     std::pow(std::cos(V - a * (V + B)) / W, (1.0 - a) / a);
    return p.escala() * std::pow(dt, 1.0 / a) * X;
}

PrecioMC fmls_mc(const FMLS& p, bool call, double S0, double K, double T, double r, double q, long long n,
                 uint64_t semilla) {
    comprobar(p);
    std::mt19937_64 gen(semilla);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::exponential_distribution<double> ed(1.0);
    const double F = S0 * std::exp((r - q) * T), DF = std::exp(-r * T);
    std::vector<double> S((size_t)n);
    double media = 0.0;
    for (long long i = 0; i < n; ++i) {
        double u;
        do { u = ud(gen); } while (u <= 0.0 || u >= 1.0);
        const double W = ed(gen);
        S[(size_t)i] = F * std::exp(p.mu() * T + cms_fmls(p, T, u, W));
        media += S[(size_t)i];
    }
    media /= (double)n;
    const double c = F / media; // ajuste del forward
    double s1 = 0, s2 = 0;
    for (double s : S) {
        const double v = DF * std::max(call ? c * s - K : K - c * s, 0.0);
        s1 += v;
        s2 += v * v;
    }
    const double m = s1 / n;
    return {m, std::sqrt(std::max(s2 / n - m * m, 0.0) / n)};
}

// ===================================================================== //
// EDP fraccionaria en espacio                                            //
// ===================================================================== //

double fmls_edp(const FMLS& p, bool call, double S0, double K, double T, double r, double q, double barrera,
                const OpcionesEDP& o) {
    comprobar(p);
    const bool con_barrera = barrera > 0.0;
    if (con_barrera && S0 <= barrera) return 0.0;
    if (!con_barrera && !call) {
        // Put europea por paridad (fuera de la malla la put no es despreciable)
        const double c = fmls_edp(p, true, S0, K, T, r, q, 0.0, o);
        return c - S0 * std::exp(-q * T) + K * std::exp(-r * T);
    }
    const double a = p.alpha, s = p.escala();
    const double kap = -std::pow(s, a) / std::cos(PI * a / 2.0); // > 0
    const double b = r - q - kap;                                // deriva de ln S (μ = -κ_α)
    const double z = p.sigma * std::sqrt(std::max(T, 1e-4));
    const double xK = std::log(K), x0 = std::log(S0);
    const double xL = con_barrera ? std::log(barrera) : std::min(x0, xK) - 12.0 * z - 0.5;
    const double xR = std::max(x0, xK) + 8.0 * z + 0.25;
    const int nx = std::max(o.nx, 50), nt = std::max(o.nt, 4);
    const double h = (xR - xL) / nx, dtau = T / nt;

    // Pesos WSGD (p, q) = (1, 0): w_0 = (α/2)g_0, w_k = (α/2)g_k + ((2-α)/2)g_{k-1}
    std::vector<double> g((size_t)nx + 2), w((size_t)nx + 2);
    g[0] = 1.0;
    for (int k = 1; k <= nx + 1; ++k) g[(size_t)k] = g[(size_t)k - 1] * (1.0 - (a + 1.0) / k);
    w[0] = 0.5 * a * g[0];
    for (int k = 1; k <= nx + 1; ++k) w[(size_t)k] = 0.5 * a * g[(size_t)k] + 0.5 * (2.0 - a) * g[(size_t)k - 1];

    // Operador en los nodos interiores i = 1..nx-1 (índice m = i - 1):
    //  (AV)_m = sup·V_{m+1} + Σ_{j=0}^{m} t_j·V_{m-j}
    const double c_frac = kap * std::pow(h, -a);
    const double sup = b / (2.0 * h) + c_frac * w[0];
    const int n = nx - 1;
    std::vector<double> t((size_t)n);
    for (int j = 0; j < n; ++j) t[(size_t)j] = c_frac * w[(size_t)j + 1];
    t[0] -= r;
    if (n > 1) t[1] -= b / (2.0 * h);

    // M = I - (Δτ/2)·A (la misma matriz para los semipasos implícitos de Rannacher y para
    // Crank-Nicolson): Hessenberg inferior => LU con U bidiagonal superior en O(n²)
    const double th = 0.5 * dtau;
    const double Msup = -th * sup;
    auto Mij = [&](int m, int mp) { // mp ≤ m
        return (m == mp ? 1.0 : 0.0) - th * t[(size_t)(m - mp)];
    };
    std::vector<double> L((size_t)n * (size_t)(n + 1) / 2), U((size_t)n);
    auto Lref = [&](int m, int j) -> double& { return L[(size_t)m * (m + 1) / 2 + (size_t)j]; };
    for (int m = 0; m < n; ++m) {
        double prev = 0.0;
        for (int j = 0; j < m; ++j) {
            const double v = (Mij(m, j) - (j > 0 ? prev * Msup : 0.0)) / U[(size_t)j];
            Lref(m, j) = v;
            prev = v;
        }
        U[(size_t)m] = Mij(m, m) - (m > 0 ? prev * Msup : 0.0);
        if (std::fabs(U[(size_t)m]) < 1e-300) throw std::runtime_error("EDP FMLS: pivote nulo");
    }
    auto resolver = [&](std::vector<double>& x) { // M·x = rhs (in situ)
        for (int m = 0; m < n; ++m) {
            double s0 = x[(size_t)m];
            const double* Lm = &L[(size_t)m * (m + 1) / 2];
            for (int j = 0; j < m; ++j) s0 -= Lm[j] * x[(size_t)j];
            x[(size_t)m] = s0;
        }
        x[(size_t)n - 1] /= U[(size_t)n - 1];
        for (int m = n - 2; m >= 0; --m) x[(size_t)m] = (x[(size_t)m] - Msup * x[(size_t)m + 1]) / U[(size_t)m];
    };
    auto aplicar_A = [&](const std::vector<double>& V, std::vector<double>& out) {
        for (int m = 0; m < n; ++m) {
            double s0 = (m + 1 < n) ? sup * V[(size_t)m + 1] : 0.0;
            for (int j = 0; j <= m; ++j) s0 += t[(size_t)j] * V[(size_t)(m - j)];
            out[(size_t)m] = s0;
        }
    };
    auto borde = [&](double tau) { // valor en x_R
        const double S = std::exp(xR);
        return call ? std::max(S * std::exp(-q * tau) - K * std::exp(-r * tau), 0.0) : 0.0;
    };

    // Condición inicial: pago promediado en cada celda (el codo no tiene por qué caer en un nodo)
    std::vector<double> V((size_t)n), AV((size_t)n), rhs((size_t)n);
    for (int m = 0; m < n; ++m) {
        const double xi = xL + (m + 1) * h, lo = xi - 0.5 * h, hi = xi + 0.5 * h;
        double v = 0.0;
        if (call && hi > xK) {
            const double a0 = std::max(lo, xK);
            v = (std::exp(hi) - std::exp(a0) - K * (hi - a0)) / h;
        } else if (!call && lo < xK) {
            const double b0 = std::min(hi, xK);
            v = (K * (b0 - lo) - (std::exp(b0) - std::exp(lo))) / h;
        }
        V[(size_t)m] = v;
    }
    double tau = 0.0;
    // Rannacher: dos semipasos implícitos
    for (int sp = 0; sp < 2; ++sp) {
        const double tn = tau + th;
        for (int m = 0; m < n; ++m) rhs[(size_t)m] = V[(size_t)m];
        rhs[(size_t)n - 1] += th * sup * borde(tn);
        resolver(rhs);
        V.swap(rhs);
        tau = tn;
    }
    for (int k = 1; k < nt; ++k) {
        const double tn = tau + dtau;
        aplicar_A(V, AV);
        for (int m = 0; m < n; ++m) rhs[(size_t)m] = V[(size_t)m] + th * AV[(size_t)m];
        rhs[(size_t)n - 1] += th * sup * (borde(tau) + borde(tn));
        resolver(rhs);
        V.swap(rhs);
        tau = tn;
    }
    // Interpolación cúbica en ln S0
    const double pos = (x0 - xL) / h; // índice de nodo (0 = x_L, nx = x_R)
    int i0 = (int)std::floor(pos) - 1;
    i0 = std::max(0, std::min(i0, nx - 3));
    auto valor = [&](int i) {
        if (i <= 0) return 0.0;
        if (i >= nx) return borde(T);
        return V[(size_t)i - 1];
    };
    double res = 0.0;
    for (int a1 = 0; a1 < 4; ++a1) {
        double l = 1.0;
        for (int b1 = 0; b1 < 4; ++b1)
            if (b1 != a1) l *= (pos - (i0 + b1)) / (double)(a1 - b1);
        res += l * valor(i0 + a1);
    }
    return std::max(res, 0.0);
}

} // namespace frac
