// Funciones especiales, cuadratura adaptativa, FFT y aproximación markoviana
// del núcleo fraccionario.

#include "fraccional.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace frac {

namespace {
constexpr double PI = 3.14159265358979323846;

// Gauss-Kronrod 7-15
const double XGK[8] = {0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                       0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                       0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
                       0.207784955007898467600689403773245, 0.0};
const double WGK[8] = {0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                       0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                       0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                       0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
const double WG[4] = {0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
                      0.381830050505118944950369775488975, 0.417959183673469387755102040816327};

void gk15(const std::function<double(double)>& f, double a, double b, double* K, double* err) {
    const double c = 0.5 * (a + b), h = 0.5 * (b - a);
    const double fc = f(c);
    double rk = fc * WGK[7], rg = fc * WG[3];
    for (int j = 0; j < 7; ++j) {
        const double d = h * XGK[j];
        const double s = f(c - d) + f(c + d);
        rk += WGK[j] * s;
        if (j % 2 == 1) rg += WG[j / 2] * s;
    }
    *K = rk * h;
    *err = std::fabs((rk - rg) * h);
}

double gk_adaptativo(const std::function<double(double)>& f, double a, double b, double tol, double escala,
                     int prof) {
    double K, e;
    gk15(f, a, b, &K, &e);
    if (e <= tol * std::max(escala, std::fabs(K)) || prof >= 40 || std::fabs(b - a) < 1e-14 * (1 + std::fabs(a)))
        return K;
    const double m = 0.5 * (a + b);
    return gk_adaptativo(f, a, m, tol, escala, prof + 1) + gk_adaptativo(f, m, b, tol, escala, prof + 1);
}
} // namespace

double integrar(const std::function<double(double)>& f, double a, double b, double tol) {
    if (std::isinf(b)) {
        // x = a + t/(1-t), t ∈ [0, 1)
        auto g = [&](double t) {
            if (t >= 1.0) return 0.0;
            const double u = 1.0 - t;
            return f(a + t / u) / (u * u);
        };
        double K, e;
        gk15(g, 0.0, 1.0, &K, &e);
        return gk_adaptativo(g, 0.0, 1.0, tol, std::fabs(K) * 1e-3, 0);
    }
    double K, e;
    gk15(f, a, b, &K, &e);
    return gk_adaptativo(f, a, b, tol, std::fabs(K) * 1e-3, 0);
}

double mittag_leffler(double alpha, double x) {
    if (!(alpha > 0.0) || alpha > 1.0) throw std::invalid_argument("mittag_leffler: hace falta 0 < α ≤ 1");
    if (x < 0.0) throw std::invalid_argument("mittag_leffler: solo para argumento -x con x ≥ 0");
    if (x == 0.0) return 1.0;
    if (alpha == 1.0) return std::exp(-x);
    if (x <= 1.0) {
        // Serie alternante Σ (-x)^k / Γ(αk + 1), sin cancelaciones graves para x ≤ 1
        double s = 0.0;
        const double lx = std::log(x);
        for (int k = 0; k < 400; ++k) {
            const double t = std::exp(k * lx - std::lgamma(alpha * k + 1.0));
            s += (k % 2 == 0) ? t : -t;
            if (t < 1e-18 * std::fabs(s) && k > 2) break;
        }
        return s;
    }
    // E_α(-x) = sin(απ)/(απ) ∫_0^∞ exp(-t^{1/α}) · x / (t² + 2xt·cos απ + x²) dt
    // (representación de Gorenflo-Mainardi con t = r^α); el integrando tiene un
    // pico lorentziano en t0 = -x·cos απ (si α > 1/2) de anchura x·sin απ
    const double ca = std::cos(alpha * PI), sa = std::sin(alpha * PI);
    auto f = [&](double t) {
        const double e = std::exp(-std::pow(t, 1.0 / alpha));
        const double d = (t + x * ca) * (t + x * ca) + (x * sa) * (x * sa);
        return e * x / d;
    };
    double total = 0.0;
    const double t0 = -x * ca, w = x * sa;
    const double tmax = std::pow(45.0, alpha);   // exp(-t^{1/α}) < 3e-20 más allá
    std::vector<double> cortes = {0.0};
    if (t0 > 0) {
        for (double z : {-20.0, -3.0, 0.0, 3.0, 20.0}) {
            double c = t0 + z * w;
            if (c > 0 && c < tmax) cortes.push_back(c);
        }
    }
    cortes.push_back(std::max(tmax, cortes.back() * 1.01));
    std::sort(cortes.begin(), cortes.end());
    for (size_t i = 0; i + 1 < cortes.size(); ++i)
        if (cortes[i + 1] > cortes[i]) total += integrar(f, cortes[i], cortes[i + 1], 1e-13);
    return sa / (alpha * PI) * total;
}

double nucleo(double H, double t) {
    const double a = H + 0.5;
    return std::pow(t, a - 1.0) / std::tgamma(a);
}

NodosNucleo nodos_nucleo(double H, int factores, double dt_min, double T) {
    NodosNucleo n;
    const double a = H + 0.5;
    if (a >= 1.0 - 1e-12) {
        n.x = {0.0};
        n.c = {1.0};
        return n;
    }
    if (!(H > 0.0)) throw std::invalid_argument("nodos_nucleo: H debe ser positivo");
    const int P = std::max(1, (factores - 1) / 2);
    const double xmin = 0.1 / std::max(T, 1e-6);
    const double xmax = std::max(30.0 / std::max(dt_min, 1e-9), 100.0 * xmin);
    const double G = PI / std::sin(PI * a); // Γ(α)Γ(1-α)
    // Nodo de masa para [0, x_min]: misma masa y mismo primer momento
    const double c0 = std::pow(xmin, 1.0 - a) / ((1.0 - a) * G);
    const double m0 = std::pow(xmin, 2.0 - a) / ((2.0 - a) * G);
    n.x.push_back(m0 / c0);
    n.c.push_back(c0);
    const double ya = std::log(xmin), yb = std::log(xmax), dy = (yb - ya) / P;
    const double z = 1.0 / std::sqrt(3.0);
    for (int p = 0; p < P; ++p) {
        const double mid = ya + (p + 0.5) * dy, half = 0.5 * dy;
        for (double s : {-z, z}) {
            const double y = mid + half * s;
            n.x.push_back(std::exp(y));
            n.c.push_back(half * std::exp((1.0 - a) * y) / G);
        }
    }
    return n;
}

double error_nodos(const NodosNucleo& n, double H, double dt_min, double T) {
    double peor = 0.0;
    const int m = 400;
    for (int i = 0; i <= m; ++i) {
        const double t = dt_min * std::pow(T / dt_min, (double)i / m);
        double s = 0.0;
        for (size_t j = 0; j < n.x.size(); ++j) s += n.c[j] * std::exp(-n.x[j] * t);
        const double k = nucleo(H, t);
        peor = std::max(peor, std::fabs(s - k) / k);
    }
    return peor;
}

void fft(std::vector<cd>& a, bool inversa) {
    const size_t n = a.size();
    if (n == 0 || (n & (n - 1)) != 0) throw std::invalid_argument("fft: el tamaño debe ser potencia de 2");
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * PI / (double)len * (inversa ? 1.0 : -1.0);
        const cd wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            cd w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                const cd u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
    if (inversa)
        for (auto& v : a) v /= (double)n;
}

double black(bool call, double F, double K, double DF, double v) {
    if (!(v > 0.0)) return DF * std::max(call ? F - K : K - F, 0.0);
    const double s = std::sqrt(v);
    const double d1 = (std::log(F / K) + 0.5 * v) / s, d2 = d1 - s;
    auto N = [](double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); };
    return call ? DF * (F * N(d1) - K * N(d2)) : DF * (K * N(-d2) - F * N(-d1));
}

} // namespace frac
