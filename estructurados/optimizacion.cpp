#include "optimizacion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace est::opt {

namespace {
double suma_cuadrados(const std::vector<double>& r) {
    double s = 0.0;
    for (double v : r) s += v * v;
    return 0.5 * s;
}
bool finitos(const std::vector<double>& v) {
    for (double x : v)
        if (!std::isfinite(x)) return false;
    return true;
}
} // namespace

bool resolver_cholesky(std::vector<double> A, std::vector<double> b, int n, std::vector<double>& x) {
    // A = L·Lᵀ (in situ en la parte inferior)
    for (int j = 0; j < n; ++j) {
        double s = A[(size_t)(j * n + j)];
        for (int k = 0; k < j; ++k) s -= A[(size_t)(j * n + k)] * A[(size_t)(j * n + k)];
        if (!(s > 0.0) || !std::isfinite(s)) return false;
        const double d = std::sqrt(s);
        A[(size_t)(j * n + j)] = d;
        for (int i = j + 1; i < n; ++i) {
            double t = A[(size_t)(i * n + j)];
            for (int k = 0; k < j; ++k) t -= A[(size_t)(i * n + k)] * A[(size_t)(j * n + k)];
            A[(size_t)(i * n + j)] = t / d;
        }
    }
    for (int i = 0; i < n; ++i) { // L·y = b
        double t = b[(size_t)i];
        for (int k = 0; k < i; ++k) t -= A[(size_t)(i * n + k)] * b[(size_t)k];
        b[(size_t)i] = t / A[(size_t)(i * n + i)];
    }
    for (int i = n - 1; i >= 0; --i) { // Lᵀ·x = y
        double t = b[(size_t)i];
        for (int k = i + 1; k < n; ++k) t -= A[(size_t)(k * n + i)] * b[(size_t)k];
        b[(size_t)i] = t / A[(size_t)(i * n + i)];
    }
    x = std::move(b);
    return true;
}

bool invertir_spd(const std::vector<double>& A, int n, std::vector<double>& inv) {
    inv.assign((size_t)(n * n), 0.0);
    for (int j = 0; j < n; ++j) {
        std::vector<double> e((size_t)n, 0.0), c;
        e[(size_t)j] = 1.0;
        if (!resolver_cholesky(A, e, n, c)) return false;
        for (int i = 0; i < n; ++i) inv[(size_t)(i * n + j)] = c[(size_t)i];
    }
    return true;
}

// ===================================================================== //
// Levenberg-Marquardt                                                    //
// ===================================================================== //
Resultado levenberg_marquardt(const FuncResiduos& f, std::vector<double> x, const OpcionesLM& o) {
    Resultado res;
    const int n = (int)x.size();
    std::vector<double> r;
    if (!f(x, r) || !finitos(r)) {
        res.x = x;
        res.f = std::numeric_limits<double>::infinity();
        res.motivo = "el punto inicial no es admisible";
        return res;
    }
    res.evaluaciones = 1;
    const size_t m = r.size();
    double F = suma_cuadrados(r);
    std::vector<double> J(m * (size_t)n), A((size_t)(n * n)), g((size_t)n);

    auto jacobiano = [&]() -> bool {
        std::vector<double> xp, rp;
        for (int j = 0; j < n; ++j) {
            double h = o.paso_jacobiano * std::max(1.0, std::fabs(x[(size_t)j]));
            xp = x;
            xp[(size_t)j] += h;
            bool ok = f(xp, rp) && finitos(rp);
            ++res.evaluaciones;
            if (!ok) { // hacia el otro lado
                xp[(size_t)j] = x[(size_t)j] - h;
                ok = f(xp, rp) && finitos(rp);
                ++res.evaluaciones;
                if (!ok) return false;
                h = -h;
            }
            for (size_t i = 0; i < m; ++i) J[i * (size_t)n + (size_t)j] = (rp[i] - r[i]) / h;
        }
        for (int a = 0; a < n; ++a) {
            double s = 0.0;
            for (size_t i = 0; i < m; ++i) s += J[i * (size_t)n + (size_t)a] * r[i];
            g[(size_t)a] = s;
            for (int b = 0; b <= a; ++b) {
                double t = 0.0;
                for (size_t i = 0; i < m; ++i) t += J[i * (size_t)n + (size_t)a] * J[i * (size_t)n + (size_t)b];
                A[(size_t)(a * n + b)] = A[(size_t)(b * n + a)] = t;
            }
        }
        return true;
    };
    if (!jacobiano()) {
        res.x = x;
        res.f = F;
        res.motivo = "no se pudo calcular el jacobiano en el punto inicial";
        return res;
    }
    double dmax = 0.0;
    for (int j = 0; j < n; ++j) dmax = std::max(dmax, A[(size_t)(j * n + j)]);
    double mu = o.tau * std::max(dmax, 1e-300), nu = 2.0;

    for (res.iteraciones = 0; res.iteraciones < o.max_iter; ++res.iteraciones) {
        if (F <= 1e-30) { res.convergido = true; res.motivo = "ajuste exacto"; break; }
        double gmax = 0.0;
        for (double v : g) gmax = std::max(gmax, std::fabs(v));
        if (gmax <= o.tol_g * std::max(1e-300, F)) { res.convergido = true; res.motivo = "gradiente nulo"; break; }
        // Escalado de Marquardt con suelo para direcciones sin información
        std::vector<double> D((size_t)n);
        double dm = 0.0;
        for (int j = 0; j < n; ++j) dm = std::max(dm, A[(size_t)(j * n + j)]);
        for (int j = 0; j < n; ++j) D[(size_t)j] = std::max(A[(size_t)(j * n + j)], 1e-12 * std::max(dm, 1e-300));
        std::vector<double> M = A, mg((size_t)n), d;
        for (int j = 0; j < n; ++j) { M[(size_t)(j * n + j)] += mu * D[(size_t)j]; mg[(size_t)j] = -g[(size_t)j]; }
        if (!resolver_cholesky(M, mg, n, d)) { mu *= nu; nu *= 2.0; continue; }
        double nd = 0.0, nx = 0.0;
        for (int j = 0; j < n; ++j) { nd += d[(size_t)j] * d[(size_t)j]; nx += x[(size_t)j] * x[(size_t)j]; }
        if (std::sqrt(nd) <= o.tol_x * (std::sqrt(nx) + o.tol_x)) { res.convergido = true; res.motivo = "paso despreciable"; break; }
        std::vector<double> xn((size_t)n), rn;
        for (int j = 0; j < n; ++j) xn[(size_t)j] = x[(size_t)j] + d[(size_t)j];
        bool ok = f(xn, rn) && finitos(rn);
        ++res.evaluaciones;
        double Fn = ok ? suma_cuadrados(rn) : std::numeric_limits<double>::infinity();
        double pred = 0.0;
        for (int j = 0; j < n; ++j) pred += d[(size_t)j] * (mu * D[(size_t)j] * d[(size_t)j] - g[(size_t)j]);
        pred *= 0.5;
        const double rho = (pred > 0.0) ? (F - Fn) / pred : -1.0;
        if (ok && Fn < F && rho > 0.0) {
            const bool poca_mejora = (F - Fn) <= o.tol_f * F;
            x = xn;
            r = rn;
            F = Fn;
            if (!jacobiano()) { res.motivo = "no se pudo calcular el jacobiano"; break; }
            mu *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3));
            nu = 2.0;
            if (poca_mejora) { res.convergido = true; res.motivo = "sin mejora relativa"; break; }
        } else {
            mu *= nu;
            nu *= 2.0;
            if (mu > 1e30) { res.convergido = true; res.motivo = "no hay paso que mejore (mínimo local)"; break; }
        }
    }
    if (res.motivo.empty()) res.motivo = "máximo de iteraciones";
    res.x = x;
    res.f = F;
    res.jtj = A;
    res.residuos = r;
    return res;
}

// ===================================================================== //
// Nelder-Mead (parámetros adaptativos de Gao y Han, 2012)                 //
// ===================================================================== //
Resultado nelder_mead(const FuncObjetivo& f, std::vector<double> x0, const OpcionesNM& o) {
    Resultado res;
    const int n = (int)x0.size();
    const double a = 1.0, b = 1.0 + 2.0 / n, c = 0.75 - 0.5 / n, s = 1.0 - 1.0 / n;
    auto eval = [&](const std::vector<double>& x) {
        double v = f(x);
        ++res.evaluaciones;
        return std::isfinite(v) ? v : std::numeric_limits<double>::infinity();
    };
    std::vector<std::vector<double>> P((size_t)n + 1, x0);
    std::vector<double> fv((size_t)n + 1);
    for (int i = 0; i < n; ++i) P[(size_t)i + 1][(size_t)i] += o.paso_inicial;
    for (int i = 0; i <= n; ++i) fv[(size_t)i] = eval(P[(size_t)i]);
    std::vector<int> idx((size_t)n + 1);
    while (res.evaluaciones < o.max_eval) {
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int i, int j) { return fv[(size_t)i] < fv[(size_t)j]; });
        const int mejor = idx[0], peor = idx[(size_t)n], seg = idx[(size_t)n - 1];
        double ext = 0.0;
        for (int i = 1; i <= n; ++i)
            for (int j = 0; j < n; ++j)
                ext = std::max(ext, std::fabs(P[(size_t)idx[(size_t)i]][(size_t)j] - P[(size_t)mejor][(size_t)j]));
        if (std::fabs(fv[(size_t)peor] - fv[(size_t)mejor]) <= o.tol_f * (std::fabs(fv[(size_t)mejor]) + 1e-300) &&
            ext <= o.tol_x) {
            res.convergido = true;
            res.motivo = "símplex contraído";
            break;
        }
        ++res.iteraciones;
        std::vector<double> cen((size_t)n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) cen[(size_t)j] += P[(size_t)idx[(size_t)i]][(size_t)j] / n;
        auto punto = [&](double t) {
            std::vector<double> p((size_t)n);
            for (int j = 0; j < n; ++j) p[(size_t)j] = cen[(size_t)j] + t * (P[(size_t)peor][(size_t)j] - cen[(size_t)j]);
            return p;
        };
        std::vector<double> xr = punto(-a);
        double fr = eval(xr);
        if (fr < fv[(size_t)mejor]) {
            std::vector<double> xe = punto(-a * b);
            double fe = eval(xe);
            if (fe < fr) { P[(size_t)peor] = xe; fv[(size_t)peor] = fe; }
            else { P[(size_t)peor] = xr; fv[(size_t)peor] = fr; }
        } else if (fr < fv[(size_t)seg]) {
            P[(size_t)peor] = xr;
            fv[(size_t)peor] = fr;
        } else {
            const bool fuera = fr < fv[(size_t)peor];
            std::vector<double> xc = punto(fuera ? -a * c : c);
            double fc = eval(xc);
            if (fc < std::min(fr, fv[(size_t)peor])) { P[(size_t)peor] = xc; fv[(size_t)peor] = fc; }
            else {
                for (int i = 1; i <= n; ++i) {
                    auto& p = P[(size_t)idx[(size_t)i]];
                    for (int j = 0; j < n; ++j) p[(size_t)j] = P[(size_t)mejor][(size_t)j] + s * (p[(size_t)j] - P[(size_t)mejor][(size_t)j]);
                    fv[(size_t)idx[(size_t)i]] = eval(p);
                }
            }
        }
    }
    if (res.motivo.empty()) res.motivo = "máximo de evaluaciones";
    int mejor = (int)(std::min_element(fv.begin(), fv.end()) - fv.begin());
    res.x = P[(size_t)mejor];
    res.f = fv[(size_t)mejor];
    return res;
}

// ===================================================================== //
// Evolución diferencial (DE/rand/1/bin)                                  //
// ===================================================================== //
Resultado evolucion_diferencial(const FuncObjetivo& f, const std::vector<double>& lo, const std::vector<double>& hi,
                                const OpcionesDE& o) {
    Resultado res;
    const int n = (int)lo.size();
    const int NP = o.poblacion > 0 ? o.poblacion : std::max(15, 10 * n);
    std::mt19937_64 gen(o.semilla);
    auto u01 = [&]() { return (double)(gen() >> 11) * (1.0 / 9007199254740992.0); };
    auto eval = [&](const std::vector<double>& x) {
        double v = f(x);
        ++res.evaluaciones;
        return std::isfinite(v) ? v : std::numeric_limits<double>::infinity();
    };
    std::vector<std::vector<double>> X((size_t)NP, std::vector<double>((size_t)n));
    std::vector<double> fx((size_t)NP);
    for (int i = 0; i < NP; ++i) {
        for (int j = 0; j < n; ++j) X[(size_t)i][(size_t)j] = lo[(size_t)j] + u01() * (hi[(size_t)j] - lo[(size_t)j]);
        fx[(size_t)i] = eval(X[(size_t)i]);
    }
    for (int gnr = 0; gnr < o.generaciones; ++gnr) {
        res.iteraciones = gnr + 1;
        for (int i = 0; i < NP; ++i) {
            int r1, r2, r3;
            do { r1 = (int)(u01() * NP); } while (r1 == i);
            do { r2 = (int)(u01() * NP); } while (r2 == i || r2 == r1);
            do { r3 = (int)(u01() * NP); } while (r3 == i || r3 == r1 || r3 == r2);
            const int jr = (int)(u01() * n);
            std::vector<double> u = X[(size_t)i];
            for (int j = 0; j < n; ++j) {
                if (j == jr || u01() < o.CR) {
                    double v = X[(size_t)r1][(size_t)j] + o.F * (X[(size_t)r2][(size_t)j] - X[(size_t)r3][(size_t)j]);
                    // Rebote en las cotas
                    if (v < lo[(size_t)j]) v = lo[(size_t)j] + u01() * (X[(size_t)i][(size_t)j] - lo[(size_t)j]);
                    if (v > hi[(size_t)j]) v = hi[(size_t)j] - u01() * (hi[(size_t)j] - X[(size_t)i][(size_t)j]);
                    u[(size_t)j] = v;
                }
            }
            double fu = eval(u);
            if (fu <= fx[(size_t)i]) { X[(size_t)i] = u; fx[(size_t)i] = fu; }
        }
        double fmin = *std::min_element(fx.begin(), fx.end()), fmax = *std::max_element(fx.begin(), fx.end());
        if (std::isfinite(fmax) && fmax - fmin <= o.tol_f * (1.0 + std::fabs(fmin))) {
            res.convergido = true;
            res.motivo = "población concentrada";
            break;
        }
    }
    if (res.motivo.empty()) res.motivo = "máximo de generaciones";
    int mejor = (int)(std::min_element(fx.begin(), fx.end()) - fx.begin());
    res.x = X[(size_t)mejor];
    res.f = fx[(size_t)mejor];
    return res;
}

} // namespace est::opt
