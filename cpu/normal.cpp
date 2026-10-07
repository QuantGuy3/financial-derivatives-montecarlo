#include "normal.hpp"

#include <limits>
#include <numbers>

namespace mc::cpu {

namespace {

// Polinomio de grado 7 por Horner con coeficientes en orden ascendente.
inline double poly7(const double (&c)[8], double x) {
    double r = c[7];
    for (int i = 6; i >= 0; --i) r = r * x + c[i];
    return r;
}

// Coeficientes de AS 241 (PPND16)
constexpr double A[8] = {3.3871328727963666080,    1.3314166789178437745e+2,
                         1.9715909503065514427e+3, 1.3731693765509461125e+4,
                         4.5921953931549871457e+4, 6.7265770927008700853e+4,
                         3.3430575583588128105e+4, 2.5090809287301226727e+3};
constexpr double B[8] = {1.0,                      4.2313330701600911252e+1,
                         6.8718700749205790830e+2, 5.3941960214247511077e+3,
                         2.1213794301586595867e+4, 3.9307895800092710610e+4,
                         2.8729085735721942674e+4, 5.2264952788528545610e+3};
constexpr double C[8] = {1.42343711074968357734,     4.63033784615654529590,
                         5.76949722146069140550,     3.64784832476320460504,
                         1.27045825245236838258,     2.41780725177450611770e-1,
                         2.27238449892691845833e-2,  7.74545014278341407640e-4};
constexpr double D[8] = {1.0,                        2.05319162663775882187,
                         1.67638483018380384940,     6.89767334985100004550e-1,
                         1.48103976427480074590e-1,  1.51986665636164571966e-2,
                         5.47593808499534494600e-4,  1.05075007164441684324e-9};
constexpr double E[8] = {6.65790464350110377720,     5.46378491116411436990,
                         1.78482653991729133580,     2.96560571828504891230e-1,
                         2.65321895265761230930e-2,  1.24266094738807843860e-3,
                         2.71155556874348757815e-5,  2.01033439929228813265e-7};
constexpr double F[8] = {1.0,                        5.99832206555887937690e-1,
                         1.36929880922735805310e-1,  1.48753612908506148525e-2,
                         7.86869131145613259100e-4,  1.84631831751005468180e-5,
                         1.42151175831644588870e-7,  2.04426310338993978564e-15};

} // namespace

double norm_inv_cdf(double p) {
    if (!(p > 0.0)) return p == 0.0 ? -std::numeric_limits<double>::infinity()
                                    :  std::numeric_limits<double>::quiet_NaN();
    if (!(p < 1.0)) return p == 1.0 ?  std::numeric_limits<double>::infinity()
                                    :  std::numeric_limits<double>::quiet_NaN();

    const double q = p - 0.5;
    if (std::abs(q) <= 0.425) {
        const double r = 0.180625 - q * q;
        return q * poly7(A, r) / poly7(B, r);
    }
    // Colas: se trabaja con la probabilidad más pequeña (simetría) para no perder precisión.
    double r = (q < 0.0) ? p : 1.0 - p;
    r = std::sqrt(-std::log(r));
    double val;
    if (r <= 5.0) {
        r -= 1.6;
        val = poly7(C, r) / poly7(D, r);
    } else {
        r -= 5.0;
        val = poly7(E, r) / poly7(F, r);
    }
    return (q < 0.0) ? -val : val;
}

// ---- Ziggurat (Marsaglia & Tsang 2000, 256 capas) ------------------------------------------------------------
//
// El 98.8 % de las veces basta una multiplicación y una comparación; solo se evalúa exp en la cuña
// (~1.2 %) y log en la cola (~0.03 %). Constantes de la tabla de 256 capas:
//   R = 3.6541528853610088 (abscisa de la última capa), V = 0.00492867323399 (área de cada capa).

namespace {

struct ZigTables {
    static constexpr double R = 3.6541528853610088;
    static constexpr double V = 0.00492867323399;
    double x[257];
    double f[257];

    ZigTables() {
        auto pdf = [](double v) { return std::exp(-0.5 * v * v); };
        x[0] = V / pdf(R);
        x[1] = R;
        for (int i = 2; i < 256; i++) x[i] = std::sqrt(-2.0 * std::log(V / x[i - 1] + pdf(x[i - 1])));
        x[256] = 0.0;
        for (int i = 0; i <= 256; i++) f[i] = pdf(x[i]);
    }
};

const ZigTables& zig_tables() {
    static const ZigTables t;
    return t;
}

inline double zig_normal(Xoshiro256pp& g, const ZigTables& T) {
    for (;;) {
        const uint64_t bits = g.next();
        const int i = (int)(bits & 0xFF);
        // 53 bits con signo -> u en [-1, 1)
        const double u = (double)((int64_t)bits >> 11) * (1.0 / 4503599627370496.0);
        const double x = u * T.x[i];
        if (std::abs(x) < T.x[i + 1]) return x;                       // camino rápido
        if (i == 0) {                                                  // cola: algoritmo de Marsaglia
            for (;;) {
                const double x1 = -std::log(g.uniform_open()) / ZigTables::R;
                const double y = -std::log(g.uniform_open());
                if (y + y >= x1 * x1) return u < 0.0 ? -(ZigTables::R + x1) : ZigTables::R + x1;
            }
        }
        // cuña: aceptación por comparación con la densidad
        if (T.f[i + 1] + (T.f[i] - T.f[i + 1]) * g.uniform() < std::exp(-0.5 * x * x)) return x;
    }
}

} // namespace

void fill_normals(NormalMethod method, Xoshiro256pp& g, double* z, int n) {
    switch (method) {
    case NormalMethod::BoxMuller: {
        const double two_pi = 2.0 * std::numbers::pi;
        int i = 0;
        for (; i + 1 < n; i += 2) {
            const double u1 = g.uniform_open();
            const double u2 = g.uniform();
            const double r = std::sqrt(-2.0 * std::log(u1));
            const double th = two_pi * u2;
            z[i]     = r * std::cos(th);
            z[i + 1] = r * std::sin(th);
        }
        if (i < n) {   // n impar: se descarta la segunda normal del último par
            const double u1 = g.uniform_open();
            const double u2 = g.uniform();
            z[i] = std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
        }
        break;
    }
    case NormalMethod::InverseCdf:
        for (int i = 0; i < n; i++) z[i] = norm_inv_cdf(g.uniform_open());
        break;
    case NormalMethod::Ziggurat: {
        const ZigTables& T = zig_tables();
        for (int i = 0; i < n; i++) z[i] = zig_normal(g, T);
        break;
    }
    }
}

} // namespace mc::cpu
