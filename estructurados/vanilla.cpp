#include "vanilla.hpp"
#include "plan.hpp"
#include "referencias.hpp"
#include "../fraccional/fraccional.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace est {

namespace {

constexpr double PI = 3.14159265358979323846;
using cd = std::complex<double>;

double ncdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
double npdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * PI); }

// Nodos y pesos de Gauss-Legendre de 32 puntos en [-1, 1]
struct GL32 {
    double x[32], w[32];
    GL32() {
        const int n = 32;
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
            x[i] = z;
            w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
        }
    }
};
const GL32& gl32() {
    static const GL32 g;
    return g;
}

} // namespace

// ===================================================================== //
// Integral de Lewis con control Black-Scholes                            //
// ===================================================================== //

IntegralLewis::IntegralLewis(const FuncCar& phi, double T, int nodos_tramo, double tol) : T_(T) {
    // Volatilidad de referencia: varianza de X_T por diferencias en u = 0,
    // c2 = -d²/du² ln φ(0)
    const auto t0 = std::chrono::steady_clock::now();
    const double h = 1e-3;
    double c2 = -(std::log(phi(cd(h, 0.0))) + std::log(phi(cd(-h, 0.0)))).real() / (h * h);
    const double coste = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 2.0;
    if (!std::isfinite(c2) || c2 <= 1e-12) c2 = 0.04 * T;
    sigma_ref_ = std::sqrt(c2 / T);
    const double s2T = sigma_ref_ * sigma_ref_ * T;
    auto phi_bs = [s2T](cd u) { return std::exp(-0.5 * s2T * (cd(0.0, 1.0) * u + u * u)); };

    // Tramos de anchura 4 hasta que la envolvente |φ| + |φ_BS| sobre (u² + 1/4) sea
    // despreciable (se mira el extremo de cada tramo). Con 16 nodos por tramo (φ cara),
    // a partir de u = 40 los tramos miden 8: el integrando ya es una cola suave.
    const double umax = 2.0e4;
    std::vector<double> ini;   // inicio de cada tramo
    std::vector<double> anc;   // anchura de cada tramo
    int pequenos = 0;
    for (double a = 0.0; a < umax;) {
        const double ancho = (nodos_tramo == 16 && a >= 40.0) ? 8.0 : 4.0;
        ini.push_back(a);
        anc.push_back(ancho);
        const double b = a + ancho;
        a = b;
        cd zb(b, -0.5);
        double env = (std::abs(phi(zb)) + std::abs(phi_bs(zb))) / (b * b + 0.25);
        if (!std::isfinite(env)) throw std::runtime_error("función característica no finita en u = " + std::to_string(b));
        pequenos = (env < tol) ? pequenos + 1 : 0;
        if (pequenos >= 2) break;
    }
    const size_t tramos = ini.size();
    // Nodos de Gauss-Legendre de cada tramo
    std::vector<double> gx, gw;
    if (nodos_tramo == 16) {
        static const std::vector<double> x16 = [] {
            std::vector<double> x(16);
            for (int i = 0; i < 16; ++i) {
                double z = std::cos(PI * (i + 0.75) / 16.5);
                for (int it = 0; it < 100; ++it) {
                    double p0 = 1.0, p1 = 0.0;
                    for (int j = 1; j <= 16; ++j) { double p2 = p1; p1 = p0; p0 = ((2.0 * j - 1.0) * z * p1 - (j - 1.0) * p2) / j; }
                    const double dp = 16 * (z * p0 - p1) / (z * z - 1.0), z1 = z;
                    z = z1 - p0 / dp;
                    if (std::fabs(z - z1) < 1e-15) break;
                }
                x[(size_t)i] = z;
            }
            return x;
        }();
        gx = x16;
        for (double z : gx) {
            double p0 = 1.0, p1 = 0.0;
            for (int j = 1; j <= 16; ++j) { double p2 = p1; p1 = p0; p0 = ((2.0 * j - 1.0) * z * p1 - (j - 1.0) * p2) / j; }
            const double dp = 16 * (z * p0 - p1) / (z * z - 1.0);
            gw.push_back(2.0 / ((1.0 - z * z) * dp * dp));
        }
    } else {
        const GL32& g = gl32();
        gx.assign(g.x, g.x + 32);
        gw.assign(g.w, g.w + 32);
    }
    const size_t m = gx.size(), n = tramos * m;
    u_.resize(n);
    cr_.resize(n);
    ci_.resize(n);
    auto nodo = [&](size_t k) {
        const size_t t = k / m, i = k % m;
        const double c = ini[t] + 0.5 * anc[t], r = 0.5 * anc[t];
        const double u = c + r * gx[i];
        const cd z(u, -0.5);
        const cd peso = (phi(z) - phi_bs(z)) * (r * gw[i] / (u * u + 0.25));
        u_[k] = u;
        cr_[k] = peso.real();
        ci_[k] = peso.imag();
    };
    // Con una φ cara (p. ej. la de rough Heston, que integra una ecuación de Riccati
    // fraccionaria en cada u) los nodos se reparten entre hilos
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int nh = (coste * (double)n > 2e-3 && hw > 1) ? (int)std::min<size_t>(hw, n / 16 + 1) : 1;
    if (nh <= 1) {
        for (size_t k = 0; k < n; ++k) nodo(k);
    } else {
        std::vector<std::thread> th;
        for (int t = 0; t < nh; ++t)
            th.emplace_back([&, t] {
                for (size_t k = (size_t)t; k < n; k += (size_t)nh) nodo(k);
            });
        for (auto& x : th) x.join();
    }
}

void IntegralLewis::delta(double k, double* d, double* dprima) const {
    double s = 0.0, sp = 0.0;
    for (size_t i = 0; i < u_.size(); ++i) {
        const double uk = u_[i] * k, cs = std::cos(uk), sn = std::sin(uk);
        s += cs * cr_[i] - sn * ci_[i];
        sp += -u_[i] * (sn * cr_[i] + cs * ci_[i]);
    }
    *d = s / PI;
    if (dprima) *dprima = sp / PI;
}

double IntegralLewis::call(double S, double K, double r, double q) const {
    const double k = std::log(S / K) + (r - q) * T_;
    double d = 0.0;
    delta(k, &d, nullptr);
    const double A = std::sqrt(S * K) * std::exp(-(r + q) * T_ / 2.0);
    return ref::bs_precio(true, S, K, T_, r, q, sigma_ref_) - A * d;
}

namespace {
double black_fwd(bool call, double F, double K, double T, double DF, double s) {
    const double sv = s * std::sqrt(T);
    if (!(sv > 0)) return DF * std::max(call ? F - K : K - F, 0.0);
    const double d1 = (std::log(F / K) + 0.5 * sv * sv) / sv, d2 = d1 - sv;
    return call ? DF * (F * ncdf(d1) - K * ncdf(d2)) : DF * (K * ncdf(-d2) - F * ncdf(-d1));
}
} // namespace

double IntegralLewis::call_fwd(double F, double K, double DF) const {
    double d = 0.0;
    delta(std::log(F / K), &d, nullptr);
    return black_fwd(true, F, K, T_, DF, sigma_ref_) - DF * std::sqrt(F * K) * d;
}

double IntegralLewis::put_fwd(double F, double K, double DF) const {
    double d = 0.0;
    delta(std::log(F / K), &d, nullptr);
    return black_fwd(false, F, K, T_, DF, sigma_ref_) - DF * std::sqrt(F * K) * d;
}

double IntegralLewis::digital(double S, double K, double r, double q) const {
    // D = -∂C/∂K; con C = C_BS - √S·e^{-(r+q)T/2}·√K·Δ(k), dk/dK = -1/K
    const double k = std::log(S / K) + (r - q) * T_;
    double d = 0.0, dp = 0.0;
    delta(k, &d, &dp);
    const double A = std::sqrt(S) * std::exp(-(r + q) * T_ / 2.0);
    return ref::bs_digital(true, S, K, T_, r, q, sigma_ref_) + A * (d / (2.0 * std::sqrt(K)) - dp / std::sqrt(K));
}

// ===================================================================== //
// PreciosVanilla                                                         //
// ===================================================================== //

struct PreciosVanilla::Muestra {
    double T = 0, DF = 1, F = 0;
    double media = 1;        // media de las S simuladas (sin reescalar)
    std::vector<double> S;   // simuladas, ordenadas (sin reescalar)
    std::vector<double> pre; // pre[i] = Σ_{j<i} S[j]
    std::vector<double> pre2; // pre2[i] = Σ_{j<i} S[j]²
    // E[(c·S - K)⁺] y E[(K - c·S)⁺] con c = F/media (ajuste del forward)
    double call_esp(double K, double Fobj) const {
        const double c = Fobj / media, k = K / c;
        const size_t n = S.size();
        size_t i = (size_t)(std::upper_bound(S.begin(), S.end(), k) - S.begin());
        return c * ((pre[n] - pre[i]) - k * (double)(n - i)) / (double)n;
    }
    double put_esp(double K, double Fobj) const {
        const double c = Fobj / media, k = K / c;
        size_t i = (size_t)(std::upper_bound(S.begin(), S.end(), k) - S.begin());
        return c * (k * (double)i - pre[i]) / (double)S.size();
    }
    // Desviación típica del pago (c·S - K)⁺ o (K - c·S)⁺
    double desv(double K, double Fobj, bool call) const {
        const double c = Fobj / media, k = K / c;
        const size_t n = S.size();
        const size_t i = (size_t)(std::upper_bound(S.begin(), S.end(), k) - S.begin());
        double m1, m2;
        if (call) {
            const double s1 = pre[n] - pre[i], s2 = pre2[n] - pre2[i], cnt = (double)(n - i);
            m1 = c * (s1 - k * cnt) / n;
            m2 = c * c * (s2 - 2.0 * k * s1 + k * k * cnt) / n;
        } else {
            const double s1 = pre[i], s2 = pre2[i], cnt = (double)i;
            m1 = c * (k * cnt - s1) / n;
            m2 = c * c * (k * k * cnt - 2.0 * k * s1 + s2) / n;
        }
        return std::sqrt(std::max(m2 - m1 * m1, 0.0));
    }
    double prob_mayor_igual(double K, double Fobj) const {
        const double k = K * media / Fobj;
        size_t i = (size_t)(std::lower_bound(S.begin(), S.end(), k) - S.begin());
        return (double)(S.size() - i) / (double)S.size();
    }
};

PreciosVanilla::PreciosVanilla(const Activo& a, const Curva& c, OpcionesMCVanilla mc, bool forzar_mc)
    : act_(a), curva_(c), S0_(a.spot), q_(a.dividendos), mc_(mc) {
    switch (a.modelo.tipo) {
        case TipoModelo::BlackScholes: metodo_ = Metodo::BS; break;
        case TipoModelo::Merton:       metodo_ = Metodo::MERTON; break;
        case TipoModelo::Heston:
        case TipoModelo::Bates:
        case TipoModelo::NIG:
        case TipoModelo::RoughHeston:
        case TipoModelo::FMLS:         metodo_ = Metodo::FOURIER; break;
        case TipoModelo::SABR:         metodo_ = Metodo::HAGAN; break;
        default:                       metodo_ = Metodo::MC; break;
    }
    if (forzar_mc) metodo_ = Metodo::MC;
}

PreciosVanilla::~PreciosVanilla() = default;

std::string PreciosVanilla::metodo() const {
    switch (metodo_) {
        case Metodo::BS:      return "fórmula cerrada de Black-Scholes";
        case Metodo::MERTON:  return "serie de Merton";
        case Metodo::FOURIER:
            return act_.modelo.tipo == TipoModelo::RoughHeston
                       ? "Fourier (Lewis) con la función característica por la ecuación de Riccati fraccionaria"
                       : "Fourier (Lewis) con la función característica del modelo";
        case Metodo::HAGAN:   return "fórmula de Hagan (aproximación SABR)";
        default:
            return "Monte Carlo (" + std::to_string(mc_.trayectorias) + " trayectorias con antitéticas, " +
                   (modelo_exacto(act_.modelo.tipo) ? std::string("paso exacto") :
                    std::to_string((int)mc_.pasos_por_anio) + " pasos/año") + ")";
    }
}

double PreciosVanilla::r_efectivo(double T) const {
    return T > 1e-12 ? curva_.integral(T) / T : curva_.tipo(0.0);
}
double PreciosVanilla::df(double T) const { return curva_.df(T); }
double PreciosVanilla::forward(double T) const { return S0_ * std::exp(-q_ * T) / curva_.df(T); }

IntegralLewis::FuncCar PreciosVanilla::func_car(double T) const {
    const Modelo& m = act_.modelo;
    switch (m.tipo) {
        case TipoModelo::BlackScholes: return ref::fc_black_scholes(m["volatilidad"], T);
        case TipoModelo::Merton:
            return ref::fc_merton(m["volatilidad"], m["lambda"], m["media_salto"], m["vol_salto"], T);
        case TipoModelo::Heston: {
            double vi = m["vol_inicial"], vl = m["vol_largo_plazo"];
            return ref::fc_heston(vi * vi, m["kappa"], vl * vl, m["xi"], m["rho"], T);
        }
        case TipoModelo::Bates: {
            double vi = m["vol_inicial"], vl = m["vol_largo_plazo"];
            return ref::fc_bates(vi * vi, m["kappa"], vl * vl, m["xi"], m["rho"], m["lambda"], m["media_salto"],
                                 m["vol_salto"], T);
        }
        case TipoModelo::NIG: return ref::fc_nig(m["sigma"], m["theta"], m["kappa"], T);
        case TipoModelo::RoughHeston: {
            frac::RoughHeston p;
            p.H = m["hurst"];
            p.V0 = std::pow(m["vol_inicial"], 2.0);
            p.theta = std::pow(m["vol_largo_plazo"], 2.0);
            p.kappa = m["kappa"];
            p.xi = m["xi"];
            p.rho = m["rho"];
            const int N = T <= 1.0 ? 128 : 192;
            return [p, T, N](std::complex<double> u) { return frac::fc_rough_heston(p, u, T, N); };
        }
        case TipoModelo::FMLS: {
            frac::FMLS p;
            p.alpha = m["alpha"];
            p.sigma = m["volatilidad"];
            return [p, T](std::complex<double> u) { return frac::fc_fmls(p, u, T); };
        }
        default: throw std::runtime_error("el modelo " + def_modelo(m.tipo).nombre + " no tiene función característica");
    }
}

const IntegralLewis& PreciosVanilla::lewis(double T) {
    auto it = lewis_.find(T);
    if (it != lewis_.end()) return *it->second;
    // La función característica de rough Heston es cara: Gauss-Legendre de 16 nodos por tramo
    // (sobra para un integrando tan suave) y corte de la cola algo menos estricto
    const bool cara = act_.modelo.tipo == TipoModelo::RoughHeston;
    auto p = std::make_unique<IntegralLewis>(func_car(T), T, cara ? 16 : 32, cara ? 1e-14 : 1e-16);
    const IntegralLewis& ref = *p;
    lewis_.emplace(T, std::move(p));
    return ref;
}

void PreciosVanilla::preparar(const std::vector<double>& vencimientos) {
    if (metodo_ != Metodo::MC) return;
    std::vector<double> nuevos;
    for (double T : vencimientos)
        if (T > 0 && !muestras_.count(T)) nuevos.push_back(T);
    if (!nuevos.empty()) simular(nuevos);
}

const PreciosVanilla::Muestra& PreciosVanilla::muestra(double T) {
    auto it = muestras_.find(T);
    if (it == muestras_.end()) {
        simular({T});
        it = muestras_.find(T);
    }
    return *it->second;
}

void PreciosVanilla::simular(const std::vector<double>& Ts_in) {
    std::vector<double> Ts = Ts_in;
    std::sort(Ts.begin(), Ts.end());
    Ts.erase(std::unique(Ts.begin(), Ts.end()), Ts.end());
    ParamsActivo<double> pa = preparar_activo<double>(act_);
    const bool exacto = modelo_exacto(act_.modelo.tipo);

    // Rejilla: los vencimientos son nodos; entre ellos, pasos de 1/pasos_por_anio
    std::vector<double> t_ini, dt, sqdt, rdt, rcum;
    std::vector<int> marca;
    double t_prev = 0.0;
    for (size_t j = 0; j < Ts.size(); ++j) {
        double delta = Ts[j] - t_prev;
        int m = exacto ? 1 : std::max(1, (int)std::ceil(delta * mc_.pasos_por_anio - 1e-9));
        for (int s = 1; s <= m; ++s) {
            double a = t_prev + delta * (s - 1) / m, b = (s == m) ? Ts[j] : t_prev + delta * s / m;
            t_ini.push_back(a);
            dt.push_back(b - a);
            sqdt.push_back(std::sqrt(b - a));
            rdt.push_back(curva_.integral(b) - curva_.integral(a));
            rcum.push_back(curva_.integral(a));
            marca.push_back(s == m ? (int)j : -1);
        }
        t_prev = Ts[j];
    }
    const int npasos = (int)dt.size();
    const uint32_t dim_paso = 1u + (uint32_t)pa.n_extra;
    std::vector<double> tabla; // modelos rugosos: tabla por pasos del núcleo
    if (modelo_rugoso(act_.modelo.tipo)) {
        preparar_rugoso(pa, act_.modelo, dt, tabla);
        pa.ranura = 0;
    }
    const long long npares = std::max(1LL, mc_.trayectorias / 2);
    const long long n = 2 * npares;
    std::vector<std::vector<double>> S(Ts.size(), std::vector<double>((size_t)n));

    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nh = mc_.hilos > 0 ? mc_.hilos : (int)hw;
    nh = (int)std::min<long long>(nh, std::max(1LL, npares / 256));
    auto trabajo = [&](int h) {
        const long long ini = npares * h / nh, fin = npares * (h + 1) / nh;
        for (long long par = ini; par < fin; ++par) {
            for (int sgn = 0; sgn < 2; ++sgn) {
                const double sg = sgn ? -1.0 : 1.0;
                RuidoPhilox<double> rz;
                rz.iniciar(mc_.semilla, (uint64_t)par, 7u);
                double x = pa.x0, v = estado_aux_inicial(pa);
                double u[MAX_FAC_HD];
                for (int j = 0; j < pa.n_fac; ++j) u[j] = 0.0;
                for (int k = 0; k < npasos; ++k) {
                    const uint32_t base = (uint32_t)k * dim_paso;
                    double w = sg * rz.normal(base);
                    double e0 = 0, e1 = 0, e2 = 0;
                    if (pa.n_extra > 0) e0 = sg * rz.normal(base + 1u);
                    if (pa.n_extra > 1) e1 = sg * rz.normal(base + 2u);
                    if (pa.n_extra > 2) e2 = sg * rz.normal(base + 3u);
                    const double* tb = pa.tab >= 0 ? tabla.data() + pa.tab + (size_t)k * (size_t)pa.tab_paso : nullptr;
                    paso_modelo(pa, x, v, t_ini[(size_t)k], dt[(size_t)k], sqdt[(size_t)k], rdt[(size_t)k],
                                rcum[(size_t)k], w, e0, e1, e2, pa.n_fac > 0 ? u : nullptr, tb);
                    if (marca[(size_t)k] >= 0)
                        S[(size_t)marca[(size_t)k]][(size_t)(2 * par + sgn)] = act_.nivel_inicial * std::exp(x);
                }
            }
        }
    };
    if (nh <= 1) trabajo(0);
    else {
        std::vector<std::thread> th;
        for (int h = 0; h < nh; ++h) th.emplace_back(trabajo, h);
        for (auto& t : th) t.join();
    }

    for (size_t j = 0; j < Ts.size(); ++j) {
        auto mu = std::make_unique<Muestra>();
        mu->T = Ts[j];
        mu->DF = curva_.df(Ts[j]);
        mu->F = forward(Ts[j]);
        // Ajuste del forward al valorar (los precios cumplen exactamente la paridad put-call)
        double media = std::accumulate(S[j].begin(), S[j].end(), 0.0) / (double)n;
        mu->media = (media > 0 && std::isfinite(media)) ? media : mu->F;
        mu->S = std::move(S[j]);
        std::sort(mu->S.begin(), mu->S.end());
        mu->pre.assign(mu->S.size() + 1, 0.0);
        mu->pre2.assign(mu->S.size() + 1, 0.0);
        for (size_t i = 0; i < mu->S.size(); ++i) {
            mu->pre[i + 1] = mu->pre[i] + mu->S[i];
            mu->pre2[i + 1] = mu->pre2[i] + mu->S[i] * mu->S[i];
        }
        muestras_[Ts[j]] = std::move(mu);
    }
}

double PreciosVanilla::call(double T, double K) {
    if (T <= 0) return std::max(S0_ - K, 0.0);
    const double r = r_efectivo(T);
    const Modelo& m = act_.modelo;
    switch (metodo_) {
        case Metodo::BS: return ref::bs_precio(true, S0_, K, T, r, q_, m["volatilidad"]);
        case Metodo::MERTON:
            return ref::merton_precio(true, S0_, K, T, r, q_, m["volatilidad"], m["lambda"], m["media_salto"], m["vol_salto"]);
        case Metodo::FOURIER: return std::max(0.0, lewis(T).call(S0_, K, r, q_));
        case Metodo::HAGAN: return ref::sabr_precio(true, S0_, K, T, r, q_, m["alpha"], m["beta"], m["rho"], m["nu"]);
        default: {
            const Muestra& mu = muestra(T);
            return mu.DF * mu.call_esp(K, mu.F);
        }
    }
}

double PreciosVanilla::put(double T, double K) {
    if (T <= 0) return std::max(K - S0_, 0.0);
    const double r = r_efectivo(T);
    const Modelo& m = act_.modelo;
    switch (metodo_) {
        case Metodo::BS: return ref::bs_precio(false, S0_, K, T, r, q_, m["volatilidad"]);
        case Metodo::MERTON:
            return ref::merton_precio(false, S0_, K, T, r, q_, m["volatilidad"], m["lambda"], m["media_salto"], m["vol_salto"]);
        case Metodo::FOURIER: {
            // Misma corrección de Fourier sobre la put de Black-Scholes (sin cancelaciones)
            const IntegralLewis& L = lewis(T);
            double c = L.call(S0_, K, r, q_);
            double c_bs = ref::bs_precio(true, S0_, K, T, r, q_, L.sigma_ref());
            double p_bs = ref::bs_precio(false, S0_, K, T, r, q_, L.sigma_ref());
            return std::max(0.0, p_bs + (c - c_bs));
        }
        case Metodo::HAGAN: return ref::sabr_precio(false, S0_, K, T, r, q_, m["alpha"], m["beta"], m["rho"], m["nu"]);
        default: {
            const Muestra& mu = muestra(T);
            return mu.DF * mu.put_esp(K, mu.F);
        }
    }
}

double PreciosVanilla::call_fwd(double T, double K, double F, double DF) {
    if (T <= 0) return DF * std::max(F - K, 0.0);
    const Modelo& m = act_.modelo;
    switch (metodo_) {
        case Metodo::BS: return black_fwd(true, F, K, T, DF, m["volatilidad"]);
        case Metodo::FOURIER: return std::max(0.0, lewis(T).call_fwd(F, K, DF));
        case Metodo::MC: return DF * muestra(T).call_esp(K, F);
        default: {
            // Merton / Hagan: precio con r y q equivalentes a (F, DF)
            const double r = -std::log(DF) / T, q = r - std::log(F / S0_) / T;
            if (metodo_ == Metodo::MERTON)
                return ref::merton_precio(true, S0_, K, T, r, q, m["volatilidad"], m["lambda"], m["media_salto"], m["vol_salto"]);
            return ref::sabr_precio(true, S0_, K, T, r, q, m["alpha"], m["beta"], m["rho"], m["nu"]);
        }
    }
}

double PreciosVanilla::put_fwd(double T, double K, double F, double DF) {
    if (T <= 0) return DF * std::max(K - F, 0.0);
    const Modelo& m = act_.modelo;
    switch (metodo_) {
        case Metodo::BS: return black_fwd(false, F, K, T, DF, m["volatilidad"]);
        case Metodo::FOURIER: return std::max(0.0, lewis(T).put_fwd(F, K, DF));
        case Metodo::MC: return DF * muestra(T).put_esp(K, F);
        default: {
            const double r = -std::log(DF) / T, q = r - std::log(F / S0_) / T;
            if (metodo_ == Metodo::MERTON)
                return ref::merton_precio(false, S0_, K, T, r, q, m["volatilidad"], m["lambda"], m["media_salto"], m["vol_salto"]);
            return ref::sabr_precio(false, S0_, K, T, r, q, m["alpha"], m["beta"], m["rho"], m["nu"]);
        }
    }
}

double PreciosVanilla::error_mc(double T, double K, bool es_call) {
    if (metodo_ != Metodo::MC || T <= 0) return -1.0;
    const Muestra& mu = muestra(T);
    return mu.DF * mu.desv(K, mu.F, es_call) / std::sqrt((double)mu.S.size());
}

double PreciosVanilla::digital(double T, double K, bool es_call) {
    if (T <= 0) return es_call ? (S0_ >= K ? 1.0 : 0.0) : (S0_ < K ? 1.0 : 0.0);
    const double r = r_efectivo(T);
    double dc;
    switch (metodo_) {
        case Metodo::BS: dc = ref::bs_digital(true, S0_, K, T, r, q_, act_.modelo["volatilidad"]); break;
        case Metodo::FOURIER: dc = lewis(T).digital(S0_, K, r, q_); break;
        case Metodo::MC: {
            const Muestra& mu = muestra(T);
            dc = mu.DF * mu.prob_mayor_igual(K, mu.F);
        } break;
        default: {
            const double h = 1e-4 * K;
            dc = (call(T, K - h) - call(T, K + h)) / (2.0 * h);
        } break;
    }
    dc = std::min(std::max(dc, 0.0), df(T));
    return es_call ? dc : df(T) - dc;
}

// ===================================================================== //
// Volatilidad implícita                                                  //
// ===================================================================== //

namespace {
// Precio de Black normalizado de una opción fuera del dinero: x = ln(F/K),
// s = σ√T, θ = +1 call (x ≤ 0) o -1 put (x ≥ 0); b = precio / (DF·√(F·K))
double b_norm(double x, double s, double th) {
    if (s <= 0) return std::max(th * (std::exp(x / 2) - std::exp(-x / 2)), 0.0);
    const double d1 = x / s + s / 2, d2 = x / s - s / 2;
    return th * (std::exp(x / 2) * ncdf(th * d1) - std::exp(-x / 2) * ncdf(th * d2));
}
double vega_norm(double x, double s) { return std::exp(x / 2) * npdf(x / s + s / 2); }
} // namespace

double vol_implicita(double precio, bool call, double S, double K, double T, double r, double q, bool* ok) {
    if (ok) *ok = false;
    if (!(T > 0) || !(S > 0) || !(K > 0) || !std::isfinite(precio)) return NAN;
    const double DF = std::exp(-r * T), F = S * std::exp((r - q) * T);
    double c = precio / DF; // sin descontar
    // Se pasa a la opción fuera del dinero por paridad
    if (call && F > K) { c -= (F - K); call = false; }
    else if (!call && F < K) { c -= (K - F); call = true; }
    const double x = std::log(F / K);
    const double th = call ? 1.0 : -1.0;
    const double beta = c / std::sqrt(F * K);
    const double bmax = call ? std::exp(x / 2) : std::exp(-x / 2);
    if (!(beta > 0) || !(beta < bmax)) return NAN;

    double s;
    if (std::fabs(x) < 1e-14) {
        // En el dinero: b = 2Φ(s/2) - 1 se invierte en forma cerrada
        s = 2.0 * hd::inv_norm_as241(0.5 * (1.0 + beta));
    } else {
        const double sc = std::sqrt(2.0 * std::fabs(x)); // punto de inflexión de b(s)
        const bool alas = beta < b_norm(x, sc, th);       // región convexa: Newton sobre ln b
        double lo = 0.0, hi = sc;
        while (b_norm(x, hi, th) < beta && hi < 100.0) { lo = hi; hi *= 2.0; }
        if (b_norm(x, hi, th) < beta) return NAN;
        if (alas) hi = sc; else lo = sc;
        s = sc;
        for (int it = 0; it < 100; ++it) {
            const double b = b_norm(x, s, th), v = vega_norm(x, s);
            if (std::fabs(b - beta) <= 1e-15 * beta) break; // convergido (antes de tocar el intervalo)
            if (b > beta) hi = std::min(hi, s); else lo = std::max(lo, s);
            double paso;
            if (alas) paso = (b > 0 && v > 0) ? (std::log(b) - std::log(beta)) * b / v : NAN;
            else paso = v > 0 ? (b - beta) / v : NAN;
            double sn = s - paso;
            if (!std::isfinite(sn) || sn <= lo || sn >= hi) sn = 0.5 * (lo + hi); // salvaguarda: bisección
            if (std::fabs(sn - s) <= 1e-15 * std::max(1.0, s)) { s = sn; break; }
            s = sn;
        }
    }
    if (ok) *ok = std::isfinite(s) && s > 0;
    return s / std::sqrt(T);
}

} // namespace est
