#include "historico.hpp"
#include "optimizacion.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <ostream>
#include <stdexcept>

namespace est {

namespace {

constexpr double PI = 3.14159265358979323846;

std::string clave(const std::string& s0) {
    std::string s;
    for (unsigned char c : s0)
        if (std::isalnum(c)) s += (char)std::tolower(c);
    return s;
}
std::string recortar(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    std::string r = s.substr(a, b - a);
    if (r.size() >= 2 && r.front() == '"' && r.back() == '"') r = r.substr(1, r.size() - 2);
    return r;
}
std::vector<std::string> dividir(const std::string& l, char sep) {
    std::vector<std::string> out;
    std::string cur;
    bool q = false;
    for (char c : l) {
        if (c == '"') { q = !q; cur += c; }
        else if (c == sep && !q) { out.push_back(recortar(cur)); cur.clear(); }
        else cur += c;
    }
    out.push_back(recortar(cur));
    return out;
}

// Pendiente de una regresión lineal y su error estándar
void regresion(const std::vector<double>& x, const std::vector<double>& y, double* b, double* se) {
    const double n = (double)x.size();
    double mx = std::accumulate(x.begin(), x.end(), 0.0) / n, my = std::accumulate(y.begin(), y.end(), 0.0) / n;
    double sxx = 0, sxy = 0;
    for (size_t i = 0; i < x.size(); ++i) { sxx += (x[i] - mx) * (x[i] - mx); sxy += (x[i] - mx) * (y[i] - my); }
    *b = sxy / sxx;
    double a = my - *b * mx, s2 = 0;
    for (size_t i = 0; i < x.size(); ++i) { double e = y[i] - a - *b * x[i]; s2 += e * e; }
    *se = n > 2 ? std::sqrt(s2 / (n - 2) / sxx) : -1;
}

// Exponente de Hurst por el variograma: E|X(t+Δ) - X(t)|² ∝ Δ^{2H}, con retardos en
// escala geométrica entre 1 y lag_max (regresión log-log). Solo el valor de H.
bool hurst_pendiente(const std::vector<double>& x, size_t ini, size_t fin, int lag_max, double* H) {
    std::vector<int> lags;
    for (int i = 0; i < 12; ++i) {
        int d = (int)std::lround(std::exp(std::log((double)lag_max) * i / 11.0));
        if (lags.empty() || d > lags.back()) lags.push_back(d);
    }
    std::vector<double> lx, ly;
    for (int d : lags) {
        double sm = 0;
        int n = 0;
        for (size_t t = ini; t + (size_t)d < fin; ++t) { double z = x[t + (size_t)d] - x[t]; sm += z * z; ++n; }
        if (n < 10 || !(sm > 0)) return false;
        lx.push_back(std::log((double)d));
        ly.push_back(std::log(sm / n));
    }
    if (lx.size() < 3) return false;
    double b = 0, e = 0;
    regresion(lx, ly, &b, &e);
    *H = b / 2.0;
    return std::isfinite(*H);
}

// H con todos los datos y su error estándar por subperíodos: H en K tramos
// consecutivos; error ≈ desviación entre tramos / √K (no supone independencia
// entre retardos, que en el variograma están muy correlacionados)
bool hurst_variograma(const std::vector<double>& x, int lag_max, double* H, double* se) {
    if ((int)x.size() < 10 * lag_max || lag_max < 3) return false;
    if (!hurst_pendiente(x, 0, x.size(), lag_max, H)) return false;
    const int K = 8;
    const size_t largo = x.size() / K;
    const int lag_sub = std::max(3, std::min(lag_max, (int)largo / 10));
    std::vector<double> hs;
    for (int k = 0; k < K; ++k) {
        double h = 0;
        if (largo >= 40 && hurst_pendiente(x, k * largo, (k + 1) * largo, lag_sub, &h)) hs.push_back(h);
    }
    *se = -1;
    if (hs.size() >= 4) {
        double m = std::accumulate(hs.begin(), hs.end(), 0.0) / hs.size(), v = 0;
        for (double h : hs) v += (h - m) * (h - m);
        *se = std::sqrt(v / (hs.size() - 1)) / std::sqrt((double)hs.size());
    }
    return true;
}

double loglik_normal(const std::vector<double>& r, double m, double v) {
    double ll = 0;
    for (double x : r) ll += -0.5 * std::log(2 * PI * v) - (x - m) * (x - m) / (2 * v);
    return ll;
}

double loglik_merton(const std::vector<double>& r, double dt, double mu, double s, double lam, double mj, double sj) {
    const double kap = std::exp(mj + 0.5 * sj * sj) - 1.0;
    const double m0 = (mu - 0.5 * s * s - lam * kap) * dt, ld = lam * dt;
    std::vector<double> pk, mk, vk;
    double p = std::exp(-ld);
    for (int k = 0; k < 40; ++k) {
        if (k > 0) p *= ld / k;
        pk.push_back(p);
        mk.push_back(m0 + k * mj);
        vk.push_back(s * s * dt + k * sj * sj);
        if (k > ld && p < 1e-12) break;
    }
    double ll = 0;
    for (double x : r) {
        double d = 0;
        for (size_t k = 0; k < pk.size(); ++k)
            d += pk[k] * std::exp(-(x - mk[k]) * (x - mk[k]) / (2 * vk[k])) / std::sqrt(2 * PI * vk[k]);
        if (!(d > 0)) return -std::numeric_limits<double>::infinity();
        ll += std::log(d);
    }
    return ll;
}

// P(χ²₃ > x)
double chi2_3_cola(double x) {
    if (x <= 0) return 1.0;
    return std::erfc(std::sqrt(x / 2.0)) + std::sqrt(2.0 * x / PI) * std::exp(-x / 2.0);
}

} // namespace

SerieHistorica leer_serie_csv(const std::string& ruta) {
    std::ifstream f(ruta);
    if (!f) throw std::runtime_error("no se puede abrir la serie de precios \"" + ruta + "\"");
    SerieHistorica s;
    s.ruta = ruta;
    std::string l;
    std::vector<std::string> filas;
    while (std::getline(f, l)) {
        if (!l.empty() && l.back() == '\r') l.pop_back();
        if (filas.empty() && l.size() >= 3 && (unsigned char)l[0] == 0xEF) l = l.substr(3);
        std::string t = recortar(l);
        if (t.empty()) continue;
        if (t[0] == '#') {
            size_t dp = t.find(':');
            if (dp != std::string::npos && clave(t.substr(1, dp - 1)) == "subyacente") s.nombre = recortar(t.substr(dp + 1));
            continue;
        }
        filas.push_back(l);
    }
    if (filas.size() < 3) throw std::runtime_error("la serie \"" + ruta + "\" tiene muy pocos datos");
    const std::string& cab = filas[0];
    char sep = ',';
    if (std::count(cab.begin(), cab.end(), ';') > std::count(cab.begin(), cab.end(), ',')) sep = ';';
    else if (std::count(cab.begin(), cab.end(), '\t') > std::count(cab.begin(), cab.end(), ',')) sep = '\t';
    std::vector<std::string> cols = dividir(cab, sep);
    int c_fecha = -1, c_precio = -1, prioridad = 99;
    static const char* nombres_precio[] = {"adjclose", "cierreajustado", "close", "cierre", "precio", "price", "valor",
                                           "value", "ultimo", "last"};
    for (size_t i = 0; i < cols.size(); ++i) {
        std::string k = clave(cols[i]);
        if (c_fecha < 0 && (k == "fecha" || k == "date" || k == "dia" || k == "day" || k == "datetime")) c_fecha = (int)i;
        for (int j = 0; j < 10; ++j)
            if (k == nombres_precio[j] && j < prioridad) { prioridad = j; c_precio = (int)i; }
    }
    size_t ini = 1;
    if (c_precio < 0) {
        // Sin cabecera reconocible: una sola columna numérica (o fecha, precio)
        bool ok = false;
        texto_a_real(cols.back(), &ok);
        if (!ok) throw std::runtime_error("la serie \"" + ruta + "\" no tiene una columna de precio reconocible (cierre, close, precio...)");
        c_precio = (int)cols.size() - 1;
        c_fecha = cols.size() > 1 ? 0 : -1;
        ini = 0;
    }
    std::vector<std::pair<int, double>> obs;
    const bool coma = (sep != ',');
    for (size_t i = ini; i < filas.size(); ++i) {
        std::vector<std::string> v = dividir(filas[i], sep);
        if ((int)v.size() <= c_precio) continue;
        std::string tp = v[(size_t)c_precio];
        if (coma) std::replace(tp.begin(), tp.end(), ',', '.');
        bool ok = false;
        double p = texto_a_real(tp, &ok);
        if (!ok || !(p > 0)) continue;
        int d = -1;
        if (c_fecha >= 0 && c_fecha < (int)v.size() && !texto_a_fecha(v[(size_t)c_fecha].substr(0, 10), d)) d = -1;
        obs.emplace_back(d, p);
    }
    const bool con_fechas = std::all_of(obs.begin(), obs.end(), [](const auto& x) { return x.first >= 0; });
    if (con_fechas) std::stable_sort(obs.begin(), obs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [d, p] : obs) { s.dias.push_back(con_fechas ? d : -1); s.precios.push_back(p); }
    if (s.precios.size() < 30) throw std::runtime_error("la serie \"" + ruta + "\" tiene menos de 30 precios válidos");
    return s;
}

ResultadoHistorico estimar_historico(const SerieHistorica& s, const std::string& modelo) {
    ResultadoHistorico R;
    std::vector<double> r;
    for (size_t i = 1; i < s.precios.size(); ++i) r.push_back(std::log(s.precios[i] / s.precios[i - 1]));
    R.n = (int)r.size();
    // Años cubiertos: por fechas si las hay; si no, 252 observaciones por año
    if (!s.dias.empty() && s.dias.front() >= 0 && s.dias.back() > s.dias.front())
        R.anios = (s.dias.back() - s.dias.front()) / 365.25;
    else
        R.anios = R.n / 252.0;
    R.obs_por_anio = R.n / R.anios;
    const double dt = 1.0 / R.obs_por_anio;
    const double n = (double)R.n;
    double m = std::accumulate(r.begin(), r.end(), 0.0) / n, m2 = 0, m3 = 0, m4 = 0;
    for (double x : r) { double d = x - m; m2 += d * d; m3 += d * d * d; m4 += d * d * d * d; }
    const double var = m2 / (n - 1);
    m2 /= n; m3 /= n; m4 /= n;
    R.asimetria = m3 / std::pow(m2, 1.5);
    R.curtosis = m4 / (m2 * m2) - 3.0;
    R.vol = std::sqrt(var / dt);
    R.media = m / dt + 0.5 * R.vol * R.vol;
    R.loglik_bs = loglik_normal(r, m, m2);
    R.bs = {{"mu", R.media, R.vol / std::sqrt(R.anios), true}, {"volatilidad", R.vol, R.vol / std::sqrt(2.0 * n), true}};

    // ---------------- Merton por máxima verosimilitud ----------------
    // θ = (μ, ln σ, ln λ, μ_J, ln δ)
    auto natural = [&](const std::vector<double>& t) {
        return std::vector<double>{t[0], std::exp(t[1]), std::exp(t[2]), t[3], std::exp(t[4])};
    };
    auto nll = [&](const std::vector<double>& t) {
        auto p = natural(t);
        if (p[1] > 5 || p[2] > 2000 || p[4] > 2 || std::fabs(p[3]) > 2) return std::numeric_limits<double>::infinity();
        double ll = loglik_merton(r, dt, p[0], p[1], p[2], p[3], p[4]);
        return std::isfinite(ll) ? -ll : std::numeric_limits<double>::infinity();
    };
    // Búsqueda amplia y barata desde varios puntos y refinado del mejor (con
    // reinicios del símplex, que evitan parones prematuros)
    opt::Resultado mejor;
    mejor.f = std::numeric_limits<double>::infinity();
    opt::OpcionesNM corta;
    corta.max_eval = 400;
    corta.paso_inicial = 0.3;
    corta.tol_f = 1e-9;
    for (double lam : {3.0, 25.0})
        for (double mj : {-0.03, 0.0})
            for (double sj : {0.015, 0.05}) {
                std::vector<double> t0 = {R.media, std::log(0.8 * R.vol), std::log(lam), mj, std::log(sj)};
                if (!std::isfinite(nll(t0))) continue;
                opt::Resultado a = opt::nelder_mead(nll, t0, corta);
                if (a.f < mejor.f) mejor = a;
            }
    if (std::isfinite(mejor.f)) {
        opt::OpcionesNM larga;
        larga.max_eval = 2500;
        larga.paso_inicial = 0.15;
        for (int rep = 0; rep < 3; ++rep) {
            opt::Resultado a = opt::nelder_mead(nll, mejor.x, larga);
            const bool igual = std::fabs(a.f - mejor.f) < 1e-9 * std::max(1.0, std::fabs(mejor.f));
            if (a.f < mejor.f) mejor = a;
            if (igual) break;
        }
    }
    if (std::isfinite(mejor.f)) {
        R.merton_ok = true;
        R.loglik_merton = -mejor.f;
        R.razon_verosimilitud = std::max(0.0, 2.0 * (R.loglik_merton - R.loglik_bs));
        R.p_valor = chi2_3_cola(R.razon_verosimilitud);
        auto p = natural(mejor.x);
        // Errores estándar: hessiana numérica de -log L en los parámetros naturales
        auto nll_nat = [&](const std::vector<double>& q) {
            double ll = loglik_merton(r, dt, q[0], q[1], q[2], q[3], q[4]);
            return std::isfinite(ll) ? -ll : 1e300;
        };
        const int k = 5;
        std::vector<double> H((size_t)(k * k)), h((size_t)k);
        for (int i = 0; i < k; ++i) h[(size_t)i] = 1e-4 * std::max(std::fabs(p[(size_t)i]), i == 2 ? 1.0 : 0.01);
        const double f0 = nll_nat(p);
        for (int i = 0; i < k; ++i)
            for (int j = 0; j <= i; ++j) {
                auto q = p;
                double v;
                if (i == j) {
                    q[(size_t)i] += h[(size_t)i];
                    double fp = nll_nat(q);
                    q[(size_t)i] -= 2 * h[(size_t)i];
                    double fm = nll_nat(q);
                    v = (fp - 2 * f0 + fm) / (h[(size_t)i] * h[(size_t)i]);
                } else {
                    double s4[4];
                    int idx = 0;
                    for (int a : {1, -1})
                        for (int b : {1, -1}) {
                            q = p;
                            q[(size_t)i] += a * h[(size_t)i];
                            q[(size_t)j] += b * h[(size_t)j];
                            s4[idx++] = nll_nat(q);
                        }
                    v = (s4[0] - s4[1] - s4[2] + s4[3]) / (4 * h[(size_t)i] * h[(size_t)j]);
                }
                H[(size_t)(i * k + j)] = H[(size_t)(j * k + i)] = v;
            }
        std::vector<double> inv;
        bool hay_err = opt::invertir_spd(H, k, inv);
        const char* claves[] = {"mu", "volatilidad", "lambda", "media_salto", "vol_salto"};
        const bool pct[] = {true, true, false, false, false};
        for (int i = 0; i < k; ++i)
            R.merton.push_back({claves[i], p[(size_t)i], hay_err ? std::sqrt(std::max(0.0, inv[(size_t)(i * k + i)])) : -1, pct[i]});
        if (!hay_err) R.notas.push_back("la hessiana de Merton no es definida positiva: los saltos no están bien identificados");
    }

    // ---------------- Exponentes de Hurst ----------------
    {
        std::vector<double> x(1, 0.0);
        for (double v : r) x.push_back(x.back() + v);
        int L = std::min(20, R.n / 20);
        (void)hurst_variograma(x, L, &R.hurst_rend, &R.hurst_rend_error);
        // Log-varianza realizada por bloques de 5 observaciones (semanal con datos diarios)
        const int w = 5;
        std::vector<double> lrv;
        for (size_t b = 0; b + w <= r.size(); b += w) {
            double rv = 0;
            for (int j = 0; j < w; ++j) rv += r[b + (size_t)j] * r[b + (size_t)j];
            if (rv > 0) lrv.push_back(0.5 * std::log(rv));
        }
        R.ventana_vol = w;
        int Lv = std::min(10, (int)lrv.size() / 12);
        if (!hurst_variograma(lrv, Lv, &R.hurst_vol, &R.hurst_vol_error)) R.hurst_vol = -1;
    }

    // ---------------- Modelo elegido ----------------
    bool usa_merton = (modelo == "merton") || (modelo == "auto" && R.merton_ok && R.p_valor < 0.01);
    if (modelo == "merton" && !R.merton_ok) throw std::runtime_error("no se pudo estimar el modelo de Merton");
    if (usa_merton) {
        R.modelo.tipo = TipoModelo::Merton;
        for (const auto& d : def_modelo(TipoModelo::Merton).params) R.modelo.valores.push_back(d.defecto);
        R.modelo["volatilidad"] = R.merton[1].valor;
        R.modelo["lambda"] = R.merton[2].valor;
        R.modelo["media_salto"] = R.merton[3].valor;
        R.modelo["vol_salto"] = R.merton[4].valor;
    } else {
        R.modelo.tipo = TipoModelo::BlackScholes;
        for (const auto& d : def_modelo(TipoModelo::BlackScholes).params) R.modelo.valores.push_back(d.defecto);
        R.modelo["volatilidad"] = R.vol;
    }
    if (modelo == "auto")
        R.notas.push_back(usa_merton ? "los saltos mejoran claramente el ajuste (p < 1%): se propone Merton"
                                     : "los saltos no mejoran el ajuste de forma significativa: se propone Black-Scholes");
    return R;
}

void imprimir_historico(std::ostream& os, const SerieHistorica& s, const ResultadoHistorico& r) {
    const std::string L(86, '=');
    os << "\n" << L << "\n ESTIMACIÓN HISTÓRICA" << (s.nombre.empty() ? "" : ": " + s.nombre) << "\n" << L << "\n";
    os << " Serie ........ " << s.ruta << ": " << s.precios.size() << " precios, " << r.n << " rendimientos en "
       << fmt(r.anios, 2) << " años (" << fmt(r.obs_por_anio, 0) << " por año)\n";
    os << " Rendimientos . media anual " << porcentaje(r.media, 2) << ", volatilidad anual " << porcentaje(r.vol, 2)
       << ", asimetría " << fmt(r.asimetria, 3) << ", exceso de curtosis " << fmt(r.curtosis, 3) << "\n";
    auto tabla = [&](const char* titulo, const std::vector<EstimacionParam>& ps) {
        os << "\n " << titulo << "\n";
        for (const auto& p : ps) {
            auto f = [&](double x) { return p.porcentaje ? porcentaje(x, 3) : fmt(x, 5); };
            os << "   " << p.clave << std::string(p.clave.size() < 16 ? 16 - p.clave.size() : 1, ' ') << f(p.valor);
            if (p.error >= 0) os << "   (error est. " << f(p.error) << ")";
            os << "\n";
        }
    };
    tabla("Black-Scholes (máxima verosimilitud)", r.bs);
    os << "   log-verosimilitud " << fmt(r.loglik_bs, 2) << "\n";
    if (r.merton_ok) {
        tabla("Merton (difusión con saltos, máxima verosimilitud)", r.merton);
        os << "   log-verosimilitud " << fmt(r.loglik_merton, 2) << "; razón de verosimilitudes frente a BS "
           << fmt(r.razon_verosimilitud, 2) << " (p = " << fmt(r.p_valor, 4) << ", χ² con 3 g.l.)\n";
    }
    if (r.hurst_rend >= 0 || r.hurst_vol >= 0) {
        os << "\n Exponentes de Hurst (variograma)\n";
        if (r.hurst_rend >= 0)
            os << "   del precio (log) ............ H = " << fmt(r.hurst_rend, 3)
               << (r.hurst_rend_error >= 0 ? " ± " + fmt(r.hurst_rend_error, 3) : std::string())
               << "   (0.5 = browniano; > 0.5 memoria larga, < 0.5 reversión)\n";
        if (r.hurst_vol >= 0)
            os << "   de la log-volatilidad ....... H = " << fmt(r.hurst_vol, 3)
               << (r.hurst_vol_error >= 0 ? " ± " + fmt(r.hurst_vol_error, 3) : std::string())
               << "   (con varianza realizada por bloques de " << r.ventana_vol << " observaciones; "
               << "la volatilidad rugosa da H ≈ 0.1)\n";
        // (H ≈ 0 indica que la log-volatilidad realizada es solo ruido de medida: volatilidad constante)
        if (r.hurst_vol > 0.04 && r.hurst_vol < 0.3 &&
            (r.hurst_vol_error < 0 || r.hurst_vol + 2.0 * r.hurst_vol_error < 0.5))
            os << "   → volatilidad rugosa: los modelos rough_heston y rough_bergomi usan este H directamente\n"
                  "     (\"hurst\"; ver estructurados/FRACCIONAL.md y 'estructurados fraccional')\n";
    }
    for (const auto& n : r.notas) os << "   · " << n << "\n";
    os << L << "\n Modelo propuesto (medida real; para valorar, la deriva será r - q):\n   ";
    json::Valor o = json::Valor::objeto();
    const DefModelo& d = def_modelo(r.modelo.tipo);
    o.poner("tipo", d.clave);
    for (size_t i = 0; i < d.params.size(); ++i) {
        if (d.params[i].clave == "vol_max" || d.params[i].clave == "factores") continue;
        if (d.params[i].porcentaje) o.poner(d.params[i].clave, porcentaje(r.modelo.valores[i], 4));
        else o.poner(d.params[i].clave, std::round(r.modelo.valores[i] * 1e6) / 1e6);
    }
    os << json::escribir(o, 0) << "\n";
}

} // namespace est
