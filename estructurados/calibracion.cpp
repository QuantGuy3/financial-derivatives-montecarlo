#include "calibracion.hpp"
#include "optimizacion.hpp"
#include "vanilla.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <ostream>
#include <random>
#include <stdexcept>

namespace est {

namespace {

using Reloj = std::chrono::steady_clock;

size_t ancho_visual(const std::string& s) {
    size_t v = 0;
    for (unsigned char c : s) v += ((c & 0xC0) != 0x80);
    return v;
}
std::string col(std::string s, size_t w) {
    size_t v = ancho_visual(s);
    if (v < w) s += std::string(w - v, ' ');
    return s;
}
std::string dcha(std::string s, size_t w) {
    size_t v = ancho_visual(s);
    if (v < w) s = std::string(w - v, ' ') + s;
    return s;
}
std::string linea(char c = '-', int n = 86) { return std::string((size_t)n, c); }

struct Cota { double lo, hi; };

// Rango de búsqueda de cada parámetro en la calibración (más estrecho que el
// rango de validez del catálogo: fuera de él los precios dejan de ser fiables)
Cota cota_calibracion(TipoModelo t, const DefParam& d) {
    const std::string& k = d.clave;
    if (k == "rho") return {-0.999, 0.999};
    if (d.porcentaje) return {0.005, 2.0};
    if (k == "kappa") return t == TipoModelo::NIG ? Cota{0.001, 10.0} : Cota{0.01, 20.0};
    if (k == "xi") return {0.01, 5.0};
    if (k == "lambda") return {1e-4, 5.0};
    if (k == "media_salto") return {-1.0, 0.5};
    if (k == "vol_salto") return {0.005, 1.0};
    if (k == "beta") return t == TipoModelo::SABR ? Cota{0.0, 1.0} : Cota{-1.0, 2.5};
    if (k == "alpha") return {-3.0, 3.0};
    if (k == "nu") return {0.001, 5.0};
    if (k == "theta") return {-2.0, 2.0};
    double lo = std::max(d.min, -10.0), hi = std::min(d.max, 10.0);
    if (d.min_abierto) lo += 1e-6 * std::max(1.0, std::fabs(lo));
    return {lo, hi};
}

double a_x(double u, const Cota& c) { return c.lo + (c.hi - c.lo) / (1.0 + std::exp(-u)); }
double a_u(double x, const Cota& c) {
    const double eps = 1e-9 * (c.hi - c.lo);
    x = std::min(std::max(x, c.lo + eps), c.hi - eps);
    return std::log((x - c.lo) / (c.hi - x));
}
double dxdu(double u, const Cota& c) {
    const double s = 1.0 / (1.0 + std::exp(-u));
    return (c.hi - c.lo) * s * (1.0 - s);
}

// Restricciones del modelo que no recoge el rango de cada parámetro
bool modelo_admisible(const Modelo& mod) {
    if (mod.tipo == TipoModelo::NIG) {
        const double s = mod["sigma"], th = mod["theta"], k = mod["kappa"];
        return 1.0 - s * s * k - 2.0 * th * k > 1e-6;
    }
    return true;
}

// Precio del modelo en cada punto (con el forward y el descuento del mercado)
bool precios_modelo(const Modelo& mod, const DatosMercado& m, const OpcionesCalibracion& o, std::vector<double>& out,
                    std::string* metodo = nullptr) {
    if (!modelo_admisible(mod)) return false;
    Activo a;
    a.nombre = m.subyacente.empty() ? "X" : m.subyacente;
    a.spot = a.nivel_inicial = m.spot;
    a.dividendos = m.q;
    a.modelo = mod;
    Curva c;
    c.t = {1.0};
    c.r = {m.r};
    OpcionesMCVanilla mc;
    mc.trayectorias = o.trayectorias_mc;
    mc.pasos_por_anio = o.pasos_mc;
    mc.semilla = o.semilla;
    PreciosVanilla pv(a, c, mc, false);
    if (metodo) *metodo = pv.metodo();
    if (pv.es_mc()) {
        std::vector<double> Ts;
        for (const auto& v : m.vencimientos) Ts.push_back(v.T);
        pv.preparar(Ts);
    }
    out.resize(m.puntos.size());
    try {
        for (size_t i = 0; i < m.puntos.size(); ++i) {
            const PuntoMercado& p = m.puntos[i];
            const VencimientoMercado& v = m.vencimientos[(size_t)p.v];
            out[i] = p.call ? pv.call_fwd(p.T, p.K, v.F, v.DF) : pv.put_fwd(p.T, p.K, v.F, v.DF);
            if (!std::isfinite(out[i])) return false;
        }
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

double vol_de_precio(const PuntoMercado& p, const VencimientoMercado& v, double precio) {
    return vol_implicita_fwd(precio, p.call, v.F, p.K, p.T, v.DF);
}

std::vector<double> pesos_de(const DatosMercado& m, const std::string& modo) {
    std::vector<double> w(m.puntos.size(), 1.0);
    for (size_t i = 0; i < w.size(); ++i) {
        const PuntoMercado& p = m.puntos[i];
        if (modo == "vega") w[i] = p.vega / std::max(1e-300, m.vencimientos[(size_t)p.v].DF * m.vencimientos[(size_t)p.v].F);
        else if (modo == "horquilla" && p.vol_bid > 0 && p.vol_ask > p.vol_bid) {
            double h = std::max(p.vol_ask - p.vol_bid, 1e-4);
            w[i] = 1.0 / (h * h);
        }
        w[i] *= p.peso;
    }
    double media = 0;
    for (double x : w) media += x;
    media /= (double)w.size();
    if (media > 0) for (double& x : w) x /= media;
    return w;
}

// ------------------------------------------------------------------ //
// SVI (Gatheral 2004): w(k) = a + b·(ρ(k-m) + √((k-m)² + σ²))          //
// ------------------------------------------------------------------ //
double svi_w(const double* p, double k) {
    const double d = k - p[3];
    return p[0] + p[1] * (p[2] * d + std::sqrt(d * d + p[4] * p[4]));
}
// g(k) de la densidad (Gatheral y Jacquier, 2014): g < 0 ⇒ arbitraje de mariposa
double svi_g(const double* p, double k) {
    const double d = k - p[3], r = std::sqrt(d * d + p[4] * p[4]);
    const double w = svi_w(p, k), w1 = p[1] * (p[2] + d / r), w2 = p[1] * p[4] * p[4] / (r * r * r);
    if (!(w > 0)) return -1.0;
    const double t = 1.0 - k * w1 / (2.0 * w);
    return t * t - w1 * w1 / 4.0 * (1.0 / w + 0.25) + w2 / 2.0;
}

ResultadoCalibracion calibrar_svi(const DatosMercado& m, const OpcionesCalibracion& o) {
    ResultadoCalibracion R;
    R.modelo_clave = "svi";
    R.modelo_nombre = "SVI (parametrización de la varianza total por vencimiento)";
    R.metodo_precios = "fórmula de Black con la volatilidad SVI";
    R.optimizador = "Levenberg-Marquardt por vencimiento desde varios puntos de partida";
    const auto t0 = Reloj::now();
    R.vol_modelo.assign(m.puntos.size(), 0.0);
    R.precio_modelo.assign(m.puntos.size(), 0.0);
    const std::vector<double> w = pesos_de(m, o.pesos);
    for (size_t v = 0; v < m.vencimientos.size(); ++v) {
        const VencimientoMercado& vm = m.vencimientos[v];
        std::vector<double> k, iv, wi;
        double W = 0, wmin = 1e300, kmin = 0;
        for (int i : vm.puntos) {
            const PuntoMercado& p = m.puntos[(size_t)i];
            k.push_back(std::log(p.K / vm.F));
            iv.push_back(p.vol);
            wi.push_back(w[(size_t)i]);
            double tv = p.vol * p.vol * vm.T;
            W = std::max(W, tv);
            if (tv < wmin) { wmin = tv; kmin = k.back(); }
        }
        AjusteSVI aj;
        if (k.size() < 5) {
            // Pocos puntos: SVI degenerado (varianza plana)
            aj.a = wmin; aj.b = 0; aj.rho = 0; aj.m = 0; aj.sigma = 0.1;
            R.notas.push_back("vencimiento " + vm.etiqueta + ": menos de 5 strikes, se ajusta una varianza plana");
        } else {
            const Cota ca{-W, 2.0 * W}, cb{1e-6, 2.0}, cr{-0.999, 0.999}, cm{-1.0, 1.0}, cs{1e-4, 2.0};
            const Cota cotas[5] = {ca, cb, cr, cm, cs};
            auto residuos = [&](const std::vector<double>& u, std::vector<double>& r) {
                double p[5];
                for (int j = 0; j < 5; ++j) p[j] = a_x(u[(size_t)j], cotas[j]);
                if (p[0] + p[1] * p[4] * std::sqrt(1.0 - p[2] * p[2]) < 0) return false; // varianza negativa
                r.resize(k.size());
                for (size_t i = 0; i < k.size(); ++i) {
                    double tv = svi_w(p, k[i]);
                    if (!(tv > 0)) return false;
                    r[i] = std::sqrt(wi[i]) * (std::sqrt(tv / vm.T) - iv[i]);
                }
                return true;
            };
            opt::Resultado mejor;
            mejor.f = std::numeric_limits<double>::infinity();
            for (double sg : {0.05, 0.15, 0.4})
                for (double rh : {-0.6, -0.2, 0.3}) {
                    double b0 = 0.1, a0 = std::max(wmin - b0 * sg * std::sqrt(1 - rh * rh), -0.5 * W);
                    std::vector<double> u0 = {a_u(a0, ca), a_u(b0, cb), a_u(rh, cr), a_u(kmin, cm), a_u(sg, cs)};
                    std::vector<double> rr;
                    if (!residuos(u0, rr)) continue;
                    opt::Resultado r = opt::levenberg_marquardt(residuos, u0);
                    R.evaluaciones += r.evaluaciones;
                    if (r.f < mejor.f) mejor = r;
                }
            if (!std::isfinite(mejor.f)) throw std::runtime_error("no se pudo ajustar SVI en el vencimiento " + vm.etiqueta);
            aj.a = a_x(mejor.x[0], ca); aj.b = a_x(mejor.x[1], cb); aj.rho = a_x(mejor.x[2], cr);
            aj.m = a_x(mejor.x[3], cm); aj.sigma = a_x(mejor.x[4], cs);
        }
        const double p[5] = {aj.a, aj.b, aj.rho, aj.m, aj.sigma};
        double s2 = 0;
        for (size_t j = 0; j < vm.puntos.size(); ++j) {
            const int i = vm.puntos[j];
            const PuntoMercado& pt = m.puntos[(size_t)i];
            double vol = std::sqrt(std::max(svi_w(p, std::log(pt.K / vm.F)), 0.0) / vm.T);
            R.vol_modelo[(size_t)i] = vol;
            R.precio_modelo[(size_t)i] = black(pt.call, vm.F, pt.K, vm.T, vm.DF, vol);
            s2 += (vol - pt.vol) * (vol - pt.vol);
        }
        aj.rmse = std::sqrt(s2 / (double)vm.puntos.size());
        // Mariposa: g(k) en un tramo amplio alrededor de los strikes
        double k0 = *std::min_element(k.begin(), k.end()) - 0.5, k1 = *std::max_element(k.begin(), k.end()) + 0.5;
        aj.g_min = 1e300;
        for (int i = 0; i <= 400; ++i) aj.g_min = std::min(aj.g_min, svi_g(p, k0 + (k1 - k0) * i / 400.0));
        if (aj.g_min < -1e-8)
            R.notas.push_back("vencimiento " + vm.etiqueta + ": arbitraje de mariposa (g mínima " + fmt(aj.g_min, 4) +
                              " < 0): la densidad implícita es negativa en algún strike");
        R.svi.push_back(aj);
    }
    // Calendario: la varianza total no puede bajar con el plazo
    for (size_t v = 1; v < R.svi.size(); ++v) {
        double peor = 0, kpeor = 0;
        for (int i = 0; i <= 200; ++i) {
            double kk = -1.0 + 2.0 * i / 200.0;
            const AjusteSVI& a1 = R.svi[v - 1];
            const AjusteSVI& a2 = R.svi[v];
            double d = a2.w(kk) - a1.w(kk);
            if (d < peor) { peor = d; kpeor = kk; }
        }
        if (peor < -1e-6)
            R.notas.push_back("arbitraje de calendario entre " + m.vencimientos[v - 1].etiqueta + " y " +
                              m.vencimientos[v].etiqueta + " (la varianza total baja en ln(K/F) = " + fmt(kpeor, 2) + ")");
    }
    double s2 = 0, mx = 0;
    for (size_t i = 0; i < m.puntos.size(); ++i) {
        double e = R.vol_modelo[i] - m.puntos[i].vol;
        s2 += e * e;
        if (std::fabs(e) > mx) { mx = std::fabs(e); R.i_max = (int)i; }
    }
    R.rmse = std::sqrt(s2 / (double)m.puntos.size());
    R.max_err = mx;
    R.segundos = std::chrono::duration<double>(Reloj::now() - t0).count();
    R.motivo = "ajuste por vencimiento";
    return R;
}

} // namespace

double AjusteSVI::w(double k) const {
    const double p[5] = {a, b, rho, m, sigma};
    return svi_w(p, k);
}

Modelo modelo_por_defecto(TipoModelo t) {
    Modelo m;
    m.tipo = t;
    for (const auto& d : def_modelo(t).params) m.valores.push_back(d.defecto);
    return m;
}

json::Valor modelo_a_json(const Modelo& mod) {
    json::Valor o = json::Valor::objeto();
    const DefModelo& d = def_modelo(mod.tipo);
    o.poner("tipo", d.clave);
    for (size_t i = 0; i < d.params.size(); ++i) {
        const DefParam& p = d.params[i];
        double v = mod.valores[i];
        if (p.clave == "vol_max" && v == p.defecto) continue;
        if (p.porcentaje) o.poner(p.clave, porcentaje(v, 4));
        else {
            double r = std::round(v * 1e6) / 1e6;
            o.poner(p.clave, json::Valor(std::fabs(r) > 0 ? r : v));
        }
    }
    return o;
}

ResultadoCalibracion calibrar(const DatosMercado& m, const OpcionesCalibracion& o) {
    std::string clave = o.modelo;
    for (auto& c : clave) c = (char)std::tolower((unsigned char)c);
    if (clave == "svi") return calibrar_svi(m, o);
    const DefModelo* dm = buscar_modelo(clave);
    if (!dm) throw std::runtime_error("modelo desconocido \"" + o.modelo + "\" (ver 'estructurados modelos', o svi)");
    const auto t0 = Reloj::now();
    ResultadoCalibracion R;
    R.modelo_clave = dm->clave;
    R.modelo_nombre = dm->nombre;

    // ---------------- Punto de partida ----------------
    Modelo base = modelo_por_defecto(dm->tipo);
    if (o.partida && o.partida->tipo == dm->tipo) base = *o.partida;
    else {
        // Niveles de volatilidad a partir de la vol ATM del mercado
        const double corto = m.vol_atm(0), largo = m.vol_atm((int)m.vencimientos.size() - 1);
        const double medio = 0.5 * (corto + largo);
        switch (dm->tipo) {
            case TipoModelo::BlackScholes: base["volatilidad"] = medio; break;
            case TipoModelo::Heston: base["vol_inicial"] = corto; base["vol_largo_plazo"] = largo; break;
            case TipoModelo::Bates: base["vol_inicial"] = 0.9 * corto; base["vol_largo_plazo"] = 0.9 * largo;
                                    base["lambda"] = 0.3; base["media_salto"] = -0.1; base["vol_salto"] = 0.1; break;
            case TipoModelo::Merton: base["volatilidad"] = 0.8 * medio; base["lambda"] = 0.3; break;
            case TipoModelo::NIG: base["sigma"] = medio; break;
            case TipoModelo::SABR: base["alpha"] = medio; break;
            case TipoModelo::VolLocal: base["sigma0"] = corto; break;
            default: break;
        }
    }
    for (const auto& [k, v] : o.iniciales) base[k] = v;
    for (const auto& [k, v] : o.fijos) base[k] = v;

    // ---------------- Parámetros libres ----------------
    std::vector<int> libres;
    std::vector<Cota> cotas;
    for (size_t i = 0; i < dm->params.size(); ++i) {
        const DefParam& d = dm->params[i];
        ParamCalibrado pc;
        pc.clave = d.clave;
        pc.descripcion = d.descripcion;
        pc.porcentaje = d.porcentaje;
        pc.inicial = base.valores[i];
        Cota c = cota_calibracion(dm->tipo, d);
        pc.lo = c.lo;
        pc.hi = c.hi;
        bool fijo = (d.clave == "vol_max") || o.fijos.count(d.clave);
        if (dm->tipo == TipoModelo::SABR && d.clave == "beta" &&
            std::find(o.liberar.begin(), o.liberar.end(), "beta") == o.liberar.end())
            fijo = true;
        pc.libre = !fijo;
        if (pc.libre) {
            libres.push_back((int)i);
            cotas.push_back(c);
            R.libres.push_back(d.clave);
        }
        R.params.push_back(pc);
    }
    if (libres.empty()) throw std::runtime_error("no queda ningún parámetro libre que calibrar");
    if (m.puntos.size() < libres.size())
        throw std::runtime_error("hay menos cotizaciones (" + std::to_string(m.puntos.size()) + ") que parámetros (" +
                                 std::to_string(libres.size()) + ")");
    const int nl = (int)libres.size();
    auto a_modelo = [&](const std::vector<double>& u) {
        Modelo mod = base;
        for (int j = 0; j < nl; ++j) mod.valores[(size_t)libres[(size_t)j]] = a_x(u[(size_t)j], cotas[(size_t)j]);
        return mod;
    };
    std::vector<double> u0((size_t)nl);
    for (int j = 0; j < nl; ++j) u0[(size_t)j] = a_u(base.valores[(size_t)libres[(size_t)j]], cotas[(size_t)j]);

    // ---------------- Objetivo ----------------
    const std::vector<double> w = pesos_de(m, o.pesos);
    std::vector<double> sw(w.size());
    for (size_t i = 0; i < w.size(); ++i) sw[i] = std::sqrt(w[i]);
    int n_eval = 0;
    auto residuos = [&](const std::vector<double>& u, std::vector<double>& r) {
        std::vector<double> pr;
        ++n_eval;
        if (!precios_modelo(a_modelo(u), m, o, pr)) return false;
        r.resize(pr.size());
        for (size_t i = 0; i < pr.size(); ++i) {
            const PuntoMercado& p = m.puntos[i];
            double v = vol_de_precio(p, m.vencimientos[(size_t)p.v], pr[i]);
            r[i] = std::isfinite(v) ? sw[i] * (v - p.vol)
                                    : sw[i] * (pr[i] - p.precio) / std::max(p.vega, 1e-10 * m.spot);
        }
        return true;
    };
    auto objetivo = [&](const std::vector<double>& u) {
        std::vector<double> r;
        if (!residuos(u, r)) return std::numeric_limits<double>::infinity();
        double s = 0;
        for (double x : r) s += x * x;
        return 0.5 * s;
    };
    {
        std::vector<double> pr;
        if (!precios_modelo(base, m, o, pr, &R.metodo_precios))
            throw std::runtime_error("el punto de partida no es válido para el modelo " + dm->nombre);
    }
    const bool es_mc = R.metodo_precios.rfind("Monte Carlo", 0) == 0;
    {
        double f0 = objetivo(u0);
        R.rmse_inicial = std::sqrt(2.0 * f0 / (double)m.puntos.size());
    }

    // ---------------- Optimización ----------------
    std::string opt_nombre = o.optimizador;
    if (opt_nombre == "auto") opt_nombre = "lm";
    opt::OpcionesLM olm;
    if (es_mc) olm.paso_jacobiano = 2e-3; // la superficie MC es suave (números comunes) pero no a escala 1e-6
    opt::Resultado mejor;
    mejor.f = std::numeric_limits<double>::infinity();
    auto progreso = [&](const std::string& s) { if (o.progreso) *o.progreso << "  · " << s << "\n" << std::flush; };

    if (opt_nombre == "de") {
        std::vector<double> lo((size_t)nl), hi((size_t)nl);
        for (int j = 0; j < nl; ++j) { lo[(size_t)j] = -5.0; hi[(size_t)j] = 5.0; }
        opt::OpcionesDE ode;
        ode.semilla = o.semilla;
        ode.generaciones = es_mc ? 30 : 80;
        progreso("evolución diferencial (" + std::to_string(std::max(15, 10 * nl)) + " individuos, hasta " +
                 std::to_string(ode.generaciones) + " generaciones)...");
        opt::Resultado rde = opt::evolucion_diferencial(objetivo, lo, hi, ode);
        progreso("mejor RMSE de la población: " + fmt(100.0 * std::sqrt(2.0 * rde.f / (double)m.puntos.size()), 3) + " puntos de vol" +
                 "; se refina con Levenberg-Marquardt");
        mejor = opt::levenberg_marquardt(residuos, rde.x, olm);
        mejor.evaluaciones += rde.evaluaciones;
        R.optimizador = "evolución diferencial + Levenberg-Marquardt";
    } else if (opt_nombre == "nm") {
        opt::OpcionesNM onm;
        onm.max_eval = es_mc ? 800 : 3000;
        opt::Resultado rnm = opt::nelder_mead(objetivo, u0, onm);
        progreso("Nelder-Mead: RMSE " + fmt(100.0 * std::sqrt(2.0 * rnm.f / (double)m.puntos.size()), 3) + " puntos de vol" +
                 "; se refina con Levenberg-Marquardt");
        mejor = opt::levenberg_marquardt(residuos, rnm.x, olm);
        mejor.evaluaciones += rnm.evaluaciones;
        R.optimizador = "Nelder-Mead + Levenberg-Marquardt";
    } else if (opt_nombre == "lm") {
        const int arranques = std::max(1, es_mc ? std::min(o.arranques, 2) : o.arranques);
        std::mt19937_64 gen(o.semilla);
        auto normal = [&]() {
            double u1 = ((double)(gen() >> 11) + 0.5) / 9007199254740992.0, u2 = (double)(gen() >> 11) / 9007199254740992.0;
            return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
        };
        for (int a = 0; a < arranques; ++a) {
            std::vector<double> u = u0;
            if (a > 0) for (double& x : u) x += 1.2 * normal(); // arranques perturbados (reproducibles)
            std::vector<double> rr;
            if (!residuos(u, rr)) continue;
            opt::Resultado r = opt::levenberg_marquardt(residuos, u, olm);
            progreso("arranque " + std::to_string(a + 1) + "/" + std::to_string(arranques) + ": RMSE " +
                     fmt(100.0 * std::sqrt(2.0 * r.f / (double)m.puntos.size()), 4) + " puntos de vol (" + r.motivo + ")");
            if (r.f < mejor.f) mejor = r;
        }
        R.optimizador = "Levenberg-Marquardt desde " + std::to_string(arranques) +
                        (arranques == 1 ? " punto de partida" : " puntos de partida");
    } else {
        throw std::runtime_error("optimizador desconocido \"" + o.optimizador + "\" (auto, lm, de o nm)");
    }
    if (!std::isfinite(mejor.f)) throw std::runtime_error("la calibración no encontró ningún punto válido");
    R.motivo = mejor.motivo;
    R.iteraciones = mejor.iteraciones;

    // ---------------- Resultado ----------------
    R.modelo = a_modelo(mejor.x);
    for (int j = 0; j < nl; ++j) R.params[(size_t)libres[(size_t)j]].valor = R.modelo.valores[(size_t)libres[(size_t)j]];
    for (size_t i = 0; i < R.params.size(); ++i)
        if (!R.params[i].libre) R.params[i].valor = R.modelo.valores[i];
    precios_modelo(R.modelo, m, o, R.precio_modelo);
    R.vol_modelo.resize(m.puntos.size());
    double s2 = 0, mx = 0;
    int dentro = 0, con_horquilla = 0;
    std::vector<double> s2v(m.vencimientos.size(), 0.0);
    for (size_t i = 0; i < m.puntos.size(); ++i) {
        const PuntoMercado& p = m.puntos[i];
        double v = vol_de_precio(p, m.vencimientos[(size_t)p.v], R.precio_modelo[i]);
        R.vol_modelo[i] = std::isfinite(v) ? v : NAN;
        double e = std::isfinite(v) ? v - p.vol : (R.precio_modelo[i] - p.precio) / std::max(p.vega, 1e-300);
        s2 += e * e;
        s2v[(size_t)p.v] += e * e;
        if (std::fabs(e) > mx) { mx = std::fabs(e); R.i_max = (int)i; }
        if (p.vol_bid > 0 && p.vol_ask > 0) {
            ++con_horquilla;
            if (std::isfinite(v) && v >= p.vol_bid - 1e-12 && v <= p.vol_ask + 1e-12) ++dentro;
        }
    }
    R.rmse = std::sqrt(s2 / (double)m.puntos.size());
    R.max_err = mx;
    for (size_t v = 0; v < m.vencimientos.size(); ++v)
        R.rmse_venc.push_back(std::sqrt(s2v[v] / std::max<size_t>(1, m.vencimientos[v].puntos.size())));
    if (con_horquilla > 0) R.dentro_horquilla = (double)dentro / con_horquilla;

    // Errores estándar y correlaciones (Cov_u = s²·(JᵀJ)⁻¹, s² = Σr²/(m - n))
    const int mm = (int)m.puntos.size();
    std::vector<double> inv;
    if (mm > nl && (int)mejor.jtj.size() == nl * nl && opt::invertir_spd(mejor.jtj, nl, inv)) {
        const double s2r = 2.0 * mejor.f / (double)(mm - nl);
        R.correlacion.assign((size_t)(nl * nl), 0.0);
        for (int a = 0; a < nl; ++a) {
            const double var_u = s2r * inv[(size_t)(a * nl + a)];
            R.params[(size_t)libres[(size_t)a]].error = dxdu(mejor.x[(size_t)a], cotas[(size_t)a]) * std::sqrt(std::max(var_u, 0.0));
            for (int b = 0; b < nl; ++b)
                R.correlacion[(size_t)(a * nl + b)] = inv[(size_t)(a * nl + b)] /
                    std::sqrt(inv[(size_t)(a * nl + a)] * inv[(size_t)(b * nl + b)]);
        }
        double peor = 0;
        int pa = -1, pb = -1;
        for (int a = 0; a < nl; ++a)
            for (int b = a + 1; b < nl; ++b)
                if (std::fabs(R.correlacion[(size_t)(a * nl + b)]) > std::fabs(peor)) {
                    peor = R.correlacion[(size_t)(a * nl + b)];
                    pa = a;
                    pb = b;
                }
        if (pa >= 0 && std::fabs(peor) > 0.95)
            R.notas.push_back("\"" + R.libres[(size_t)pa] + "\" y \"" + R.libres[(size_t)pb] + "\" tienen correlación " +
                              fmt(peor, 3) + ": los datos fijan una combinación de ambos, no cada uno por separado");
    } else if (mm > nl) {
        R.notas.push_back("JᵀJ es singular: algún parámetro no influye en los precios (no se pueden dar errores estándar)");
    }
    for (int j = 0; j < nl; ++j)
        if (std::fabs(mejor.x[(size_t)j]) > 9.0)
            R.notas.push_back("\"" + R.libres[(size_t)j] + "\" ha quedado en el borde de su rango [" +
                              fmt(cotas[(size_t)j].lo, 4) + ", " + fmt(cotas[(size_t)j].hi, 4) + "]");
    if (dm->tipo == TipoModelo::Heston || dm->tipo == TipoModelo::Bates) {
        const double kappa = R.modelo["kappa"], th = std::pow(R.modelo["vol_largo_plazo"], 2.0), xi = R.modelo["xi"];
        R.notas.push_back(std::string("condición de Feller: 2κθ = ") + fmt(2 * kappa * th, 4) +
                          (2 * kappa * th >= xi * xi ? " ≥ " : " < ") + "ξ² = " + fmt(xi * xi, 4) +
                          (2 * kappa * th >= xi * xi ? " (se cumple)" : " (no se cumple: la varianza puede tocar 0; el motor lo admite)"));
    }
    if (dm->tipo == TipoModelo::SABR && !R.params.empty())
        for (const auto& pc : R.params)
            if (pc.clave == "beta" && !pc.libre)
                R.notas.push_back("β de SABR fijo en " + fmt(pc.valor, 3) + " (habitual: los datos de un día apenas lo "
                                  "distinguen de ρ); --liberar beta lo calibra también");
    if (es_mc)
        R.notas.push_back("precios por Monte Carlo con " + std::to_string(o.trayectorias_mc) +
                          " trayectorias y números aleatorios comunes: el ajuste es suave, pero cada precio tiene error "
                          "de muestreo (--trayectorias para más precisión)");
    R.evaluaciones = n_eval;
    R.segundos = std::chrono::duration<double>(Reloj::now() - t0).count();
    return R;
}

// ===================================================================== //
// Informe                                                                //
// ===================================================================== //
void imprimir_calibracion(std::ostream& os, const DatosMercado& m, const ResultadoCalibracion& r) {
    os << "\n" << linea('=') << "\n CALIBRACIÓN: " << r.modelo_nombre << " a " << m.puntos.size() << " cotizaciones";
    if (!m.subyacente.empty()) os << " de " << m.subyacente;
    os << " (" << m.vencimientos.size() << (m.vencimientos.size() == 1 ? " vencimiento" : " vencimientos") << ")\n"
       << linea('=') << "\n";
    os << " Datos ........ " << m.fuente << ": spot " << fmt(m.spot, 4);
    bool impl = false;
    for (const auto& v : m.vencimientos) impl |= v.forward_implicito;
    if (impl) os << "; forwards y descuentos implícitos por paridad put-call";
    else os << "; r " << porcentaje(m.r, 3) << ", q " << porcentaje(m.q, 3) << " (continuos)";
    os << "\n";
    if (m.descartados > 0) os << "                " << m.descartados << " cotizaciones descartadas por los filtros\n";
    os << " Objetivo ..... volatilidad implícita (error en puntos de volatilidad)\n";
    os << " Precios ...... " << r.metodo_precios << "\n";
    os << " Optimizador .. " << r.optimizador << " (" << r.motivo << ")\n";
    os << " Resultado .... RMSE " << fmt(100.0 * r.rmse, 3) << " puntos de vol";
    if (r.svi.empty()) os << " (en el punto de partida " << fmt(100.0 * r.rmse_inicial, 3) << ")";
    os << "; error máximo " << fmt(100.0 * r.max_err, 3);
    if (r.i_max >= 0) {
        const PuntoMercado& p = m.puntos[(size_t)r.i_max];
        os << " (" << m.vencimientos[(size_t)p.v].etiqueta << ", strike " << fmt(p.K, 2) << ")";
    }
    os << "\n";
    if (r.dentro_horquilla >= 0)
        os << "                el " << porcentaje(r.dentro_horquilla, 1) << " de las vols del modelo cae dentro de la horquilla bid-ask\n";
    os << " Tiempo ....... " << (r.segundos < 0.01 ? std::string("< 0.01") : fmt(r.segundos, 2)) << " s, " << r.evaluaciones
       << (r.svi.empty() ? " valoraciones de la superficie\n" : " evaluaciones de los ajustes por vencimiento\n");

    if (!r.svi.empty()) {
        os << "\n Parámetros SVI por vencimiento: w(k) = a + b·(ρ(k-m) + √((k-m)² + σ²)), k = ln(K/F)\n   "
           << col("Vto.", 9) << dcha("a", 10) << dcha("b", 10) << dcha("ρ", 9) << dcha("m", 9) << dcha("σ", 9)
           << dcha("RMSE (pb)", 11) << dcha("g mín.", 10) << "\n";
        for (size_t v = 0; v < r.svi.size(); ++v) {
            const AjusteSVI& a = r.svi[v];
            os << "   " << col(m.vencimientos[v].etiqueta, 9) << dcha(fmt(a.a, 5), 10) << dcha(fmt(a.b, 5), 10)
               << dcha(fmt(a.rho, 4), 9) << dcha(fmt(a.m, 4), 9) << dcha(fmt(a.sigma, 4), 9)
               << dcha(fmt(1e4 * a.rmse, 1), 11) << dcha(fmt(a.g_min, 3), 10) << "\n";
        }
    } else {
        os << "\n Parámetros" << std::string(16, ' ') << dcha("Valor", 12) << dcha("Error est.", 13) << "   Rango de búsqueda\n";
        for (const auto& p : r.params) {
            if (p.clave == "vol_max" && !p.libre) continue;
            auto f = [&](double x) { return p.porcentaje ? porcentaje(x, 3) : fmt(x, 5); };
            os << "   " << col(p.clave, 23) << dcha(f(p.valor), 12)
               << dcha(p.libre ? (p.error >= 0 ? f(p.error) : "-") : "fijo", 13) << "   ";
            if (p.libre) os << "[" << f(p.lo) << ", " << f(p.hi) << "]";
            os << "\n";
        }
    }
    if (!r.notas.empty()) {
        os << "\n";
        for (const auto& n : r.notas) os << "   · " << n << "\n";
    }

    os << "\n Ajuste por vencimiento (volatilidades implícitas en %)\n";
    for (size_t v = 0; v < m.vencimientos.size(); ++v) {
        const VencimientoMercado& vm = m.vencimientos[v];
        const double rv = v < r.rmse_venc.size() ? r.rmse_venc[v] : 0.0;
        os << "   " << vm.etiqueta << " (" << fmt(vm.T, 3) << " años, F = " << fmt(vm.F, 2)
           << (vm.forward_implicito ? " implícito" : "") << ")   RMSE " << fmt(100.0 * rv, 3)
           << "\n     " << col("Strike", 12) << col("K/F", 9) << dcha("Mercado", 9) << dcha("Modelo", 9) << dcha("Dif. (pb)", 11);
        bool hb = false;
        for (int i : vm.puntos) hb |= (m.puntos[(size_t)i].vol_bid > 0);
        if (hb) os << dcha("Bid", 9) << dcha("Ask", 9);
        os << "\n";
        // como mucho 15 strikes por vencimiento en pantalla (todos en --csv)
        const size_t n = vm.puntos.size(), paso = std::max<size_t>(1, (n + 14) / 15);
        for (size_t j = 0; j < n; j += paso) {
            const int i = vm.puntos[j];
            const PuntoMercado& p = m.puntos[(size_t)i];
            double vm_ = r.vol_modelo[(size_t)i];
            os << "     " << col(fmt(p.K, 2), 12) << col(fmt(100.0 * p.K / vm.F, 1) + "%", 9) << dcha(fmt(100 * p.vol, 2), 9)
               << dcha(std::isfinite(vm_) ? fmt(100 * vm_, 2) : "-", 9)
               << dcha(std::isfinite(vm_) ? fmt(1e4 * (vm_ - p.vol), 1) : "-", 11);
            if (hb) os << dcha(p.vol_bid > 0 ? fmt(100 * p.vol_bid, 2) : "-", 9) << dcha(p.vol_ask > 0 ? fmt(100 * p.vol_ask, 2) : "-", 9);
            os << "\n";
        }
    }
    os << linea('=') << "\n";
    if (r.svi.empty()) os << " Modelo calibrado (para el campo \"modelo\" de un subyacente):\n   " << json::escribir(modelo_a_json(r.modelo), 0) << "\n";
}

json::Valor calibracion_a_json(const DatosMercado& m, const ResultadoCalibracion& r) {
    json::Valor o = json::Valor::objeto();
    o.poner("formato", "calibracion/1");
    o.poner("subyacente", m.subyacente);
    o.poner("datos", m.fuente);
    o.poner("spot", m.spot);
    o.poner("modelo_tipo", r.modelo_clave);
    if (r.svi.empty()) o.poner("modelo", modelo_a_json(r.modelo));
    o.poner("rmse_vol", r.rmse);
    o.poner("error_max_vol", r.max_err);
    o.poner("optimizador", r.optimizador);
    o.poner("precios", r.metodo_precios);
    if (r.svi.empty()) {
        json::Valor ps = json::Valor::objeto();
        for (const auto& p : r.params) {
            if (!p.libre) continue;
            json::Valor x = json::Valor::objeto();
            x.poner("valor", p.valor);
            if (p.error >= 0) x.poner("error_estandar", p.error);
            ps.poner(p.clave, std::move(x));
        }
        o.poner("parametros", std::move(ps));
    } else {
        json::Valor ls = json::Valor::lista();
        for (size_t v = 0; v < r.svi.size(); ++v) {
            json::Valor x = json::Valor::objeto();
            x.poner("vencimiento", m.vencimientos[v].T);
            x.poner("forward", m.vencimientos[v].F);
            x.poner("a", r.svi[v].a);
            x.poner("b", r.svi[v].b);
            x.poner("rho", r.svi[v].rho);
            x.poner("m", r.svi[v].m);
            x.poner("sigma", r.svi[v].sigma);
            x.poner("rmse_vol", r.svi[v].rmse);
            x.poner("g_min", r.svi[v].g_min);
            ls.anadir(std::move(x));
        }
        o.poner("svi", std::move(ls));
    }
    json::Valor ns = json::Valor::lista();
    for (const auto& n : r.notas) ns.anadir(json::Valor(n));
    o.poner("notas", std::move(ns));
    return o;
}

void guardar_ajuste_csv(const std::string& ruta, const DatosMercado& m, const ResultadoCalibracion& r) {
    std::ofstream f(ruta);
    if (!f) throw std::runtime_error("no se puede escribir " + ruta);
    f << "vencimiento,T,forward,descuento,strike,K_sobre_F,tipo,precio_mercado,vol_mercado,vol_bid,vol_ask,precio_modelo,vol_modelo\n";
    for (size_t i = 0; i < m.puntos.size(); ++i) {
        const PuntoMercado& p = m.puntos[i];
        const VencimientoMercado& v = m.vencimientos[(size_t)p.v];
        f << v.etiqueta << "," << fmt(p.T, 8) << "," << fmt(v.F, 8) << "," << fmt(v.DF, 10) << "," << fmt(p.K, 8) << ","
          << fmt(p.K / v.F, 8) << "," << (p.call ? "call" : "put") << "," << fmt(p.precio, 10) << "," << fmt(p.vol, 10) << ","
          << (p.vol_bid > 0 ? fmt(p.vol_bid, 10) : "") << "," << (p.vol_ask > 0 ? fmt(p.vol_ask, 10) : "") << ","
          << fmt(r.precio_modelo[i], 10) << "," << (std::isfinite(r.vol_modelo[i]) ? fmt(r.vol_modelo[i], 10) : "") << "\n";
    }
}

} // namespace est
