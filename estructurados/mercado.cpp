#include "mercado.hpp"
#include "cotizaciones.hpp"
#include "json.hpp"
#include "vanilla.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace est {

namespace {
constexpr double PI = 3.14159265358979323846;
double ncdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

std::string carpeta_de(const std::string& ruta) {
    size_t p = ruta.find_last_of("/\\");
    return p == std::string::npos ? std::string() : ruta.substr(0, p + 1);
}

bool termina_en(const std::string& s, const std::string& suf) {
    if (s.size() < suf.size()) return false;
    std::string a = s.substr(s.size() - suf.size());
    for (auto& c : a) c = (char)std::tolower((unsigned char)c);
    return a == suf;
}

// Cotización tal como viene (antes de elegir forward e instrumento)
struct Cruda {
    double T = 0;
    std::string etiqueta;
    double K = 0;
    int tipo = -1;            // 1 call, 0 put, -1 sin especificar (vol de una superficie)
    double precio = -1, bid = -1, ask = -1, vol = -1, peso = 1, volumen = -1;
};

double real_de(const json::Valor& v, const std::string& que) {
    bool ok = false;
    double x = v.es_numero() ? v.numero() : (v.es_texto() ? texto_a_real(v.texto(), &ok) : NAN);
    if (v.es_numero()) ok = true;
    if (!ok || !std::isfinite(x)) throw std::runtime_error("valor no válido en " + que + v.donde());
    return x;
}
double plazo_de(const json::Valor& v, int dias_val, const std::string& que) {
    if (v.es_numero()) return v.numero();
    if (!v.es_texto()) throw std::runtime_error("plazo no válido en " + que + v.donde());
    int d = 0;
    if (texto_a_fecha(v.texto(), d)) {
        if (dias_val < 0) throw std::runtime_error("los vencimientos son fechas: indique \"fecha\" en el fichero de mercado");
        return (d - dias_val) / 365.0;
    }
    bool ok = false;
    double t = texto_a_plazo(v.texto(), &ok);
    if (!ok) throw std::runtime_error("plazo no válido \"" + v.texto() + "\" en " + que + v.donde());
    return t;
}
// Strike: "90%" = 90% del spot; un número es absoluto
double strike_de(const json::Valor& v, double S, const std::string& que) {
    if (v.es_texto()) {
        std::string t = v.texto();
        bool pct = !t.empty() && t.back() == '%';
        double x = real_de(v, que);
        return pct ? x * S : x;
    }
    return real_de(v, que);
}
} // namespace

double black(bool call, double F, double K, double T, double DF, double vol) {
    const double sv = vol * std::sqrt(T);
    if (!(sv > 0)) return DF * std::max(call ? F - K : K - F, 0.0);
    const double d1 = (std::log(F / K) + 0.5 * sv * sv) / sv, d2 = d1 - sv;
    return call ? DF * (F * ncdf(d1) - K * ncdf(d2)) : DF * (K * ncdf(-d2) - F * ncdf(-d1));
}

double vega_black(double F, double K, double T, double DF, double vol) {
    const double sv = vol * std::sqrt(T);
    if (!(sv > 0)) return 0.0;
    const double d1 = (std::log(F / K) + 0.5 * sv * sv) / sv;
    return DF * F * std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * PI) * std::sqrt(T);
}

double vol_implicita_fwd(double precio, bool call, double F, double K, double T, double DF) {
    // Con S = F y q = r se tiene S·e^{(r-q)T} = F
    const double r = -std::log(DF) / T;
    bool ok = false;
    double v = vol_implicita(precio, call, F, K, T, r, r, &ok);
    return ok ? v : NAN;
}

double DatosMercado::vol_atm(int v) const {
    const VencimientoMercado& vm = vencimientos[(size_t)v];
    std::vector<std::pair<double, double>> kv;
    for (int i : vm.puntos) kv.emplace_back(std::log(puntos[(size_t)i].K / vm.F), puntos[(size_t)i].vol);
    if (kv.empty()) return 0.2;
    std::sort(kv.begin(), kv.end());
    if (kv.front().first >= 0) return kv.front().second;
    if (kv.back().first <= 0) return kv.back().second;
    for (size_t i = 1; i < kv.size(); ++i)
        if (kv[i].first >= 0) {
            double w = (0 - kv[i - 1].first) / (kv[i].first - kv[i - 1].first);
            return kv[i - 1].second + w * (kv[i].second - kv[i - 1].second);
        }
    return kv.back().second;
}

DatosMercado cargar_mercado(const std::string& ruta, const OpcionesMercado& o) {
    DatosMercado m;
    m.fuente = ruta;
    std::vector<Cruda> crudas;
    double r = -1e9, q = -1e9;
    int dias = o.dias_valoracion;

    auto desde_cadena = [&](const std::string& rc) {
        CadenaOpciones c = leer_cadena_csv(rc, dias);
        for (const auto& a : c.avisos) m.avisos.push_back(a);
        if (m.subyacente.empty()) m.subyacente = c.subyacente;
        if (m.fecha.empty()) m.fecha = c.fecha;
        if (m.spot <= 0 && c.spot > 0) m.spot = c.spot;
        if (r < -1e8 && c.tipo_interes > -1e8) r = c.tipo_interes;
        if (q < -1e8 && c.dividendos > -1e8) q = c.dividendos;
        for (const auto& x : c.cot) {
            Cruda cr;
            cr.T = x.T;
            cr.etiqueta = x.vencimiento;
            cr.K = x.K;
            cr.tipo = x.call ? 1 : 0;
            cr.precio = x.referencia();
            cr.bid = x.bid;
            cr.ask = x.ask;
            cr.vol = x.vol;
            cr.peso = x.peso;
            cr.volumen = x.volumen;
            crudas.push_back(cr);
        }
    };

    if (termina_en(ruta, ".csv") || termina_en(ruta, ".txt")) {
        desde_cadena(ruta);
    } else {
        json::Valor d = json::leer_fichero(ruta);
        if (!d.es_objeto()) throw std::runtime_error("el fichero de mercado debe ser un objeto JSON");
        if (auto* x = d.buscar("subyacente")) m.subyacente = x->texto();
        if (auto* x = d.buscar("fecha")) {
            m.fecha = x->texto();
            int dd = 0;
            if (dias < 0 && texto_a_fecha(m.fecha, dd)) dias = dd;
        }
        if (auto* x = d.buscar("spot")) m.spot = real_de(*x, "spot");
        const json::Valor* bm = d.buscar("mercado");
        auto leer_tipo = [&](const json::Valor& obj) {
            if (auto* x = obj.buscar("tipo_interes")) r = real_de(*x, "tipo_interes");
            if (auto* x = obj.buscar("dividendos")) q = real_de(*x, "dividendos");
        };
        leer_tipo(d);
        if (bm && bm->es_objeto()) leer_tipo(*bm);
        if (auto* x = d.buscar("superficie")) {
            if (m.spot <= 0) throw std::runtime_error("falta el \"spot\" en el fichero de mercado");
            const json::Valor* vs = x->buscar("vencimientos");
            const json::Valor* ks = x->buscar("strikes");
            const json::Valor* vol = x->buscar("volatilidades");
            if (!vol) vol = x->buscar("vols");
            if (!vs || !ks || !vol || !vs->es_lista() || !ks->es_lista() || !vol->es_lista())
                throw std::runtime_error("\"superficie\" necesita las listas \"vencimientos\", \"strikes\" y \"volatilidades\"");
            if (vol->tamano() != vs->tamano())
                throw std::runtime_error("\"volatilidades\" debe tener una fila por vencimiento");
            for (size_t i = 0; i < vs->tamano(); ++i) {
                const json::Valor& fila = (*vol)[i];
                if (!fila.es_lista() || fila.tamano() != ks->tamano())
                    throw std::runtime_error("la fila " + std::to_string(i + 1) + " de \"volatilidades\" debe tener un valor por strike" + fila.donde());
                double T = plazo_de((*vs)[i], dias, "vencimientos");
                for (size_t j = 0; j < ks->tamano(); ++j) {
                    if (fila[j].es_nulo()) continue; // hueco en la superficie
                    Cruda c;
                    c.T = T;
                    c.etiqueta = (*vs)[i].es_texto() ? (*vs)[i].texto() : json::numero_a_texto((*vs)[i].numero());
                    c.K = strike_de((*ks)[j], m.spot, "strikes");
                    c.vol = real_de(fila[j], "volatilidades");
                    crudas.push_back(c);
                }
            }
        }
        if (auto* x = d.buscar("cotizaciones")) {
            if (!x->es_lista()) throw std::runtime_error("\"cotizaciones\" debe ser una lista");
            for (size_t i = 0; i < x->tamano(); ++i) {
                const json::Valor& e = (*x)[i];
                Cruda c;
                const json::Valor* pv = e.buscar("vencimiento");
                const json::Valor* pk = e.buscar("strike");
                if (!pv || !pk) throw std::runtime_error("cada cotización necesita \"vencimiento\" y \"strike\"" + e.donde());
                c.T = plazo_de(*pv, dias, "vencimiento");
                c.etiqueta = pv->es_texto() ? pv->texto() : json::numero_a_texto(pv->numero());
                c.K = strike_de(*pk, m.spot > 0 ? m.spot : 1.0, "strike");
                if (auto* t = e.buscar("tipo")) {
                    std::string s = t->texto();
                    for (auto& ch : s) ch = (char)std::tolower((unsigned char)ch);
                    c.tipo = (s == "call" || s == "c") ? 1 : (s == "put" || s == "p") ? 0 : -1;
                }
                if (auto* v = e.buscar("vol")) c.vol = real_de(*v, "vol");
                if (auto* v = e.buscar("vol_implicita")) c.vol = real_de(*v, "vol_implicita");
                if (auto* v = e.buscar("precio")) c.precio = real_de(*v, "precio");
                if (auto* v = e.buscar("bid")) c.bid = real_de(*v, "bid");
                if (auto* v = e.buscar("ask")) c.ask = real_de(*v, "ask");
                if (auto* v = e.buscar("peso")) c.peso = real_de(*v, "peso");
                if (auto* v = e.buscar("volumen")) c.volumen = real_de(*v, "volumen");
                if (c.precio < 0 && c.bid > 0 && c.ask >= c.bid) c.precio = 0.5 * (c.bid + c.ask);
                if (c.precio > 0 && c.tipo < 0)
                    throw std::runtime_error("una cotización con precio necesita \"tipo\" (call o put)" + e.donde());
                crudas.push_back(c);
            }
        }
        if (auto* x = d.buscar("cadena")) {
            std::string rc = x->texto();
            if (!rc.empty() && rc[0] != '/') rc = carpeta_de(ruta) + rc;
            desde_cadena(rc);
        }
    }
    if (o.spot > 0) m.spot = o.spot;
    if (o.tipo > -1e8) r = o.tipo;
    if (o.dividendos > -1e8) q = o.dividendos;
    if (!(m.spot > 0)) throw std::runtime_error("no se conoce el spot del subyacente (indíquelo con --spot o en el fichero)");
    if (crudas.empty()) throw std::runtime_error("el fichero de mercado no tiene cotizaciones");
    const bool r_dado = r > -1e8, q_dado = q > -1e8;
    m.r = r_dado ? r : 0.0;
    m.q = q_dado ? q : 0.0;

    // ---------------- Vencimientos ----------------
    std::sort(crudas.begin(), crudas.end(), [](const Cruda& a, const Cruda& b) {
        return a.T < b.T || (a.T == b.T && (a.K < b.K || (a.K == b.K && a.tipo < b.tipo)));
    });
    std::vector<std::vector<const Cruda*>> grupos;
    std::vector<double> Ts;
    for (const auto& c : crudas) {
        if (Ts.empty() || std::fabs(c.T - Ts.back()) > 1e-9) { Ts.push_back(c.T); grupos.emplace_back(); }
        grupos.back().push_back(&c);
    }
    bool algun_implicito = false, algun_dado = false;
    for (size_t g = 0; g < grupos.size(); ++g) {
        const double T = Ts[g];
        if (T < o.T_min || T > o.T_max) { m.descartados += (int)grupos[g].size(); continue; }
        VencimientoMercado vm;
        vm.T = T;
        vm.etiqueta = grupos[g].front()->etiqueta;
        vm.DF = std::exp(-m.r * T);
        vm.F = m.spot * std::exp((m.r - m.q) * T);
        // Paridad put-call con los strikes que tienen call y put cerca del spot
        if (o.forward != "dado") {
            std::map<double, std::pair<double, double>> cp; // K -> (call, put)
            for (const Cruda* c : grupos[g]) {
                if (c->precio <= 0 || c->tipo < 0) continue;
                auto& e = cp[c->K];
                if (e.first == 0 && e.second == 0) e = {-1, -1};
                (c->tipo == 1 ? e.first : e.second) = c->precio;
            }
            std::vector<std::pair<double, double>> xy; // (K, C - P)
            for (const auto& [K, e] : cp)
                if (e.first > 0 && e.second > 0 && std::fabs(K / m.spot - 1.0) <= 0.25) xy.emplace_back(K, e.first - e.second);
            if (xy.size() >= 3) {
                double sx = 0, sy = 0, sxx = 0, sxy = 0, n = (double)xy.size();
                for (auto& [x, y] : xy) { sx += x; sy += y; sxx += x * x; sxy += x * y; }
                double b = (n * sxy - sx * sy) / (n * sxx - sx * sx), a = (sy - b * sx) / n;
                double DF = -b, F = (DF > 0) ? a / DF : -1;
                if (DF > 0.3 && DF < 1.2 && F > 0 && std::fabs(F / m.spot - 1.0) < 0.6) {
                    vm.DF = DF;
                    vm.F = F;
                    vm.forward_implicito = true;
                    algun_implicito = true;
                } else {
                    m.avisos.push_back("vencimiento " + vm.etiqueta + ": la paridad put-call da un forward incoherente; se usan r y q dados");
                }
            } else if (o.forward == "paridad") {
                m.avisos.push_back("vencimiento " + vm.etiqueta + ": no hay calls y puts del mismo strike para la paridad; se usan r y q dados");
            }
        }
        if (!vm.forward_implicito) algun_dado = true;
        // Una cotización por strike: la opción fuera del dinero
        std::map<double, std::vector<const Cruda*>> por_k;
        for (const Cruda* c : grupos[g]) por_k[c->K].push_back(c);
        const int v = (int)m.vencimientos.size();
        for (const auto& [K, lista] : por_k) {
            const double mon = K / vm.F;
            if (mon < o.mon_min || mon > o.mon_max) { m.descartados += (int)lista.size(); continue; }
            const bool otm_call = K >= vm.F;
            const Cruda* elegido = nullptr;
            for (const Cruda* c : lista)
                if (c->tipo < 0 || (c->tipo == 1) == otm_call) { elegido = c; break; }
            if (!elegido && o.incluir_dentro_del_dinero) elegido = lista.front();
            if (!elegido) { m.descartados += (int)lista.size(); continue; }
            if (o.volumen_min > 0 && elegido->volumen >= 0 && elegido->volumen < o.volumen_min) {
                m.descartados += (int)lista.size();
                continue;
            }
            PuntoMercado p;
            p.v = v;
            p.T = T;
            p.K = K;
            p.call = (elegido->tipo < 0) ? otm_call : (elegido->tipo == 1);
            p.peso = elegido->peso;
            p.volumen = elegido->volumen;
            if (elegido->precio > 0) {
                p.precio = elegido->precio;
                p.vol = vol_implicita_fwd(p.precio, p.call, vm.F, K, T, vm.DF);
                if (!std::isfinite(p.vol)) {
                    // precio fuera de los límites de no arbitraje: se usa la vol del fichero si la hay
                    if (elegido->vol > 0) { p.vol = elegido->vol; p.precio = black(p.call, vm.F, K, T, vm.DF, p.vol); }
                    else { ++m.descartados; continue; }
                }
            } else if (elegido->vol > 0) {
                p.vol = elegido->vol;
                p.precio = black(p.call, vm.F, K, T, vm.DF, p.vol);
            } else {
                ++m.descartados;
                continue;
            }
            if (elegido->bid > 0) { double x = vol_implicita_fwd(elegido->bid, p.call, vm.F, K, T, vm.DF); if (std::isfinite(x)) p.vol_bid = x; }
            if (elegido->ask > 0) { double x = vol_implicita_fwd(elegido->ask, p.call, vm.F, K, T, vm.DF); if (std::isfinite(x)) p.vol_ask = x; }
            p.vega = vega_black(vm.F, K, T, vm.DF, p.vol);
            vm.puntos.push_back((int)m.puntos.size());
            m.puntos.push_back(p);
        }
        if (vm.puntos.empty()) continue;
        m.vencimientos.push_back(vm);
    }
    if (m.puntos.empty()) throw std::runtime_error("ninguna cotización supera los filtros (vencimiento, strikes, volumen)");
    if (algun_dado && (!r_dado || !q_dado))
        m.avisos.push_back(std::string("no se conocen ") + (!r_dado && !q_dado ? "el tipo de interés ni los dividendos" :
                           !r_dado ? "el tipo de interés" : "los dividendos") +
                           " y algún vencimiento no admite la paridad put-call: se toma 0 (use --tipo/--dividendos)");
    if (algun_implicito && !algun_dado) {
        // r y q de referencia: los del vencimiento más largo implícito
        const auto& u = m.vencimientos.back();
        m.r = -std::log(u.DF) / u.T;
        m.q = m.r - std::log(u.F / m.spot) / u.T;
    }
    return m;
}

} // namespace est
