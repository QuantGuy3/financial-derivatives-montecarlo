#include "cotizaciones.hpp"
#include "producto.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace est {

namespace {

// minúsculas, sin tildes y sin espacios/guiones/puntos/subrayados
std::string clave_columna(const std::string& s0) {
    std::string s;
    for (size_t i = 0; i < s0.size(); ++i) {
        unsigned char c = (unsigned char)s0[i];
        if (c == 0xC3 && i + 1 < s0.size()) {
            unsigned char d = (unsigned char)s0[++i];
            const char* m = nullptr;
            switch (d) {
                case 0xA1: case 0x81: m = "a"; break;
                case 0xA9: case 0x89: m = "e"; break;
                case 0xAD: case 0x8D: m = "i"; break;
                case 0xB3: case 0x93: m = "o"; break;
                case 0xBA: case 0x9A: case 0xBC: m = "u"; break;
                case 0xB1: case 0x91: m = "n"; break;
                default: break;
            }
            if (m) s += m;
            continue;
        }
        if (std::isalnum(c)) s += (char)std::tolower(c);
    }
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

// Divide una línea CSV respetando comillas
std::vector<std::string> dividir(const std::string& l, char sep) {
    std::vector<std::string> out;
    std::string cur;
    bool comillas = false;
    for (char c : l) {
        if (c == '"') { comillas = !comillas; cur += c; }
        else if (c == sep && !comillas) { out.push_back(recortar(cur)); cur.clear(); }
        else cur += c;
    }
    out.push_back(recortar(cur));
    return out;
}

enum Col { C_VENC, C_TIPO, C_STRIKE, C_BID, C_ASK, C_PRECIO, C_ULTIMO, C_VOL, C_VOLUMEN, C_OI, C_CONTRATO, C_PESO, C_N };

int columna_de(const std::string& nombre) {
    static const std::map<std::string, int> m = {
        {"vencimiento", C_VENC}, {"expiration", C_VENC}, {"expiry", C_VENC}, {"expirationdate", C_VENC},
        {"maturity", C_VENC}, {"fechavencimiento", C_VENC}, {"venc", C_VENC}, {"plazo", C_VENC},
        {"tipo", C_TIPO}, {"type", C_TIPO}, {"optiontype", C_TIPO}, {"cp", C_TIPO}, {"callput", C_TIPO},
        {"right", C_TIPO}, {"clase", C_TIPO},
        {"strike", C_STRIKE}, {"ejercicio", C_STRIKE}, {"precioejercicio", C_STRIKE}, {"k", C_STRIKE},
        {"bid", C_BID}, {"compra", C_BID}, {"preciocompra", C_BID},
        {"ask", C_ASK}, {"venta", C_ASK}, {"precioventa", C_ASK}, {"oferta", C_ASK},
        {"precio", C_PRECIO}, {"mid", C_PRECIO}, {"medio", C_PRECIO}, {"price", C_PRECIO}, {"prima", C_PRECIO},
        {"ultimo", C_ULTIMO}, {"last", C_ULTIMO}, {"lastprice", C_ULTIMO},
        {"volimplicita", C_VOL}, {"vol", C_VOL}, {"iv", C_VOL}, {"impliedvolatility", C_VOL},
        {"volatilidad", C_VOL}, {"volatilidadimplicita", C_VOL}, {"sigma", C_VOL},
        {"volumen", C_VOLUMEN}, {"volume", C_VOLUMEN},
        {"interesabierto", C_OI}, {"openinterest", C_OI}, {"oi", C_OI}, {"posicionabierta", C_OI},
        {"contractsymbol", C_CONTRATO}, {"contrato", C_CONTRATO}, {"simbolo", C_CONTRATO}, {"symbol", C_CONTRATO},
        {"peso", C_PESO}, {"weight", C_PESO}};
    auto it = m.find(clave_columna(nombre));
    return it == m.end() ? -1 : it->second;
}

} // namespace

double Cotizacion::referencia() const {
    if (bid > 0 && ask > 0 && ask >= bid) return 0.5 * (bid + ask);
    if (precio > 0) return precio;
    if (ultimo > 0) return ultimo;
    return -1;
}

std::vector<double> CadenaOpciones::vencimientos() const {
    std::vector<double> v;
    for (const auto& q : cot) v.push_back(q.T);
    std::sort(v.begin(), v.end());
    std::vector<double> u;
    for (double t : v)
        if (u.empty() || t - u.back() > 1e-9) u.push_back(t);
    return u;
}

double CadenaOpciones::vencimiento_cercano(double T, double* diferencia) const {
    double mejor = -1, dmin = 1e300;
    for (double t : vencimientos())
        if (std::fabs(t - T) < dmin) { dmin = std::fabs(t - T); mejor = t; }
    if (diferencia) *diferencia = dmin;
    return mejor;
}

CadenaOpciones leer_cadena_csv(const std::string& ruta, int dias_val) {
    std::ifstream f(ruta);
    if (!f) throw std::runtime_error("no se puede abrir la cadena de opciones \"" + ruta + "\"");
    CadenaOpciones c;
    c.ruta = ruta;
    std::string linea;
    std::vector<std::string> datos;
    std::vector<int> num_linea;
    int nl = 0;
    while (std::getline(f, linea)) {
        ++nl;
        if (!linea.empty() && linea.back() == '\r') linea.pop_back();
        if (nl == 1 && linea.size() >= 3 && (unsigned char)linea[0] == 0xEF) linea = linea.substr(3); // BOM
        std::string t = recortar(linea);
        if (t.empty()) continue;
        if (t[0] == '#') {
            size_t dp = t.find(':');
            if (dp == std::string::npos) continue;
            std::string k = clave_columna(t.substr(1, dp - 1)), v = recortar(t.substr(dp + 1));
            bool ok = false;
            if (k == "spot" || k == "subyacenteprecio" || k == "precio") { double x = texto_a_real(v, &ok); if (ok) c.spot = x; }
            else if (k == "fecha" || k == "fechavaloracion" || k == "date") c.fecha = v;
            else if (k == "subyacente" || k == "ticker" || k == "underlying") c.subyacente = v;
            else if (k == "tipointeres" || k == "tipo" || k == "rate") { double x = texto_a_real(v, &ok); if (ok) c.tipo_interes = x; }
            else if (k == "dividendos" || k == "dividendo" || k == "q") { double x = texto_a_real(v, &ok); if (ok) c.dividendos = x; }
            else if (k == "fuente" || k == "source") c.fuente = v;
            continue;
        }
        datos.push_back(linea);
        num_linea.push_back(nl);
    }
    if (datos.empty()) throw std::runtime_error("la cadena \"" + ruta + "\" no tiene datos");

    // Separador: el más frecuente en la cabecera
    const std::string& cab = datos[0];
    char sep = ',';
    size_t nc = std::count(cab.begin(), cab.end(), ','), np = std::count(cab.begin(), cab.end(), ';'),
           nt = std::count(cab.begin(), cab.end(), '\t');
    if (np > nc && np >= nt) sep = ';';
    else if (nt > nc && nt > np) sep = '\t';
    const bool coma_decimal = (sep != ',');

    std::vector<std::string> nombres = dividir(cab, sep);
    std::vector<int> col(nombres.size(), -1);
    int pos[C_N];
    std::fill(pos, pos + C_N, -1);
    for (size_t i = 0; i < nombres.size(); ++i) {
        col[i] = columna_de(nombres[i]);
        if (col[i] >= 0 && pos[col[i]] < 0) pos[col[i]] = (int)i;
    }
    if (pos[C_STRIKE] < 0) throw std::runtime_error("la cadena \"" + ruta + "\" no tiene columna de strike");
    if (pos[C_TIPO] < 0 && pos[C_CONTRATO] < 0)
        throw std::runtime_error("la cadena \"" + ruta + "\" no indica si cada opción es call o put "
                                 "(columna \"tipo\" o el símbolo del contrato)");
    if (pos[C_VENC] < 0) throw std::runtime_error("la cadena \"" + ruta + "\" no tiene columna de vencimiento");

    int dv = dias_val;
    if (dv < 0 && !c.fecha.empty() && !texto_a_fecha(c.fecha, dv)) dv = -1;
    c.dias_valoracion = dv;

    auto numero = [&](const std::string& s0, bool* ok) -> double {
        std::string s = s0;
        if (coma_decimal) std::replace(s.begin(), s.end(), ',', '.');
        if (s.empty() || s == "nan" || s == "NaN" || s == "-" || s == "null") { *ok = false; return -1; }
        return texto_a_real(s, ok);
    };

    for (size_t r = 1; r < datos.size(); ++r) {
        std::vector<std::string> v = dividir(datos[r], sep);
        auto campo = [&](int c_) -> std::string { return (pos[c_] >= 0 && pos[c_] < (int)v.size()) ? v[(size_t)pos[c_]] : ""; };
        const std::string donde = ruta + ":" + std::to_string(num_linea[r]);
        Cotizacion q;
        q.linea = num_linea[r];
        bool ok = false;
        q.K = numero(campo(C_STRIKE), &ok);
        if (!ok || !(q.K > 0)) { c.avisos.push_back(donde + ": strike no válido; se ignora la línea"); continue; }

        // Tipo
        std::string tipo = clave_columna(campo(C_TIPO));
        if (tipo.empty() && pos[C_CONTRATO] >= 0) {
            // Símbolo OCC: raíz + AAMMDD + C/P + strike×1000 en 8 cifras
            std::string s = campo(C_CONTRATO);
            if (s.size() > 9) {
                char cp = (char)std::toupper((unsigned char)s[s.size() - 9]);
                if (cp == 'C') tipo = "call";
                else if (cp == 'P') tipo = "put";
            }
        }
        if (tipo == "call" || tipo == "c" || tipo == "calls" || tipo == "compra" || tipo == "opciondecompra") q.call = true;
        else if (tipo == "put" || tipo == "p" || tipo == "puts" || tipo == "venta" || tipo == "opciondeventa") q.call = false;
        else { c.avisos.push_back(donde + ": no se sabe si es call o put; se ignora la línea"); continue; }

        // Vencimiento
        q.vencimiento = campo(C_VENC);
        int dias = 0;
        std::string fv = q.vencimiento.substr(0, 10);
        if (texto_a_fecha(fv, dias)) {
            if (dv < 0)
                throw std::runtime_error(donde + ": los vencimientos son fechas; indique la fecha de los datos "
                                         "(línea \"# fecha: AAAA-MM-DD\" en el fichero o --fecha)");
            q.T = (dias - dv) / 365.0;
        } else {
            bool okp = false;
            q.T = texto_a_plazo(q.vencimiento, &okp);
            if (!okp) { c.avisos.push_back(donde + ": vencimiento \"" + q.vencimiento + "\" no válido; se ignora"); continue; }
        }
        if (!(q.T > 0)) { c.avisos.push_back(donde + ": opción ya vencida; se ignora"); continue; }

        double x;
        if ((x = numero(campo(C_BID), &ok)), ok) q.bid = x;
        if ((x = numero(campo(C_ASK), &ok)), ok) q.ask = x;
        if ((x = numero(campo(C_PRECIO), &ok)), ok) q.precio = x;
        if ((x = numero(campo(C_ULTIMO), &ok)), ok) q.ultimo = x;
        if ((x = numero(campo(C_VOLUMEN), &ok)), ok) q.volumen = x;
        if ((x = numero(campo(C_OI), &ok)), ok) q.interes_abierto = x;
        if ((x = numero(campo(C_PESO), &ok)), ok && x >= 0) q.peso = x;
        std::string sv = campo(C_VOL);
        if ((x = numero(sv, &ok)), ok && x > 0) {
            // Sin '%': un número mayor que 3 se entiende como puntos porcentuales (24.5 = 24.5%)
            if (sv.find('%') == std::string::npos && x > 3.0) x /= 100.0;
            q.vol = x;
        }
        c.cot.push_back(q);
    }
    if (c.cot.empty()) throw std::runtime_error("la cadena \"" + ruta + "\" no tiene ninguna cotización válida");
    return c;
}

std::vector<StrikeCotizado> strikes_fuera_del_dinero(const CadenaOpciones& c, double T, double S0,
                                                     double volumen_min, double K_min, double K_max) {
    std::vector<StrikeCotizado> out;
    for (const auto& q : c.cot) {
        if (std::fabs(q.T - T) > 1e-9) continue;
        if (q.call ? !(q.K > S0) : !(q.K < S0)) continue;
        if (q.K < K_min || q.K > K_max) continue;
        if (volumen_min > 0 && !(q.volumen >= volumen_min)) continue;
        if (!q.tiene_precio()) continue;
        out.push_back(StrikeCotizado{q.K, q.call, &q});
    }
    std::sort(out.begin(), out.end(), [](const StrikeCotizado& a, const StrikeCotizado& b) { return a.K < b.K; });
    // Un strike repetido (misma opción dos veces) se queda con la primera
    out.erase(std::unique(out.begin(), out.end(), [](const StrikeCotizado& a, const StrikeCotizado& b) {
                  return std::fabs(a.K - b.K) < 1e-12 * std::max(1.0, a.K);
              }), out.end());
    return out;
}

} // namespace est
