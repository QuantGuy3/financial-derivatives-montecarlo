#include "producto.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>

namespace est {

// ===================================================================== //
// Catálogo de modelos                                                    //
// ===================================================================== //

namespace {

DefParam P(std::string clave, std::vector<std::string> alias, std::string desc, double def,
           bool pct, double mn, double mx, bool abierto, std::string vol_de = "") {
    return DefParam{std::move(clave), std::move(alias), std::move(desc), def, pct, mn, mx, abierto,
                    std::move(vol_de)};
}

std::vector<DefModelo> construir_catalogo() {
    std::vector<DefModelo> c;

    const DefParam vol_heston = P("vol_inicial", {"sqrt_v0"}, "volatilidad instantánea inicial √v0",
                                  0.20, true, 0.0, 5.0, false, "v0");
    const DefParam kappa = P("kappa", {"reversion", "velocidad_reversion"},
                             "velocidad de reversión a la media de la varianza", 1.5, false, 0.0, 50.0, true);
    const DefParam vol_lp = P("vol_largo_plazo", {"sqrt_theta"}, "volatilidad de largo plazo √θ",
                              0.20, true, 0.0, 5.0, true, "theta");
    const DefParam xi = P("xi", {"vol_de_vol", "sigma_v"}, "volatilidad de la varianza (vol de vol)",
                          0.50, false, 0.0, 10.0, true);
    const DefParam rho = P("rho", {"correlacion_vol"}, "correlación precio-volatilidad", -0.70, false,
                           -1.0, 1.0, false);
    const DefParam lam = P("lambda", {"intensidad_saltos", "intensidad"},
                           "intensidad de saltos (saltos esperados por año)", 0.50, false, 0.0, 100.0, false);
    const DefParam muj = P("media_salto", {"mu_j", "mu_salto"},
                           "media del log-salto (-0.10 ≈ caída media del 9.5%)", -0.10, false, -5.0, 5.0, false);
    const DefParam sigj = P("vol_salto", {"sigma_j", "delta"}, "desviación típica del log-salto",
                            0.15, false, 0.0, 5.0, false);
    const DefParam volmax = P("vol_max", {}, "tope de la volatilidad instantánea", 3.0, true, 0.0,
                              50.0, true);

    c.push_back(DefModelo{TipoModelo::BlackScholes, "black_scholes", {"bs", "gbm", "lognormal"},
        "Black-Scholes",
        "Volatilidad constante (movimiento browniano geométrico). Simulación exacta.",
        {P("volatilidad", {"vol", "sigma"}, "volatilidad anual", 0.20, true, 0.0, 5.0, false)},
        {"volatilidad"}});

    c.push_back(DefModelo{TipoModelo::Heston, "heston", {},
        "Heston",
        "Volatilidad estocástica con reversión a la media (sonrisa/skew de volatilidad).",
        {vol_heston, kappa, vol_lp, xi, rho},
        {"vol_inicial", "vol_largo_plazo"}});

    c.push_back(DefModelo{TipoModelo::Merton, "merton", {"merton_saltos", "salto_difusion"},
        "Merton (difusión con saltos)",
        "Black-Scholes más saltos lognormales de Poisson (caídas bruscas). Simulación exacta.",
        {P("volatilidad", {"vol", "sigma"}, "volatilidad de la parte difusiva", 0.15, true, 0.0, 5.0, false),
         lam, muj, sigj},
        {"volatilidad"}});

    c.push_back(DefModelo{TipoModelo::Bates, "bates", {"heston_saltos"},
        "Bates (Heston con saltos)",
        "Volatilidad estocástica de Heston más saltos de Merton.",
        {vol_heston, kappa, vol_lp, xi, rho, lam, muj, sigj},
        {"vol_inicial", "vol_largo_plazo"}});

    c.push_back(DefModelo{TipoModelo::VolLocal, "vol_local", {"dupire", "cev", "volatilidad_local"},
        "Volatilidad local paramétrica",
        "σ(S,t) = σ0·exp(-α·t)·(S/S0)^(β-1): β<1 da skew (tipo CEV), α hace decaer la vol con el plazo.",
        {P("sigma0", {"volatilidad", "vol"}, "volatilidad en el spot inicial", 0.20, true, 0.0, 5.0, true),
         P("alpha", {"decaimiento"}, "decaimiento temporal α de la volatilidad", 0.0, false, -10.0, 10.0, false),
         P("beta", {"elasticidad"}, "elasticidad β (1 = Black-Scholes, <1 = skew negativo)", 0.7, false,
           -2.0, 3.0, false),
         volmax},
        {"sigma0"}});

    c.push_back(DefModelo{TipoModelo::SABR, "sabr", {},
        "SABR",
        "Volatilidad estocástica lognormal con elasticidad β (Hagan et al. 2002), sobre el forward.",
        {P("alpha", {"volatilidad", "vol_atm"}, "volatilidad inicial α (en el forward inicial)", 0.20, true,
           0.0, 5.0, true),
         P("beta", {"elasticidad"}, "elasticidad β ∈ [0,1]", 0.70, false, 0.0, 1.0, false),
         P("nu", {"vol_de_vol"}, "volatilidad de la volatilidad ν", 0.40, false, 0.0, 10.0, false),
         P("rho", {"correlacion_vol"}, "correlación precio-volatilidad", -0.50, false, -1.0, 1.0, false),
         volmax},
        {"alpha"}});

    c.push_back(DefModelo{TipoModelo::NIG, "nig", {"normal_inverse_gaussian"},
        "NIG (Normal Inverse Gaussian)",
        "Proceso de Lévy de saltos puros: colas gruesas y asimetría. Simulación exacta.",
        {P("sigma", {"volatilidad", "vol"}, "volatilidad σ del browniano subordinado", 0.20, true, 0.0, 5.0, true),
         P("theta", {"asimetria"}, "asimetría θ (negativa = skew bajista)", -0.10, false, -5.0, 5.0, false),
         P("kappa", {"curtosis"}, "varianza del subordinador κ (más alto = colas más gruesas)", 0.20, false,
           0.0, 20.0, true)},
        {"sigma"}});
    return c;
}

std::string minusculas(std::string s) {
    for (auto& ch : s) ch = (char)std::tolower((unsigned char)ch);
    return s;
}

std::string recortar(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

// Quita tildes comunes (UTF-8) y pasa a minúsculas para comparar nombres
std::string normalizar(const std::string& s0) {
    std::string s = minusculas(recortar(s0));
    static const char* pares[][2] = {
        {"á", "a"}, {"é", "e"}, {"í", "i"}, {"ó", "o"}, {"ú", "u"}, {"ñ", "n"},
        {"Á", "a"}, {"É", "e"}, {"Í", "i"}, {"Ó", "o"}, {"Ú", "u"}, {"Ñ", "n"}};
    for (auto& pr : pares) {
        size_t pos;
        while ((pos = s.find(pr[0])) != std::string::npos) s.replace(pos, std::strlen(pr[0]), pr[1]);
    }
    for (auto& ch : s) if (ch == ' ' || ch == '-') ch = '_';
    return s;
}

size_t distancia_edicion(const std::string& a, const std::string& b) {
    std::vector<size_t> prev(b.size() + 1), cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) prev[j] = j;
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= b.size(); ++j) {
            size_t c = (a[i - 1] == b[j - 1]) ? 0 : 1;
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + c});
        }
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

} // namespace

const std::vector<DefModelo>& catalogo_modelos() {
    static const std::vector<DefModelo> c = construir_catalogo();
    return c;
}

const DefModelo& def_modelo(TipoModelo t) {
    for (const auto& d : catalogo_modelos())
        if (d.tipo == t) return d;
    throw ErrorProducto("modelo desconocido");
}

const DefModelo* buscar_modelo(const std::string& nombre) {
    std::string n = normalizar(nombre);
    for (const auto& d : catalogo_modelos()) {
        if (normalizar(d.clave) == n) return &d;
        for (const auto& a : d.alias)
            if (normalizar(a) == n) return &d;
    }
    return nullptr;
}

double Modelo::operator[](const std::string& clave) const {
    const auto& d = def_modelo(tipo);
    for (size_t i = 0; i < d.params.size(); ++i)
        if (d.params[i].clave == clave) return valores.at(i);
    throw ErrorProducto("el modelo " + d.nombre + " no tiene el parámetro '" + clave + "'");
}

double& Modelo::operator[](const std::string& clave) {
    const auto& d = def_modelo(tipo);
    for (size_t i = 0; i < d.params.size(); ++i)
        if (d.params[i].clave == clave) return valores.at(i);
    throw ErrorProducto("el modelo " + d.nombre + " no tiene el parámetro '" + clave + "'");
}

std::string Modelo::resumen() const {
    const auto& d = def_modelo(tipo);
    std::string s = d.nombre + " (";
    for (size_t i = 0; i < d.params.size(); ++i) {
        if (i) s += ", ";
        s += d.params[i].clave + "=";
        s += d.params[i].porcentaje ? porcentaje(valores[i]) : fmt(valores[i], 4);
    }
    return s + ")";
}

// ===================================================================== //
// Utilidades de texto                                                    //
// ===================================================================== //

std::string fmt(double x, int dec) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", dec, x);
    std::string s = buf;
    // quitar ceros finales superfluos
    if (s.find('.') != std::string::npos) {
        while (!s.empty() && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
    }
    if (s == "-0") s = "0";
    return s;
}

std::string porcentaje(double x, int dec) {
    if (x >= HD_INFINITO * 0.5) return "+inf";
    if (x <= -HD_INFINITO * 0.5) return "-inf";
    return fmt(100.0 * x, std::max(dec, 0) + 2) + "%";
}

double texto_a_real(const std::string& s0, bool* ok) {
    std::string s = minusculas(recortar(s0));
    if (ok) *ok = false;
    if (s.empty()) return 0.0;
    double escala = 1.0;
    if (!s.empty() && s.back() == '%') { escala = 0.01; s.pop_back(); }
    else if (s.size() > 2 && (s.compare(s.size() - 2, 2, "pb") == 0 || s.compare(s.size() - 2, 2, "bp") == 0)) {
        escala = 1e-4; s.resize(s.size() - 2);
    }
    s = recortar(s);
    if (s.find('.') == std::string::npos) std::replace(s.begin(), s.end(), ',', '.');
    if (s == "inf" || s == "+inf" || s == "infinito") { if (ok) *ok = true; return HD_INFINITO; }
    if (s == "-inf" || s == "-infinito") { if (ok) *ok = true; return -HD_INFINITO; }
    char* fin = nullptr;
    double x = std::strtod(s.c_str(), &fin);
    if (fin == s.c_str()) return 0.0;
    while (*fin && std::isspace((unsigned char)*fin)) ++fin;
    if (*fin != '\0') return 0.0;
    if (ok) *ok = true;
    return x * escala;
}

double texto_a_plazo(const std::string& s0, bool* ok) {
    std::string s = minusculas(recortar(s0));
    if (ok) *ok = false;
    if (s.empty()) return 0.0;
    // Número a secas = años
    bool solo_num = true;
    for (char ch : s)
        if (!(std::isdigit((unsigned char)ch) || ch == '.' || ch == ',' || ch == '-' || ch == '+' ||
              ch == 'e')) { solo_num = false; break; }
    if (solo_num) return texto_a_real(s, ok);
    double total = 0.0;
    size_t i = 0;
    bool algo = false;
    while (i < s.size()) {
        while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
        if (i >= s.size()) break;
        size_t ini = i;
        while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '.' || s[i] == ',')) ++i;
        if (i == ini) return 0.0;
        std::string num = s.substr(ini, i - ini);
        std::replace(num.begin(), num.end(), ',', '.');
        double v = std::strtod(num.c_str(), nullptr);
        while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
        if (i >= s.size()) return 0.0;
        char u = s[i++];
        // admitir palabras completas: "meses", "anos", "años", "semanas", "dias"
        while (i < s.size() && std::isalpha((unsigned char)s[i])) ++i;
        while (i < s.size() && (unsigned char)s[i] >= 0x80) ++i; // "ñ"/tildes
        while (i < s.size() && std::isalpha((unsigned char)s[i])) ++i;
        switch (u) {
            case 'd': total += v / 365.0; break;
            case 's': case 'w': total += v * 7.0 / 365.0; break;
            case 'm': total += v / 12.0; break;
            case 'a': case 'y': total += v; break;
            default: return 0.0;
        }
        algo = true;
    }
    if (!algo) return 0.0;
    if (ok) *ok = true;
    return total;
}

int texto_a_frecuencia(const std::string& s0) {
    std::string s = normalizar(s0);
    if (s == "mensual") return 12;
    if (s == "bimestral") return 6;
    if (s == "trimestral") return 4;
    if (s == "cuatrimestral") return 3;
    if (s == "semestral") return 2;
    if (s == "anual") return 1;
    if (s == "semanal") return 52;
    if (s == "diaria") return 252;
    bool ok = false;
    double t = texto_a_plazo(s, &ok);
    if (ok && t > 0 && s.find_first_of("dmswya") != std::string::npos) {
        double f = 1.0 / t;
        if (std::fabs(f - std::round(f)) < 1e-6) return (int)std::round(f);
    }
    return 0;
}

namespace {
long dias_desde_civil(long y, unsigned m, unsigned d) { // H. Hinnant
    y -= m <= 2;
    const long era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = (unsigned)(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + (long)doe - 719468;
}
} // namespace

bool texto_a_fecha(const std::string& s0, int& dias) {
    std::string s = recortar(s0);
    int y = 0, m = 0, d = 0;
    char c1 = 0, c2 = 0;
    if (std::sscanf(s.c_str(), "%d%c%d%c%d", &y, &c1, &m, &c2, &d) != 5) return false;
    if (c1 == '-' && c2 == '-' && y > 1900) {
        // AAAA-MM-DD
    } else if ((c1 == '/' || c1 == '-' || c1 == '.') && c1 == c2 && d > 1900) {
        std::swap(y, d); // DD/MM/AAAA
    } else return false;
    if (m < 1 || m > 12 || d < 1 || d > 31) return false;
    static const int dm[] = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (d > dm[m - 1]) return false;
    dias = (int)dias_desde_civil(y, (unsigned)m, (unsigned)d);
    return true;
}

double Curva::tipo(double tt) const {
    if (t.empty()) return 0.0;
    if (t.size() == 1 || tt <= t.front()) return r.front();
    if (tt >= t.back()) return r.back();
    size_t k = (size_t)(std::upper_bound(t.begin(), t.end(), tt) - t.begin());
    double w = (tt - t[k - 1]) / (t[k] - t[k - 1]);
    return r[k - 1] + w * (r[k] - r[k - 1]);
}

int Producto::indice_cliquet() const {
    for (size_t b = 0; b < bloques.size(); ++b)
        if (bloques[b].tipo == BLQ_CLIQUET) return (int)b;
    return -1;
}

std::string Producto::nombre_cesta() const {
    if (activos.size() == 1) return "un único subyacente";
    switch (cesta) {
        case CESTA_PEOR:  return "el peor de (worst-of)";
        case CESTA_MEJOR: return "el mejor de (best-of)";
        case CESTA_MEDIA: return "media equiponderada";
        default:          return "media ponderada";
    }
}

// ===================================================================== //
// Lectura del documento con contexto (ruta + línea) para los mensajes    //
// ===================================================================== //

namespace {

class Nodo {
public:
    Nodo(const json::Valor* v, std::string ruta, std::vector<std::string>* avisos, int dias_val)
        : v_(v), ruta_(std::move(ruta)), avisos_(avisos), dias_val_(dias_val) {}

    bool existe() const { return v_ != nullptr && !v_->es_nulo(); }
    const json::Valor& valor() const { return *v_; }
    const std::string& ruta() const { return ruta_; }

    Nodo campo(const std::string& clave) const {
        const json::Valor* c = (v_ && v_->es_objeto()) ? v_->buscar(clave) : nullptr;
        return Nodo(c, sub(clave), avisos_, dias_val_);
    }
    // Primera de varias claves alternativas que exista
    Nodo alguno(std::initializer_list<const char*> claves) const {
        for (const char* k : claves) {
            Nodo n = campo(k);
            if (n.existe()) return n;
        }
        return campo(*claves.begin());
    }
    Nodo elemento(size_t i) const {
        return Nodo(&(*v_)[i], ruta_ + "[" + std::to_string(i) + "]", avisos_, dias_val_);
    }
    size_t tamano() const { return v_ ? v_->tamano() : 0; }
    bool es_lista() const { return v_ && v_->es_lista(); }
    bool es_objeto() const { return v_ && v_->es_objeto(); }
    bool es_texto() const { return v_ && v_->es_texto(); }
    bool es_numero() const { return v_ && v_->es_numero(); }

    [[noreturn]] void error(const std::string& msg) const {
        std::string donde = v_ ? v_->donde() : "";
        throw ErrorProducto("Error en '" + (ruta_.empty() ? std::string("(documento)") : ruta_) + "'" +
                            donde + ": " + msg);
    }
    void aviso(const std::string& msg) const {
        if (avisos_) avisos_->push_back("'" + ruta_ + "': " + msg);
    }

    void requerir(const std::string& que) const {
        if (!existe()) error("falta " + que);
    }

    double real() const {
        requerir("un valor numérico");
        if (v_->es_numero()) return v_->numero();
        if (v_->es_texto()) {
            bool ok = false;
            double x = texto_a_real(v_->texto(), &ok);
            if (ok) return x;
            error("no se entiende el número \"" + v_->texto() + "\" (ejemplos válidos: 0.6, \"60%\", \"25pb\")");
        }
        error(std::string("se esperaba un número y hay un ") + v_->nombre_tipo());
    }
    double real(double defecto) const { return existe() ? real() : defecto; }
    double real_en(double mn, double mx, bool abierto_min = false) const {
        double x = real();
        comprobar_rango(x, mn, mx, abierto_min);
        return x;
    }
    void comprobar_rango(double x, double mn, double mx, bool abierto_min) const {
        bool mal = abierto_min ? !(x > mn) : !(x >= mn);
        if (mal || x > mx) {
            std::string lim = (abierto_min ? "(" : "[") + fmt(mn, 6) + ", " + fmt(mx, 6) + "]";
            error("valor fuera de rango: " + fmt(x, 8) + " (debe estar en " + lim + ")");
        }
    }

    // Plazo en años: número, "18M", "2A6M" o fecha (si hay fecha de valoración)
    double plazo() const {
        requerir("un plazo o una fecha");
        if (v_->es_numero()) return v_->numero();
        if (!v_->es_texto()) error(std::string("se esperaba un plazo y hay un ") + v_->nombre_tipo());
        const std::string& s = v_->texto();
        int dias = 0;
        if (texto_a_fecha(s, dias)) {
            if (dias_val_ < 0)
                error("para usar fechas (\"" + s + "\") indique \"fecha_valoracion\" en el producto");
            return (dias - dias_val_) / 365.0;
        }
        bool ok = false;
        double t = texto_a_plazo(s, &ok);
        if (!ok) error("no se entiende el plazo \"" + s + "\" (ejemplos: 1.5, \"18M\", \"3A\", \"2026-12-31\")");
        return t;
    }

    bool booleano() const {
        requerir("un valor sí/no");
        if (v_->es_booleano()) return v_->booleano();
        if (v_->es_texto()) {
            std::string s = normalizar(v_->texto());
            if (s == "si" || s == "s" || s == "true" || s == "verdadero" || s == "yes") return true;
            if (s == "no" || s == "n" || s == "false" || s == "falso") return false;
        }
        if (v_->es_numero()) return v_->numero() != 0.0;
        error("se esperaba true/false");
    }
    bool booleano(bool defecto) const { return existe() ? booleano() : defecto; }

    std::string texto() const {
        requerir("un texto");
        if (v_->es_texto()) return v_->texto();
        if (v_->es_numero()) return json::numero_a_texto(v_->numero());
        error(std::string("se esperaba un texto y hay un ") + v_->nombre_tipo());
    }
    std::string texto(const std::string& defecto) const { return existe() ? texto() : defecto; }

    int entero() const {
        double x = real();
        if (std::fabs(x - std::round(x)) > 1e-9) error("se esperaba un número entero");
        return (int)std::lround(x);
    }
    int entero(int defecto) const { return existe() ? entero() : defecto; }

    // Avisa de las claves que no se reconocen (posibles erratas)
    void claves_permitidas(const std::vector<std::string>& validas) const {
        if (!v_ || !v_->es_objeto()) return;
        for (const auto& k : v_->claves()) {
            if (std::find(validas.begin(), validas.end(), k) != validas.end()) continue;
            if (!k.empty() && (k[0] == '_' || k.rfind("nota", 0) == 0 || k.rfind("coment", 0) == 0))
                continue; // comentarios libres
            std::string mejor;
            size_t dmin = 3;
            for (const auto& c : validas) {
                size_t d = distancia_edicion(k, c);
                if (d < dmin) { dmin = d; mejor = c; }
            }
            std::string msg = "clave desconocida \"" + k + "\"" + v_->buscar(k)->donde() + ": se ignora";
            if (!mejor.empty()) msg += " (¿quería decir \"" + mejor + "\"?)";
            aviso(msg);
        }
    }

    // Escalar o lista de n valores reales
    std::vector<double> lista_reales(size_t n, const std::string& que) const {
        std::vector<double> out;
        if (es_lista()) {
            if (tamano() != n)
                error("la lista de " + que + " tiene " + std::to_string(tamano()) +
                      " elementos y hay " + std::to_string(n) + " fechas de observación");
            for (size_t i = 0; i < n; ++i) out.push_back(elemento(i).real());
        } else {
            out.assign(n, real());
        }
        return out;
    }

private:
    const json::Valor* v_;
    std::string ruta_;
    std::vector<std::string>* avisos_;
    int dias_val_;
    std::string sub(const std::string& k) const { return ruta_.empty() ? k : ruta_ + "." + k; }
};

// Etiqueta legible de un plazo en años: "3M", "1A", "1A6M" o "t=0.083"
std::string etiqueta_plazo(double t) {
    double meses = t * 12.0;
    long m = std::lround(meses);
    if (std::fabs(meses - (double)m) < 1e-6 && m > 0) {
        long a = m / 12, r = m % 12;
        std::string s;
        if (a) s += std::to_string(a) + "A";
        if (r) s += std::to_string(r) + "M";
        return s;
    }
    return "t=" + fmt(t, 4);
}

// ----------------------- //
// Lectura de cada sección //
// ----------------------- //

Modelo leer_modelo(const Nodo& nm) {
    if (!nm.existe())
        nm.error("falta el modelo del subyacente, p. ej. \"modelo\": {\"tipo\": \"black_scholes\", "
                 "\"volatilidad\": \"20%\"}");
    Nodo ntipo = nm.es_texto() ? nm : nm.campo("tipo");
    std::string nombre = ntipo.texto("");
    if (nombre.empty()) nm.error("falta \"tipo\" del modelo");
    const DefModelo* d = buscar_modelo(nombre);
    if (!d) {
        std::string lista;
        for (const auto& m : catalogo_modelos()) lista += (lista.empty() ? "" : ", ") + m.clave;
        ntipo.error("modelo \"" + nombre + "\" desconocido. Modelos disponibles: " + lista);
    }
    Modelo m;
    m.tipo = d->tipo;
    std::vector<std::string> validas = {"tipo"};
    for (const auto& prm : d->params) {
        validas.push_back(prm.clave);
        for (const auto& a : prm.alias) validas.push_back(a);
        if (!prm.es_vol_de.empty()) validas.push_back(prm.es_vol_de);
        double val;
        Nodo n = nm.campo(prm.clave);
        for (size_t k = 0; !n.existe() && k < prm.alias.size(); ++k) n = nm.campo(prm.alias[k]);
        if (n.existe()) {
            val = n.real();
        } else if (!prm.es_vol_de.empty() && nm.campo(prm.es_vol_de).existe()) {
            // se ha dado la varianza (v0, theta): convertir a volatilidad
            Nodo nv = nm.campo(prm.es_vol_de);
            double var = nv.real();
            if (var < 0) nv.error("la varianza no puede ser negativa");
            val = std::sqrt(var);
            n = nv;
        } else if (prm.clave == "vol_max") {
            val = prm.defecto; // opcional
        } else {
            nm.error("falta el parámetro \"" + prm.clave + "\" del modelo " + d->nombre + " (" +
                     prm.descripcion + ")");
        }
        if (n.existe()) n.comprobar_rango(val, prm.min, prm.max, prm.min_abierto);
        m.valores.push_back(val);
    }
    nm.claves_permitidas(validas);

    // Comprobaciones conjuntas
    if (m.tipo == TipoModelo::NIG) {
        double s = m["sigma"], th = m["theta"], k = m["kappa"];
        if (1.0 - s * s * k - 2.0 * th * k <= 0.0)
            nm.error("parámetros NIG incompatibles: hace falta 1 - sigma²·kappa - 2·theta·kappa > 0 "
                     "(si no, el precio descontado no es una martingala)");
    }
    // (Si no se cumple la condición de Feller 2κθ ≥ ξ², algo habitual en
    // calibraciones reales, la varianza puede tocar 0: el esquema de truncamiento
    // completo lo admite, así que no se avisa.)
    return m;
}

void leer_correlacion(const Nodo& nc, Producto& p) {
    const size_t n = p.activos.size();
    p.correlacion.assign(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) p.correlacion[i * n + i] = 1.0;
    if (n == 1) return;
    if (!nc.existe())
        nc.error("hay " + std::to_string(n) + " subyacentes: indique \"correlacion\" (un número común, "
                 "una matriz, o {\"comun\": 0.5, \"A/B\": 0.7})");
    auto poner = [&](size_t i, size_t j, double r, const Nodo& donde) {
        if (r < -1.0 || r > 1.0) donde.error("correlación fuera de [-1, 1]: " + fmt(r, 6));
        p.correlacion[i * n + j] = r;
        p.correlacion[j * n + i] = r;
    };
    if (nc.es_numero() || nc.es_texto()) {
        double r = nc.real();
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j) poner(i, j, r, nc);
    } else if (nc.es_lista()) {
        if (nc.tamano() != n) nc.error("la matriz de correlación debe tener " + std::to_string(n) + " filas");
        for (size_t i = 0; i < n; ++i) {
            Nodo fila = nc.elemento(i);
            if (!fila.es_lista() || fila.tamano() != n)
                fila.error("cada fila de la matriz debe tener " + std::to_string(n) + " elementos");
            for (size_t j = 0; j < n; ++j) {
                double r = fila.elemento(j).real();
                if (i == j && std::fabs(r - 1.0) > 1e-9) fila.elemento(j).error("la diagonal debe ser 1");
                if (j < i && std::fabs(r - p.correlacion[i * n + j]) > 1e-9)
                    fila.elemento(j).error("la matriz no es simétrica (" + fmt(r, 6) + " frente a " +
                                           fmt(p.correlacion[i * n + j], 6) + ")");
                if (j > i) poner(i, j, r, fila.elemento(j));
            }
        }
    } else if (nc.es_objeto()) {
        Nodo com = nc.alguno({"comun", "común", "resto"});
        double r0 = com.real(0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j) poner(i, j, r0, nc);
        const json::Valor& o = nc.valor();
        for (size_t k = 0; k < o.claves().size(); ++k) {
            const std::string& clave = o.claves()[k];
            if (normalizar(clave) == "comun" || clave == "resto") continue;
            size_t barra = clave.find('/');
            Nodo nv = nc.campo(clave);
            if (barra == std::string::npos) nv.error("las parejas se escriben \"NOMBRE1/NOMBRE2\"");
            std::string a = recortar(clave.substr(0, barra)), b = recortar(clave.substr(barra + 1));
            size_t ia = n, ib = n;
            for (size_t i = 0; i < n; ++i) {
                if (p.activos[i].nombre == a) ia = i;
                if (p.activos[i].nombre == b) ib = i;
            }
            if (ia == n || ib == n || ia == ib) nv.error("pareja de subyacentes desconocida \"" + clave + "\"");
            poner(ia, ib, nv.real(), nv);
        }
    } else {
        nc.error("formato de correlación no reconocido");
    }

    // Cholesky (Banachiewicz). Se admite semidefinida (pivote ~0) con tolerancia.
    p.chol.assign(n * n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double s = p.correlacion[i * n + j];
            for (size_t k = 0; k < j; ++k) s -= p.chol[i * n + k] * p.chol[j * n + k];
            if (i == j) {
                if (s < -1e-10)
                    nc.error("la matriz de correlación no es semidefinida positiva (falla el pivote " +
                             std::to_string(i + 1) + " de Cholesky); revise las correlaciones cruzadas");
                p.chol[i * n + i] = std::sqrt(std::max(s, 0.0));
            } else {
                p.chol[i * n + j] = (p.chol[j * n + j] > 1e-12) ? s / p.chol[j * n + j] : 0.0;
            }
        }
    }
}

int leer_observacion(const Nodo& n, int defecto) {
    if (!n.existe()) return defecto;
    std::string s = normalizar(n.texto());
    if (s == "vencimiento" || s == "europea" || s == "final" || s == "a_vencimiento") return OBS_VENCIMIENTO;
    if (s == "fechas" || s == "fechas_observacion" || s == "discreta" || s == "observaciones") return OBS_FECHAS;
    if (s == "diaria" || s == "cierre_diario") return OBS_DIARIA;
    if (s == "continua" || s == "americana") return OBS_CONTINUA;
    if (s == "ninguna" || s == "no") return OBS_NINGUNA;
    n.error("tipo de observación \"" + n.texto() + "\" desconocido (use vencimiento, fechas, diaria o continua)");
}

int leer_condicion(const Nodo& n) {
    if (!n.existe()) return COND_SIEMPRE;
    std::string s = normalizar(n.texto());
    if (s == "siempre") return COND_SIEMPRE;
    if (s == "barrera_tocada" || s == "con_barrera" || s == "si_barrera" || s == "ki" ||
        s == "barrera_capital_tocada" || s == "evento_barrera")
        return COND_CON_BARRERA;
    if (s == "barrera_no_tocada" || s == "sin_barrera" || s == "no_barrera" || s == "sin_ki" ||
        s == "barrera_capital_no_tocada")
        return COND_SIN_BARRERA;
    n.error("condición \"" + n.texto() + "\" desconocida (use siempre, barrera_tocada o barrera_no_tocada)");
}

Bloque leer_bloque(const Nodo& nb) {
    Bloque b;
    std::string tipo = normalizar(nb.campo("tipo").texto(""));
    if (tipo.empty()) nb.error("falta el \"tipo\" del componente (fijo, lineal, call, put, put_vendida, "
                               "digital, digital_put o cliquet)");
    b.cond = leer_condicion(nb.campo("condicion"));
    std::vector<std::string> validas = {"tipo", "condicion", "descripcion"};
    auto rd = [&](const char* clave, double def) {
        validas.push_back(clave);
        return nb.campo(clave).real(def);
    };
    if (tipo == "fijo" || tipo == "capital" || tipo == "bono") {
        b.tipo = BLQ_FIJO;
        b.importe = rd("importe", 1.0);
        b.descripcion = porcentaje(b.importe) + " del nominal";
    } else if (tipo == "lineal" || tipo == "subyacente" || tipo == "rendimiento") {
        b.tipo = BLQ_LINEAL;
        b.part = rd("participacion", 1.0);
        b.suelo = rd("suelo", -HD_INFINITO);
        b.techo = rd("techo", HD_INFINITO);
        b.descripcion = porcentaje(b.part) + " × P";
        if (b.suelo > -HD_INFINITO * 0.5 || b.techo < HD_INFINITO * 0.5)
            b.descripcion += " (P acotado a [" + porcentaje(b.suelo) + ", " + porcentaje(b.techo) + "])";
    } else if (tipo == "call") {
        b.tipo = BLQ_CALL;
        b.part = rd("participacion", 1.0);
        b.strike = rd("strike", 1.0);
        b.techo = rd("techo", HD_INFINITO);
        b.descripcion = porcentaje(b.part) + " × max(" +
                        (b.techo < HD_INFINITO * 0.5 ? "min(P, " + porcentaje(b.techo) + ")" : std::string("P")) +
                        " - " + porcentaje(b.strike) + ", 0)";
    } else if (tipo == "put") {
        b.tipo = BLQ_PUT;
        b.part = rd("participacion", 1.0);
        b.strike = rd("strike", 1.0);
        b.suelo = rd("suelo", -HD_INFINITO);
        b.descripcion = porcentaje(b.part) + " × max(" + porcentaje(b.strike) + " - " +
                        (b.suelo > -HD_INFINITO * 0.5 ? "max(P, " + porcentaje(b.suelo) + ")" : std::string("P")) +
                        ", 0)";
    } else if (tipo == "put_vendida" || tipo == "perdida" || tipo == "put_corta") {
        // Pérdida de capital "a la reverse convertible": -part · max(K - P, 0) / K
        b.tipo = BLQ_PUT;
        double part = rd("participacion", 1.0);
        b.strike = rd("strike", 1.0);
        b.suelo = rd("suelo", -HD_INFINITO);
        if (!(b.strike > 0)) nb.campo("strike").error("el strike de una put vendida debe ser > 0");
        b.part = -part / b.strike;
        b.descripcion = "- " + porcentaje(part) + " × max(" + porcentaje(b.strike) + " - P, 0) / " +
                        porcentaje(b.strike) + "   (pérdida proporcional a la caída)";
    } else if (tipo == "digital" || tipo == "digital_call" || tipo == "bonus") {
        b.tipo = BLQ_DIGITAL;
        b.importe = rd("importe", 0.0);
        b.strike = rd("strike", 1.0);
        b.descripcion = porcentaje(b.importe) + " si P ≥ " + porcentaje(b.strike);
    } else if (tipo == "digital_put") {
        b.tipo = BLQ_DIGITAL_PUT;
        b.importe = rd("importe", 0.0);
        b.strike = rd("strike", 1.0);
        b.descripcion = porcentaje(b.importe) + " si P < " + porcentaje(b.strike);
    } else if (tipo == "cliquet") {
        b.tipo = BLQ_CLIQUET;
        b.part = rd("participacion", 1.0);
        b.suelo_local = rd("suelo_local", -HD_INFINITO);
        b.techo_local = rd("techo_local", HD_INFINITO);
        b.suelo = rd("suelo_global", 0.0);
        b.techo = rd("techo_global", HD_INFINITO);
        b.descripcion = porcentaje(b.part) + " × clamp(Σ clamp(P_j/P_{j-1} - 1, " + porcentaje(b.suelo_local) +
                        ", " + porcentaje(b.techo_local) + "), " + porcentaje(b.suelo) + ", " +
                        porcentaje(b.techo) + ")";
    } else {
        nb.campo("tipo").error("tipo de componente \"" + tipo + "\" desconocido (use fijo, lineal, call, put, "
                               "put_vendida, digital, digital_put o cliquet)");
    }
    if (b.techo < b.suelo) nb.error("el techo es menor que el suelo");
    if (nb.campo("descripcion").existe()) b.descripcion = nb.campo("descripcion").texto();
    nb.claves_permitidas(validas);
    return b;
}

} // namespace

// ===================================================================== //
// compilar()                                                             //
// ===================================================================== //

Producto compilar(const json::Valor& doc) {
    if (!doc.es_objeto())
        throw ErrorProducto("el producto debe ser un objeto JSON { ... }");
    Producto p;

    // Fecha de valoración (opcional): permite escribir fechas absolutas
    int dias_val = -1;
    if (const json::Valor* fv = doc.buscar("fecha_valoracion")) {
        if (!fv->es_texto() || !texto_a_fecha(fv->texto(), dias_val))
            throw ErrorProducto("Error en 'fecha_valoracion'" + fv->donde() +
                                ": use el formato \"AAAA-MM-DD\" o \"DD/MM/AAAA\"");
    }
    Nodo raiz(&doc, "", &p.avisos, dias_val);
    raiz.claves_permitidas({"formato", "plantilla", "nombre", "descripcion", "nominal", "moneda",
                            "fecha_valoracion", "vencimiento", "mercado", "tipo_interes", "subyacentes",
                            "subyacente", "correlacion", "cesta", "observaciones", "cupones", "cupon",
                            "autocancelacion", "barrera_capital", "barrera_desactivante",
                            "amortizacion_final", "amortizacion", "estado", "simulacion"});

    p.nombre = raiz.campo("nombre").texto("Producto sin nombre");
    p.descripcion = raiz.campo("descripcion").texto("");
    p.moneda = raiz.campo("moneda").texto("EUR");
    p.plantilla = raiz.campo("plantilla").texto("");
    p.nominal = raiz.campo("nominal").real(100.0);
    if (!(p.nominal > 0)) raiz.campo("nominal").error("el nominal debe ser positivo");

    // ---------------- Calendario: vencimiento y observaciones ---------------- //
    // Las fechas ya pasadas (productos vivos descritos con su calendario completo)
    // se omiten; las listas por fecha pueden darse para el calendario completo o
    // solo para las fechas futuras, y "desde"/"hasta" cuentan en el completo.
    Nodo nobs = raiz.campo("observaciones");
    Nodo nvto = raiz.campo("vencimiento");
    std::vector<double> fechas;      // solo futuras
    std::vector<double> fechas_tot;  // calendario completo (las pasadas con t <= 0)
    std::vector<std::string> etiquetas;
    size_t n_pasadas = 0;
    Nodo nfechas = nobs.es_lista() ? nobs : nobs.campo("fechas");
    if (nfechas.existe()) {
        if (!nfechas.es_lista() || nfechas.tamano() == 0)
            nfechas.error("\"fechas\" debe ser una lista no vacía de plazos o fechas");
        for (size_t i = 0; i < nfechas.tamano(); ++i) {
            Nodo e = nfechas.elemento(i);
            double t = e.plazo();
            if (!fechas_tot.empty() && !(t > fechas_tot.back() + 1e-9))
                e.error("las fechas de observación deben ser estrictamente crecientes");
            fechas_tot.push_back(t);
            if (!(t > 1e-9)) { ++n_pasadas; continue; }
            fechas.push_back(t);
            etiquetas.push_back(e.es_texto() ? e.texto() : etiqueta_plazo(t));
        }
        if (fechas.empty())
            nfechas.error("todas las fechas de observación son anteriores a la fecha de valoración: el producto ya ha vencido");
        if (n_pasadas > 0)
            nfechas.aviso("se omiten " + std::to_string(n_pasadas) + " fechas de observación ya pasadas (use \"estado\" "
                          "para la memoria de cupones y la barrera ya tocada)");
        if (nvto.existe()) {
            p.T = nvto.plazo();
            if (std::fabs(fechas.back() - p.T) > 1.5 / 365.0)
                nvto.error("la última fecha de observación (" + fmt(fechas.back(), 4) +
                           " años) debe coincidir con el vencimiento (" + fmt(p.T, 4) + " años)");
            fechas.back() = p.T;
            fechas_tot.back() = p.T;
        } else {
            p.T = fechas.back();
        }
    } else {
        nvto.requerir("el vencimiento del producto (p. ej. \"vencimiento\": \"3A\")");
        p.T = nvto.plazo();
        if (!(p.T > 0)) nvto.error("el vencimiento debe ser posterior a la fecha de valoración");
        Nodo nfrec = nobs.es_texto() || nobs.es_numero() ? nobs : nobs.campo("frecuencia");
        if (nfrec.existe()) {
            int f = nfrec.es_numero() ? nfrec.entero() : texto_a_frecuencia(nfrec.texto());
            if (f <= 0)
                nfrec.error("frecuencia no reconocida (use mensual, trimestral, semestral, anual, o un plazo "
                            "como \"3M\")");
            double dtf = 1.0 / f;
            int nf = (int)std::ceil(p.T / dtf - 1e-6);
            for (int j = 1; j <= nf; ++j) {
                double t = p.T - (nf - j) * dtf;
                fechas.push_back(t);
                etiquetas.push_back(etiqueta_plazo(t));
            }
            if (p.T / dtf - std::floor(p.T / dtf + 1e-6) > 1e-6)
                nfrec.aviso("el vencimiento no es múltiplo del periodo: el primer periodo es más corto");
        } else {
            fechas.push_back(p.T);
            etiquetas.push_back(etiqueta_plazo(p.T));
        }
    }
    if (nobs.es_objeto()) nobs.claves_permitidas({"frecuencia", "fechas"});
    if ((int)fechas.size() > 4096) nobs.error("demasiadas fechas de observación (máximo 4096)");
    if (fechas_tot.empty()) fechas_tot = fechas; // calendario por frecuencia: no hay fechas pasadas
    p.t_obs = fechas;
    p.etiqueta_obs = etiquetas;
    const size_t nobs_n = fechas.size();
    const size_t n_tot = fechas_tot.size(); // = n_pasadas + nobs_n

    // Lista por fecha en el calendario COMPLETO: escalar, lista completa o lista
    // solo de las fechas futuras (se rellena por delante); recortar() la deja
    // luego en las fechas futuras.
    auto lista_completa = [&](const Nodo& nd, const std::string& que) {
        std::vector<double> v;
        if (nd.es_lista() && nd.tamano() == nobs_n && n_pasadas > 0) {
            v.assign(n_pasadas, 0.0);
            for (size_t i = 0; i < nobs_n; ++i) v.push_back(nd.elemento(i).real());
            return v;
        }
        return nd.lista_reales(n_tot, que);
    };
    auto recortar = [&](std::vector<double>& v) { v.erase(v.begin(), v.begin() + (long)n_pasadas); };

    // ---------------- Mercado ---------------- //
    Nodo nmer = raiz.campo("mercado");
    Nodo nr = nmer.campo("tipo_interes");
    if (!nr.existe()) nr = raiz.campo("tipo_interes");
    Nodo ncurva = nmer.campo("curva");
    if (ncurva.existe()) {
        if (!ncurva.es_lista() || ncurva.tamano() == 0)
            ncurva.error("la curva debe ser una lista de pares [plazo, tipo], p. ej. [[\"1A\", \"3%\"], "
                         "[\"5A\", \"3.2%\"]]");
        std::vector<std::pair<double, double>> pts;
        for (size_t i = 0; i < ncurva.tamano(); ++i) {
            Nodo e = ncurva.elemento(i);
            double t, r;
            if (e.es_lista() && e.tamano() == 2) { t = e.elemento(0).plazo(); r = e.elemento(1).real(); }
            else if (e.es_objeto()) { t = e.campo("plazo").plazo(); r = e.campo("tipo").real(); }
            else e.error("cada punto de la curva debe ser [plazo, tipo]");
            if (!(t > 0)) e.error("el plazo de la curva debe ser positivo");
            pts.push_back({t, r});
        }
        std::sort(pts.begin(), pts.end());
        for (auto& pt : pts) { p.curva.t.push_back(pt.first); p.curva.r.push_back(pt.second); }
        if (nr.existe()) nr.aviso("se ha dado curva y tipo plano: se usa la curva");
    } else {
        nr.requerir("el tipo de interés libre de riesgo (\"mercado\": {\"tipo_interes\": \"3%\"})");
        double r = nr.real_en(-0.2, 1.0);
        p.curva.t = {1.0};
        p.curva.r = {r};
    }
    nmer.claves_permitidas({"tipo_interes", "curva"});

    // ---------------- Subyacentes ---------------- //
    Nodo nsub = raiz.alguno({"subyacentes", "subyacente"});
    nsub.requerir("la lista de \"subyacentes\"");
    std::vector<Nodo> lista_sub;
    if (nsub.es_lista()) for (size_t i = 0; i < nsub.tamano(); ++i) lista_sub.push_back(nsub.elemento(i));
    else lista_sub.push_back(nsub);
    if (lista_sub.empty()) nsub.error("hace falta al menos un subyacente");
    if ((int)lista_sub.size() > MAX_ACT_HD)
        nsub.error("como máximo " + std::to_string(MAX_ACT_HD) + " subyacentes");
    std::set<std::string> nombres;
    for (size_t i = 0; i < lista_sub.size(); ++i) {
        const Nodo& ns = lista_sub[i];
        if (!ns.es_objeto()) ns.error("cada subyacente debe ser un objeto {\"nombre\": ..., \"spot\": ..., ...}");
        Activo a;
        a.nombre = ns.campo("nombre").texto("S" + std::to_string(i + 1));
        if (!nombres.insert(a.nombre).second) ns.campo("nombre").error("nombre de subyacente repetido");
        a.spot = ns.alguno({"spot", "precio"}).real(100.0);
        if (!(a.spot > 0)) ns.campo("spot").error("el spot debe ser positivo");
        Nodo ni = ns.alguno({"nivel_inicial", "fixing_inicial", "strike_inicial"});
        a.nivel_inicial = ni.real(a.spot);
        if (!(a.nivel_inicial > 0)) ni.error("el nivel inicial debe ser positivo");
        a.dividendos = ns.alguno({"dividendos", "dividendo", "q"}).real(0.0);
        if (a.dividendos < -0.2 || a.dividendos > 1.0)
            ns.campo("dividendos").error("rentabilidad por dividendo fuera de rango");
        a.modelo = leer_modelo(ns.campo("modelo"));
        ns.claves_permitidas({"nombre", "spot", "precio", "nivel_inicial", "fixing_inicial", "strike_inicial",
                              "dividendos", "dividendo", "q", "modelo"});
        p.activos.push_back(a);
    }
    const size_t n = p.activos.size();
    leer_correlacion(raiz.campo("correlacion"), p);

    // ---------------- Cesta ---------------- //
    Nodo ncesta = raiz.campo("cesta");
    p.cesta = CESTA_PEOR;
    if (ncesta.existe()) {
        Nodo nt = ncesta.es_objeto() ? ncesta.campo("tipo") : ncesta;
        std::string s = normalizar(nt.texto("peor"));
        if (s == "peor" || s == "worst" || s == "worst_of" || s == "peor_de" || s == "minimo") p.cesta = CESTA_PEOR;
        else if (s == "mejor" || s == "best" || s == "best_of" || s == "mejor_de" || s == "maximo") p.cesta = CESTA_MEJOR;
        else if (s == "media" || s == "promedio" || s == "equiponderada") p.cesta = CESTA_MEDIA;
        else if (s == "ponderada") p.cesta = CESTA_PONDERADA;
        else nt.error("tipo de cesta \"" + nt.texto() + "\" desconocido (peor, mejor, media o ponderada)");
        if (p.cesta == CESTA_PONDERADA) {
            Nodo npes = ncesta.campo("pesos");
            if (!npes.es_lista() || npes.tamano() != n)
                ncesta.error("la cesta ponderada necesita \"pesos\" con " + std::to_string(n) + " valores");
            double s2 = 0;
            for (size_t i = 0; i < n; ++i) { p.pesos.push_back(npes.elemento(i).real()); s2 += p.pesos.back(); }
            if (!(s2 > 0)) npes.error("los pesos deben sumar una cantidad positiva");
            if (std::fabs(s2 - 1.0) > 1e-9) {
                npes.aviso("los pesos no suman 100%: se normalizan");
                for (auto& w : p.pesos) w /= s2;
            }
        }
        if (ncesta.es_objeto()) ncesta.claves_permitidas({"tipo", "pesos"});
    }
    if (p.pesos.empty()) p.pesos.assign(n, 1.0 / (double)n);

    // ---------------- Cupones ---------------- //
    Nodo ncup = raiz.alguno({"cupones", "cupon"});
    if (ncup.existe()) {
        std::vector<Nodo> patas;
        if (ncup.es_lista()) for (size_t i = 0; i < ncup.tamano(); ++i) patas.push_back(ncup.elemento(i));
        else patas.push_back(ncup);
        if ((int)patas.size() > MAX_CUP_HD) ncup.error("como máximo " + std::to_string(MAX_CUP_HD) + " patas de cupón");
        for (const Nodo& nc : patas) {
            Cupon c;
            std::string tipo = normalizar(nc.campo("tipo").texto(nc.campo("barrera").existe() ? "condicional" : "fijo"));
            if (tipo == "fijo" || tipo == "garantizado") c.tipo = CUP_FIJO;
            else if (tipo == "condicional" || tipo == "digital" || tipo == "phoenix") c.tipo = CUP_CONDICIONAL;
            else if (tipo == "rango" || tipo == "range_accrual" || tipo == "acumulativo_rango") c.tipo = CUP_RANGO;
            else nc.campo("tipo").error("tipo de cupón \"" + tipo + "\" desconocido (fijo, condicional o rango)");

            // Importe por fecha: "importe" (por periodo) o "tasa_anual" (× duración del periodo)
            Nodo nimp = nc.campo("importe"), ntasa = nc.campo("tasa_anual");
            if (nimp.existe() == ntasa.existe())
                nc.error("indique \"importe\" (por periodo) o \"tasa_anual\" (anualizada), uno de los dos");
            // (todo en el calendario completo; al final se recortan las fechas pasadas)
            if (nimp.existe()) c.importe = lista_completa(nimp, "importes");
            else {
                std::vector<double> tasa = lista_completa(ntasa, "tasas");
                for (size_t j = 0; j < n_tot; ++j)
                    c.importe.push_back(tasa[j] * (fechas_tot[j] - (j ? fechas_tot[j - 1] : 0.0)));
            }

            int desde = nc.campo("desde").entero(1), hasta = nc.campo("hasta").entero((int)n_tot);
            if (desde < 1 || hasta > (int)n_tot || desde > hasta)
                nc.error("\"desde\"/\"hasta\" deben cumplir 1 ≤ desde ≤ hasta ≤ " + std::to_string(n_tot));

            c.barrera.assign(n_tot, -HD_INFINITO);
            if (c.tipo == CUP_CONDICIONAL) {
                Nodo nbar = nc.campo("barrera");
                nbar.requerir("la \"barrera\" del cupón condicional (p. ej. \"70%\")");
                c.barrera = lista_completa(nbar, "barreras de cupón");
                for (size_t j = n_pasadas; j < n_tot; ++j)
                    if (!(c.barrera[j] > 0)) nbar.error("las barreras deben ser positivas");
                c.memoria = nc.campo("memoria").booleano(false);
            } else if (nc.campo("memoria").existe() && nc.campo("memoria").booleano()) {
                nc.campo("memoria").aviso("la memoria solo tiene sentido en cupones condicionales: se ignora");
            }
            if (c.tipo == CUP_RANGO) {
                Nodo nrg = nc.campo("rango");
                if (nrg.existe()) {
                    if (!nrg.es_lista() || nrg.tamano() != 2) nrg.error("\"rango\" debe ser [inferior, superior]");
                    c.inf = nrg.elemento(0).real();
                    c.sup = nrg.elemento(1).real();
                } else {
                    c.inf = nc.campo("limite_inferior").real(0.0);
                    c.sup = nc.campo("limite_superior").real(HD_INFINITO);
                }
                if (!(c.sup > c.inf)) nc.error("el rango del range accrual está vacío");
            }
            // Fechas sin cupón: importe 0 y barrera inalcanzable (la memoria se conserva)
            for (int j = 0; j < (int)n_tot; ++j)
                if (j + 1 < desde || j + 1 > hasta) { c.importe[j] = 0.0; c.barrera[j] = HD_INFINITO; }
            recortar(c.importe);
            recortar(c.barrera);
            for (double x : c.importe) if (x < 0) { nc.aviso("hay importes de cupón negativos"); break; }
            nc.claves_permitidas({"tipo", "importe", "tasa_anual", "barrera", "memoria", "rango",
                                  "limite_inferior", "limite_superior", "desde", "hasta"});
            p.cupones.push_back(c);
        }
    }

    // ---------------- Autocancelación ---------------- //
    Nodo nac = raiz.campo("autocancelacion");
    if (nac.existe() && !(nac.es_objeto())) nac.error("\"autocancelacion\" debe ser un objeto");
    if (nac.existe()) {
        p.hay_autocall = true;
        // Índices (desde/hasta, nº de observación de la prima acumulada) en el
        // calendario completo, incluidas las fechas ya pasadas.
        int desde = nac.campo("desde").entero(1);
        bool en_vto = nac.campo("en_vencimiento").booleano(true);
        int hasta = nac.campo("hasta").entero((int)n_tot - (en_vto ? 0 : 1));
        if (desde < 1 || hasta > (int)n_tot || desde > hasta)
            nac.error("\"desde\"/\"hasta\" deben cumplir 1 ≤ desde ≤ hasta ≤ " + std::to_string(n_tot));
        Nodo nbar = nac.campo("barrera");
        nbar.requerir("la \"barrera\" de autocancelación (p. ej. \"100%\")");
        std::vector<double> bar;
        if (nbar.es_lista()) {
            bar = lista_completa(nbar, "barreras de autocancelación");
            if (nac.campo("escalon").existe()) nac.campo("escalon").aviso("se ignora \"escalon\" porque la barrera es una lista");
        } else {
            double b0 = nbar.real(), esc = nac.campo("escalon").real(0.0);
            bar.assign(n_tot, b0);
            for (int j = desde - 1; j < (int)n_tot; ++j) bar[j] = b0 + esc * (j - (desde - 1));
        }
        double imp = nac.campo("importe").real(1.0);
        Nodo nprima = nac.alguno({"prima", "cupon"});
        Nodo nprima_a = nac.campo("prima_anual");
        bool acum = nac.campo("acumulativa").booleano(false);
        std::vector<double> prima(n_tot, 0.0);
        if (nprima.existe() && nprima_a.existe()) nac.error("indique \"prima\" o \"prima_anual\", no ambas");
        if (nprima.existe()) {
            prima = lista_completa(nprima, "primas");
            if (acum) for (size_t j = 0; j < n_tot; ++j) prima[j] *= (double)(j + 1);
        } else if (nprima_a.existe()) {
            double pa = nprima_a.real();
            for (size_t j = 0; j < n_tot; ++j) prima[j] = pa * fechas_tot[j];
        }
        std::vector<double> activa(n_tot, 0.0), barrera(n_tot, HD_INFINITO), importe(n_tot, 0.0);
        for (int j = desde - 1; j < hasta; ++j) {
            if (j >= (int)n_pasadas && !(bar[j] > 0)) nbar.error("las barreras de autocancelación deben ser positivas");
            activa[j] = 1.0;
            barrera[j] = bar[j];
            importe[j] = imp + prima[j];
        }
        recortar(activa);
        recortar(barrera);
        recortar(importe);
        p.ac_barrera = barrera;
        p.ac_importe = importe;
        for (double a : activa) p.ac_activa.push_back(a != 0.0 ? 1 : 0);
        nac.claves_permitidas({"barrera", "escalon", "desde", "hasta", "en_vencimiento", "importe", "prima",
                               "cupon", "prima_anual", "acumulativa"});
    }

    // ---------------- Barreras ---------------- //
    Nodo nki = raiz.campo("barrera_capital");
    if (nki.existe()) {
        if (!nki.es_objeto()) nki.error("\"barrera_capital\" debe ser un objeto {\"nivel\": ..., \"observacion\": ...}");
        p.ki_nivel = nki.campo("nivel").real_en(0.0, 10.0, true);
        p.ki_obs = leer_observacion(nki.campo("observacion"), OBS_VENCIMIENTO);
        p.ki_bgk = nki.campo("correccion_bgk").booleano(true);
        nki.claves_permitidas({"nivel", "observacion", "correccion_bgk"});
    }
    Nodo nko = raiz.campo("barrera_desactivante");
    if (nko.existe()) {
        if (!nko.es_objeto()) nko.error("\"barrera_desactivante\" debe ser un objeto");
        p.ko_nivel = nko.campo("nivel").real_en(0.0, 100.0, true);
        p.ko_obs = leer_observacion(nko.campo("observacion"), OBS_CONTINUA);
        if (p.ko_obs == OBS_VENCIMIENTO || p.ko_obs == OBS_NINGUNA)
            nko.campo("observacion").error("la barrera desactivante se observa en \"fechas\", \"diaria\" o "
                                           "\"continua\" (a vencimiento equivale a un componente digital)");
        p.ko_importe = nko.campo("importe").real(1.0);
        std::string pago = normalizar(nko.campo("pago").texto("vencimiento"));
        if (pago == "vencimiento" || pago == "a_vencimiento") p.ko_pago_venc = true;
        else if (pago == "inmediato" || pago == "inmediatamente") p.ko_pago_venc = false;
        else nko.campo("pago").error("\"pago\" debe ser \"vencimiento\" o \"inmediato\"");
        p.ko_bgk = nko.campo("correccion_bgk").booleano(true);
        nko.claves_permitidas({"nivel", "observacion", "importe", "pago", "correccion_bgk"});
    }

    // ---------------- Amortización final ---------------- //
    Nodo nam = raiz.alguno({"amortizacion_final", "amortizacion"});
    nam.requerir("la \"amortizacion_final\" (lista de componentes; p. ej. [{\"tipo\": \"fijo\", \"importe\": "
                 "\"100%\"}] para devolver el nominal)");
    Nodo ncomp = nam;
    if (nam.es_objeto()) {
        std::string rend = normalizar(nam.campo("rendimiento").texto("final"));
        if (rend == "final" || rend == "vencimiento") p.rend_medio = false;
        else if (rend == "medio" || rend == "media" || rend == "promedio" || rend == "asiatico") p.rend_medio = true;
        else nam.campo("rendimiento").error("\"rendimiento\" debe ser \"final\" o \"medio\"");
        ncomp = nam.campo("componentes");
        nam.claves_permitidas({"rendimiento", "componentes"});
    }
    if (!ncomp.es_lista()) ncomp.error("los componentes de la amortización deben ir en una lista [ ... ]");
    if ((int)ncomp.tamano() > MAX_BLOQUES_HD) ncomp.error("como máximo " + std::to_string(MAX_BLOQUES_HD) + " componentes");
    for (size_t i = 0; i < ncomp.tamano(); ++i) p.bloques.push_back(leer_bloque(ncomp.elemento(i)));
    int n_cliq = 0;
    for (const auto& b : p.bloques) n_cliq += (b.tipo == BLQ_CLIQUET);
    if (n_cliq > 1) ncomp.error("solo se admite un componente cliquet");

    // ---------------- Estado (productos ya vivos) ---------------- //
    Nodo nest = raiz.campo("estado");
    p.memoria_inicial.assign(p.cupones.size(), 0.0);
    if (nest.existe()) {
        p.ki_inicial = nest.campo("barrera_capital_tocada").booleano(false);
        Nodo nmem = nest.campo("cupones_en_memoria");
        if (nmem.existe()) {
            if (nmem.es_lista()) {
                if (nmem.tamano() != p.cupones.size()) nmem.error("un valor por pata de cupón");
                for (size_t c = 0; c < p.cupones.size(); ++c) p.memoria_inicial[c] = nmem.elemento(c).real();
            } else {
                double m = nmem.real();
                bool puesto = false;
                for (size_t c = 0; c < p.cupones.size(); ++c)
                    if (p.cupones[c].memoria) { p.memoria_inicial[c] = m; puesto = true; break; }
                if (!puesto && m != 0.0) nmem.error("no hay cupones con memoria");
            }
        }
        nest.claves_permitidas({"barrera_capital_tocada", "cupones_en_memoria"});
    }

    // ---------------- Simulación ---------------- //
    Nodo nsim = raiz.campo("simulacion");
    if (nsim.existe()) {
        ConfigSimulacion& s = p.sim;
        s.metodo = normalizar(nsim.campo("metodo").texto("auto"));
        if (s.metodo != "auto" && s.metodo != "mc" && s.metodo != "qmc")
            nsim.campo("metodo").error("\"metodo\" debe ser auto, mc o qmc");
        s.dispositivo = normalizar(nsim.campo("dispositivo").texto("auto"));
        if (s.dispositivo != "auto" && s.dispositivo != "gpu" && s.dispositivo != "cpu")
            nsim.campo("dispositivo").error("\"dispositivo\" debe ser auto, gpu o cpu");
        if (nsim.campo("error_objetivo").existe())
            s.error_objetivo = nsim.campo("error_objetivo").real_en(0.0, 1.0, true);
        if (nsim.campo("error_objetivo_importe").existe())
            s.error_objetivo = nsim.campo("error_objetivo_importe").real_en(0.0, 1e30, true) / p.nominal;
        Nodo nppa = nsim.campo("pasos_por_anio");
        if (nppa.existe() && !(nppa.es_texto() && normalizar(nppa.texto()) == "auto"))
            s.pasos_por_anio = nppa.real_en(0.0, 100000.0);
        s.semilla = (uint32_t)nsim.campo("semilla").real((double)s.semilla);
        s.trayectorias = (long long)nsim.campo("trayectorias").real(0.0);
        s.trayectorias_max = (long long)nsim.campo("trayectorias_max").real((double)s.trayectorias_max);
        s.replicas_qmc = nsim.campo("replicas_qmc").entero(s.replicas_qmc);
        if (s.replicas_qmc < 2 || s.replicas_qmc > 1024) nsim.campo("replicas_qmc").error("entre 2 y 1024 réplicas");
        std::string prec = normalizar(nsim.campo("precision_gpu").texto("simple"));
        if (prec == "simple" || prec == "float") s.doble_precision = false;
        else if (prec == "doble" || prec == "double") s.doble_precision = true;
        else nsim.campo("precision_gpu").error("\"precision_gpu\" debe ser simple o doble");
        s.hilos = nsim.campo("hilos").entero(0);
        nsim.claves_permitidas({"metodo", "dispositivo", "error_objetivo", "error_objetivo_importe",
                                "pasos_por_anio", "semilla", "trayectorias", "trayectorias_max",
                                "replicas_qmc", "precision_gpu", "hilos"});
    }

    // ---------------- Coherencia global (avisos) ---------------- //
    bool algun_cond = false;
    for (const auto& b : p.bloques) algun_cond |= (b.cond != COND_SIEMPRE);
    if (p.ki_obs != OBS_NINGUNA && !algun_cond)
        nki.aviso("ningún componente de la amortización final depende de la barrera de capital "
                  "(use \"condicion\": \"barrera_tocada\" en el componente que la active)");
    if (p.ki_obs == OBS_NINGUNA && algun_cond)
        nam.aviso("hay componentes condicionados a la barrera de capital pero el producto no tiene "
                  "\"barrera_capital\": los de \"barrera_tocada\" nunca pagan");
    if (p.hay_autocall) {
        for (const auto& c : p.cupones) {
            if (c.tipo != CUP_CONDICIONAL) continue;
            for (size_t j = 0; j < nobs_n; ++j)
                if (p.ac_activa[j] && p.ac_barrera[j] < c.barrera[j] - 1e-12) {
                    nac.aviso("en la observación " + std::to_string(j + 1) + " la barrera de autocancelación (" +
                              porcentaje(p.ac_barrera[j]) + ") es menor que la de cupón (" +
                              porcentaje(c.barrera[j]) + ")");
                    break;
                }
        }
    }
    if (p.bloques.empty() && !p.hay_autocall && p.ko_obs == OBS_NINGUNA && p.cupones.empty())
        nam.aviso("el producto no paga nada: no tiene cupones, ni autocancelación, ni componentes finales");
    (void)n;
    return p;
}

Producto cargar(const std::string& ruta, json::Valor* doc) {
    json::Valor d;
    try {
        d = json::leer_fichero(ruta);
    } catch (const json::Error& e) {
        throw ErrorProducto(e.what());
    }
    Producto p = compilar(d);
    if (doc) *doc = std::move(d);
    return p;
}

double pasos_por_anio_efectivos(const Producto& p) {
    bool vigilancia = p.ki_obs >= OBS_DIARIA || p.ko_obs >= OBS_DIARIA;
    for (const auto& c : p.cupones) vigilancia |= (c.tipo == CUP_RANGO);
    if (p.sim.pasos_por_anio >= 0.0) {
        if (vigilancia && p.sim.pasos_por_anio < 252.0) return 252.0;
        return p.sim.pasos_por_anio;
    }
    if (vigilancia) return 252.0;
    bool todos_exactos = true;
    for (const auto& a : p.activos) todos_exactos &= modelo_exacto(a.modelo.tipo);
    return todos_exactos ? 0.0 : 252.0;
}

} // namespace est
