#include "asistente.hpp"

#include "plantillas.hpp"
#include "producto.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <istream>
#include <ostream>
#include <utility>
#include <vector>

namespace est {

namespace {

std::string recortar(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

std::string sin_tildes_ascii(const std::string& s) {
    static const char* pares[][2] = {{"á", "a"}, {"é", "e"}, {"í", "i"}, {"ó", "o"}, {"ú", "u"},
                                     {"ñ", "n"}, {"Á", "a"}, {"É", "e"}, {"Í", "i"}, {"Ó", "o"},
                                     {"Ú", "u"}, {"Ñ", "n"}, {"ü", "u"}};
    std::string r = s;
    for (auto& p : pares) {
        size_t pos;
        while ((pos = r.find(p[0])) != std::string::npos) r.replace(pos, std::string(p[0]).size(), p[1]);
    }
    return r;
}

// "Phoenix Worst-of SX5E (3 años)" -> "phoenix_worst_of_sx5e_3_anos.json"
std::string nombre_fichero(const std::string& nombre) {
    std::string s = sin_tildes_ascii(nombre), out;
    for (char c : s) {
        if (std::isalnum((unsigned char)c)) out += (char)std::tolower((unsigned char)c);
        else if (!out.empty() && out.back() != '_') out += '_';
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    if (out.size() > 60) out.resize(60);
    if (out.empty()) out = "producto";
    return out + ".json";
}

class Dialogo {
public:
    Dialogo(std::istream& in, std::ostream& out) : in_(in), out_(out) {}

    std::ostream& out() { return out_; }

    std::string leer(const std::string& prompt) {
        out_ << prompt << std::flush;
        std::string s;
        if (!std::getline(in_, s)) {
            out_ << "\n";
            throw Cancelado{};
        }
        if (!s.empty() && s.back() == '\r') s.pop_back();
        std::string t = recortar(s);
        if (t == "salir" || t == "SALIR") throw Cancelado{};
        return t;
    }

    std::string preguntar(const Pregunta& q, const std::string& defecto = "") {
        std::string def = defecto.empty() ? q.defecto : defecto, def_can, err;
        if (!normalizar_respuesta(q, def, def_can, err)) def_can = def;
        for (;;) {
            std::string r = leer("   " + q.texto + " [" + mostrar(q.tipo, def_can) + "]: ");
            if (r == "?") { ayuda(q); continue; }
            if (r.empty()) return def_can;
            std::string can;
            if (!normalizar_respuesta(q, r, can, err)) {
                out_ << "     ✗ " << err << " (escriba ? para ayuda)\n";
                continue;
            }
            if ((q.tipo == TipoDato::Porcentaje || q.tipo == TipoDato::PctOpcional) && r.back() != '%' && can != "no")
                out_ << "     → " << can << "\n";
            return can;
        }
    }

    // Pregunta numérica (o porcentual) con rango
    double numero(const std::string& texto, double defecto, bool pct, double mn, double mx, bool abierto = false,
                  const std::string& ayuda_txt = "") {
        Pregunta q{"", texto, pct ? porcentaje(defecto, 4) : fmt(defecto, 10),
                   pct ? TipoDato::Porcentaje : TipoDato::Real, ayuda_txt, {}, ""};
        for (;;) {
            std::string c = preguntar(q);
            double x = texto_a_real(c);
            bool mal = abierto ? !(x > mn) : !(x >= mn);
            if (mal || x > mx) {
                out_ << "     ✗ debe estar entre " << (pct ? porcentaje(mn) : fmt(mn, 6)) << " y "
                     << (pct ? porcentaje(mx) : fmt(mx, 6)) << (abierto ? " (sin incluir el mínimo)" : "") << "\n";
                continue;
            }
            return x;
        }
    }

    int entero(const std::string& texto, int defecto, int mn, int mx) {
        Pregunta q{"", texto, std::to_string(defecto), TipoDato::Entero, "", {}, ""};
        for (;;) {
            int x = std::stoi(preguntar(q));
            if (x < mn || x > mx) {
                out_ << "     ✗ debe estar entre " << mn << " y " << mx << "\n";
                continue;
            }
            return x;
        }
    }

    bool si_no(const std::string& texto, bool defecto) {
        Pregunta q{"", texto, defecto ? "si" : "no", TipoDato::SiNo, "", {}, ""};
        return preguntar(q) == "si";
    }

    std::string texto(const std::string& t, const std::string& defecto) {
        std::string r = leer("   " + t + (defecto.empty() ? "" : " [" + defecto + "]") + ": ");
        return r.empty() ? defecto : r;
    }

    // Menú numerado; devuelve el índice elegido
    int menu(const std::vector<std::pair<std::string, std::string>>& ops, int defecto) {
        for (size_t i = 0; i < ops.size(); ++i) {
            out_ << "     " << (i + 1 < 10 ? " " : "") << (i + 1) << ". " << ops[i].first;
            if (!ops[i].second.empty()) out_ << "  —  " << ops[i].second;
            out_ << "\n";
        }
        for (;;) {
            std::string r = leer("   Elija una opción [" + std::to_string(defecto + 1) + "]: ");
            if (r.empty()) return defecto;
            bool ok = false;
            double x = texto_a_real(r, &ok);
            if (ok && x >= 1 && x <= (double)ops.size() && std::fabs(x - std::round(x)) < 1e-9)
                return (int)std::llround(x) - 1;
            for (size_t i = 0; i < ops.size(); ++i)
                if (ops[i].first.rfind(r, 0) == 0) return (int)i;
            out_ << "     ✗ escriba un número entre 1 y " << ops.size() << "\n";
        }
    }

    void seccion(const std::string& titulo) { out_ << "\n── " << titulo << " " << std::string(70 - std::min<size_t>(68, titulo.size()), '-') << "\n"; }

private:
    std::istream& in_;
    std::ostream& out_;

    static std::string mostrar(TipoDato t, const std::string& c) {
        if (t == TipoDato::SiNo) return c == "si" ? "s" : "n";
        return c;
    }
    void ayuda(const Pregunta& q) {
        if (!q.ayuda.empty()) out_ << "     · " << q.ayuda << "\n";
        switch (q.tipo) {
            case TipoDato::Porcentaje:  out_ << "     · porcentaje: 70 o 70% = 0.70; admite decimales (2.5) y negativos\n"; break;
            case TipoDato::PctOpcional: out_ << "     · porcentaje, o 'no' para no poner límite\n"; break;
            case TipoDato::Plazo:       out_ << "     · plazo: 3A (años), 18M (meses), 26S (semanas), 2.5 (años)\n"; break;
            case TipoDato::Frecuencia:  out_ << "     · mensual, bimestral, trimestral, cuatrimestral, semestral, anual o 3M, 6M...\n"; break;
            case TipoDato::Observacion: out_ << "     · vencimiento | fechas | diaria | continua\n"; break;
            case TipoDato::SiNo:        out_ << "     · s / n\n"; break;
            case TipoDato::Opcion: {
                out_ << "     · opciones:";
                for (const auto& o : q.opciones) out_ << " " << o;
                out_ << "\n";
            } break;
            default: break;
        }
        out_ << "     · Intro acepta el valor entre corchetes; 'salir' cancela el asistente\n";
    }
};

// ------------------------------------------------------------ //
// Secciones comunes                                             //
// ------------------------------------------------------------ //

json::Valor preguntar_modelo(Dialogo& d, const std::string& vol_def) {
    std::vector<std::pair<std::string, std::string>> ops;
    for (const auto& m : catalogo_modelos()) ops.push_back({m.clave, m.descripcion});
    d.out() << "   Modelo de evolución del precio:\n";
    int k = d.menu(ops, 0);
    const DefModelo& def = catalogo_modelos()[(size_t)k];
    json::Valor m = json::Valor::objeto();
    m.poner("tipo", def.clave);
    for (const auto& p : def.params) {
        if (p.clave == "vol_max") continue; // parámetro técnico: se deja su valor por defecto
        double dv = p.defecto;
        bool es_vol = (p.clave == "volatilidad" || p.clave == "vol_inicial" || p.clave == "sigma0" ||
                       (def.tipo == TipoModelo::SABR && p.clave == "alpha") ||
                       (def.tipo == TipoModelo::NIG && p.clave == "sigma"));
        if (es_vol) dv = texto_a_real(vol_def);
        double x = d.numero(p.clave + ": " + p.descripcion + (p.porcentaje ? " (%)" : ""), dv, p.porcentaje, p.min,
                            p.max, p.min_abierto);
        if (p.porcentaje) m.poner(p.clave, porcentaje(x, 4));
        else m.poner(p.clave, x);
    }
    if (def.tipo == TipoModelo::NIG) {
        double s = texto_a_real(m.buscar("sigma")->texto()), th = m.buscar("theta")->numero(),
               ka = m.buscar("kappa")->numero();
        if (1.0 - s * s * ka - 2.0 * th * ka <= 0.0) {
            d.out() << "     ✗ parámetros NIG incompatibles (hace falta 1 - sigma²·kappa - 2·theta·kappa > 0); "
                       "vuelva a introducirlos\n";
            return preguntar_modelo(d, vol_def);
        }
    }
    return m;
}

bool cholesky_ok(const std::vector<double>& c, size_t n) {
    std::vector<double> L(n * n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j <= i; ++j) {
            double s = c[i * n + j];
            for (size_t k = 0; k < j; ++k) s -= L[i * n + k] * L[j * n + k];
            if (i == j) {
                if (s < -1e-10) return false;
                L[i * n + i] = std::sqrt(std::max(s, 0.0));
            } else {
                L[i * n + j] = L[j * n + j] > 1e-12 ? s / L[j * n + j] : 0.0;
            }
        }
    return true;
}

void preguntar_subyacentes(Dialogo& d, const Plantilla& pl, DatosComunes& c) {
    d.seccion("Subyacentes");
    int ndef = (int)std::max<size_t>(1, pl.subyacentes_def.size());
    int n = pl.nominal_es_spot ? 1 : d.entero("Número de subyacentes", ndef, 1, MAX_ACT_HD);
    c.subyacentes = json::Valor::lista();
    std::vector<std::string> nombres;
    for (int i = 0; i < n; ++i) {
        SubyacenteDef sd = (i < (int)pl.subyacentes_def.size()) ? pl.subyacentes_def[(size_t)i]
                                                                : SubyacenteDef{"S" + std::to_string(i + 1), 100.0, "0%", "20%"};
        d.out() << "\n   Subyacente " << (i + 1) << "\n";
        json::Valor a = json::Valor::objeto();
        std::string nom;
        for (;;) {
            nom = d.texto("Nombre", sd.nombre);
            bool rep = false;
            for (const auto& x : nombres) rep |= (x == nom);
            if (!rep) break;
            d.out() << "     ✗ ese nombre ya está usado\n";
        }
        nombres.push_back(nom);
        a.poner("nombre", nom);
        double spot = d.numero("Precio actual (spot)", sd.spot, false, 0.0, 1e12, true);
        a.poner("spot", spot);
        double ini = d.numero("Nivel inicial (fijación de referencia; Intro = spot)", spot, false, 0.0, 1e12, true);
        if (std::fabs(ini - spot) > 1e-12 * spot) a.poner("nivel_inicial", ini);
        double q = d.numero("Rentabilidad por dividendo (%)", texto_a_real(sd.dividendos), true, -0.2, 1.0);
        a.poner("dividendos", porcentaje(q, 4));
        a.poner("modelo", preguntar_modelo(d, sd.vol));
        c.subyacentes.anadir(a);
    }
    if (pl.nominal_es_spot) {
        c.nominal = c.subyacentes[0].buscar("spot")->numero();
        d.out() << "   (nominal = spot = " << fmt(c.nominal, 4) << ": el precio es el de una opción sobre una unidad)\n";
    }
    if (n > 1) {
        d.seccion("Correlación y cesta");
        for (;;) {
            std::vector<double> corr((size_t)(n * n), 0.0);
            for (int i = 0; i < n; ++i) corr[(size_t)(i * n + i)] = 1.0;
            if (d.si_no("¿Usar la misma correlación para todos los pares?", true)) {
                double r = d.numero("Correlación entre subyacentes (%)", 0.6, true, -1.0, 1.0);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j)
                        if (i != j) corr[(size_t)(i * n + j)] = r;
                c.correlacion = json::Valor(porcentaje(r, 4));
            } else {
                json::Valor o = json::Valor::objeto();
                for (int i = 0; i < n; ++i)
                    for (int j = i + 1; j < n; ++j) {
                        double r = d.numero("Correlación " + nombres[(size_t)i] + "/" + nombres[(size_t)j] + " (%)", 0.6,
                                            true, -1.0, 1.0);
                        corr[(size_t)(i * n + j)] = corr[(size_t)(j * n + i)] = r;
                        o.poner(nombres[(size_t)i] + "/" + nombres[(size_t)j], porcentaje(r, 4));
                    }
                c.correlacion = o;
            }
            if (cholesky_ok(corr, (size_t)n)) break;
            d.out() << "     ✗ esas correlaciones no forman una matriz válida (no es semidefinida positiva); "
                       "vuelva a introducirlas\n";
        }
        d.out() << "   ¿Cómo se combinan los rendimientos de la cesta (P)?\n";
        int k = d.menu({{"peor", "worst-of: manda el subyacente que peor lo hace (lo habitual)"},
                        {"mejor", "best-of: manda el que mejor lo hace"},
                        {"media", "media equiponderada de los rendimientos"}},
                       0);
        c.cesta = (k == 0) ? "peor" : (k == 1) ? "mejor" : "media";
    }
}

// ------------------------------------------------------------ //
// Producto personalizado: sección a sección                     //
// ------------------------------------------------------------ //

void personalizado(Dialogo& d, json::Valor& doc, double T) {
    d.seccion("Calendario de observación");
    Pregunta qf{"", "Frecuencia de observación", "trimestral", TipoDato::Frecuencia, "", {}, ""};
    std::string frec = d.preguntar(qf);
    json::Valor obs = json::Valor::objeto();
    obs.poner("frecuencia", frec);
    doc.poner("observaciones", obs);
    int f = texto_a_frecuencia(frec);
    int nobs = std::max(1, (int)std::ceil(T * f - 1e-6));
    d.out() << "   (" << nobs << " fechas de observación hasta el vencimiento)\n";

    Pregunta qobs{"", "Observación", "vencimiento", TipoDato::Observacion,
                  "vencimiento | fechas | diaria | continua", {}, ""};

    d.seccion("Cupones");
    json::Valor cups = json::Valor::lista();
    while (cups.tamano() < (size_t)MAX_CUP_HD &&
           d.si_no(cups.tamano() == 0 ? "¿Paga cupones?" : "¿Añadir otra pata de cupón?", cups.tamano() == 0)) {
        json::Valor c = json::Valor::objeto();
        int k = d.menu({{"condicional", "se cobra si P ≥ barrera (Phoenix), con memoria opcional"},
                        {"fijo", "se cobra siempre"},
                        {"rango", "range accrual: proporcional a los días con P dentro de un rango"}},
                       0);
        c.poner("tipo", k == 0 ? "condicional" : k == 1 ? "fijo" : "rango");
        if (k == 0) {
            c.poner("importe", porcentaje(d.numero("Cupón por periodo (%)", 0.02, true, 0.0, 10.0), 4));
            c.poner("barrera", porcentaje(d.numero("Barrera de cupón (%)", 0.7, true, 0.0, 10.0, true), 4));
            c.poner("memoria", d.si_no("¿Con memoria?", true));
        } else {
            c.poner("tasa_anual", porcentaje(d.numero("Cupón anual (%)", 0.05, true, 0.0, 10.0), 4));
            if (k == 2) {
                json::Valor rg = json::Valor::lista();
                rg.anadir(porcentaje(d.numero("Límite inferior del rango (%)", 0.9, true, 0.0, 10.0), 4));
                rg.anadir(porcentaje(d.numero("Límite superior del rango (%)", 1.1, true, 0.0, 10.0), 4));
                c.poner("rango", rg);
            }
        }
        int desde = d.entero("Primera fecha con cupón", 1, 1, nobs);
        if (desde > 1) c.poner("desde", desde);
        int hasta = d.entero("Última fecha con cupón", nobs, desde, nobs);
        if (hasta < nobs) c.poner("hasta", hasta);
        cups.anadir(c);
    }
    if (cups.tamano() == 1) doc.poner("cupones", cups[0]);
    else if (cups.tamano() > 1) doc.poner("cupones", cups);

    d.seccion("Autocancelación");
    if (d.si_no("¿Es autocancelable (amortización anticipada)?", true)) {
        json::Valor ac = json::Valor::objeto();
        ac.poner("barrera", porcentaje(d.numero("Barrera de autocancelación (%)", 1.0, true, 0.0, 10.0, true), 4));
        double esc = d.numero("Variación de la barrera por fecha (step-down, %)", 0.0, true, -1.0, 1.0);
        if (esc != 0.0) ac.poner("escalon", porcentaje(esc, 4));
        ac.poner("desde", d.entero("Primera fecha en que puede autocancelar", std::min(2, nobs), 1, nobs));
        double prima = d.numero("Prima adicional al autocancelar (% por periodo)", 0.0, true, 0.0, 10.0);
        if (prima != 0.0) {
            ac.poner("prima", porcentaje(prima, 4));
            ac.poner("acumulativa", d.si_no("¿La prima se acumula por periodo transcurrido (Athena)?", true));
        }
        doc.poner("autocancelacion", ac);
    }

    d.seccion("Barreras");
    bool ki = d.si_no("¿Tiene barrera de capital (pérdida si la cesta cae por debajo)?", true);
    if (ki) {
        json::Valor b = json::Valor::objeto();
        b.poner("nivel", porcentaje(d.numero("Nivel de la barrera de capital (%)", 0.6, true, 0.0, 10.0, true), 4));
        b.poner("observacion", d.preguntar(qobs));
        doc.poner("barrera_capital", b);
    }
    if (d.si_no("¿Tiene barrera desactivante (knock-out al alza)?", false)) {
        json::Valor b = json::Valor::objeto();
        b.poner("nivel", porcentaje(d.numero("Nivel de la barrera desactivante (%)", 1.3, true, 0.0, 100.0, true), 4));
        Pregunta qo{"", "Observación", "continua", TipoDato::Observacion, "fechas | diaria | continua", {}, ""};
        std::string o;
        for (;;) {
            o = d.preguntar(qo);
            if (o != "vencimiento") break;
            d.out() << "     ✗ la desactivante se observa en fechas, diaria o continua\n";
        }
        b.poner("observacion", o);
        b.poner("importe", porcentaje(d.numero("Importe que se paga al desactivarse (%)", 1.0, true, 0.0, 100.0), 4));
        b.poner("pago", d.si_no("¿Se paga a vencimiento (s) o en el momento (n)?", true) ? "vencimiento" : "inmediato");
        doc.poner("barrera_desactivante", b);
    }

    d.seccion("Amortización a vencimiento");
    d.out() << "   Se construye sumando componentes en función de P (rendimiento final de la cesta).\n";
    json::Valor comp = json::Valor::lista();
    std::vector<std::pair<std::string, std::string>> tipos = {
        {"fijo", "importe fijo (p. ej. 100% = devolver el nominal)"},
        {"put_vendida", "pérdida si P < strike: -(strike - P)/strike (reverse convertible)"},
        {"call", "participación en la subida: part × max(P - strike, 0)"},
        {"put", "participación en la caída: part × max(strike - P, 0)"},
        {"lineal", "part × P (replica al subyacente)"},
        {"digital", "importe si P ≥ strike"},
        {"digital_put", "importe si P < strike"},
        {"cliquet", "suma de rendimientos periódicos acotados"},
        {"terminar", "no añadir más componentes"}};
    for (;;) {
        d.out() << "   Componente " << (comp.tamano() + 1) << ":\n";
        int k = d.menu(tipos, comp.tamano() == 0 ? 0 : (int)tipos.size() - 1);
        const std::string t = tipos[(size_t)k].first;
        if (t == "terminar") break;
        json::Valor b = json::Valor::objeto();
        b.poner("tipo", t);
        if (t == "fijo") b.poner("importe", porcentaje(d.numero("Importe (%)", 1.0, true, -10.0, 100.0), 4));
        if (t == "digital" || t == "digital_put")
            b.poner("importe", porcentaje(d.numero("Importe (%)", 0.05, true, -10.0, 100.0), 4));
        if (t == "call" || t == "put" || t == "put_vendida" || t == "digital" || t == "digital_put")
            b.poner("strike", porcentaje(d.numero("Strike (%)", 1.0, true, 0.0, 100.0, t == "put_vendida"), 4));
        if (t == "call" || t == "put" || t == "lineal" || t == "cliquet" || t == "put_vendida")
            b.poner("participacion", porcentaje(d.numero("Participación (%)", 1.0, true, -100.0, 100.0), 4));
        if (t == "call" || t == "lineal") {
            Pregunta qt{"", "Tope de P (%, o 'no')", "no", TipoDato::PctOpcional, "", {}, ""};
            std::string tp = d.preguntar(qt);
            if (tp != "no") b.poner("techo", tp);
        }
        if (t == "cliquet") {
            b.poner("suelo_local", porcentaje(d.numero("Suelo local (%)", 0.0, true, -10.0, 10.0), 4));
            b.poner("techo_local", porcentaje(d.numero("Techo local (%)", 0.08, true, -10.0, 10.0), 4));
            b.poner("suelo_global", porcentaje(d.numero("Suelo global (%)", 0.0, true, -100.0, 100.0), 4));
        }
        if (ki) {
            int c = d.menu({{"siempre", ""}, {"barrera_tocada", "solo si se ha tocado la barrera de capital"},
                            {"barrera_no_tocada", "solo si NO se ha tocado"}},
                           t == "put_vendida" ? 1 : 0);
            if (c == 1) b.poner("condicion", "barrera_tocada");
            if (c == 2) b.poner("condicion", "barrera_no_tocada");
        }
        comp.anadir(b);
        if (comp.tamano() >= (size_t)MAX_BLOQUES_HD) break;
    }
    doc.poner("amortizacion_final", comp);
}

} // namespace

bool preguntar_si_no(std::istream& in, std::ostream& out, const std::string& texto, bool defecto) {
    Dialogo d(in, out);
    return d.si_no(texto, defecto);
}

json::Valor asistente_nuevo_producto(std::istream& in, std::ostream& out, const std::string& ruta_sugerida,
                                     std::string* ruta) {
    if (ruta) ruta->clear();
    Dialogo d(in, out);
    out << "\n=== Asistente para definir un producto estructurado ===\n"
        << " Responda y pulse Intro; entre corchetes está el valor por defecto.\n"
        << " Porcentajes en puntos (70 = 70%), plazos como 3A, 18M o 2.5 años.\n"
        << " Escriba ? para ver la ayuda de una pregunta y 'salir' para cancelar.\n";

    d.seccion("1. Tipo de producto");
    std::vector<std::pair<std::string, std::string>> ops;
    for (const auto& p : plantillas()) ops.push_back({p.titulo, ""});
    const Plantilla& pl = plantillas()[(size_t)d.menu(ops, 0)];
    out << "   " << pl.descripcion << "\n";

    DatosComunes c = comunes_por_defecto(pl);
    d.seccion("2. Datos generales");
    c.nombre = d.texto("Nombre del producto", pl.titulo);
    if (!pl.nominal_es_spot) c.nominal = d.numero("Nominal", c.nominal, false, 0.0, 1e15, true);
    c.moneda = d.texto("Moneda", c.moneda);
    Pregunta qv{"", "Vencimiento (3A, 18M, 2.5...)", pl.vencimiento_def, TipoDato::Plazo, "", {}, ""};
    c.vencimiento = d.preguntar(qv);
    c.tipo_interes = porcentaje(d.numero("Tipo de interés libre de riesgo (continuo, %)", 0.03, true, -0.2, 1.0), 4);

    preguntar_subyacentes(d, pl, c);

    Respuestas r;
    if (pl.id != "personalizado") {
        d.seccion("4. Características del producto");
        for (const auto& q : pl.preguntas) {
            if (!q.solo_si.empty()) {
                size_t eq = q.solo_si.find('=');
                std::string k = q.solo_si.substr(0, eq), vals = q.solo_si.substr(eq + 1);
                bool cumple = false;
                size_t ini = 0;
                while (ini <= vals.size()) {
                    size_t bar = vals.find('|', ini);
                    std::string v = vals.substr(ini, bar == std::string::npos ? std::string::npos : bar - ini);
                    if (r.count(k) && r[k] == v) cumple = true;
                    if (bar == std::string::npos) break;
                    ini = bar + 1;
                }
                if (!cumple) continue;
            }
            r[q.clave] = d.preguntar(q);
        }
        c.descripcion = d.texto("Descripción (Intro = se genera automáticamente)", "");
    }

    d.seccion("5. Simulación");
    json::Valor sim = json::Valor::objeto();
    Pregunta qm{"", "Método: auto, mc (Monte Carlo) o qmc (cuasi Monte Carlo)", "auto", TipoDato::Opcion,
                "auto elige QMC si la dimensión es moderada y MC si no", {"auto", "mc", "qmc"}, ""};
    sim.poner("metodo", d.preguntar(qm));
    sim.poner("error_objetivo",
              porcentaje(d.numero("Error estándar objetivo (% del nominal)", 0.0005, true, 0.0, 1.0, true,
                                  "precisión deseada del precio; 0.05 = 0,05% del nominal"), 4));
    if (d.si_no("¿Opciones avanzadas (pasos por año, semilla, dispositivo, precisión)?", false)) {
        std::string ppa = d.texto("Pasos de simulación por año ('auto' o un número)", "auto");
        if (ppa != "auto") sim.poner("pasos_por_anio", texto_a_real(ppa));
        sim.poner("semilla", d.entero("Semilla", 2024, 0, 2147483647));
        Pregunta qd{"", "Dispositivo: auto, gpu o cpu", "auto", TipoDato::Opcion, "", {"auto", "gpu", "cpu"}, ""};
        sim.poner("dispositivo", d.preguntar(qd));
        Pregunta qp{"", "Precisión en GPU: simple o doble", "simple", TipoDato::Opcion, "", {"simple", "doble"}, ""};
        sim.poner("precision_gpu", d.preguntar(qp));
    }
    c.simulacion = sim;

    json::Valor doc = construir_producto(pl, c, r);
    if (pl.id == "personalizado") {
        // se sustituyen las secciones del ejemplo por las que responda el usuario
        for (const char* k : {"observaciones", "cupones", "autocancelacion", "barrera_capital", "barrera_desactivante",
                              "amortizacion_final", "estado"})
            doc.quitar(k);
        doc.quitar("simulacion");
        personalizado(d, doc, texto_a_plazo(c.vencimiento));
        doc.poner("simulacion", sim);
        if (c.descripcion.empty()) {
            c.descripcion = d.texto("Descripción", "");
            if (!c.descripcion.empty()) doc.poner("descripcion", c.descripcion);
        }
    }

    // Validación y ficha
    try {
        Producto p = compilar(doc);
        imprimir_ficha(out, p);
    } catch (const std::exception& e) {
        out << "\n ✗ El producto no es válido: " << e.what() << "\n";
        if (!d.si_no("¿Guardarlo de todos modos para corregirlo a mano?", false)) return doc;
    }

    // Guardar
    d.seccion("Guardar");
    std::string def = ruta_sugerida.empty() ? nombre_fichero(c.nombre) : ruta_sugerida;
    for (;;) {
        std::string f = d.texto("Fichero donde guardar el producto (vacío = no guardar)", def);
        if (f.empty() || f == "-") break;
        std::ifstream existe(f);
        if (existe.good() && !d.si_no("El fichero ya existe. ¿Sobrescribirlo?", false)) continue;
        try {
            json::escribir_fichero(f, doc);
            out << "   ✓ Guardado en " << f << "\n";
            if (ruta) *ruta = f;
        } catch (const std::exception& e) {
            out << "   ✗ " << e.what() << "\n";
            continue;
        }
        break;
    }
    return doc;
}

} // namespace est
