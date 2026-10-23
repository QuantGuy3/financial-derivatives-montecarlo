// Ficha descriptiva (term sheet) de un producto ya validado: permite comprobar
// de un vistazo que el fichero dice lo que se quería decir antes de valorarlo.

#include "producto.hpp"

#include <algorithm>
#include <ostream>
#include <sstream>

namespace est {

namespace {

std::string linea(char c = '-', int n = 78) { return std::string((size_t)n, c); }

size_t ancho_visual(const std::string& s) {
    size_t vis = 0;
    for (unsigned char ch : s) vis += ((ch & 0xC0) != 0x80);
    return vis;
}
std::string col(std::string s, size_t w) {
    size_t v = ancho_visual(s);
    if (v < w) s += std::string(w - v, ' ');
    return s;
}

std::string nombre_obs(int o, bool bgk) {
    switch (o) {
        case OBS_VENCIMIENTO: return "solo a vencimiento (europea)";
        case OBS_FECHAS:      return "en cada fecha de observación";
        case OBS_DIARIA:      return "diaria (cada paso de la simulación, sin corrección)";
        case OBS_CONTINUA:    return std::string("continua (americana)") +
                                     (bgk ? ", con corrección de continuidad BGK" : ", sin corrección");
        default:              return "no hay";
    }
}

std::string condicion(int c) {
    switch (c) {
        case COND_CON_BARRERA: return "  [solo si se ha tocado la barrera de capital]";
        case COND_SIN_BARRERA: return "  [solo si NO se ha tocado la barrera de capital]";
        default:               return "";
    }
}

} // namespace

void imprimir_ficha(std::ostream& os, const Producto& p) {
    const int n = (int)p.activos.size();
    os << "\n" << linea('=') << "\n " << p.nombre << "\n" << linea('=') << "\n";
    if (!p.descripcion.empty()) os << " " << p.descripcion << "\n\n";
    os << " Nominal ............ " << fmt(p.nominal, 2) << " " << p.moneda << "\n";
    os << " Vencimiento ........ " << fmt(p.T, 4) << " años\n";
    if (!p.plantilla.empty()) os << " Plantilla .......... " << p.plantilla << "\n";
    if (p.curva.t.size() == 1) {
        os << " Tipo de interés .... " << porcentaje(p.curva.r[0]) << " (continuo, curva plana)\n";
    } else {
        os << " Curva de tipos ..... ";
        for (size_t i = 0; i < p.curva.t.size(); ++i)
            os << (i ? ", " : "") << fmt(p.curva.t[i], 2) << "a: " << porcentaje(p.curva.r[i]);
        os << " (cero, continuos)\n";
    }

    os << "\n Subyacentes (rendimiento P_i = S_i / nivel inicial)\n";
    os << "   " << col("#", 4) << col("Nombre", 12) << col("Spot", 12) << col("Nivel inicial", 15)
       << col("Div.", 9) << "Modelo\n";
    for (int i = 0; i < n; ++i) {
        const Activo& a = p.activos[(size_t)i];
        os << "   " << col(std::to_string(i + 1), 4) << col(a.nombre, 12) << col(fmt(a.spot, 4), 12)
           << col(fmt(a.nivel_inicial, 4), 15) << col(porcentaje(a.dividendos), 9) << a.modelo.resumen() << "\n";
    }
    if (n > 1) {
        os << "\n Correlación\n   " << col("", 12);
        for (int j = 0; j < n; ++j) os << col(p.activos[(size_t)j].nombre.substr(0, 8), 9);
        os << "\n";
        for (int i = 0; i < n; ++i) {
            os << "   " << col(p.activos[(size_t)i].nombre.substr(0, 11), 12);
            for (int j = 0; j < n; ++j) os << col(fmt(p.correlacion[(size_t)(i * n + j)], 3), 9);
            os << "\n";
        }
        os << "\n Cesta: P = " << p.nombre_cesta();
        if (p.cesta == CESTA_PONDERADA) {
            os << " (pesos:";
            for (int i = 0; i < n; ++i) os << " " << porcentaje(p.pesos[(size_t)i]);
            os << ")";
        }
        os << " de los rendimientos\n";
    }

    // Calendario
    const int nobs = p.n_obs();
    bool hay_cup = !p.cupones.empty();
    os << "\n Calendario de observación (" << nobs << (nobs == 1 ? " fecha" : " fechas") << ")\n";
    os << "   " << col("#", 5) << col("Fecha", 12) << col("t (años)", 10);
    if (p.hay_autocall) os << col("Autocancelación", 26);
    if (hay_cup) os << "Cupones";
    os << "\n";
    for (int j = 0; j < nobs; ++j) {
        os << "   " << col(std::to_string(j + 1), 5) << col(p.etiqueta_obs[(size_t)j], 12)
           << col(fmt(p.t_obs[(size_t)j], 4), 10);
        if (p.hay_autocall) {
            std::string s = "-";
            if (p.ac_activa[(size_t)j])
                s = "P≥" + porcentaje(p.ac_barrera[(size_t)j]) + " → paga " + porcentaje(p.ac_importe[(size_t)j]);
            os << col(s, 26);
        }
        if (hay_cup) {
            std::string s;
            for (size_t c = 0; c < p.cupones.size(); ++c) {
                const Cupon& cu = p.cupones[c];
                double imp = cu.importe[(size_t)j];
                if (cu.tipo == CUP_CONDICIONAL && cu.barrera[(size_t)j] >= HD_INFINITO * 0.5) continue;
                if (imp == 0.0 && cu.tipo != CUP_CONDICIONAL) continue;
                if (!s.empty()) s += " + ";
                if (cu.tipo == CUP_FIJO) s += porcentaje(imp);
                else if (cu.tipo == CUP_CONDICIONAL)
                    s += porcentaje(imp) + " si P≥" + porcentaje(cu.barrera[(size_t)j]) + (cu.memoria ? " (memoria)" : "");
                else
                    s += porcentaje(imp) + " × fracción de días con " + porcentaje(cu.inf) + "≤P≤" + porcentaje(cu.sup);
            }
            os << (s.empty() ? "-" : s);
        }
        os << "\n";
    }

    // Barreras
    os << "\n Barrera de capital (knock-in) .. ";
    if (p.ki_obs == OBS_NINGUNA) os << "no hay";
    else os << "P ≤ " << porcentaje(p.ki_nivel) << ", observación " << nombre_obs(p.ki_obs, p.ki_bgk);
    if (p.ki_inicial) os << "  [YA TOCADA]";
    os << "\n";
    if (p.ko_obs != OBS_NINGUNA) {
        os << " Barrera desactivante (KO) ...... P ≥ " << porcentaje(p.ko_nivel) << ", observación "
           << nombre_obs(p.ko_obs, p.ko_bgk) << "\n"
           << "   si se toca, el producto termina y paga " << porcentaje(p.ko_importe)
           << (p.ko_pago_venc ? " a vencimiento" : " en ese momento") << "\n";
    }

    // Amortización final
    os << "\n Amortización a vencimiento (si el producto sigue vivo), en % del nominal,\n con P = "
       << (p.rend_medio ? "media de los rendimientos de la cesta en las fechas de observación"
                        : "rendimiento de la cesta en la fecha final")
       << ":\n";
    if (p.bloques.empty()) os << "   (no hay pago final)\n";
    for (const auto& b : p.bloques) {
        if (b.oculto) continue; // piezas internas de una curva (se describe en su primer bloque)
        bool neg = !b.descripcion.empty() && b.descripcion[0] == '-';
        os << "   " << (neg ? b.descripcion : "+ " + b.descripcion) << condicion(b.cond) << "\n";
    }

    // Perfil de pago a vencimiento
    if (!p.bloques.empty()) {
        std::vector<BloqueSim<double>> bs;
        for (const auto& b : p.bloques) {
            BloqueSim<double> x{};
            x.tipo = b.tipo; x.cond = b.cond; x.importe = b.importe; x.part = b.part;
            x.strike = b.strike; x.suelo = b.suelo; x.techo = b.techo;
            x.ini = b.formula_ini; x.len = b.formula_len;
            bs.push_back(x);
        }
        SimProducto<double> sp{};
        sp.n_bloques = (int)bs.size();
        sp.bloques = bs.data();
        sp.formulas = p.formulas.empty() ? nullptr : p.formulas.data();
        bool dos = (p.ki_obs != OBS_NINGUNA);
        os << "\n Perfil de pago a vencimiento (sin cupones" << (p.indice_cliquet() >= 0 ? "; cliquet acumulado = 0" : "")
           << ")\n   " << col("P final", 10) << (dos ? col("Sin tocar barrera", 20) + "Barrera tocada" : std::string("Amortización"))
           << "\n";
        // Un P final por debajo de la barrera de capital implica haberla tocado (todas las
        // formas de observación incluyen la fecha final); con observación solo a
        // vencimiento, por encima de la barrera no puede estar tocada. Por encima de la
        // desactivante el producto ya habría terminado.
        auto celda = [&](double P, int ki) -> std::string {
            const double tol = 1e-9;
            if (p.ko_obs != OBS_NINGUNA && P >= p.ko_nivel - tol) return "(desactivado: " + porcentaje(p.ko_importe) + ")";
            if (dos && ki == 0 && P <= p.ki_nivel + tol) return "-";
            if (dos && ki == 1 && P > p.ki_nivel + tol && p.ki_obs == OBS_VENCIMIENTO && !p.ki_inicial) return "-";
            return porcentaje(amortizacion_final(sp, P, ki, 0.0));
        };
        for (int k = 0; k <= 15; ++k) {
            double P = 0.1 * k;
            os << "   " << col(porcentaje(P), 10);
            if (dos) os << col(celda(P, 0), 20) << celda(P, 1);
            else os << celda(P, 0);
            os << "\n";
        }
        if (dos) os << "   ('-' = escenario imposible con esa forma de observación de la barrera)\n";
    }

    // Simulación
    const ConfigSimulacion& s = p.sim;
    double ppa = pasos_por_anio_efectivos(p);
    os << "\n Simulación: método " << s.metodo << ", dispositivo " << s.dispositivo << ", error objetivo "
       << porcentaje(s.error_objetivo, 3) << " del nominal, ";
    if (ppa > 0) os << fmt(ppa, 1) << " pasos/año";
    else os << "solo fechas de observación (modelos exactos)";
    os << ", semilla " << s.semilla << "\n";

    if (!p.avisos.empty()) {
        os << "\n Avisos:\n";
        for (const auto& a : p.avisos) os << "   ! " << a << "\n";
    }
    os << linea('=') << "\n";
}

} // namespace est
