#include "replicacion.hpp"
#include "perfil.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <ostream>
#include <stdexcept>

namespace est {

namespace {

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
std::string linea(char c = '-', int n = 100) { return std::string((size_t)n, c); }

// Etiqueta de un plazo en años ("1A6M" si son meses enteros)
std::string plazo_txt(double t) {
    double m = t * 12.0;
    long mm = std::lround(m);
    if (std::fabs(m - (double)mm) < 1e-6 && mm > 0) {
        std::string s;
        if (mm / 12) s += std::to_string(mm / 12) + "A";
        if (mm % 12) s += std::to_string(mm % 12) + "M";
        return s;
    }
    return fmt(t, 3) + "a";
}

// Número con decimales significativos según su tamaño
std::string num(double x) {
    double a = std::fabs(x);
    int dec = a >= 1000 ? 2 : a >= 1 ? 4 : a >= 0.01 ? 6 : 8;
    return fmt(x, dec);
}

const char* nombre_obs(int o) {
    switch (o) {
        case OBS_FECHAS: return "en cada fecha";
        case OBS_DIARIA: return "a diario";
        case OBS_CONTINUA: return "de forma continua";
        default: return "a vencimiento";
    }
}

} // namespace

bool es_replicable(const Producto& p, std::string* motivo) {
    auto no = [&](const std::string& m) {
        if (motivo) *motivo = m;
        return false;
    };
    if (p.activos.size() != 1)
        return no("tiene " + std::to_string(p.activos.size()) + " subyacentes: el pago de una cesta no se "
                  "replica con opciones sobre cada subyacente por separado");
    if (p.hay_autocall) return no("es autocancelable: lo que paga depende de si se canceló antes (de la trayectoria)");
    if (p.ko_obs != OBS_NINGUNA) return no("tiene barrera desactivante: depende de la trayectoria");
    bool usa_barrera = false;
    for (const auto& b : p.bloques) usa_barrera |= (b.cond != COND_SIEMPRE);
    if (usa_barrera && !p.ki_inicial && p.ki_obs != OBS_NINGUNA && p.ki_obs != OBS_VENCIMIENTO)
        return no(std::string("la barrera de capital se observa ") + nombre_obs(p.ki_obs) +
                  ": depende de la trayectoria (solo es replicable si se observa a vencimiento)");
    if (p.rend_medio) return no("el rendimiento final es una media de fechas (asiático): depende de la trayectoria");
    if (p.indice_cliquet() >= 0) return no("incluye un cliquet: depende del rendimiento de cada periodo");
    for (size_t c = 0; c < p.cupones.size(); ++c) {
        const Cupon& cu = p.cupones[c];
        if (cu.tipo == CUP_RANGO)
            return no("tiene un cupón range accrual: necesitaría una digital por cada día de observación");
        if (cu.tipo == CUP_CONDICIONAL && cu.memoria)
            return no("tiene cupones con memoria: cada pago depende de las fechas anteriores");
        if (c < p.memoria_inicial.size() && p.memoria_inicial[c] != 0.0)
            return no("arrastra cupones en memoria de fechas pasadas");
    }
    return true;
}

// Fuera de los strikes extremos, la cartera sigue la pendiente local del pago justo
// al otro lado del strike (con una opción en ese mismo strike), no la del último tramo
void ajustar_extremos(PerfilLineal& f, const EvaluadorFinal& g) {
    if (f.nodos.empty()) return;
    const double a = f.nodos.front().p, b = f.nodos.back().p;
    const double hb = 1e-4 * std::max(1.0, b);
    f.pend_der = (g(b + hb) - g(b)) / hb;
    if (a > 0) {
        const double ha = std::min(1e-4 * std::max(1.0, a), 0.5 * a);
        f.pend_izq = (g(a) - g(a - ha)) / ha;
    }
}

Replica replicar(const Producto& p, const OpcionesReplica& o) {
    std::string motivo;
    if (!es_replicable(p, &motivo)) throw std::runtime_error("el producto no se puede replicar estáticamente: " + motivo);
    const Activo& a = p.activos[0];
    Replica r;
    r.S_ref = a.nivel_inicial;
    r.P0 = a.spot / a.nivel_inicial;
    r.nominal = p.nominal;
    const double T = p.T, N = p.nominal, Sref = r.S_ref, P0 = r.P0;
    EvaluadorFinal g(p);
    bool hay_formula = false;
    for (const auto& b : p.bloques) hay_formula |= (b.tipo == BLQ_FORMULA);
    PerfilLineal f;
    const bool exacto = !hay_formula && perfil_exacto(p, &f);
    if (!hay_formula && !exacto) throw std::runtime_error("no se pudo obtener el perfil lineal del pago final");

    Descomposicion d;
    double T_opc = T; // vencimiento de las opciones (el de la cadena, si se usa)
    std::vector<const Cotizacion*> cot_put, cot_call; // cotización de cada pieza (modo cadena)
    auto cot_de = [&](double k, bool call) -> const Cotizacion* {
        if (!o.cadena) return nullptr;
        for (const auto& q : o.cadena->cot)
            if (q.call == call && std::fabs(q.T - T_opc) < 1e-9 && std::fabs(q.K - k * Sref) <= 1e-9 * std::max(1.0, q.K))
                return &q;
        return nullptr;
    };

    if (o.cadena) {
        double dif = 0;
        T_opc = o.cadena->vencimiento_cercano(T, &dif);
        if (T_opc < 0 || dif > o.tolerancia_vencimiento)
            throw std::runtime_error("la cadena no tiene ningún vencimiento cercano al del producto (" + plazo_txt(T) +
                                     "); el más próximo está a " + fmt(dif * 365.0, 0) + " días");
        if (dif > 1e-9)
            r.notas.push_back("las opciones vencen a " + fmt(T_opc, 4) + " años y el producto a " + fmt(T, 4) +
                              " (diferencia de " + fmt(dif * 365.0, 1) + " días)");
        double Kmin = o.p_min > 0 ? o.p_min * Sref : 0.0, Kmax = o.p_max > 0 ? o.p_max * Sref : 1e300;
        auto st = strikes_fuera_del_dinero(*o.cadena, T_opc, a.spot, o.volumen_min, Kmin, Kmax);
        if (st.size() < 2)
            throw std::runtime_error("menos de dos opciones fuera del dinero cumplen el filtro (volumen ≥ " +
                                     fmt(o.volumen_min, 0) + ") en ese vencimiento; baje el volumen mínimo");
        int n_put = 0, n_call = 0;
        std::vector<double> nodos;
        for (const auto& s : st) { nodos.push_back(s.K / Sref); (s.call ? n_call : n_put)++; }
        r.p_lo = nodos.front();
        r.p_hi = nodos.back();
        if (o.pesos == "curvatura") {
            if (!hay_formula)
                throw std::runtime_error("el método de curvatura necesita un pago suave definido con una fórmula; "
                                         "con pagos lineales a trozos use la interpolación (exacta en los strikes)");
            // Cuadratura de Carr-Madan: w_i = A''(k_i)·ancho_i (como TooManyOptions)
            auto d1 = [&](double x) { double h = 1e-4 * std::max(1.0, x); return (g(x + h) - g(x - h)) / (2 * h); };
            auto d2 = [&](double x) { double h = 1e-3 * std::max(1.0, x); return (g(x + h) - 2 * g(x) + g(x - h)) / (h * h); };
            d.P0 = P0;
            d.base = g(P0);
            d.pendiente = d1(P0);
            const size_t m = nodos.size();
            for (size_t i = 0; i < m; ++i) {
                double ancho = (i == 0) ? nodos[1] - nodos[0]
                             : (i + 1 == m) ? nodos[m - 1] - nodos[m - 2] : 0.5 * (nodos[i + 1] - nodos[i - 1]);
                double w = d2(nodos[i]) * ancho;
                if (std::fabs(w) < 1e-14) continue;
                if (nodos[i] < P0) d.puts.push_back({nodos[i], w});
                else d.calls.push_back({nodos[i], w});
            }
            r.metodo = "strikes cotizados con pesos por curvatura A''(K)·ΔK (cuadratura de Carr-Madan)";
        } else {
            f = interpolar(g, nodos);
            ajustar_extremos(f, g);
            d = descomponer(f, P0);
            r.metodo = "strikes cotizados: interpolación lineal del pago en cada strike (exacta en los strikes)";
        }
        r.metodo += "; " + std::to_string(n_put) + " puts y " + std::to_string(n_call) + " calls" +
                    (o.volumen_min > 0 ? " con volumen ≥ " + fmt(o.volumen_min, 0) : std::string());
    } else if (exacto) {
        d = descomponer(f, P0);
        r.exacta = true;
        r.metodo = "exacta: el pago es lineal a trozos y basta con opciones en sus nodos";
        double ult = f.nodos.empty() ? 1.0 : f.nodos.back().p;
        r.p_lo = 0.0;
        r.p_hi = o.p_max > 0 ? o.p_max : std::max(3.0 * P0, 1.5 * ult);
    } else {
        const double paso = o.paso > 0 ? o.paso : 0.05;
        r.p_lo = o.p_min >= 0 ? o.p_min : paso;
        r.p_hi = o.p_max > 0 ? o.p_max : std::max(2.5 * P0, 2.5);
        if (!(r.p_hi > r.p_lo)) throw std::runtime_error("el rango de strikes está vacío");
        std::vector<double> nodos;
        for (long k = (long)std::ceil(r.p_lo / paso - 1e-9); k * paso <= r.p_hi + 1e-12; ++k)
            if (k * paso > 0) nodos.push_back(k * paso);
        nodos.push_back(P0);
        // Nodos exactos de los componentes lineales a trozos (strikes, topes, barrera)
        for (const auto& b : p.bloques) {
            for (double x : {b.strike, b.suelo, b.techo})
                if (b.tipo != BLQ_FORMULA && b.tipo != BLQ_FIJO && x > r.p_lo && x < r.p_hi && std::fabs(x) < 1e20)
                    nodos.push_back(x);
        }
        if (p.ki_obs == OBS_VENCIMIENTO && p.ki_nivel > r.p_lo && p.ki_nivel < r.p_hi) nodos.push_back(p.ki_nivel);
        std::sort(nodos.begin(), nodos.end());
        f = interpolar(g, nodos);
        ajustar_extremos(f, g);
        d = descomponer(f, P0);
        r.metodo = "rejilla de strikes cada " + porcentaje(paso) + " entre " + porcentaje(r.p_lo) + " y " +
                   porcentaje(r.p_hi) + " (el pago tiene fórmulas: se interpola en la rejilla)";
    }

    // ------------------- Valoración de las piezas ------------------- //
    PreciosVanilla pv(a, p.curva, o.mc, o.precios_mc);
    std::vector<double> vtos = {T, T_opc};
    for (size_t j = 0; j < p.t_obs.size(); ++j) vtos.push_back(p.t_obs[j]);
    pv.preparar(vtos);
    r.metodo_precios = def_modelo(a.modelo.tipo).nombre + ": " + pv.metodo();
    if (o.cadena) r.metodo_precios = "opciones a precio de mercado (medio de la horquilla) de " +
                                     (o.cadena->subyacente.empty() ? o.cadena->ruta : o.cadena->subyacente) +
                                     "; el resto con " + r.metodo_precios;

    auto anadir = [&](LineaReplica l) {
        l.valor = l.cantidad * l.precio;
        r.lineas.push_back(l);
    };
    const std::string fin = "amortización final";
    const double tol = 1e-12;
    double efectivo = d.base - d.pendiente * d.P0;
    if (std::fabs(efectivo) > tol) {
        LineaReplica l;
        l.instrumento = "Bono cupón cero";
        l.origen = fin;
        l.T = T;
        l.peso = efectivo;
        l.cantidad = N * efectivo;
        l.precio = pv.df(T);
        anadir(l);
    }
    if (std::fabs(d.pendiente) > tol) {
        LineaReplica l;
        l.instrumento = "Subyacente (entrega en T)";
        l.origen = fin;
        l.T = T;
        l.peso = d.pendiente;
        l.cantidad = N * d.pendiente / Sref;
        l.precio = a.spot * std::exp(-a.dividendos * T); // forward prepagado
        anadir(l);
    }
    auto opcion = [&](const char* nombre, bool call, const Descomposicion::Pieza& pz) {
        LineaReplica l;
        l.instrumento = nombre;
        l.origen = fin;
        l.T = T_opc;
        l.k = pz.k;
        l.K = pz.k * Sref;
        l.peso = pz.w;
        l.cantidad = N * pz.w / Sref;
        const Cotizacion* q = cot_de(pz.k, call);
        if (q) {
            l.precio = q->referencia();
            l.bid = q->bid;
            l.ask = q->ask;
            l.de_mercado = true;
        } else {
            l.precio = call ? pv.call(T_opc, l.K) : pv.put(T_opc, l.K);
        }
        anadir(l);
    };
    for (const auto& pz : d.puts) opcion("Put", false, pz);
    for (const auto& pz : d.calls) opcion("Call", true, pz);
    auto digital = [&](const char* nombre, bool call, const Descomposicion::Pieza& pz, double t, const std::string& origen) {
        LineaReplica l;
        l.instrumento = nombre;
        l.origen = origen;
        l.T = t;
        l.k = pz.k;
        l.K = pz.k * Sref;
        l.peso = pz.w;
        l.cantidad = N * pz.w; // digitales que pagan 1 unidad monetaria
        l.precio = pv.digital(t, l.K, call);
        anadir(l);
    };
    for (const auto& pz : d.dig_puts) digital("Digital put", false, pz, T, fin);
    for (const auto& pz : d.dig_calls) digital("Digital call", true, pz, T, fin);

    // Cupones: fijos = bonos; condicionales sin memoria = digitales en su fecha
    for (size_t c = 0; c < p.cupones.size(); ++c) {
        const Cupon& cu = p.cupones[c];
        for (int j = 0; j < p.n_obs(); ++j) {
            const double imp = cu.importe[(size_t)j], t = p.t_obs[(size_t)j];
            if (imp == 0.0) continue;
            std::string origen = "cupón " + std::to_string(j + 1) + " (" + p.etiqueta_obs[(size_t)j] + ")";
            if (cu.tipo == CUP_FIJO) {
                LineaReplica l;
                l.instrumento = "Bono cupón cero";
                l.origen = origen;
                l.T = t;
                l.peso = imp;
                l.cantidad = N * imp;
                l.precio = pv.df(t);
                anadir(l);
            } else if (cu.barrera[(size_t)j] < HD_INFINITO * 0.5) {
                digital("Digital call", true, {cu.barrera[(size_t)j], imp}, t, origen);
            }
        }
    }

    r.valor = 0;
    for (const auto& l : r.lineas) r.valor += l.valor;
    if (o.cadena) {
        // Montar la cartera cruzando la horquilla: se compra al ask y se vende al bid
        bool horquilla = true;
        double compra = 0, venta = 0;
        for (const auto& l : r.lineas) {
            if (l.de_mercado && l.bid > 0 && l.ask > 0) {
                compra += l.cantidad * (l.cantidad > 0 ? l.ask : l.bid);
                venta += l.cantidad * (l.cantidad > 0 ? l.bid : l.ask);
            } else {
                if (l.de_mercado) horquilla = false;
                compra += l.valor;
                venta += l.valor;
            }
        }
        if (horquilla) { r.valor_ask = compra; r.valor_bid = venta; r.con_horquilla = true; }
    }

    // ------------------- Error del pago final ------------------- //
    auto replica_en = [&](double P) {
        double v = d.base + d.pendiente * (P - d.P0);
        for (const auto& pz : d.puts) v += pz.w * std::max(pz.k - P, 0.0);
        for (const auto& pz : d.calls) v += pz.w * std::max(P - pz.k, 0.0);
        for (const auto& pz : d.dig_puts) v += (P < pz.k) ? pz.w : 0.0;
        for (const auto& pz : d.dig_calls) v += (P >= pz.k) ? pz.w : 0.0;
        return v;
    };
    const int M = 1200;
    double s2 = 0;
    r.error_max = 0;
    for (int i = 0; i <= M; ++i) {
        double P = r.p_lo + (r.p_hi - r.p_lo) * i / M;
        double y = g(P), z = replica_en(P);
        r.P.push_back(P);
        r.objetivo.push_back(y);
        r.replicado.push_back(z);
        double e = std::fabs(z - y);
        r.error_max = std::max(r.error_max, e);
        s2 += e * e;
    }
    r.error_rms = std::sqrt(s2 / (M + 1));
    if (!r.exacta) {
        std::string n = "fuera de [" + porcentaje(r.p_lo) + ", " + porcentaje(r.p_hi) +
                        "] la cartera sigue en línea recta con la pendiente del pago en los strikes extremos";
        // ¿El pago sigue curvándose fuera del rango cubierto?
        double e_bajo = r.p_lo > 0 ? std::fabs(replica_en(0.5 * r.p_lo) - g(0.5 * r.p_lo)) : 0.0;
        double e_alto = std::fabs(replica_en(1.5 * r.p_hi) - g(1.5 * r.p_hi));
        if (std::max(e_bajo, e_alto) > std::max(10.0 * r.error_max, 1e-4))
            n += "; ahí el pago objetivo se aparta (en P = " + porcentaje(e_alto >= e_bajo ? 1.5 * r.p_hi : 0.5 * r.p_lo) +
                 " la diferencia es de " + porcentaje(std::max(e_bajo, e_alto), 3) +
                 " del nominal) y la réplica no recoge el valor de esas colas";
        r.notas.push_back(n);
    }
    if (a.modelo.tipo == TipoModelo::SABR && !o.precios_mc && !o.cadena)
        r.notas.push_back("SABR se valora con la aproximación de Hagan; --precios-mc usa el Monte Carlo del modelo");
    return r;
}

void imprimir_replica(std::ostream& os, const Producto& p, const Replica& r) {
    const Activo& a = p.activos[0];
    os << "\n" << linea('=') << "\n RÉPLICA ESTÁTICA: " << p.nombre << "\n" << linea('=') << "\n";
    os << " Se replica cada flujo con instrumentos que pagan en su fecha según el precio del subyacente:\n"
       << "   pago final A(P) con P = S_T / " << fmt(r.S_ref, 4) << " (nivel inicial de " << a.nombre << ")";
    if (!p.cupones.empty()) os << " y los cupones";
    os << "\n Método ........ " << r.metodo << "\n";
    os << " Desarrollo .... alrededor del spot: P0 = " << porcentaje(r.P0) << " (S0 = " << fmt(a.spot, 4) << ")\n";
    os << " Precios ....... " << r.metodo_precios << "\n";

    os << "\n Cartera para un nominal de " << fmt(r.nominal, 2) << " " << p.moneda
       << "  (opciones sobre 1 unidad del subyacente; digitales que pagan 1 " << p.moneda << ")\n";
    os << "   " << col("Instrumento", 27) << col("Vto.", 8) << dcha("Strike %", 10) << dcha("Strike", 12)
       << dcha("Peso %nom", 12) << dcha("Cantidad", 15) << dcha("Precio", 14) << dcha("Valor", 14) << "   Origen\n";
    for (const auto& l : r.lineas) {
        os << "   " << col(l.instrumento + (l.de_mercado ? " *" : ""), 27) << col(plazo_txt(l.T), 8)
           << dcha(l.k >= 0 ? porcentaje(l.k) : "-", 10) << dcha(l.K >= 0 ? fmt(l.K, 2) : "-", 12)
           << dcha(porcentaje(l.peso, 3), 12) << dcha(num(l.cantidad), 15) << dcha(num(l.precio), 14)
           << dcha(fmt(l.valor, 2), 14) << "   " << l.origen << "\n";
    }
    bool mercado = false;
    for (const auto& l : r.lineas) mercado |= l.de_mercado;
    if (mercado) os << "   (* = precio de mercado)\n";
    os << " " << linea('-') << "\n";
    os << "   Valor de la réplica .......... " << fmt(r.valor, 2) << " " << p.moneda << "   ("
       << porcentaje(r.valor / r.nominal, 4) << " del nominal)\n";
    if (r.con_horquilla)
        os << "   Con la horquilla ............. comprar la cartera cuesta " << fmt(r.valor_ask, 2) << " " << p.moneda
           << "; venderla da " << fmt(r.valor_bid, 2) << " " << p.moneda << "\n";
    os << "   Error del pago a vencimiento . ";
    if (r.exacta && r.error_max < 1e-9) os << "nulo (réplica exacta)";
    else os << "máximo " << porcentaje(r.error_max, 4) << " y cuadrático medio " << porcentaje(r.error_rms, 4)
            << " del nominal";
    os << ", en P ∈ [" << porcentaje(r.p_lo) << ", " << porcentaje(r.p_hi) << "]\n";
    for (const auto& n : r.notas) os << "   · " << n << "\n";

    // Perfil comparado (pocas filas)
    os << "\n Pago a vencimiento (% del nominal, sin cupones)\n   " << col("P final", 10) << dcha("Objetivo", 12)
       << dcha("Réplica", 12) << dcha("Diferencia", 13) << "\n";
    const int filas = 12;
    for (int i = 0; i <= filas; ++i) {
        size_t k = (size_t)std::lround((double)i / filas * (double)(r.P.size() - 1));
        os << "   " << col(porcentaje(r.P[k]), 10) << dcha(porcentaje(r.objetivo[k], 3), 12)
           << dcha(porcentaje(r.replicado[k], 3), 12) << dcha(porcentaje(r.replicado[k] - r.objetivo[k], 4), 13) << "\n";
    }
    os << linea('=') << "\n";
}

void guardar_replica_csv(const std::string& ruta, const Replica& r) {
    std::ofstream f(ruta);
    if (!f) throw std::runtime_error("no se puede escribir " + ruta);
    f << "instrumento,origen,vencimiento,strike_pct,strike,peso,cantidad,precio,valor,bid,ask,de_mercado\n";
    for (const auto& l : r.lineas) {
        f << "\"" << l.instrumento << "\",\"" << l.origen << "\"," << fmt(l.T, 8) << ","
          << (l.k >= 0 ? fmt(l.k, 10) : "") << "," << (l.K >= 0 ? fmt(l.K, 8) : "") << "," << fmt(l.peso, 12) << ","
          << fmt(l.cantidad, 12) << "," << fmt(l.precio, 12) << "," << fmt(l.valor, 10) << ","
          << (l.bid > 0 ? fmt(l.bid, 8) : "") << "," << (l.ask > 0 ? fmt(l.ask, 8) : "") << ","
          << (l.de_mercado ? 1 : 0) << "\n";
    }
}

void guardar_perfil_csv(const std::string& ruta, const Replica& r) {
    std::ofstream f(ruta);
    if (!f) throw std::runtime_error("no se puede escribir " + ruta);
    f << "P,objetivo,replica,error\n";
    for (size_t i = 0; i < r.P.size(); ++i)
        f << fmt(r.P[i], 8) << "," << fmt(r.objetivo[i], 12) << "," << fmt(r.replicado[i], 12) << ","
          << fmt(r.replicado[i] - r.objetivo[i], 12) << "\n";
}

json::Valor replica_a_json(const Producto& p, const Replica& r) {
    json::Valor o = json::Valor::objeto();
    o.poner("producto", json::Valor(p.nombre));
    o.poner("metodo", json::Valor(r.metodo));
    o.poner("precios", json::Valor(r.metodo_precios));
    o.poner("P0", json::Valor(r.P0));
    o.poner("nominal", json::Valor(r.nominal));
    o.poner("valor", json::Valor(r.valor));
    o.poner("valor_pct_nominal", json::Valor(r.valor / r.nominal));
    if (r.con_horquilla) {
        o.poner("coste_compra_horquilla", json::Valor(r.valor_ask));
        o.poner("ingreso_venta_horquilla", json::Valor(r.valor_bid));
    }
    o.poner("exacta", json::Valor(r.exacta));
    o.poner("error_max", json::Valor(r.error_max));
    o.poner("error_rms", json::Valor(r.error_rms));
    json::Valor ls = json::Valor::lista();
    for (const auto& l : r.lineas) {
        json::Valor x = json::Valor::objeto();
        x.poner("instrumento", json::Valor(l.instrumento));
        x.poner("origen", json::Valor(l.origen));
        x.poner("vencimiento", json::Valor(l.T));
        if (l.k >= 0) { x.poner("strike_pct", json::Valor(l.k)); x.poner("strike", json::Valor(l.K)); }
        x.poner("peso", json::Valor(l.peso));
        x.poner("cantidad", json::Valor(l.cantidad));
        x.poner("precio", json::Valor(l.precio));
        x.poner("valor", json::Valor(l.valor));
        if (l.de_mercado) x.poner("precio_de_mercado", json::Valor(true));
        ls.anadir(std::move(x));
    }
    o.poner("cartera", std::move(ls));
    return o;
}

} // namespace est
