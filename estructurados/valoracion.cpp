#include "valoracion.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace est {

namespace {

using Reloj = std::chrono::steady_clock;

std::unique_ptr<Motor> crear_motor(const Producto& q, const ConfigSimulacion& c) {
    const bool pide_gpu = (c.dispositivo == "gpu");
    if (pide_gpu || c.dispositivo == "auto") {
        std::string desc;
        if (gpu_disponible(&desc)) return crear_motor_gpu(q, c.semilla, c.doble_precision);
        if (pide_gpu) throw std::runtime_error("se ha pedido GPU pero no está disponible: " + desc);
    }
    return crear_motor_cpu(q, c.semilla, c.hilos);
}

// Con "auto" se usa QMC si el motor lo admite y la dimensión del problema es
// moderada (pocas fechas/pasos): ahí la baja discrepancia de Sobol compensa. Con
// miles de dimensiones (p. ej. vigilancia diaria) la ganancia es pequeña y MC,
// más barato por trayectoria, suele llegar antes al error objetivo.
bool usa_qmc(const ConfigSimulacion& c, const Motor& m, long long dimension) {
    if (c.metodo == "qmc") {
        if (!m.admite_qmc())
            throw std::runtime_error("QMC no disponible en este motor (" + m.descripcion() +
                                     "); use \"metodo\": \"mc\"");
        return true;
    }
    if (c.metodo == "mc") return false;
    return m.admite_qmc() && dimension <= 1024;
}

long long pot2_ceil(long long x) {
    long long p = 1;
    while (p < x) p <<= 1;
    return p;
}

std::string miles(long long x) {
    std::string s = std::to_string(x), out;
    int c = 0;
    for (int i = (int)s.size() - 1; i >= 0; --i) {
        out.insert(out.begin(), s[(size_t)i]);
        if (++c % 3 == 0 && i > 0 && s[(size_t)i - 1] != '-') out.insert(out.begin(), ' ');
    }
    return out;
}

void rellenar(ResultadoValoracion& r, const Acumulado& a) {
    const double n = std::max(a.n, 1.0);
    r.vp_cupones = a.s_cup / n;
    r.vp_anticipada = a.s_ant / n;
    r.vp_final = a.s_fin / n;
    r.vida_media = a.s_vida / n;
    r.prob_ki = a.n_ki / n;
    r.prob_ko = a.n_ko / n;
    r.prob_perdida = a.n_perd / n;
    r.perdida_media = a.n_perd > 0 ? a.s_perd / a.n_perd : 0.0;
    r.prob_autocall.resize(a.n_ac.size());
    r.prob_cupon.resize(a.n_cup.size());
    for (size_t j = 0; j < a.n_ac.size(); ++j) {
        r.prob_autocall[j] = a.n_ac[j] / n;
        r.prob_cupon[j] = a.n_cup[j] / n;
    }
}

// Precio con un número FIJO de trayectorias (MC) o de puntos por réplica (QMC):
// lo usan las griegas para que todas las revaluaciones compartan el ruido.
double precio_fijo(const Producto& q, const ConfigSimulacion& c, bool qmc, long long N, int R, long long Nper) {
    auto motor = crear_motor(q, c);
    Acumulado acc;
    acc.iniciar(q.n_obs());
    if (qmc) {
        double suma = 0.0;
        for (int r = 0; r < R; ++r) {
            Acumulado ar;
            ar.iniciar(q.n_obs());
            motor->qmc(r, 0, Nper, ar);
            suma += ar.s_pv / ar.n;
        }
        return suma / R;
    }
    motor->mc(0, N, acc);
    return acc.s_pv / acc.n;
}

} // namespace

ConfigSimulacion configuracion_efectiva(const Producto& p, const OpcionesValoracion& o) {
    ConfigSimulacion c = p.sim;
    if (!o.metodo.empty()) c.metodo = o.metodo;
    if (!o.dispositivo.empty()) c.dispositivo = o.dispositivo;
    if (o.error_objetivo > 0) c.error_objetivo = o.error_objetivo;
    if (o.trayectorias >= 0) c.trayectorias = o.trayectorias;
    if (o.trayectorias_max > 0) c.trayectorias_max = o.trayectorias_max;
    if (o.semilla >= 0) c.semilla = (uint32_t)o.semilla;
    if (o.replicas_qmc > 0) c.replicas_qmc = o.replicas_qmc;
    if (o.hilos >= 0) c.hilos = o.hilos;
    if (o.doble_precision >= 0) c.doble_precision = (o.doble_precision == 1);
    if (o.pasos_por_anio > -2.0) c.pasos_por_anio = o.pasos_por_anio;
    return c;
}

ResultadoValoracion valorar(const Producto& p, const OpcionesValoracion& o, std::ostream* log) {
    const auto t0 = Reloj::now();
    ConfigSimulacion c = configuracion_efectiva(p, o);
    Producto q = p;
    q.sim = c;
    auto motor = crear_motor(q, c);

    ResultadoValoracion res;
    res.motor = motor->descripcion();
    res.semilla = c.semilla;
    res.pasos_por_anio = pasos_por_anio_efectivos(q);
    res.error_objetivo = c.error_objetivo;
    {
        // número de pasos y dimensión del problema (normales por trayectoria)
        double ppa = res.pasos_por_anio, tp = 0.0;
        long long pasos = 0;
        for (double t : q.t_obs) {
            pasos += (ppa > 0) ? std::max(1, (int)std::ceil((t - tp) * ppa - 1e-9)) : 1;
            tp = t;
        }
        int dim_paso = 0;
        for (const auto& a : q.activos) dim_paso += 1 + ruidos_extra(a.modelo.tipo);
        res.pasos = (int)pasos;
        res.dimension = pasos * dim_paso;
    }
    const bool qmc = usa_qmc(c, *motor, res.dimension);
    const double objetivo = c.error_objetivo;
    const int nobs = q.n_obs();

    if (!qmc) {
        res.metodo = "MC (Philox)";
        Acumulado acc;
        acc.iniciar(nobs);
        if (c.trayectorias > 0) {
            motor->mc(0, c.trayectorias, acc);
        } else {
            const long long piloto = motor->es_gpu() ? (1LL << 16) : (1LL << 13);
            motor->mc(0, std::min(piloto, c.trayectorias_max), acc);
            for (int it = 0; it < 6; ++it) {
                double m = acc.s_pv / acc.n;
                double var = std::max(0.0, acc.s_pv2 / acc.n - m * m) * acc.n / std::max(acc.n - 1.0, 1.0);
                double se = std::sqrt(var / acc.n);
                if (se <= objetivo) break;
                long long necesario = (long long)std::ceil(1.02 * var / (objetivo * objetivo));
                if (necesario > c.trayectorias_max) necesario = c.trayectorias_max;
                if (necesario <= (long long)acc.n) { res.objetivo_alcanzado = false; break; }
                if (log)
                    *log << "  · " << miles((long long)acc.n) << " trayectorias: error " << porcentaje(se, 3)
                         << " (objetivo " << porcentaje(objetivo, 3) << ") -> se simulan hasta "
                         << miles(necesario) << "\n";
                motor->mc((long long)acc.n, necesario, acc);
            }
        }
        double m = acc.s_pv / acc.n;
        double var = std::max(0.0, acc.s_pv2 / acc.n - m * m) * acc.n / std::max(acc.n - 1.0, 1.0);
        res.precio = m;
        res.error = std::sqrt(var / acc.n);
        res.trayectorias = (long long)acc.n;
        rellenar(res, acc);
    } else {
        const int R = c.replicas_qmc;
        res.replicas = R;
        long long Nper = motor->es_gpu() ? 4096 : 512;
        if (c.trayectorias > 0) Nper = pot2_ceil(std::max(1LL, (c.trayectorias + R - 1) / R));
        std::vector<Acumulado> accr((size_t)R);
        for (auto& a : accr) a.iniciar(nobs);
        long long hecho = 0;
        double media = 0.0, se = 0.0;
        for (;;) {
            for (int r = 0; r < R; ++r) motor->qmc(r, hecho, Nper, accr[(size_t)r]);
            hecho = Nper;
            media = 0.0;
            for (const auto& a : accr) media += a.s_pv / a.n;
            media /= R;
            double v = 0.0;
            for (const auto& a : accr) { double d = a.s_pv / a.n - media; v += d * d; }
            se = std::sqrt(v / ((double)R * (R - 1)));
            if (c.trayectorias > 0 || se <= objetivo) break;
            if ((long long)R * Nper * 2 > c.trayectorias_max || Nper * 2 > 4294967296LL) {
                res.objetivo_alcanzado = false;
                break;
            }
            if (log)
                *log << "  · " << R << " réplicas × " << miles(Nper) << " puntos: error " << porcentaje(se, 3)
                     << " (objetivo " << porcentaje(objetivo, 3) << ") -> se duplica\n";
            Nper *= 2;
        }
        Acumulado tot;
        tot.iniciar(nobs);
        for (const auto& a : accr) tot.combinar(a);
        res.metodo = "QMC (Sobol + scrambling Hong-Hickernell)";
        res.precio = media;
        res.error = se;
        res.trayectorias = (long long)tot.n;
        res.puntos_por_replica = Nper;
        rellenar(res, tot);
    }
    if (res.error > objetivo * 1.0001 && c.trayectorias <= 0) res.objetivo_alcanzado = false;
    res.tiempo_s = std::chrono::duration<double>(Reloj::now() - t0).count();
    return res;
}

Griegas calcular_griegas(const Producto& p, const OpcionesValoracion& o, const ResultadoValoracion& base,
                         std::ostream* log) {
    ConfigSimulacion c = configuracion_efectiva(p, o);
    Producto q = p;
    q.sim = c;
    const bool qmc = base.replicas > 0;
    const long long N = base.trayectorias;
    const int R = base.replicas;
    const long long Nper = base.puntos_por_replica;
    auto V = [&](const Producto& x) { return precio_fijo(x, c, qmc, N, R, Nper); };

    Griegas g;
    const double h = g.salto;
    const double V0 = V(q);
    for (size_t i = 0; i < q.activos.size(); ++i) {
        if (log) *log << "  · griegas de " << q.activos[i].nombre << "...\n";
        const double S = q.activos[i].spot;
        Producto up = q, dn = q;
        up.activos[i].spot_ancla = S;
        dn.activos[i].spot_ancla = S;
        up.activos[i].spot = S * (1.0 + h);
        dn.activos[i].spot = S * (1.0 - h);
        double Vu = V(up), Vd = V(dn);
        g.nombres.push_back(q.activos[i].nombre);
        g.delta.push_back(q.nominal * (Vu - Vd) / (2.0 * h * S));
        g.delta_pct.push_back(100.0 * (Vu - Vd) / 2.0 * (0.01 / h));
        g.gamma_pct.push_back(100.0 * (Vu - 2.0 * V0 + Vd) * (0.01 / h) * (0.01 / h));
        Producto vg = q;
        const DefModelo& d = def_modelo(vg.activos[i].modelo.tipo);
        for (const auto& k : d.claves_vol) vg.activos[i].modelo[k] += 0.01;
        g.vega.push_back(100.0 * (V(vg) - V0));
    }
    Producto pr = q;
    for (auto& r : pr.curva.r) r += 0.001;
    g.rho = 100.0 * (V(pr) - V0) * 10.0;
    return g;
}

// ---------------------------------------------------------------------- //
// Informes                                                                //
// ---------------------------------------------------------------------- //

namespace {
std::string linea(char c = '-', int n = 78) { return std::string((size_t)n, c); }
std::string dinero(double x, const Producto& p) { return fmt(x, 2) + " " + p.moneda; }
std::string rellenar_derecha(std::string s, size_t w) {
    // ancho visual aproximado (cuenta caracteres UTF-8, no bytes)
    size_t vis = 0;
    for (unsigned char ch : s) vis += ((ch & 0xC0) != 0x80);
    if (vis < w) s += std::string(w - vis, ' ');
    return s;
}
} // namespace

void imprimir_resultado(std::ostream& os, const Producto& p, const ResultadoValoracion& r) {
    os << "\n" << linea('=') << "\n VALORACIÓN: " << p.nombre << "\n" << linea('=') << "\n";
    double z = 1.959963984540054;
    os << " Precio ................ " << dinero(r.precio * p.nominal, p) << "   (" << porcentaje(r.precio, 2)
       << " del nominal)\n";
    os << " Error estándar ........ " << dinero(r.error * p.nominal, p) << "   (" << porcentaje(r.error, 3)
       << ")    IC 95%: [" << fmt((r.precio - z * r.error) * p.nominal, 2) << ", "
       << fmt((r.precio + z * r.error) * p.nominal, 2) << "]\n";
    os << " Método ................ " << r.metodo;
    if (r.replicas > 0) os << ", " << r.replicas << " réplicas × " << miles(r.puntos_por_replica) << " puntos";
    os << "\n Motor ................. " << r.motor << "\n";
    os << " Trayectorias .......... " << miles(r.trayectorias) << "    Pasos: " << r.pasos;
    if (r.pasos_por_anio > 0) os << " (" << fmt(r.pasos_por_anio, 1) << " por año)";
    else os << " (solo fechas de observación: modelos exactos)";
    os << "    Dimensión: " << r.dimension << "\n";
    os << " Tiempo ................ " << fmt(r.tiempo_s, 3) << " s    Semilla: " << r.semilla << "\n";
    if (!r.objetivo_alcanzado)
        os << " AVISO: no se alcanzó el error objetivo (" << porcentaje(r.error_objetivo, 3)
           << ") dentro del máximo de trayectorias.\n";

    os << "\n Descomposición del valor actual (% del nominal)\n";
    if (!p.cupones.empty())                       os << "   Cupones ......................... " << porcentaje(r.vp_cupones) << "\n";
    if (p.hay_autocall || p.ko_obs != OBS_NINGUNA) os << "   Amortización anticipada ......... " << porcentaje(r.vp_anticipada) << "\n";
    os << "   Amortización a vencimiento ...... " << porcentaje(r.vp_final) << "\n";

    os << "\n Escenarios\n";
    if (p.hay_autocall) {
        double pac = 0;
        for (double x : r.prob_autocall) pac += x;
        os << "   Probabilidad de autocancelación ......... " << porcentaje(pac, 1) << "\n";
    }
    if (p.ko_obs != OBS_NINGUNA) os << "   Probabilidad de desactivación (KO) ...... " << porcentaje(r.prob_ko, 1) << "\n";
    os << "   Vida media esperada ..................... " << fmt(r.vida_media, 2) << " años (máx. " << fmt(p.T, 2) << ")\n";
    if (p.ki_obs != OBS_NINGUNA || p.ki_inicial)
        os << "   Probabilidad de tocar la barrera capital  " << porcentaje(r.prob_ki, 1) << "\n";
    bool hay_capital = false;
    for (const auto& b : p.bloques)
        hay_capital |= (b.tipo == BLQ_FIJO || b.tipo == BLQ_LINEAL || b.tipo == BLQ_FORMULA);
    if (hay_capital) {
        os << "   Prob. de amortizar < 100% a vencimiento   " << porcentaje(r.prob_perdida, 1);
        if (r.prob_perdida > 0) os << "   (pérdida media en ese caso: " << porcentaje(r.perdida_media, 1) << ")";
        os << "\n";
    }

    bool hay_cond = false;
    for (const auto& cu : p.cupones) hay_cond |= (cu.tipo != CUP_FIJO);
    if ((p.hay_autocall || hay_cond) && p.n_obs() > 1) {
        os << "\n Por fecha de observación\n   " << rellenar_derecha("#", 5) << rellenar_derecha("Fecha", 12);
        if (p.hay_autocall) os << rellenar_derecha("P(autocancela)", 16) << rellenar_derecha("P(acumulada)", 14);
        if (hay_cond) os << "P(cobra cupón)";
        os << "\n";
        double acum = 0;
        for (int j = 0; j < p.n_obs(); ++j) {
            acum += (j < (int)r.prob_autocall.size()) ? r.prob_autocall[(size_t)j] : 0.0;
            os << "   " << rellenar_derecha(std::to_string(j + 1), 5) << rellenar_derecha(p.etiqueta_obs[(size_t)j], 12);
            if (p.hay_autocall)
                os << rellenar_derecha(p.ac_activa[(size_t)j] ? porcentaje(r.prob_autocall[(size_t)j], 1) : "-", 16)
                   << rellenar_derecha(porcentaje(acum, 1), 14);
            if (hay_cond) os << porcentaje(r.prob_cupon[(size_t)j], 1);
            os << "\n";
        }
    }
    os << linea('=') << "\n";
}

void imprimir_griegas(std::ostream& os, const Producto& p, const Griegas& g) {
    os << "\n Sensibilidades (diferencias finitas con números aleatorios comunes, salto del spot ±"
       << porcentaje(g.salto, 1) << ")\n";
    os << "   " << rellenar_derecha("Subyacente", 14) << rellenar_derecha("Delta", 16)
       << rellenar_derecha("Delta 1%", 14) << rellenar_derecha("Gamma 1%", 14) << "Vega 1pt\n";
    for (size_t i = 0; i < g.nombres.size(); ++i) {
        os << "   " << rellenar_derecha(g.nombres[i], 14) << rellenar_derecha(fmt(g.delta[i], 4), 16)
           << rellenar_derecha(fmt(g.delta_pct[i], 4) + "%", 14) << rellenar_derecha(fmt(g.gamma_pct[i], 4) + "%", 14)
           << fmt(g.vega[i], 4) << "%\n";
    }
    os << "   Rho (+1% de tipos): " << fmt(g.rho, 4) << "% del nominal\n";
    os << "   Delta = " << p.moneda << " por unidad de subyacente; Delta/Gamma/Vega 1% = variación del precio en\n"
       << "   puntos porcentuales del nominal por +1% del spot / +1 punto de volatilidad.\n";
}

json::Valor resultado_a_json(const Producto& p, const ResultadoValoracion& r, const Griegas* g) {
    json::Valor o = json::Valor::objeto();
    o.poner("producto", p.nombre);
    o.poner("moneda", p.moneda);
    o.poner("nominal", p.nominal);
    o.poner("precio", r.precio * p.nominal);
    o.poner("precio_pct_nominal", r.precio);
    o.poner("error_estandar", r.error * p.nominal);
    o.poner("metodo", r.metodo);
    o.poner("motor", r.motor);
    o.poner("trayectorias", (double)r.trayectorias);
    if (r.replicas > 0) {
        o.poner("replicas_qmc", r.replicas);
        o.poner("puntos_por_replica", (double)r.puntos_por_replica);
    }
    o.poner("pasos", r.pasos);
    o.poner("pasos_por_anio", r.pasos_por_anio);
    o.poner("tiempo_s", r.tiempo_s);
    o.poner("semilla", (double)r.semilla);
    o.poner("objetivo_alcanzado", r.objetivo_alcanzado);
    json::Valor d = json::Valor::objeto();
    d.poner("cupones", r.vp_cupones);
    d.poner("amortizacion_anticipada", r.vp_anticipada);
    d.poner("amortizacion_vencimiento", r.vp_final);
    o.poner("descomposicion_pct_nominal", d);
    json::Valor e = json::Valor::objeto();
    e.poner("vida_media_anios", r.vida_media);
    e.poner("prob_barrera_capital", r.prob_ki);
    e.poner("prob_desactivacion", r.prob_ko);
    e.poner("prob_amortizacion_bajo_par", r.prob_perdida);
    e.poner("perdida_media_condicionada", r.perdida_media);
    o.poner("escenarios", e);
    json::Valor f = json::Valor::lista();
    for (int j = 0; j < p.n_obs(); ++j) {
        json::Valor x = json::Valor::objeto();
        x.poner("fecha", p.etiqueta_obs[(size_t)j]);
        x.poner("t", p.t_obs[(size_t)j]);
        x.poner("prob_autocancelacion", r.prob_autocall[(size_t)j]);
        x.poner("prob_cupon", r.prob_cupon[(size_t)j]);
        f.anadir(x);
    }
    o.poner("por_fecha", f);
    if (g) {
        json::Valor gr = json::Valor::lista();
        for (size_t i = 0; i < g->nombres.size(); ++i) {
            json::Valor x = json::Valor::objeto();
            x.poner("subyacente", g->nombres[i]);
            x.poner("delta", g->delta[i]);
            x.poner("delta_1pct", g->delta_pct[i] / 100.0);
            x.poner("gamma_1pct", g->gamma_pct[i] / 100.0);
            x.poner("vega_1pt", g->vega[i] / 100.0);
            gr.anadir(x);
        }
        o.poner("griegas", gr);
        o.poner("rho_1pct", g->rho / 100.0);
    }
    return o;
}

} // namespace est
