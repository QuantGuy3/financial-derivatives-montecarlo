#include "perfil.hpp"

#include <algorithm>
#include <cmath>

namespace est {

// ------------------------------------------------------------------ //
// PerfilLineal                                                        //
// ------------------------------------------------------------------ //

double PerfilLineal::pendiente_tras(size_t i) const {
    if (i + 1 < nodos.size()) return (nodos[i + 1].izq - nodos[i].der) / (nodos[i + 1].p - nodos[i].p);
    return pend_der;
}

double PerfilLineal::operator()(double P) const {
    const size_t u = nodos.size();
    if (u == 0) return 0.0;
    if (P < nodos[0].p) return nodos[0].izq + pend_izq * (P - nodos[0].p);
    size_t i = (size_t)(std::upper_bound(nodos.begin(), nodos.end(), P,
                                         [](double x, const NodoPerfil& n) { return x < n.p; }) -
                        nodos.begin()) - 1;
    return nodos[i].der + pendiente_tras(i) * (P - nodos[i].p);
}

Descomposicion descomponer(const PerfilLineal& f, double P0) {
    Descomposicion d;
    d.P0 = P0;
    const size_t u = f.nodos.size();
    if (u == 0) return d;

    // Valor y pendiente justo a la izquierda de P0
    size_t ult_menor = u; // último nodo con p < P0
    for (size_t i = 0; i < u; ++i)
        if (f.nodos[i].p < P0) ult_menor = i;
    bool en_nodo = false;
    size_t j_nodo = 0;
    for (size_t i = 0; i < u; ++i)
        if (f.nodos[i].p == P0) { en_nodo = true; j_nodo = i; }
    if (ult_menor == u) d.pendiente = f.pend_izq;
    else d.pendiente = f.pendiente_tras(ult_menor);
    if (en_nodo) d.base = f.nodos[j_nodo].izq;
    else d.base = f(P0);

    // Escala para descartar pesos que son ruido de redondeo
    double escala = std::fabs(d.base) + std::fabs(d.pendiente);
    for (const auto& n : f.nodos) escala = std::max(escala, std::fabs(n.der));
    const double tol = 1e-13 * std::max(1.0, escala);

    for (size_t k = 0; k < u; ++k) {
        const double antes = (k == 0) ? f.pend_izq : f.pendiente_tras(k - 1);
        const double ds = f.pendiente_tras(k) - antes;
        const double J = f.nodos[k].der - f.nodos[k].izq;
        const double K = f.nodos[k].p;
        if (K < P0) {
            if (std::fabs(ds) > tol) d.puts.push_back({K, ds});
            if (std::fabs(J) > tol) d.dig_puts.push_back({K, -J});
        } else {
            if (std::fabs(ds) > tol) d.calls.push_back({K, ds});
            if (std::fabs(J) > tol) d.dig_calls.push_back({K, J});
        }
    }
    return d;
}

PerfilLineal interpolar(const std::function<double(double)>& g, const std::vector<double>& nodos) {
    PerfilLineal f;
    for (double p : nodos) {
        if (!f.nodos.empty() && p <= f.nodos.back().p) continue;
        double v = g(p);
        f.nodos.push_back(NodoPerfil{p, v, v});
    }
    if (f.nodos.size() >= 2) {
        f.pend_izq = f.pendiente_tras(0);
        f.pend_der = (f.nodos[f.nodos.size() - 1].der - f.nodos[f.nodos.size() - 2].der) /
                     (f.nodos[f.nodos.size() - 1].p - f.nodos[f.nodos.size() - 2].p);
    }
    return f;
}

// ------------------------------------------------------------------ //
// Pago final del producto                                             //
// ------------------------------------------------------------------ //

EvaluadorFinal::EvaluadorFinal(const Producto& p) {
    for (const auto& b : p.bloques) {
        BloqueSim<double> x{};
        x.tipo = b.tipo; x.cond = b.cond; x.importe = b.importe; x.part = b.part;
        x.strike = b.strike; x.suelo = b.suelo; x.techo = b.techo;
        x.ini = b.formula_ini; x.len = b.formula_len;
        bloques_.push_back(x);
    }
    sp_.n_bloques = (int)bloques_.size();
    sp_.formulas = p.formulas.empty() ? nullptr : p.formulas.data();
    ki_obs_ = p.ki_obs;
    ki_nivel_ = p.ki_nivel;
    ki_inicial_ = p.ki_inicial;
}

double EvaluadorFinal::con_barrera(double P, int ki) const {
    SimProducto<double> sp = sp_;
    sp.bloques = bloques_.data();
    return amortizacion_final(sp, P, ki, 0.0);
}

double EvaluadorFinal::operator()(double P) const {
    int ki = ki_inicial_ ? 1 : 0;
    if (!ki && ki_obs_ != OBS_NINGUNA && P <= ki_nivel_) ki = 1; // la fecha final siempre se observa
    return con_barrera(P, ki);
}

bool perfil_exacto(const Producto& p, PerfilLineal* out) {
    std::vector<double> nodos;
    bool barrera_cuenta = false;
    for (const auto& b : p.bloques) {
        if (b.tipo == BLQ_FORMULA || b.tipo == BLQ_CLIQUET) return false;
        auto anadir = [&](double x) { if (std::isfinite(x) && std::fabs(x) < HD_INFINITO * 0.5 && x > 0) nodos.push_back(x); };
        switch (b.tipo) {
            case BLQ_LINEAL: anadir(b.suelo); anadir(b.techo); break;
            case BLQ_CALL: anadir(b.strike); anadir(b.techo); break;
            case BLQ_PUT: anadir(b.strike); anadir(b.suelo); break;
            case BLQ_DIGITAL: case BLQ_DIGITAL_PUT: anadir(b.strike); break;
            default: break;
        }
        barrera_cuenta |= (b.cond != COND_SIEMPRE);
    }
    if (barrera_cuenta && !p.ki_inicial && p.ki_obs != OBS_NINGUNA && p.ki_nivel > 0) nodos.push_back(p.ki_nivel);
    std::sort(nodos.begin(), nodos.end());
    std::vector<double> u;
    for (double x : nodos)
        if (u.empty() || x - u.back() > 1e-12 * std::max(1.0, x)) u.push_back(x);

    EvaluadorFinal g(p);
    PerfilLineal f;
    // Recta del tramo (a, b) evaluada en x, a partir de dos puntos interiores
    auto recta = [&](double a, double b, double x) {
        double x1 = a + 0.25 * (b - a), x2 = a + 0.75 * (b - a);
        double y1 = g(x1), y2 = g(x2);
        return y2 + (y2 - y1) / (x2 - x1) * (x - x2);
    };
    auto pendiente = [&](double a, double b) {
        double x1 = a + 0.25 * (b - a), x2 = a + 0.75 * (b - a);
        return (g(x2) - g(x1)) / (x2 - x1);
    };
    if (u.empty()) {
        double v = g(1.0);
        f.nodos.push_back(NodoPerfil{1.0, v, v});
        f.pend_izq = f.pend_der = pendiente(0.5, 1.5);
    } else {
        // P nunca es negativo: el primer tramo empieza en 0 (la función puede tener
        // otro codo en P = 0, p. ej. una curva con extrapolación constante)
        for (size_t k = 0; k < u.size(); ++k) {
            double a = (k == 0) ? 0.0 : u[k - 1];
            double b = (k + 1 < u.size()) ? u[k + 1] : u[k] + std::max(1.0, u[k]);
            f.nodos.push_back(NodoPerfil{u[k], recta(a, u[k], u[k]), recta(u[k], b, u[k])});
        }
        f.pend_izq = pendiente(0.0, u[0]);
        f.pend_der = pendiente(u.back(), u.back() + std::max(1.0, u.back()));
    }
    // Comprobación: el perfil reproduce el pago fuera de los nodos
    double escala = 1.0;
    for (const auto& n : f.nodos) escala = std::max(escala, std::fabs(n.der));
    const double pmax = std::max(3.0, u.empty() ? 3.0 : 1.5 * u.back());
    for (int k = 1; k < 3000; ++k) {
        double P = pmax * (k + 0.37) / 3000.0;
        bool cerca = false;
        for (double x : u) cerca |= std::fabs(P - x) < 1e-9;
        if (cerca) continue;
        if (std::fabs(f(P) - g(P)) > 1e-9 * escala) return false;
    }
    *out = std::move(f);
    return true;
}

} // namespace est
